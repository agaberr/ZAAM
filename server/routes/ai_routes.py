from flask import Blueprint, jsonify, request, session, url_for
from models.ai_processor import AIProcessor
import sys
import os
from pathlib import Path
import json
import jwt
import datetime
import pytz
import re
import traceback
from models.reminder import ReminderNLP, Reminder, ReminderDB
from bson.objectid import ObjectId
sys.path.append(str(Path(__file__).parent.parent / "Weather"))
from Weather.run import get_weather_response
conversation_qa_path = Path(__file__).parent.parent / "ConversationQA"
sys.path.append(str(conversation_qa_path))
from ConversationQA.text_summarization import article_summarize
from ConversationQA.qa_singleton import get_qa_instance
qa_system = get_qa_instance()
from cognitive_game import CognitiveGame

def ai_routes_funcitons(app, mongo):
    
    JWT_SECRET = os.getenv("JWT_SECRET", "super-secret-key")
    

    def get_user_loggedin():
        auth_header = request.headers.get('Authorization')
        token = None
        
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
        
        if not token:
            if 'token' in session:
                token = session['token']
            elif request.cookies.get('token'):
                token = request.cookies.get('token')
            elif 'user_id' in session:
                return session['user_id']
        
        # decode el token 3shan ageeb kol el data bta3t el user
        if token:
            try:
                payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
                return payload.get('user_id')
            except Exception as e:
                print("can't decode the token, what did you do??")
        
        try:
            session_id = request.cookies.get('session_id')
            if session_id:
                user_session = mongo.db.sessions.find_one({"_id": session_id})
                if user_session and 'user_id' in user_session:
                    return user_session['user_id']
        except :
            print("can't auth user")
            
        # mafeeesh authentication
        return None
    


    def newsmodel_function(text):            
            result, _ = qa_system.process_query(text)
            if not result:
                return {"response": "news service is not available.."}
            return {"response": result}
    


    def remindermodel_function(text):
        try:
            reminder_nlp = ReminderNLP()
            if not reminder_nlp:
                return {"response": "reminder model is not available..."}
            
            user_id = get_user_loggedin()
            res = reminder_nlp.process_text(text)
            
            # get all predictions mn el result of the query
            predicted_intent = res.get("predicted_intent", "")
            predicted_action = res.get("predicted_action", "")
            predicted_time = res.get("predicted_time", "")
            days_offset = res.get("days_offset", 0)
            

            # there are 2 predictions, el awal is get time table and el tany  is to create an event
            if predicted_intent == "get_timetable":
                
                final_date = datetime.datetime.now(pytz.timezone('Africa/Cairo')) + datetime.timedelta(days=days_offset)
                
                reminders = ReminderDB.getDayReminders(user_id, final_date, db=mongo.db)
                response = ReminderDB.formatReminderResponse(reminders, days_offset)
                
                return {"response": response}
                
            elif predicted_intent == "create_event":
             
                if not predicted_action:
                    return {"response": "I couldn't understand the action, please be more clear."}
                
                start_time = datetime.datetime.now(pytz.timezone('Africa/Cairo')) + datetime.timedelta(days=days_offset)

                if predicted_time:
                    try:
                        if hasattr(reminder_nlp, 'parse_time'):
                            start_time = reminder_nlp.parse_time(predicted_time, start_time)
                        else:
                            start_time = Reminder.parse_time(predicted_time, start_time)
                    except:
                        return {"response": "error parsing the time.."}
                else:

                    # el default value bta3 el duration byab2a 1 hr
                    start_time = start_time.replace(hour=(start_time.hour + 1) % 24, minute=0, second=0, microsecond=0)
                
                    reminder = ReminderDB.createReminder(
                        user_id=user_id, 
                        title=predicted_action, 
                        start_time=start_time,
                        description=text,
                        db=mongo.db
                    )
                try:
                    
                    new_time = start_time.strftime("%I:%M %p")
                    new_date = start_time.strftime("%A, %B %d")

                    # make days offset tb2a readable by human
                    if days_offset == 0:
                        time_context = "today"
                    elif days_offset == 1:
                        time_context = "tomorrow"
                    else:
                        time_context = f"on {new_date}"
                    
                    response = f"I've added '{predicted_action}' to your reminders for {new_time} {time_context}. It's scheduled for one hour."
                    
                    return {"response": response}

                except:
                    return {"response": f"Sorry, I couldn't create your reminder"}
            
            return {"response": "I couldn't understand your request, please be more clear"}
            
        except:
            return {"response": "error occured processsing the reminder..."}




    def weathermodel_function(text):
        try:
            
            response = get_weather_response(text)
            return {"response": response}
            
        except:
            return {"response": "error processing the weather quey.."}
    



    def gamemodel_function(text):
        try:

            user_id = get_user_loggedin()
            db = app.config.get("DATABASE")

            ans = text.strip().lower()

            if "stop game" in ans:

                # check hal el game already active aslun walla la
                if not session.get('game_active') or session.get('game_user_id') != user_id:
                    return {"response": "You didn't start a game to stop it."}
                
                totalScore = session.get('game_score', 0)
                allQ = session.get('game_questions_asked', 0)
                
                for key in ['game_active', 'game_user_id', 'game_score', 'game_questions_asked', 'current_question']:
                    session.pop(key, None)
                
                return {"response": f"Game ended! You scored {totalScore} out of {allQ} questions. Great job exercising your memory!"}
            
            if session.get('game_active') and session.get('game_user_id') == user_id:
                currQ = session.get('current_question')
                if not currQ:
                    return {"response": "error happened.. please start a new game."}
                
                game = CognitiveGame(db, user_id)
                result = game.check_answer(currQ, text)
                
                session['game_questions_asked'] = session.get('game_questions_asked', 0) + 1
                
                if result.get('correct', False):
                    session['game_score'] = session.get('game_score', 0) + 1
                
                next_question = game.generate_random_question()
                session['current_question'] = next_question
                
                fb = result.get('feedback', 'Thank you for your answer!')
                
                response = f"{fb}. Next question: {next_question}"
                
                return {"response": response}
            
            if "start game" in ans:
                game = CognitiveGame(db, user_id)
                
                memory_aids = game.get_all_memory_aids()
                if not memory_aids:
                    return {"response": "There is no memory aids"}
                
                # zabat el session storage data 
                session['game_active'] = True
                session['game_user_id'] = user_id
                session['game_score'] = 0
                session['game_questions_asked'] = 0
                
                q = game.generate_random_question()
                session['current_question'] = q

                return {"response": f"Game started, Let's test your memory! {q}."}
            
            return {"response": "Try saying start game to sstart..."}
            
       
        except:
            return {"response": "Something broke..."}



    @app.route('/api/ai/process', methods=['POST'])
    def modelRouter():
        """
        This is the main router where the text is segmentted and then wach segment is
        processed to the corresponding model
        then it concat all the result and return them
        """
        try:
            data = request.json
            if not data or 'text' not in data:
                return jsonify({"error": "No text provided"}), 400
                
            text = data['text']
            print("response text: ", text)
            
            user_id = get_user_loggedin()
            
           
            isgameactive = session.get('game_active') and session.get('game_user_id') == user_id            
            is_game = False

            if "game" in text.lower():
                is_game = True

            if isgameactive or is_game:
                game_result = gamemodel_function(text)
                
                return jsonify({"response": game_result["response"]})
            
            ai_processor = AIProcessor()
            segments = ai_processor.segmentAllTxt(text)
            print("segmenter is initialized")
            
            
            responses = {}
            allresponses = ""
            
            if "news" in segments and segments["news"]:
                print("i am in news segment now")
                news_text = " ".join(segments["news"])
                
                news_result = newsmodel_function(news_text)
                news_response = f"{news_result['response']}"
                
                responses["news"] = news_response
                allresponses += news_response
            
            if "weather" in segments and segments["weather"]:
                print("i am in weather segment now")
                weather_text = " ".join(segments["weather"])
                
                weather_result = weathermodel_function(weather_text)
                weather_response = weather_result["response"]
                
                responses["weather"] = weather_response
                allresponses += weather_response
            
            if "reminder" in segments and segments["reminder"]:
                print("i am in reminder segment now")
                reminder_text = " ".join(segments["reminder"])

                reminder_result = remindermodel_function(reminder_text)
                reminder_response = reminder_result["response"]

                responses["reminder"] = reminder_response
                allresponses += reminder_response
            
            allresponses = allresponses.strip()
            
            result = {"response": allresponses}
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    


############## FOR TESTING each model alone #####################
    @app.route('/api/ai/news', methods=['POST'])
    def newsmodel():
        try:
            data = request.json
            if not data or 'text' not in data:
                return jsonify({"error": "No text provided"}), 400
                
            text = data['text']
            
            result = newsmodel_function(text)
            return jsonify(result)
            
        except:
            return jsonify({"error": "error with news model"}), 500
    

    @app.route('/api/ai/weather', methods=['POST'])
    def weathermodel():
        try:
            data = request.json
            if not data or 'text' not in data:
                return jsonify({"error": "No text provided"}), 400
                
            text = data['text']
            
            result = get_weather_response(text)
            
            result = {"response": result}
            
            return jsonify(result)
            
        except:
            return jsonify({"error": "error in weather model"}), 500
    
    @app.route('/api/ai/reminder', methods=['POST'])
    def remindermodel():
        try:
            data = request.json
            if not data or 'text' not in data:
                return jsonify({"error": "write some text.."}), 400
                
            text = data['text']
            
            res = remindermodel_function(text)
            return jsonify(res)
            
        except:
            return jsonify({"error": "can't process reminders"}), 500
    


    ## REMINDER ROUTES
    @app.route('/api/reminder', methods=['GET'])
    def getrem():
        try:
            userid = get_user_loggedin()
 
            daysoffset = request.args.get('days_offset', 0, type=int)
            
            date = datetime.datetime.now(pytz.timezone('Africa/Cairo')) + datetime.timedelta(days=daysoffset)
            
            reminders = ReminderDB.getDayReminders(userid, date, db=mongo.db)
            
            remindersOutput = []
            for reminder in reminders:
                reminder['_id'] = str(reminder['_id'])
                reminder['start_time'] = reminder['start_time'].isoformat()
                reminder['end_time'] = reminder['end_time'].isoformat()
                reminder['created_at'] = reminder['created_at'].isoformat()
                
                remindersOutput.append(reminder)
            
            return jsonify({"reminders": remindersOutput})
            
        except:
            return jsonify({"error": "error in getting reminders"}), 500
    
    @app.route('/api/reminder/<reminder_id>', methods=['PUT'])
    def updaterem(reminder_id):
        userID = get_user_loggedin()
        data = request.json
        
        rem = mongo.db.reminders.find_one({"_id": ObjectId(reminder_id), "user_id": userID})
        if not rem:
            return jsonify({"error": "Reminder not found"}), 404
        
        update_data = {}
        if 'title' in data:
            update_data['title'] = data['title']
        if 'start_time' in data:
            update_data['start_time'] = datetime.datetime.fromisoformat(data['start_time'])
        if 'end_time' in data:
            update_data['end_time'] = datetime.datetime.fromisoformat(data['end_time'])
        if 'description' in data:
            update_data['description'] = data['description']
        if 'completed' in data:
            update_data['completed'] = data['completed']
        
        res = ReminderDB.update_reminder(reminder_id, update_data, db=mongo.db)

        if not res:
            return jsonify({"error": "Can't update it"}), 500
        
        return jsonify({"message": "Reminder updated..."})
    

    @app.route('/api/reminder/<reminder_id>', methods=['DELETE'])
    def deleterem(reminder_id):
            
        id = get_user_loggedin()
        rem = mongo.db.reminders.find_one({"_id": ObjectId(reminder_id), "user_id": id})
        
        if not rem:
            return jsonify({"error": "Reminder not found"}), 404
        
        res = ReminderDB.delete_reminder(reminder_id, db=mongo.db)

        if not res:
            return jsonify({"error": "Can't delete this"}), 500
        
        return jsonify({"message": "Reminder deleted..."})
    