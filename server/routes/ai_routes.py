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
# Add imports for reminder functionality
from models.reminder import ReminderNLP, Reminder, ReminderDB
from bson.objectid import ObjectId

# Import the Weather functionality
sys.path.append(str(Path(__file__).parent.parent / "Weather"))
from Weather.run import get_weather_response

# Add ConversationQA to the path
conversation_qa_path = Path(__file__).parent.parent / "ConversationQA"
sys.path.append(str(conversation_qa_path))

# Import ConversationQA functionality
from ConversationQA.text_summarization import article_summarize
from ConversationQA.qa_singleton import get_qa_instance

# Get the singleton instance
qa_system = get_qa_instance()

# Add imports for cognitive game functionality
from cognitive_game import CognitiveGame

def ai_routes_funcitons(app, mongo):
    
    
    # set for auth 
    JWT_SECRET = os.getenv("JWT_SECRET", "super-secret-key")
    
    ai_processor = AIProcessor()    

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
        
        if token:
            try:
                # Decode the token
                payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
                return payload.get('user_id')
            except Exception as e:
                print(f"Error decoding token: {str(e)}")
        
        try:
            session_id = request.cookies.get('session_id')
            if session_id:
                user_session = mongo.db.sessions.find_one({"_id": session_id})
                if user_session and 'user_id' in user_session:
                    return user_session['user_id']
        except Exception as e:
            print(f"Error checking session in database: {str(e)}")
            
        # No authentication found
        return None
    

##### Processing internal functions for the ai routes 
    def process_news_function(text):
        """process news query"""
        try:
            
            result, _ = qa_system.process_query(text)
            
            return {"response": result}
        except Exception as e:
            return {"response": f"Error processing news query: {str(e)}"}
    


    def process_reminder_function(text):
        """process reminder query"""
        try:
            reminder_nlp = ReminderNLP()
            if not reminder_nlp:
                return {"response": "The reminder service is not available."}
            
            user_id = get_user_loggedin()

            
            result = reminder_nlp.process_text(text)
            
            # get all predictions from the result
            predicted_intent = result.get("predicted_intent", "")
            predicted_action = result.get("predicted_action", "")
            predicted_time = result.get("predicted_time", "")
            days_offset = result.get("days_offset", 0)
            

            # there are 2 predictions, one is get time table and other is to create an event
            if predicted_intent == "get_timetable":
                
                final_date = datetime.datetime.now(pytz.timezone('Africa/Cairo')) + datetime.timedelta(days=days_offset)
                
                # Get reminders from db
                reminders = ReminderDB.get_day_reminders(user_id, final_date, db=mongo.db)
                
                response = ReminderDB.format_reminders_response(reminders, days_offset)
                
                return {"response": response}
                
            elif predicted_intent == "create_event":
             
                if not predicted_action:
                    return {"response": "I couldn't understand the action, please be more clear."}
                
                start_time = datetime.datetime.now(pytz.timezone('Africa/Cairo')) + datetime.timedelta(days=days_offset)
                
                # Parse the time if provided
                if predicted_time:
                    try:
                        if hasattr(reminder_nlp, 'parse_time'):
                            start_time = reminder_nlp.parse_time(predicted_time, start_time)
                        else:
                            start_time = Reminder.parse_time(predicted_time, start_time)
                    except ValueError as e:
                        return {"response": str(e)}
                else:

                    # set for an hour
                    start_time = start_time.replace(hour=(start_time.hour + 1) % 24, minute=0, second=0, microsecond=0)
                
                # Create the reminder
                try:
                    reminder = ReminderDB.create_reminder(
                        user_id=user_id, 
                        title=predicted_action, 
                        start_time=start_time,
                        description=text,
                        db=mongo.db
                    )
                    
                    new_time = start_time.strftime("%I:%M %p")
                    new_date = start_time.strftime("%A, %B %d")
                    # Choose appropriate time context based on days_offset
                    if days_offset == 0:
                        time_context = "today"
                    elif days_offset == 1:
                        time_context = "tomorrow"
                    else:
                        time_context = f"on {new_date}"
                    
                    response = f"I've added '{predicted_action}' to your reminders for {new_time} {time_context}. It's scheduled for one hour."
                    
                    return {"response": response}

                except Exception as e:
                    return {"response": f"Sorry, I couldn't create your reminder: {str(e)}"}
            
            return {"response": "I couldn't understand your request, please be more clear"}
            
        except Exception as e:
            return {"response": f"Error processing reminder: {str(e)}"}




    def process_weather_frunction(text):
        """process weather query"""
        try:
            
            response = get_weather_response(text)
            return {"response": response}
            
        except Exception as e:
            return {"response": f"Error processing weather query: {str(e)}"}
    



    def process_game_internal(text):
        """process game query"""
        try:

            print("game processing startinggggg")

            # get thelogged in user id and the database contain the questions
            user_id = get_user_loggedin()
            db = app.config.get("DATABASE")

            # There are three actions with three priorities:
            # user would like to stop the game
            # user would like to answer some questions
            # user would like to start game
            if "stop game" in text.lower():
                # Check if game is active
                if not session.get('game_active') or session.get('game_user_id') != user_id:
                    return {"response": "You didn't start a game to stop it."}
                
                final_score = session.get('game_score', 0)
                total_questions = session.get('game_questions_asked', 0)
                
                # Clear game session
                session.pop('game_active', None)
                session.pop('game_user_id', None)
                session.pop('game_score', None)
                session.pop('game_questions_asked', None)
                session.pop('current_question', None)
                
                # Calculate final accuracy percentage
                final_accuracy = round((final_score / total_questions) * 100, 2) if total_questions > 0 else 0
                
                return {"response": f"Game ended! You scored {final_accuracy}% accuracy across {total_questions} questions. Great job exercising your memory!"}
            
            # game is already started
            if session.get('game_active') and session.get('game_user_id') == user_id:
                current_question = session.get('current_question')
                if not current_question:
                    return {"response": "Something went wrong. Please start a new game."}
                
                # init game instance
                game = CognitiveGame(db, user_id)
                result = game.check_answer(current_question, text)
                
                # increment questions asked
                session['game_questions_asked'] = session.get('game_questions_asked', 0) + 1
                
                # Use similarity_score for scoring instead of 'correct' field
                similarity_score = result.get('similarity_score')
                if similarity_score is not None:
                    # Add the similarity score to the total score
                    session['game_score'] = session.get('game_score', 0) + similarity_score
                
                # Generate next question
                next_question = game.generate_random_question()
                session['current_question'] = next_question
                
                # Prepare response
                feedback = result.get('feedback', 'Thank you for your answer!')
                score = session.get('game_score', 0)
                questions_asked = session.get('game_questions_asked', 0)
                # Calculate accuracy as average similarity score (since scores are 0-1)
                accuracy = round((score / questions_asked) * 100, 2) if questions_asked > 0 else 0
                
                response = f"{feedback}. Next question: {next_question}"
                
                return {"response": response}
            
            # Start a new game
            if "start game" in text.lower():
                game = CognitiveGame(db, user_id)
                
                # Check if user has any memory aids
                memory_aids = game.get_all_memory_aids()
                if not memory_aids:
                    return {"response": "You don't have any memory aids yet."}
                
                # Store game session in session storage
                session['game_active'] = True
                session['game_user_id'] = user_id
                session['game_score'] = 0
                session['game_questions_asked'] = 0
                
                # create the first question
                question = game.generate_random_question()
                session['current_question'] = question
                
                response = f"Game started, Let's test your memory! Here's your first question: {question}."
                
                return {"response": response}
            
            return {"response": "say start game to start the game, or stop top stop the game"}
            
        except Exception as e:
            return {"response": f"Error processing game request: {str(e)}"}



    @app.route('/api/ai/process', methods=['POST'])
    def process_ai_request():
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
            
           
            is_game_active = session.get('game_active') and session.get('game_user_id') == user_id            
            is_game = False

            if "game" in text.lower():
                is_game = True

            if is_game_active or is_game:
                game_result = process_game_internal(text)
                
                return jsonify({"response": game_result["response"]})
            
            # init the segmenter
            ai_processor = AIProcessor()
            segments = ai_processor.segment_all_texts(text)
            print("segmenter is initialized")
            print("segments:", segments)
            
            # Check if this is a greeting
            if "greeting" in segments and segments["greeting"]:
                print("Greeting detected")
                return jsonify({"response": "I'm here to help you. What would you like to know?"})
            
            # Process each segment
            responses = {}
            all_responses = ""
            
            if "news" in segments and segments["news"]:
                print("i am in news segment now")
                news_text = " ".join(segments["news"])
                
                news_result = process_news_function(news_text)
                news_response = f"{news_result['response']}"
                
                responses["news"] = news_response
                all_responses += news_response
            
            if "weather" in segments and segments["weather"]:
                print("i am in weather segment now")
                weather_text = " ".join(segments["weather"])
                
                weather_result = process_weather_frunction(weather_text)
                weather_response = weather_result["response"]
                
                responses["weather"] = weather_response
                all_responses += weather_response
            
            if "reminder" in segments and segments["reminder"]:
                print("i am in reminder segment now")
                reminder_text = " ".join(segments["reminder"])

                reminder_result = process_reminder_function(reminder_text)
                reminder_response = reminder_result["response"]

                responses["reminder"] = reminder_response
                all_responses += reminder_response
            
            all_responses = all_responses.strip()
            
            # If no response was generated, provide a default message
            if not all_responses:
                all_responses = "I couldn't find any relevant information for your query."
            
            result = {"response": all_responses}
            
            return jsonify(result)
            
        except Exception as e:
            print(f"Error processing AI request: {str(e)}")
            return jsonify({"error": str(e), "success": False}), 500
    


############## FOR TESTING each model alone #####################
    @app.route('/api/ai/news', methods=['POST'])
    def process_news():
        try:
            data = request.json
            if not data or 'text' not in data:
                return jsonify({"error": "No text provided"}), 400
                
            text = data['text']
            
            result = process_news_function(text)
            return jsonify(result)
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    

    @app.route('/api/ai/weather', methods=['POST'])
    def process_weather():
        try:
            data = request.json
            if not data or 'text' not in data:
                return jsonify({"error": "No text provided"}), 400
                
            text = data['text']
            
            result = get_weather_response(text)
            
            result = {"response": result}
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/ai/reminder', methods=['POST'])
    def process_reminder():
        try:
            data = request.json
            if not data or 'text' not in data:
                return jsonify({"error": "No text provided"}), 400
                
            text = data['text']
            
            result = process_reminder_function(text)
            return jsonify(result)
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    


    ## REMINDER ROUTES
    @app.route('/api/reminder', methods=['GET'])
    def get_reminders():
        try:
            user_id = get_user_loggedin()
 
            days_offset = request.args.get('days_offset', 0, type=int)
            
            target_date = datetime.datetime.now(pytz.timezone('Africa/Cairo')) + datetime.timedelta(days=days_offset)
            
            reminders = ReminderDB.get_day_reminders(user_id, target_date, db=mongo.db)
            
            formatted_reminders = []
            for reminder in reminders:
                reminder['_id'] = str(reminder['_id'])
                reminder['start_time'] = reminder['start_time'].isoformat()
                reminder['end_time'] = reminder['end_time'].isoformat()
                reminder['created_at'] = reminder['created_at'].isoformat()
                formatted_reminders.append(reminder)
            
            return jsonify({"reminders": formatted_reminders})
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/reminder/<reminder_id>', methods=['PUT'])
    def update_reminder(reminder_id):
        try:
            user_id = get_user_loggedin()
            data = request.json
            
            reminder = mongo.db.reminders.find_one({"_id": ObjectId(reminder_id), "user_id": user_id})
            if not reminder:
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
            
            success = ReminderDB.update_reminder(reminder_id, update_data, db=mongo.db)

            if not success:
                return jsonify({"error": "Failed to update reminder"}), 500
            
            return jsonify({"message": "Reminder updated successfully"})
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/reminder/<reminder_id>', methods=['DELETE'])
    def delete_reminder(reminder_id):
        try:
            
            user_id = get_user_loggedin()
            reminder = mongo.db.reminders.find_one({"_id": ObjectId(reminder_id), "user_id": user_id})
            
            if not reminder:
                return jsonify({"error": "Reminder not found"}), 404
            
            # Delete the reminder
            success = ReminderDB.delete_reminder(reminder_id, db=mongo.db)

            if not success:
                return jsonify({"error": "Failed to delete reminder"}), 500
            
            return jsonify({"message": "Reminder deleted successfully"})
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500