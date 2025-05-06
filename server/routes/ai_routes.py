from flask import Blueprint, jsonify, request, session, url_for
from models.ai_processor import AIProcessor
import sys
import os
from pathlib import Path
import logging
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Add ConversationQA to the path
conversation_qa_path = Path(__file__).parent.parent / "ConversationQA"
sys.path.append(str(conversation_qa_path))

# Import ConversationQA functionality
from ConversationQA.text_summarization import article_summarize
from ConversationQA.qa_singleton import get_qa_instance

# Get the singleton instance
qa_system = get_qa_instance()

def register_ai_routes(app, mongo):
    """Register AI processing routes."""
    
    print("\n===== AI ROUTES INITIALIZATION =====")
    
    # Debug environment variables
    flask_env = os.getenv("FLASK_ENV", "production")
    print(f"[DEBUG] Current FLASK_ENV: {flask_env}")
    print(f"[DEBUG] JWT_SECRET present: {'Yes' if os.getenv('JWT_SECRET') else 'No'}")
    
    # TEMPORARY FIX: If FLASK_ENV is not set, set it to development for testing
    if not os.getenv("FLASK_ENV"):
        os.environ["FLASK_ENV"] = "development"
        print("[DEBUG] FLASK_ENV not set, temporarily forcing to 'development' mode")
    
    # Set JWT_SECRET for authentication
    JWT_SECRET = os.getenv("JWT_SECRET", "super-secret-key")
    print(f"[DEBUG] JWT_SECRET length: {len(JWT_SECRET)} characters")
    
    # Initialize the AI processor
    ai_processor = AIProcessor()
    print("[DEBUG] AI Processor initialized")
    
    # Initialize ReminderNLP model
    try:
        reminder_nlp = ReminderNLP()
        print("[DEBUG] ReminderNLP model initialized")
    except Exception as e:
        logger.error(f"Error initializing ReminderNLP: {str(e)}")
        reminder_nlp = None
        print(f"[DEBUG] Failed to initialize ReminderNLP: {str(e)}")
    
    print("===== END OF AI ROUTES INITIALIZATION =====\n")
    
    # Helper function to get authenticated user ID
    def get_authenticated_user_id():
        # First try to get the token from the Authorization header
        auth_header = request.headers.get('Authorization')
        token = None
        
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
        
        # If no token in header, try to get from session or cookies
        if not token:
            # Try to get from session
            if 'token' in session:
                token = session['token']
            # Try to get from cookies
            elif request.cookies.get('token'):
                token = request.cookies.get('token')
            # Check for user_id directly in session
            elif 'user_id' in session:
                return session['user_id']
        
        # If we have a token, decode it
        if token:
            try:
                # Decode the token
                payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
                return payload.get('user_id')
            except Exception as e:
                logger.error(f"Error decoding token: {str(e)}")
        
        # FALLBACK: Try to find currently logged in user in database
        try:
            # If there's a session ID in the cookie, try to get the user from there
            session_id = request.cookies.get('session_id')
            if session_id:
                user_session = mongo.db.sessions.find_one({"_id": session_id})
                if user_session and 'user_id' in user_session:
                    return user_session['user_id']
        except Exception as e:
            logger.error(f"Error checking session in database: {str(e)}")
        
        # For development, temporarily use a hardcoded user ID for testing - REMOVE IN PRODUCTION
        if os.getenv("FLASK_ENV") == "development":
            print("[DEBUG] No authentication found - using development fallback")
            # For development only - get the first user from the database
            try:
                default_user = mongo.db.users.find_one({})
                if default_user:
                    logger.warning("Using first user in database as fallback (DEVELOPMENT ONLY)")
                    return str(default_user["_id"])
            except Exception as e:
                logger.error(f"Error getting default user: {str(e)}")
            
        # No authentication found
        return None
    
    # Internal function to process news queries
    def process_news_internal(text):
        """Internal function to process news queries, used by the main AI route"""
        try:
            print(f"[DEBUG] process_news_internal called with: '{text}'")
            
            # For short texts, treat as a query about news
            answer, need_new_passage = qa_system.process_query(text)
            
            # Get the last turn from conversation history
            last_turn = qa_system.conversation_history[-1] if qa_system.conversation_history else {}
            confidence = last_turn.get('confidence', 0)
            
            print(f"[DEBUG] QA answer: '{answer}', confidence: {confidence}")
            
            result = {
                "response": answer,
                "category": "news",
                "success": True,
                "confidence": round(confidence, 2),
                "debug": {
                    "original_query": last_turn.get('query', text),
                    "resolved_query": last_turn.get('resolved_query', text),
                    "needed_new_passage": bool(need_new_passage)
                }
            }
            return result
        except Exception as e:
            logger.error(f"Error in process_news_internal: {str(e)}")
            print(f"[DEBUG] Error in process_news_internal: {str(e)}")
            return {"response": f"Error processing news query: {str(e)}", "success": False}
    
    # Internal function to process reminder queries
    def process_reminder_internal(text):
        """Internal function to process reminder-related queries"""
        try:
            print(f"[DEBUG] process_reminder_internal called with: '{text}'")
            
            # Check if reminder model is initialized
            if not reminder_nlp:
                return {"response": "Sorry, the reminder service is not available at the moment.", "success": False}
            
            # Get the authenticated user ID
            user_id = get_authenticated_user_id()
            if not user_id:
                return {
                    "response": "You must be logged in to use the reminder service.",
                    "category": "reminder",
                    "success": False,
                    "needs_auth": True
                }
            
            # Process the text with NLP model
            result = reminder_nlp.process_text(text)
            print(f"[DEBUG] NLP processing result: {result}")
            
            # Extract information from the result
            predicted_intent = result.get("predicted_intent", "")
            action = result.get("predicted_action", "")
            time_str = result.get("predicted_time", "")
            days_offset = result.get("days_offset", 0)
            
            if predicted_intent == "get_timetable":
                # Get reminders for the specified day
                egypt_tz = pytz.timezone('Africa/Cairo')
                target_date = datetime.datetime.now(egypt_tz) + datetime.timedelta(days=days_offset)
                
                # Get reminders from database
                reminders = ReminderDB.get_day_reminders(user_id, target_date, db=mongo.db)
                
                # Format the response
                response = ReminderDB.format_reminders_response(reminders, days_offset)
                
                return {
                    "response": response,
                    "category": "reminder",
                    "success": True,
                    "intent": "get_timetable",
                    "days_offset": days_offset,
                    "reminders_count": len(reminders)
                }
                
            elif predicted_intent == "create_event":
                # Extract a better title from the original text if action is incomplete
                if not action or len(action.split()) < 2:
                    # Try to extract a meaningful title from the original text
                    # Remove common phrases like "remind me to", "set a reminder to", etc.
                    cleaned_text = re.sub(r'^(remind me to|set a reminder to|remind me|set reminder|remind)\s+', '', text.lower())
                    
                    # Remove time related parts
                    time_pattern = r'\b(at|on|for|by)\s+\d{1,2}(?::)?\d{0,2}\s*(am|pm|a\.m\.|p\.m\.)\b'
                    cleaned_text = re.sub(time_pattern, '', cleaned_text)
                    
                    # Remove date related parts
                    for date_exp in reminder_nlp.TIME_EXPRESSIONS:
                        if date_exp in cleaned_text.lower():
                            cleaned_text = re.sub(r'\b' + re.escape(date_exp) + r'\b', '', cleaned_text, flags=re.IGNORECASE)
                    
                    # Clean up any duplicate spaces
                    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
                    
                    # Use this as the action if it's better than what we had
                    if len(cleaned_text) > len(action):
                        action = cleaned_text
                
                # Validate the action (reminder title)
                if not action:
                    return {
                        "response": "I couldn't understand what reminder to create. Could you please be more specific?",
                        "category": "reminder",
                        "success": False,
                        "intent": "create_event"
                    }
                
                # Use Egypt timezone
                egypt_tz = pytz.timezone('Africa/Cairo')
                start_time = datetime.datetime.now(egypt_tz) + datetime.timedelta(days=days_offset)
                
                # Parse the time if provided
                if time_str:
                    try:
                        # If we have our own parse_time method in the ReminderNLP class, use it
                        if hasattr(reminder_nlp, 'parse_time'):
                            start_time = reminder_nlp.parse_time(time_str, start_time)
                        else:
                            # Otherwise use the Reminder class method
                            start_time = Reminder.parse_time(time_str, start_time)
                    except ValueError as e:
                        return {
                            "response": str(e),
                            "category": "reminder",
                            "success": False,
                            "intent": "create_event"
                        }
                else:
                    # Default time if none provided - set to next hour
                    current_hour = start_time.hour
                    start_time = start_time.replace(hour=(current_hour + 1) % 24, minute=0, second=0, microsecond=0)
                
                # Create the reminder
                try:
                    reminder = ReminderDB.create_reminder(
                        user_id=user_id, 
                        title=action, 
                        start_time=start_time,
                        description=f"Created from voice command: '{text}'",
                        db=mongo.db
                    )
                    
                    # Format the response
                    formatted_time = start_time.strftime("%I:%M %p")
                    formatted_date = start_time.strftime("%A, %B %d")
                    
                    # Choose appropriate time context based on days_offset
                    if days_offset == 0:
                        time_context = "today"
                    elif days_offset == 1:
                        time_context = "tomorrow"
                    else:
                        time_context = f"on {formatted_date}"
                    
                    response = f"Perfect! I've added '{action}' to your reminders for {formatted_time} {time_context}. It's scheduled for one hour."
                    
                    return {
                        "response": response,
                        "category": "reminder",
                        "success": True,
                        "intent": "create_event",
                        "reminder": {
                            "id": str(reminder["_id"]),
                            "title": action,
                            "time": formatted_time,
                            "date": formatted_date
                        }
                    }
                except Exception as e:
                    logger.error(f"Error creating reminder: {str(e)}")
                    return {
                        "response": f"Sorry, I couldn't create your reminder: {str(e)}",
                        "category": "reminder",
                        "success": False,
                        "intent": "create_event"
                    }
            
            # If intent is not recognized
            return {
                "response": "I'm not quite sure what you'd like me to do with your reminders. Could you please rephrase that?",
                "category": "reminder",
                "success": False,
                "recognized": False
            }
            
        except Exception as e:
            logger.error(f"Error in process_reminder_internal: {str(e)}")
            print(f"[DEBUG] Error in process_reminder_internal: {str(e)}")
            return {"response": f"Error processing reminder: {str(e)}", "success": False}

    # Internal function to process weather-related queries using Weather module
    def process_weather_internal(text):
        """Internal function to process weather-related queries using the Weather module"""
        try:
            print(f"[DEBUG] process_weather_internal called with: '{text}'")
            
            # Use get_weather_response function from Weather module
            weather_response = get_weather_response(text)
            
            result = {
                "response": weather_response,
                "category": "weather",
                "success": True
            }
            return result
            
        except Exception as e:
            logger.error(f"Error in process_weather_internal: {str(e)}")
            print(f"[DEBUG] Error in process_weather_internal: {str(e)}")
            return {"response": f"Error processing weather query: {str(e)}", "success": False}
    
    @app.route('/api/ai/process', methods=['POST'])
    def process_ai_request():
        """
        Process a natural language input and categorize it.
        
        This route handles general queries by categorizing them as news, weather, or other types,
        and processes each category accordingly.
        """
        try:
            data = request.json
            if not data or 'text' not in data:
                return jsonify({"error": "No text provided in request"}), 400
                
            text = data['text']
            print(f"[DEBUG] AI processing request: '{text}'")
            
            # Get authentication info for advanced processing
            user_id = get_authenticated_user_id()
            print(f"[DEBUG] User ID: {user_id}")
            
            # Categorize text into different segments
            ai_processor = AIProcessor()
            segments = ai_processor.segment_sentences(text)
            
            print(f"[DEBUG] Categorized segments: {segments.keys()}")
            
            # Process each category
            responses = {}
            combined_response = ""
            
            # Process news segments
            if "news" in segments and segments["news"]:
                news_text = " ".join(segments["news"])
                print(f"[DEBUG] Processing news segment: '{news_text}'")
                
                news_result = process_news_internal(news_text)
                news_response = f"NEWS: {news_result['response']}"
                
                responses["news"] = news_response
                combined_response += news_response + "\n\n"
            
            # Process weather segments
            if "weather" in segments and segments["weather"]:
                weather_text = " ".join(segments["weather"])
                print(f"[DEBUG] Processing weather segment: '{weather_text}'")
                
                weather_result = process_weather_internal(weather_text)
                weather_response = weather_result["response"]
                
                responses["weather"] = weather_response
                combined_response += weather_response + "\n\n"
            
            # Process reminder segments
            if "reminder" in segments and segments["reminder"]:
                reminder_text = " ".join(segments["reminder"])
                print(f"[DEBUG] Processing reminder segment: '{reminder_text}'")
                
                reminder_result = process_reminder_internal(reminder_text)
                reminder_response = reminder_result["response"]
                
                responses["reminder"] = reminder_response
                combined_response += reminder_response + "\n\n"
            
            # Process general segments
            if "general" in segments and segments["general"]:
                general_text = " ".join(segments["general"])
                print(f"[DEBUG] Processing general segment as news: '{general_text}'")
                
                general_result = process_news_internal(general_text)
                general_response = general_result["response"]
                
                responses["general"] = general_response
                if not combined_response:  # Only add if no other responses
                    combined_response += general_response
            
            # Remove any trailing newlines
            combined_response = combined_response.strip()
            
            result = {
                "response": combined_response,
                "category_responses": responses,
                "categories": list(responses.keys()),
                "success": True
            }
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Error processing AI request: {str(e)}")
            return jsonify({"error": str(e), "success": False}), 500
    
    @app.route('/api/ai/news', methods=['POST'])
    def process_news():
        """Process news-specific requests."""
        try:
            data = request.json
            if not data or 'text' not in data:
                print("[DEBUG] /api/ai/news: No text provided in request")
                return jsonify({"error": "No text provided in request"}), 400
                
            text = data['text']
            print(f"[DEBUG] /api/ai/news received: '{text}'")
            
            # Reuse the internal news processing function
            result = process_news_internal(text)
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Error processing news request: {str(e)}")
            print(f"[DEBUG] Error in /api/ai/news: {str(e)}")
            return jsonify({"error": str(e), "success": False}), 500
    
    @app.route('/api/ai/weather', methods=['POST'])
    def process_weather():
        """Process weather-specific requests using the Weather module."""
        try:
            data = request.json
            if not data or 'text' not in data:
                print("[DEBUG] /api/ai/weather: No text provided in request")
                return jsonify({"error": "No text provided in request"}), 400
                
            text = data['text']
            print(f"[DEBUG] /api/ai/weather received: '{text}'")
            
            # Get weather response directly from Weather module
            weather_response = get_weather_response(text)
            
            result = {
                "response": weather_response,
                "category": "weather",
                "success": True
            }
            
            return jsonify(result)
            
        except Exception as e:
            print(f"[DEBUG] Error in /api/ai/weather: {str(e)}")
            return jsonify({"error": str(e), "success": False}), 500
    
    @app.route('/api/ai/reminder', methods=['POST'])
    def process_reminder():
        """Process reminder-specific requests."""
        try:
            data = request.json
            if not data or 'text' not in data:
                print("[DEBUG] /api/ai/reminder: No text provided in request")
                return jsonify({"error": "No text provided in request"}), 400
                
            text = data['text']
            print(f"[DEBUG] /api/ai/reminder received: '{text}'")
            
            # Reuse the internal reminder processing function
            result = process_reminder_internal(text)
            return jsonify(result)
            
        except Exception as e:
            print(f"[DEBUG] Error in /api/ai/reminder: {str(e)}")
            return jsonify({"error": str(e), "success": False}), 500
     
    @app.route('/api/reminder/check_upcoming', methods=['GET'])
    def check_upcoming_reminders():
        """
        Check for reminders scheduled in the next 15 minutes and send notifications.
        This endpoint is designed to be called by a scheduler every minute.
        """
        try:
            print("[DEBUG] ===== Checking for upcoming reminders =====")
            
            # Get current time with Egypt timezone
            egypt_tz = pytz.timezone('Africa/Cairo')
            now = datetime.datetime.now(egypt_tz)
            fifteen_min_later = now + datetime.timedelta(minutes=15)
            
            # Debug
            print(f"[DEBUG] Current time (Cairo): {now.strftime('%Y-%m-%d %H:%M:%S %Z%z')}")
            print(f"[DEBUG] 15 min later (Cairo): {fifteen_min_later.strftime('%Y-%m-%d %H:%M:%S %Z%z')}")
            
            # For debugging, get all reminders for today to see what's available
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            today_end = now.replace(hour=23, minute=59, second=59, microsecond=999999)
            
            all_day_reminders = list(mongo.db.reminders.find({
                "start_time": {"$gte": today_start, "$lte": today_end},
                "status": "active"
            }))
            
            print(f"[DEBUG] Found {len(all_day_reminders)} total reminders for today")
            for r in all_day_reminders:
                r_time = r.get('start_time')
                if r_time:
                    # Ensure the time is timezone-aware in Egypt time
                    if r_time.tzinfo is None:
                        r_time = egypt_tz.localize(r_time)
                    elif r_time.tzinfo != egypt_tz:
                        r_time = r_time.astimezone(egypt_tz)
                    
                    # Calculate time until reminder
                    time_diff_minutes = (r_time - now).total_seconds() / 60
                    time_diff_str = f"{time_diff_minutes:.1f} min" if time_diff_minutes >= 0 else "past"
                    
                    print(f"[DEBUG] Reminder: '{r.get('title')}' at {r_time.strftime('%Y-%m-%d %H:%M:%S %Z%z')} ({time_diff_str} from now)")
            
            # Get all users from the database
            users = list(mongo.db.users.find({}, {"_id": 1}))
            print(f"[DEBUG] Found {len(users)} users to check reminders for")
            
            upcoming_reminders = []
            notifications_sent = 0
            
            # Check each user's reminders
            for user in users:
                user_id = str(user["_id"])
                
                # Find reminders that are due in the next 15 minutes
                user_reminders = ReminderDB.get_reminders(
                    user_id=user_id,
                    time_min=now,
                    time_max=fifteen_min_later,
                    max_results=10,
                    db=mongo.db
                )
                
                print(f"[DEBUG] Found {len(user_reminders)} upcoming reminders for user {user_id}")
                
                # Add valid reminders to our list
                for reminder in user_reminders:
                    start_time = reminder.get('start_time')
                    
                    # Ensure timezone is set to Egypt time
                    if start_time.tzinfo is None:
                        start_time = egypt_tz.localize(start_time)
                        reminder['start_time'] = start_time
                    elif start_time.tzinfo != egypt_tz:
                        start_time = start_time.astimezone(egypt_tz)
                        reminder['start_time'] = start_time
                    
                    # Calculate minutes until reminder is due
                    time_diff = (start_time - now).total_seconds() / 60
                    
                    # Only include reminders that are actually in the next 15 minutes
                    if 0 <= time_diff <= 15:
                        print(f"[DEBUG] Including reminder '{reminder.get('title')}' at {start_time.strftime('%H:%M:%S')} ({time_diff:.1f} min from now)")
                        upcoming_reminders.append(reminder)
                    else:
                        print(f"[DEBUG] Excluding reminder '{reminder.get('title')}' at {start_time.strftime('%H:%M:%S')} ({time_diff:.1f} min from now)")
            
            print(f"[DEBUG] Found {len(upcoming_reminders)} total valid upcoming reminders")
            
            # Process each upcoming reminder
            for reminder in upcoming_reminders:
                user_id = reminder.get('user_id')
                reminder_title = reminder.get('title', 'Untitled reminder')
                start_time = reminder.get('start_time')
                
                # Format the time
                time_str = start_time.strftime("%I:%M %p")
                
                # Format the notification text
                notification_text = f"You have a reminder '{reminder_title}' at {time_str} (in the next 15 minutes), be prepared."
                
                # Send the notification to the AI reminder endpoint
                try:
                    # Save notification in the database
                    mongo.db.notifications.insert_one({
                        'user_id': user_id,
                        'text': notification_text,
                        'reminder_id': reminder['_id'],
                        'created_at': now,
                        'read': False,
                        'type': 'reminder_alert'
                    })
                    
                    notifications_sent += 1
                    print(f"[DEBUG] Sent notification for reminder '{reminder_title}' to user {user_id}")
                    
                except Exception as e:
                    logger.error(f"Error sending notification for reminder {reminder['_id']}: {str(e)}")
            
            print("[DEBUG] ===== Reminder check completed =====")
            
            return jsonify({
                "success": True,
                "upcoming_reminders": len(upcoming_reminders),
                "notifications_sent": notifications_sent,
                "current_time": now.strftime("%Y-%m-%d %H:%M:%S %Z%z"),
                "time_window_end": fifteen_min_later.strftime("%Y-%m-%d %H:%M:%S %Z%z"),
            })
            
        except Exception as e:
            logger.error(f"Error checking upcoming reminders: {str(e)}")
            print(f"[DEBUG] Error checking upcoming reminders: {str(e)}")
            return jsonify({"error": str(e), "success": False}), 500
    
    @app.route('/api/reminder', methods=['GET'])
    def get_reminders():
        """Get user's reminders for a specific day"""
        try:
            user_id = get_authenticated_user_id()
            if not user_id:
                return jsonify({"error": "Authentication required", "success": False}), 401
            
            # Parse query parameters
            days_offset = request.args.get('days_offset', 0, type=int)
            
            # Get target date
            egypt_tz = pytz.timezone('Africa/Cairo')
            target_date = datetime.datetime.now(egypt_tz) + datetime.timedelta(days=days_offset)
            
            # Get reminders from database
            reminders = ReminderDB.get_day_reminders(user_id, target_date, db=mongo.db)
            
            # Convert ObjectId to string for JSON serialization
            formatted_reminders = []
            for reminder in reminders:
                reminder['_id'] = str(reminder['_id'])
                # Convert datetime objects to ISO format
                reminder['start_time'] = reminder['start_time'].isoformat()
                reminder['end_time'] = reminder['end_time'].isoformat()
                reminder['created_at'] = reminder['created_at'].isoformat()
                formatted_reminders.append(reminder)
            
            return jsonify({
                "success": True,
                "reminders": formatted_reminders,
                "count": len(formatted_reminders),
                "date": target_date.strftime("%Y-%m-%d")
            })
            
        except Exception as e:
            logger.error(f"Error getting reminders: {str(e)}")
            return jsonify({"error": str(e), "success": False}), 500
    
    @app.route('/api/reminder/<reminder_id>', methods=['PUT'])
    def update_reminder(reminder_id):
        """Update a specific reminder"""
        try:
            user_id = get_authenticated_user_id()
            if not user_id:
                return jsonify({"error": "Authentication required", "success": False}), 401
            
            data = request.json
            if not data:
                return jsonify({"error": "No data provided", "success": False}), 400
            
            # Ensure the reminder belongs to the user
            reminder = mongo.db.reminders.find_one({"_id": ObjectId(reminder_id), "user_id": user_id})
            if not reminder:
                return jsonify({"error": "Reminder not found", "success": False}), 404
            
            # Prepare update data
            update_data = {}
            if 'title' in data:
                update_data['title'] = data['title']
            if 'start_time' in data:
                update_data['start_time'] = datetime.datetime.fromisoformat(data['start_time'])
            if 'end_time' in data:
                update_data['end_time'] = datetime.datetime.fromisoformat(data['end_time'])
            if 'description' in data:
                update_data['description'] = data['description']
            
            # Update the reminder
            success = ReminderDB.update_reminder(reminder_id, update_data, db=mongo.db)
            
            return jsonify({
                "success": success,
                "message": "Reminder updated successfully" if success else "Failed to update reminder"
            })
            
        except Exception as e:
            logger.error(f"Error updating reminder: {str(e)}")
            return jsonify({"error": str(e), "success": False}), 500
    
    @app.route('/api/reminder/<reminder_id>', methods=['DELETE'])
    def delete_reminder(reminder_id):
        """Delete a specific reminder"""
        try:
            user_id = get_authenticated_user_id()
            if not user_id:
                return jsonify({"error": "Authentication required", "success": False}), 401
            
            # Ensure the reminder belongs to the user
            reminder = mongo.db.reminders.find_one({"_id": ObjectId(reminder_id), "user_id": user_id})
            if not reminder:
                return jsonify({"error": "Reminder not found", "success": False}), 404
            
            # Delete the reminder
            success = ReminderDB.delete_reminder(reminder_id, db=mongo.db)
            
            return jsonify({
                "success": success,
                "message": "Reminder deleted successfully" if success else "Failed to delete reminder"
            })
            
        except Exception as e:
            logger.error(f"Error deleting reminder: {str(e)}")
            return jsonify({"error": str(e), "success": False}), 500
            
    @app.route('/api/ai/news/article', methods=['POST'])
    def upload_news_article():
        """
        Upload a news article to be used as context for future news queries.
        
        This allows setting a news article as the context for the ConversationQA system,
        enabling detailed follow-up questions about the article.
        """
        try:
            data = request.json
            if not data or 'text' not in data:
                return jsonify({"error": "No article text provided in request"}), 400
                
            # Get the article text
            article_text = data.get('text')
            title = data.get('title', 'Untitled Article')
            
            # Check if article is long enough
            if len(article_text) < 100:
                return jsonify({
                    "error": "Article text is too short", 
                    "success": False
                }), 400
                
            logger.info(f"Processing article: {title} (length: {len(article_text)})")
            
            # Generate a summary of the article
            summary = article_summarize(article_text)
            
            # Set the article as the passage for the QA system
            qa_system.set_passage(article_text)
            
            return jsonify({
                "success": True,
                "message": "Article processed successfully and set as context for future queries",
                "title": title,
                "summary": summary,
                "article_length": len(article_text),
                "context_set": True
            })
            
        except Exception as e:
            logger.error(f"Error processing news article: {str(e)}")
            return jsonify({"error": str(e), "success": False}), 500 