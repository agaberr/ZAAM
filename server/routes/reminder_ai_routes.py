from flask import jsonify, request, session
from datetime import datetime
from models.reminder_nlp import ReminderNLP
from models.reminder import Reminder
from models.google_calendar import GoogleCalendarService
from models.google_oauth import GoogleOAuthService
import jwt
import os
import pytz

# Load JWT secret from environment variables
JWT_SECRET = os.getenv("JWT_SECRET")
print(f"[REMINDER_AI_ROUTES] Loading module, JWT_SECRET length: {len(JWT_SECRET) if JWT_SECRET else 0} characters")

def register_reminder_ai_routes(app, mongo):
    """Register routes for AI-powered reminder processing."""
    
    print("\n===== REGISTERING REMINDER AI ROUTES =====")
    
    # Initialize services
    print("[REMINDER_AI_ROUTES] Initializing NLP service")
    nlp_service = ReminderNLP()
    print("[REMINDER_AI_ROUTES] Initializing OAuth service")
    oauth_service = GoogleOAuthService()
    
    print("[REMINDER_AI_ROUTES] Defining get_authenticated_user_id function")
    # Helper function to get authenticated user ID
    def get_authenticated_user_id():
        # Get the token from the Authorization header
        auth_header = request.headers.get('Authorization')
        print(f"[REMINDER_AI_ROUTES] Auth header: {auth_header[:20] + '...' if auth_header else None}")
        if not auth_header or not auth_header.startswith('Bearer '):
            print("[REMINDER_AI_ROUTES] No valid auth header found")
            return None
            
        token = auth_header.split(' ')[1]
        try:
            # Decode the token
            print(f"[REMINDER_AI_ROUTES] Decoding token (first 15 chars): {token[:15]}...")
            payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            user_id = payload.get('user_id')
            print(f"[REMINDER_AI_ROUTES] Decoded user_id: {user_id}")
            return user_id
        except Exception as e:
            print(f"[REMINDER_AI_ROUTES] Token decode error: {str(e)}")
            return None
    
    print("[REMINDER_AI_ROUTES] Registering /api/ai/reminder route")
    @app.route('/api/ai/reminder', methods=['POST'])
    def process_reminder_request():
        """Process natural language reminder requests."""
        try:
            print("\n===== REMINDER_AI_ROUTES.PROCESS_REMINDER_REQUEST CALLED =====")
            # Get authenticated user ID
            print("[REMINDER_AI_ROUTES] Getting authenticated user ID")
            user_id = get_authenticated_user_id()
            print(f"[REMINDER_AI_ROUTES] user_id: {user_id}")
            
            if not user_id:
                print("[REMINDER_AI_ROUTES] No user_id - returning authentication error")
                return jsonify({"error": "Authentication required"}), 401
                
            # Get request data
            print("[REMINDER_AI_ROUTES] Parsing request data")
            data = request.json
            print(f"[REMINDER_AI_ROUTES] Request data: {data}")
            user_input = data.get('text', '')
            
            if not user_input:
                print("[REMINDER_AI_ROUTES] No input text provided")
                return jsonify({"error": "No input text provided"}), 400
                
            # Process the input with NLP
            print(f"[REMINDER_AI_ROUTES] Processing with NLP: '{user_input}'")
            nlp_result = nlp_service.predict(user_input)
            print(f"[REMINDER_AI_ROUTES] NLP result: {nlp_result}")
            
            if not nlp_result.get('success', False):
                print("[REMINDER_AI_ROUTES] NLP processing failed")
                return jsonify({"error": "Failed to process your request"}), 500
                
            # Extract parsed data
            parsed_data = nlp_result.get('parsed_data', {})
            intent = parsed_data.get('predicted_intent')
            action = parsed_data.get('predicted_action')
            time_str = parsed_data.get('predicted_time')
            
            print(f"[REMINDER_AI_ROUTES] Parsed data - intent: {intent}, action: {action}, time: {time_str}")
            
            response_message = nlp_result.get('response', "I've processed your request.")
            
            # Handle intents
            if intent == "create_event" and action:
                print("[REMINDER_AI_ROUTES] Handling create_event intent")
                # Parse time to datetime
                print(f"[REMINDER_AI_ROUTES] Parsing time string: {time_str}")
                
                try:
                    start_time = nlp_service.parse_time_to_datetime(time_str)
                    print(f"[REMINDER_AI_ROUTES] Parsed start_time: {start_time}")
                except Exception as e:
                    print(f"[REMINDER_AI_ROUTES] Error parsing time: {str(e)}")
                    return jsonify({"error": f"Failed to parse time: {str(e)}"}), 500
                    
                end_time = start_time + datetime.timedelta(hours=1)
                
                # Create reminder object
                print("[REMINDER_AI_ROUTES] Creating reminder object")
                reminder = Reminder(
                    user_id=user_id,
                    title=action,
                    description=f"Created from voice command: '{user_input}'",
                    start_time=start_time,
                    end_time=end_time,
                    recurrence=None  # No recurrence by default for voice reminders
                )
                
                # Save reminder
                print("[REMINDER_AI_ROUTES] Saving reminder to database")
                success, errors = reminder.save(mongo.db)
                
                if not success:
                    print(f"[REMINDER_AI_ROUTES] Failed to save reminder: {errors}")
                    return jsonify({
                        "success": False,
                        "message": "Failed to create reminder",
                        "errors": errors
                    }), 400
                
                print("[REMINDER_AI_ROUTES] Reminder saved successfully")
                reminder_id_str = str(reminder._id) if reminder._id else "unknown"
                print(f"[REMINDER_AI_ROUTES] Reminder ID: {reminder_id_str}")
                
                # Try to create Google Calendar event if credentials are available
                google_credentials = oauth_service.get_credentials_for_user(mongo.db, user_id)
                if google_credentials:
                    print(f"Found Google credentials for user {user_id}, attempting to create calendar event")
                    calendar_service = GoogleCalendarService(google_credentials)
                    event_id = calendar_service.create_event(reminder)
                    if event_id:
                        print(f"Successfully created Google Calendar event with ID: {event_id}")
                        # Update reminder with Google event ID
                        reminder.google_event_id = event_id
                        reminder.save(mongo.db)
                        
                        return jsonify({
                            "success": True,
                            "message": f"{response_message} I've also added it to your Google Calendar.",
                            "reminder_id": str(reminder._id),
                            "reminder": {
                                "title": reminder.title,
                                "start_time": reminder.start_time.isoformat(),
                                "end_time": reminder.end_time.isoformat() if reminder.end_time else None,
                                "google_event_id": event_id
                            }
                        })
                    else:
                        print(f"Failed to create Google Calendar event for reminder {reminder._id}")
                else:
                    print(f"No Google credentials found for user {user_id}, skipping calendar integration")
                    print(f"Available tokens in database: {list(mongo.db.google_tokens.find())}")
                    
                    # Try to create tokens using request body directly
                    temp_user = mongo.db.users.find_one({"_id": user_id})
                    if temp_user and temp_user.get("google_id"):
                        print(f"User has Google ID but no tokens, trying to recover tokens")
                        # Look for tokens by email
                        if temp_user.get("email"):
                            other_user = mongo.db.users.find_one({"email": temp_user.get("email")})
                            if other_user:
                                print(f"Found other user with same email, trying their tokens")
                                google_credentials = oauth_service.get_credentials_for_user(mongo.db, str(other_user["_id"]))
                                if google_credentials:
                                    print(f"Found credentials for other user, trying to create calendar event")
                                    calendar_service = GoogleCalendarService(google_credentials)
                                    event_id = calendar_service.create_event(reminder)
                                    if event_id:
                                        print(f"Successfully created Google Calendar event with ID: {event_id}")
                                        reminder.google_event_id = event_id
                                        reminder.save(mongo.db)
                                        
                                        return jsonify({
                                            "success": True,
                                            "message": f"{response_message} I've also added it to your Google Calendar.",
                                            "reminder_id": str(reminder._id),
                                            "reminder": {
                                                "title": reminder.title,
                                                "start_time": reminder.start_time.isoformat(),
                                                "end_time": reminder.end_time.isoformat() if reminder.end_time else None,
                                                "google_event_id": event_id
                                            }
                                        })
                
                # If we got here, no Google Calendar event was created
                return jsonify({
                    "success": True,
                    "message": response_message,
                    "reminder_id": str(reminder._id),
                    "reminder": {
                        "title": reminder.title,
                        "start_time": reminder.start_time.isoformat(),
                        "end_time": reminder.end_time.isoformat() if reminder.end_time else None
                    }
                })
                
            elif intent == "get_timetable":
                # Use local timezone
                egypt_tz = pytz.timezone('Africa/Cairo')
                now = datetime.now(egypt_tz)
                today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
                today_end = now.replace(hour=23, minute=59, second=59, microsecond=999999)
                
                # Get reminders for today
                reminders = Reminder.find_by_user_and_timerange(
                    mongo.db, 
                    user_id, 
                    today_start, 
                    today_end
                )
                
                # Format reminders for response
                formatted_reminders = []
                for reminder in reminders:
                    formatted_time = reminder.start_time.strftime('%I:%M %p')
                    formatted_reminders.append({
                        "id": str(reminder._id),
                        "title": reminder.title,
                        "time": formatted_time,
                        "description": reminder.description,
                        "completed": reminder.completed
                    })
                
                # Generate response
                if not formatted_reminders:
                    response_message = "You don't have any reminders scheduled for today. Your schedule is clear!"
                else:
                    today_date = now.strftime("%A, %B %d")
                    response_items = [f"• {r['time']} - {r['title']}" for r in formatted_reminders]
                    items_text = "\n".join(response_items)
                    response_message = f"Here's your schedule for {today_date}:\n\n{items_text}"
                
                return jsonify({
                    "success": True,
                    "message": response_message,
                    "reminders": formatted_reminders
                })
                
            else:
                # Generic response for unhandled intents
                return jsonify({
                    "success": True,
                    "message": response_message
                })
                
        except Exception as e:
            print(f"Error in reminder AI processing: {str(e)}")
            return jsonify({
                "success": False,
                "message": "Sorry, I encountered an error while processing your request.",
                "error": str(e)
            }), 500 