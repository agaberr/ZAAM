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

def register_reminder_ai_routes(app, mongo):
    """Register routes for AI-powered reminder processing."""
    
    # Initialize services
    nlp_service = ReminderNLP()
    oauth_service = GoogleOAuthService()
    
    # Helper function to get authenticated user ID
    def get_authenticated_user_id():
        # Get the token from the Authorization header
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return None
            
        token = auth_header.split(' ')[1]
        try:
            # Decode the token
            payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            return payload.get('user_id')
        except Exception:
            return None
    
    @app.route('/api/ai/reminder', methods=['POST'])
    def process_reminder_request():
        """Process natural language reminder requests."""
        try:
            # Get authenticated user ID
            user_id = get_authenticated_user_id()
            if not user_id:
                return jsonify({"error": "Authentication required"}), 401
                
            # Get request data
            data = request.json
            user_input = data.get('text', '')
            
            if not user_input:
                return jsonify({"error": "No input text provided"}), 400
                
            # Process the input with NLP
            nlp_result = nlp_service.predict(user_input)
            
            if not nlp_result.get('success', False):
                return jsonify({"error": "Failed to process your request"}), 500
                
            # Extract parsed data
            parsed_data = nlp_result.get('parsed_data', {})
            intent = parsed_data.get('predicted_intent')
            action = parsed_data.get('predicted_action')
            time_str = parsed_data.get('predicted_time')
            
            response_message = nlp_result.get('response', "I've processed your request.")
            
            # Handle intents
            if intent == "create_event" and action:
                # Parse time to datetime
                start_time = nlp_service.parse_time_to_datetime(time_str)
                end_time = start_time + datetime.timedelta(hours=1)
                
                # Create reminder object
                reminder = Reminder(
                    user_id=user_id,
                    title=action,
                    description=f"Created from voice command: '{user_input}'",
                    start_time=start_time,
                    end_time=end_time,
                    recurrence=None  # No recurrence by default for voice reminders
                )
                
                # Save reminder
                success, errors = reminder.save(mongo.db)
                
                if not success:
                    return jsonify({
                        "success": False,
                        "message": "Failed to create reminder",
                        "errors": errors
                    }), 400
                
                # Try to create Google Calendar event if credentials are available
                google_credentials = oauth_service.get_credentials_for_user(mongo.db, user_id)
                if google_credentials:
                    calendar_service = GoogleCalendarService(google_credentials)
                    event_id = calendar_service.create_event(reminder)
                    if event_id:
                        # Update reminder with Google event ID
                        reminder.google_event_id = event_id
                        reminder.save(mongo.db)
                
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