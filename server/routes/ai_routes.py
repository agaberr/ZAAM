from flask import jsonify, request, session
from models.ai_processor import AIProcessor
import sys
import os
from pathlib import Path
import logging
import json
import jwt
import datetime
import pytz
from models.reminder_nlp import ReminderNLP
from models.reminder import Reminder
from models.google_calendar import GoogleCalendarService
from models.google_oauth import GoogleOAuthService
# Import the functionality from reminder_ai_routes but don't register routes
from routes.reminder_ai_routes import register_reminder_ai_routes

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
    
    # Initialize the AI processor
    ai_processor = AIProcessor()
    print("[DEBUG] AI Processor initialized")
    
    # Initialize reminder-related services
    nlp_service = ReminderNLP()
    oauth_service = GoogleOAuthService()
    print("[DEBUG] ReminderNLP and GoogleOAuthService initialized")
    
    # JWT secret for authentication
    JWT_SECRET = os.getenv("JWT_SECRET")
    print(f"[DEBUG] JWT_SECRET length: {len(JWT_SECRET) if JWT_SECRET else 0} characters")
    
    print("\n[DEBUG] Creating test Flask app for reminder AI routes")
    # Create a test Flask app to register the reminder AI routes without conflicts
    # This allows us to reuse the implementation while avoiding endpoint conflicts
    from flask import Flask
    reminder_app = Flask("reminder_test_app")
    reminder_modules = {}
    
    print("[DEBUG] Registering reminder AI routes in test app")
    register_reminder_ai_routes(reminder_app, mongo)
    
    # Save the implementation of process_reminder_request to use it later
    reminder_handler = None
    endpoint_found = False
    print("[DEBUG] Looking for process_reminder_request in registered endpoints")
    for rule in reminder_app.url_map.iter_rules():
        print(f"[DEBUG] Test app has endpoint: {rule.endpoint}, URL: {rule}")
        if rule.endpoint == "process_reminder_request":
            endpoint_found = True
            reminder_handler = reminder_app.view_functions["process_reminder_request"]
            print(f"[DEBUG] Found process_reminder_request handler: {reminder_handler}")
            break
    
    if not endpoint_found:
        print("[DEBUG] WARNING: Could not find process_reminder_request endpoint in test app!")
        
    print(f"[DEBUG] Successfully imported reminder AI handler: {reminder_handler is not None}")
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
            # Check for google_auth_temp_id in session (from Google OAuth flow)
            elif 'google_auth_temp_id' in session:
                print(f"[DEBUG] Found google_auth_temp_id in session: {session['google_auth_temp_id']}")
                return session['google_auth_temp_id']
        
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
                    
            # Check if there's a Google session in progress
            if 'google_auth_temp_id' in session:
                google_temp_id = session['google_auth_temp_id']
                # Look up the user by their Google temp ID
                user = mongo.db.users.find_one({"google_auth_temp_id": google_temp_id})
                if user:
                    return str(user["_id"])
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
    
    # Internal function to process reminder queries - this now uses the implementation from reminder_ai_routes.py
    def process_reminder_internal(text):
        """Internal function to process reminder queries, using the handlers from reminder_ai_routes.py"""
        try:
            print(f"\n===== PROCESS_REMINDER_INTERNAL CALLED =====")
            print(f"[DEBUG] process_reminder_internal called with: '{text}'")
            
            # Get authenticated user ID
            user_id = get_authenticated_user_id()
            print(f"[DEBUG] Authentication status: user_id={user_id}")
            
            # For development purposes, allow bypass if no user is found
            if not user_id and os.getenv("FLASK_ENV") == "development":
                print("[DEBUG] DEVELOPMENT MODE: Creating temporary user for reminder processing")
                try:
                    # Try to get the first user in the database
                    temp_user = mongo.db.users.find_one({})
                    if temp_user:
                        user_id = str(temp_user["_id"])
                        print(f"[DEBUG] Using temporary user ID: {user_id}")
                    else:
                        print("[DEBUG] No users found in the database!")
                except Exception as e:
                    print(f"[DEBUG] Error getting temporary user: {str(e)}")
            
            # If still no user ID, return authentication error
            if not user_id:
                print(f"[DEBUG] No authenticated user for reminder processing - returning auth error")
                return {
                    "response": "Authentication required to process reminders. Please log in first.",
                    "category": "reminder",
                    "success": False
                }
            
            # Check if we have the reminder handler
            print(f"[DEBUG] Reminder handler available: {reminder_handler is not None}")
            print(f"[DEBUG] JWT_SECRET available: {JWT_SECRET is not None}")
            
            # If we have the reminder handler from reminder_ai_routes.py and a valid JWT secret
            if reminder_handler and JWT_SECRET:
                try:
                    print("[DEBUG] Attempting to use reminder_handler from reminder_ai_routes.py")
                    # Create a JWT token to authenticate the request
                    token = jwt.encode({"user_id": user_id}, JWT_SECRET, algorithm="HS256")
                    print(f"[DEBUG] Generated JWT token (first 15 chars): {token[:15]}...")
                    
                    # Create test environment to call the handler
                    print("[DEBUG] Creating test request context")
                    data_json = json.dumps({"text": text})
                    print(f"[DEBUG] Request data: {data_json}")
                    
                    with reminder_app.test_request_context(
                        '/api/ai/reminder',
                        method='POST',
                        data=data_json,
                        headers={
                            'Authorization': f'Bearer {token}',
                            'Content-Type': 'application/json'
                        }
                    ):
                        print("[DEBUG] Calling reminder_handler from reminder_ai_routes.py")
                        # Call the handler function directly
                        response = reminder_handler()
                        print(f"[DEBUG] Got response type: {type(response)}")
                        
                        # Process the response
                        if hasattr(response, 'get_json'):
                            # It's a Response object
                            print("[DEBUG] Converting Response object to dict")
                            response_data = response.get_json()
                            print(f"[DEBUG] Response data: {response_data}")
                            
                            formatted_response = {
                                "response": response_data.get("message", ""),
                                "category": "reminder",
                                "success": response_data.get("success", False)
                            }
                            
                            # Copy any additional fields
                            for key in ["reminder_id", "reminder", "reminders"]:
                                if key in response_data:
                                    formatted_response[key] = response_data[key]
                            
                            print(f"[DEBUG] Returning formatted response: {formatted_response}")
                            return formatted_response
                        else:
                            print(f"[DEBUG] Response is not a Response object: {response}")
                        
                except Exception as e:
                    print(f"[DEBUG] Error calling reminder handler: {str(e)}")
                    import traceback
                    print(f"[DEBUG] Traceback: {traceback.format_exc()}")
            
            # If we couldn't use the reminder handler, fall back to our own implementation
            print("[DEBUG] Using fallback reminder implementation in ai_routes.py")
            
            # Process the input with NLP
            print("[DEBUG] Calling nlp_service.predict")
            nlp_result = nlp_service.predict(text)
            print(f"[DEBUG] NLP result: {nlp_result}")
            
            if not nlp_result.get('success', False):
                print(f"[DEBUG] NLP processing failed: {nlp_result}")
                error_message = nlp_result.get('error', 'Failed to understand your request')
                return {
                    "response": f"Sorry, I couldn't process your reminder request: {error_message}",
                    "category": "reminder",
                    "success": False,
                    "error": error_message
                }
                
            # Extract parsed data
            parsed_data = nlp_result.get('parsed_data', {})
            intent = parsed_data.get('predicted_intent')
            action = parsed_data.get('predicted_action')
            time_str = parsed_data.get('predicted_time')
            
            response_message = nlp_result.get('response', "I've processed your request.")
            
            # Handle intents
            if intent == "create_event" and action:
                # Parse time to datetime
                try:
                    start_time = nlp_service.parse_time_to_datetime(time_str)
                    end_time = start_time + datetime.timedelta(hours=1)
                    
                    # Create reminder object
                    reminder = Reminder(
                        user_id=user_id,
                        title=action,
                        description=f"Created from voice command: '{text}'",
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
                    
                except Exception as e:
                    logger.error(f"Error creating reminder: {str(e)}", exc_info=True)
                    print(f"[DEBUG] Error creating reminder: {str(e)}")
                    return jsonify({
                        "success": False,
                        "message": f"I couldn't create your reminder due to an error: {str(e)}",
                        "error": str(e)
                    }), 500
                    
            elif intent == "get_timetable":
                # Show today's schedule
                try:
                    now = datetime.datetime.now(pytz.timezone('Africa/Cairo'))
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
                    
                except Exception as e:
                    logger.error(f"Error fetching timetable: {str(e)}", exc_info=True)
                    print(f"[DEBUG] Error fetching timetable: {str(e)}")
                    return jsonify({
                        "success": False,
                        "message": f"I couldn't retrieve your schedule due to an error: {str(e)}",
                        "error": str(e)
                    }), 500
            else:
                # Use the response from the NLP service
                return jsonify({
                    "success": True,
                    "message": response_message,
                    "intent": intent
                })
                
        except Exception as e:
            logger.error(f"Error in process_reminder_internal: {str(e)}", exc_info=True)
            print(f"[DEBUG] Error in process_reminder_internal: {str(e)}")
            return {
                "response": f"Error processing reminder: {str(e)}",
                "category": "reminder", 
                "success": False
            }
    
    # Internal function to process weather queries
    def process_weather_internal(text):
        """Internal function to process weather queries, used by the main AI route"""
        try:
            print(f"[DEBUG] process_weather_internal called with: '{text}'")
            
            # Process weather text
            sentences = text.split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # Format the response
            newline = '\n'
            response = f"WEATHER: Processed weather request:{newline}- {newline}- ".join(sentences)
            
            print(f"[DEBUG] Weather response: '{response}'")
            
            result = {
                "response": response,
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
        Process an AI request with natural language understanding.
        
        This endpoint accepts natural language input and categorizes it into
        different types (news, reminders, weather), processing each category
        with specialized handlers.
        """
        try:
            # Get the request data
            data = request.json
            if not data or 'text' not in data:
                return jsonify({"error": "No text provided in request"}), 400
                
            # Get the text from the request
            text = data['text']
            print(f"[DEBUG] /api/ai/process received: '{text}'")
            
            # Segment the text into categories
            segments = ai_processor.segment_sentences(text)
            print(f"[DEBUG] Segments: {segments}")
            
            # Process each category
            responses = {}
            combined_response = ""
            
            # Process news queries
            if segments.get("news"):
                news_text = " ".join(segments["news"])
                print(f"[DEBUG] Redirecting to news processing: '{news_text}'")
                
                news_response = process_news_internal(news_text)
                print(f"[DEBUG] News response: {news_response}")
                
                responses["news"] = news_response.get("response", "")
                combined_response += responses["news"] + "\n\n"
            
            # Process reminder queries
            if segments.get("reminders"):
                reminder_text = " ".join(segments["reminders"])
                print(f"[DEBUG] Redirecting to reminder processing: '{reminder_text}'")
                
                # Get user authentication for reminder processing
                user_id = get_authenticated_user_id()
                
                # In development mode, we can proceed even without authentication
                # IMPORTANT: Remove this in production 
                dev_mode = os.getenv("FLASK_ENV") == "development"
                
                if user_id or dev_mode:
                    if user_id:
                        print(f"[DEBUG] User authenticated for reminder: {user_id}")
                    else:
                        print("[DEBUG] DEVELOPMENT MODE: Proceeding without authentication")
                    
                    # Check if NLP model is loaded
                    if not nlp_service.is_loaded:
                        print("[DEBUG] NLP model not loaded, cannot process reminder")
                        reminder_response = {
                            "response": "I'm sorry, the reminder processing service is currently unavailable. Please try again later.",
                            "category": "reminder",
                            "success": False,
                            "error": "NLP model not loaded"
                        }
                    else:
                        # Call reminder processing function
                        reminder_response = process_reminder_internal(reminder_text)
                else:
                    print("[DEBUG] No authentication for reminder processing")
                    reminder_response = {
                        "response": "Authentication required to process reminders. Please log in first.",
                        "category": "reminder",
                        "success": False
                    }
                
                print(f"[DEBUG] Reminder response: {reminder_response}")
                
                # Extract the response text from reminder_response
                if isinstance(reminder_response, dict):
                    # If it's a dictionary, extract the response field
                    responses["reminders"] = reminder_response.get("response", "")
                else:
                    # If it's a Response object, convert it to JSON and extract the message
                    try:
                        # Attempt to get the JSON data from the response
                        response_data = reminder_response.get_json()
                        responses["reminders"] = response_data.get("message", response_data.get("response", ""))
                    except Exception as e:
                        print(f"[DEBUG] Error extracting response from reminder: {e}")
                        responses["reminders"] = "I processed your reminder, but couldn't format the response properly."
                
                combined_response += responses["reminders"] + "\n\n"
            
            # Process weather queries
            if segments.get("weather"):
                weather_text = " ".join(segments["weather"])
                print(f"[DEBUG] Redirecting to weather processing: '{weather_text}'")
                
                weather_response = process_weather_internal(weather_text)
                print(f"[DEBUG] Weather response: {weather_response}")
                
                responses["weather"] = weather_response.get("response", "")
                combined_response += responses["weather"] + "\n\n"
            
            # Process uncategorized as generic responses
            if segments.get("uncategorized"):
                uncategorized_text = " ".join(segments["uncategorized"])
                print(f"[DEBUG] Processing uncategorized text: '{uncategorized_text}'")
                
                uncategorized_response = ai_processor.process_uncategorized(segments["uncategorized"])
                responses["uncategorized"] = uncategorized_response
                combined_response += uncategorized_response + "\n\n"
            
            # If no categories were found, return a message
            if not responses:
                print(f"[DEBUG] No categories found for input: '{text}'")
                return jsonify({
                    "response": "I couldn't categorize your request.",
                    "success": True,
                    "categories": {}
                })
            
            # Remove trailing newlines
            combined_response = combined_response.strip()
            print(f"[DEBUG] Final combined response: '{combined_response}'")
            
            return jsonify({
                "response": combined_response,
                "success": True,
                "categories": responses
            })
            
        except Exception as e:
            logger.error(f"Error processing AI request: {str(e)}", exc_info=True)
            print(f"[DEBUG] Error in /api/ai/process: {str(e)}")
            return jsonify({
                "error": str(e), 
                "success": False,
                "response": "Sorry, I encountered an error while processing your request."
            }), 500
    
    @app.route('/api/ai/news', methods=['POST'])
    def process_news():
        """Process news-specific requests using ConversationQA."""
        try:
            data = request.json
            if not data or 'text' not in data:
                print("[DEBUG] /api/ai/news: No text provided in request")
                return jsonify({"error": "No text provided in request"}), 400
                
            text = data['text']
            print(f"[DEBUG] /api/ai/news received: '{text}', length: {len(text)}")
            
            # If this is a news article, first summarize it and set as passage
            if len(text) > 500:  # If it's a longer text, treat as article
                print(f"[DEBUG] Processing as article (length > 500)")
                
                # Summarize and set as passage in QA system
                summary = article_summarize(text)
                qa_system.set_passage(text)  # Set full text as passage for QA
                
                print(f"[DEBUG] Article summarized. Summary length: {len(summary)}")
                
                return jsonify({
                    "response": summary,
                    "category": "news",
                    "success": True,
                    "message": "Article processed. You can now ask questions about it."
                })
            else:
                print(f"[DEBUG] Processing as news query (length <= 500)")
                
                # Reuse the internal news processing function
                result = process_news_internal(text)
                return jsonify(result)
            
        except Exception as e:
            logger.error(f"Error processing news request: {str(e)}")
            print(f"[DEBUG] Error in /api/ai/news: {str(e)}")
            return jsonify({"error": str(e), "success": False}), 500
    
    @app.route('/api/ai/weather', methods=['POST'])
    def process_weather():
        """Process weather-specific requests."""
        try:
            data = request.json
            if not data or 'text' not in data:
                print("[DEBUG] /api/ai/weather: No text provided in request")
                return jsonify({"error": "No text provided in request"}), 400
                
            text = data['text']
            print(f"[DEBUG] /api/ai/weather received: '{text}'")
            
            # Reuse the internal weather processing function
            result = process_weather_internal(text)
            return jsonify(result)
            
        except Exception as e:
            print(f"[DEBUG] Error in /api/ai/weather: {str(e)}")
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
    
    # Add back the reminder route with a unique name to avoid conflicts
    @app.route('/api/ai/reminder', methods=['POST'])
    def process_reminder():
        """Process reminder-specific requests using the handler from reminder_ai_routes.py."""
        try:
            print("\n===== /API/AI/REMINDER ENDPOINT CALLED =====")
            # If we have the imported handler from reminder_ai_routes.py, use it
            print(f"[DEBUG] reminder_handler available: {reminder_handler is not None}")
            if reminder_handler:
                print("[DEBUG] Using imported reminder_handler from reminder_ai_routes.py")
                return reminder_handler()
            
            print("[DEBUG] Fallback to local implementation - reminder_handler not available")
            # Fallback implementation if the handler is not available
            data = request.json
            if not data or 'text' not in data:
                print("[DEBUG] /api/ai/reminder: No text provided in request")
                return jsonify({"error": "No text provided in request"}), 400
                
            text = data['text']
            print(f"[DEBUG] /api/ai/reminder received: '{text}'")
            
            # Use the internal implementation
            print("[DEBUG] Calling process_reminder_internal")
            result = process_reminder_internal(text)
            print(f"[DEBUG] process_reminder_internal returned: {result}")
            
            # If result is already a Response object, return it directly
            if hasattr(result, 'get_data'):
                return result
                
            # Otherwise convert to a proper response
            if isinstance(result, dict):
                # Extract the fields we need
                success = result.get("success", False)
                message = result.get("response", "")
                
                response_data = {
                    "success": success,
                    "message": message
                }
                
                # Add any additional fields
                for key in ["reminder_id", "reminder", "reminders", "category", "error"]:
                    if key in result:
                        response_data[key] = result[key]
                
                status_code = 400 if not success else 200
                return jsonify(response_data), status_code
            
            # If we got something unexpected, return an error
            return jsonify({
                "error": "Unexpected response format", 
                "success": False
            }), 500
                
        except Exception as e:
            logger.error(f"Error in /api/ai/reminder: {str(e)}", exc_info=True)
            print(f"[DEBUG] Error in /api/ai/reminder: {str(e)}")
            return jsonify({
                "error": str(e), 
                "success": False,
                "response": "Sorry, I encountered an error while processing your reminder request."
            }), 500 