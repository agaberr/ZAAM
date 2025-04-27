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
# Remove imports related to reminders and Google Calendar
# from models.reminder_nlp import ReminderNLP
# from models.reminder import Reminder
# from models.google_calendar import GoogleCalendarService
# from models.google_oauth import GoogleOAuthService
# Remove import from reminder_ai_routes
# from routes.reminder_ai_routes import register_reminder_ai_routes

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
    
    # Remove ReminderNLP and GoogleOAuthService initialization
    
    # JWT secret for authentication
    JWT_SECRET = os.getenv("JWT_SECRET")
    print(f"[DEBUG] JWT_SECRET length: {len(JWT_SECRET) if JWT_SECRET else 0} characters")
    
    # Remove test Flask app for reminder AI routes
    
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
            # Remove Google OAuth related checks
        
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
            
            # Remove Google session check
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
    
    # Remove process_reminder_internal function

    def process_weather_internal(text):
        """Internal function to process weather-related queries"""
        try:
            print(f"[DEBUG] process_weather_internal called with: '{text}'")
            
            # For now, just extract entities and return a weather-formatted response
            # In a real implementation, this would call a weather API
            
            # Simple entity extraction for city names, dates, etc.
            location_match = re.search(r'(?:in|at|for)\s+([A-Za-z\s]+)(?:,|\?|\.|\s+|$)', text)
            location = location_match.group(1).strip() if location_match else "your current location"
            
            # Extract time reference (today, tomorrow, etc)
            time_match = re.search(r'(?:today|tomorrow|tonight|this week|next week|on\s+([A-Za-z]+))', text, re.IGNORECASE)
            time_ref = time_match.group(0) if time_match else "today"
            
            # Generate a simple response
            forecast_options = [
                f"The weather {time_ref} in {location} looks good with temperatures around 75°F (24°C).",
                f"Expect partly cloudy skies {time_ref} in {location} with a high of 72°F (22°C).",
                f"The forecast for {location} {time_ref} shows a chance of rain with temperatures around 68°F (20°C)."
            ]
            
            # Select a random forecast for demo purposes
            import random
            weather_response = random.choice(forecast_options)
            
            # Create a formatted response
            response = f"WEATHER: {weather_response}\n\nNote: This is a demonstration response. In a production environment, this would connect to a real weather API."
            
            result = {
                "response": response,
                "category": "weather",
                "success": True,
                "location": location,
                "time": time_ref
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
            
            # Remove reminder segment processing
            
            # Process uncategorized segments
            if "uncategorized" in segments and segments["uncategorized"]:
                uncategorized_text = " ".join(segments["uncategorized"])
                print(f"[DEBUG] Processing uncategorized segment: '{uncategorized_text}'")
                
                # Try to process it as a general query
                uncategorized_result = process_news_internal(uncategorized_text)
                uncategorized_response = uncategorized_result["response"]
                
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
    
    # Remove the reminder route 