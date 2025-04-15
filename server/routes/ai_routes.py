from flask import jsonify, request
from models.ai_processor import AIProcessor
import sys
import os
from pathlib import Path
import logging
import json

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
from text_summarization import article_summarize
from qa_singleton import get_qa_instance

# Get the singleton instance
qa_system = get_qa_instance()

def register_ai_routes(app, mongo):
    """Register AI processing routes."""
    
    # Initialize the AI processor
    ai_processor = AIProcessor()
    
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
        """Internal function to process reminder queries, used by the main AI route"""
        try:
            print(f"[DEBUG] process_reminder_internal called with: '{text}'")
            
            # Process reminder text
            sentences = text.split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # Format the response
            newline = '\n'
            response = f"REMINDER: Processed reminder request:{newline}- {newline}- ".join(sentences)
            
            print(f"[DEBUG] Reminder response: '{response}'")
            
            result = {
                "response": response,
                "category": "reminder",
                "success": True
            }
            return result
        except Exception as e:
            logger.error(f"Error in process_reminder_internal: {str(e)}")
            print(f"[DEBUG] Error in process_reminder_internal: {str(e)}")
            return {"response": f"Error processing reminder: {str(e)}", "success": False}
    
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
                
                reminder_response = process_reminder_internal(reminder_text)
                print(f"[DEBUG] Reminder response: {reminder_response}")
                
                responses["reminders"] = reminder_response.get("response", "")
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