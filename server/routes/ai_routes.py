from flask import jsonify, request
from models.ai_processor import AIProcessor

def register_ai_routes(app, mongo):
    """Register AI processing routes."""
    
    # Initialize the AI processor
    ai_processor = AIProcessor()
    
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
            
            # Segment the text into categories
            segments = ai_processor.segment_sentences(text)
            
            # Process each category
            responses = {}
            if segments.get("news"):
                responses["news"] = ai_processor.process_news(segments["news"])
            
            if segments.get("reminders"):
                responses["reminders"] = ai_processor.process_reminders(segments["reminders"])
            
            if segments.get("weather"):
                responses["weather"] = ai_processor.process_weather(segments["weather"])
            
            if segments.get("uncategorized"):
                responses["uncategorized"] = ai_processor.process_uncategorized(segments["uncategorized"])
            
            # If no categories were found, return a message
            if not responses:
                return jsonify({
                    "response": "I couldn't categorize your request.",
                    "success": True,
                    "categories": {}
                })
            
            return jsonify({
                "response": "Your request has been processed.",
                "success": True,
                "categories": responses
            })
            
        except Exception as e:
            return jsonify({"error": str(e), "success": False}), 500
    
    @app.route('/api/ai/news', methods=['POST'])
    def process_news():
        """Process news-specific requests."""
        try:
            data = request.json
            if not data or 'text' not in data:
                return jsonify({"error": "No text provided in request"}), 400
                
            text = data['text']
            # In a real implementation, this would call a specialized news model
            # For now, just use the basic processor
            sentences = text.split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # Fix: Use newline character variable instead of backslash in f-string
            newline = '\n'
            response = f"NEWS: Processed news request:{newline}- {newline}- ".join(sentences)
            
            return jsonify({
                "response": response,
                "category": "news",
                "success": True
            })
            
        except Exception as e:
            return jsonify({"error": str(e), "success": False}), 500
    
    @app.route('/api/ai/reminder', methods=['POST'])
    def process_reminder():
        """Process reminder-specific requests."""
        try:
            data = request.json
            if not data or 'text' not in data:
                return jsonify({"error": "No text provided in request"}), 400
                
            text = data['text']
            # In a real implementation, this would call a specialized reminder model
            # and potentially interact with the reminder database
            sentences = text.split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # Fix: Use newline character variable instead of backslash in f-string
            newline = '\n'
            response = f"REMINDER: Processed reminder request:{newline}- {newline}- ".join(sentences)
            
            return jsonify({
                "response": response,
                "category": "reminder",
                "success": True
            })
            
        except Exception as e:
            return jsonify({"error": str(e), "success": False}), 500
    
    @app.route('/api/ai/weather', methods=['POST'])
    def process_weather():
        """Process weather-specific requests."""
        try:
            data = request.json
            if not data or 'text' not in data:
                return jsonify({"error": "No text provided in request"}), 400
                
            text = data['text']
            # In a real implementation, this would call a weather API or model
            sentences = text.split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # Fix: Use newline character variable instead of backslash in f-string
            newline = '\n'
            response = f"WEATHER: Processed weather request:{newline}- {newline}- ".join(sentences)
            
            return jsonify({
                "response": response,
                "category": "weather",
                "success": True
            })
            
        except Exception as e:
            return jsonify({"error": str(e), "success": False}), 500 