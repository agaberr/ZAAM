from flask import jsonify, request
import sys
import os
from pathlib import Path
import logging

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

def register_conversation_qa_routes(app, mongo):
    """Register ConversationQA routes for advanced natural language processing"""
    
    @app.route('/api/qa/query', methods=['POST'])
    def process_qa_query():
        """Process a query using the conversational QA system"""
        try:
            data = request.json
            if not data or 'query' not in data:
                return jsonify({"error": "No query provided in request"}), 400
                
            # Get the query from the request
            user_query = data.get('query')
            
            # Process the query with ConversationQA
            answer, need_new_passage = qa_system.process_query(user_query)
            
            # Get the last turn from conversation history for additional data
            last_turn = qa_system.conversation_history[-1] if qa_system.conversation_history else {}
            
            return jsonify({
                "answer": answer,
                "confidence": round(last_turn.get('confidence', 0), 2),
                "success": True,
                "debug": {
                    "original_query": last_turn.get('query', user_query),
                    "resolved_query": last_turn.get('resolved_query', user_query),
                    "current_entity": qa_system.current_entity,
                    "needed_new_passage": bool(need_new_passage)
                }
            })
            
        except Exception as e:
            logger.error(f"Error processing QA query: {str(e)}")
            return jsonify({"error": str(e), "success": False}), 500
            
    @app.route('/api/qa/set_passage', methods=['POST'])
    def set_qa_passage():
        """Set a new passage for the QA system"""
        try:
            data = request.json
            if not data or 'passage' not in data:
                return jsonify({"error": "No passage provided in request"}), 400
                
            # Set the new passage
            passage = data.get('passage')
            
            qa_system.set_passage(passage)
            
            return jsonify({
                "message": "Passage set successfully",
                "success": True
            })
            
        except Exception as e:
            logger.error(f"Error setting passage: {str(e)}")
            return jsonify({"error": str(e), "success": False}), 500
    
    @app.route('/api/qa/summarize', methods=['POST'])
    def summarize_article():
        """Summarize an article using the text summarization functionality"""
        try:
            data = request.json
            if not data or 'text' not in data:
                return jsonify({"error": "No text provided in request"}), 400
                
            # Get the text from the request
            text = data.get('text')
            
            # Summarize the article
            summary = article_summarize(text)
            
            return jsonify({
                "summary": summary,
                "success": True
            })
            
        except Exception as e:
            logger.error(f"Error summarizing article: {str(e)}")
            return jsonify({"error": str(e), "success": False}), 500
    
    @app.route('/api/qa/status', methods=['GET'])
    def get_qa_status():
        """Get diagnostic information about the QA system's current state"""
        try:
            # Gather diagnostic information
            status = {
                "success": True,
                "has_passage": qa_system.current_passage is not None,
                "passage_length": len(qa_system.current_passage) if qa_system.current_passage else 0,
                "processed_sentences_count": len(qa_system.processed_sentences),
                "conversation_history_length": len(qa_system.conversation_history),
                "current_entity": qa_system.current_entity,
                "embedding_cache_size": len(qa_system.embedding_cache),
                "entity_cache_size": len(qa_system.entity_cache),
                "qa_model_device": str(qa_system.device)
            }
            
            # Include a snippet of the current passage for verification
            if qa_system.current_passage and len(qa_system.current_passage) > 0:
                max_preview = 100  # Show first 100 chars
                status["passage_preview"] = qa_system.current_passage[:max_preview] + "..."
            else:
                status["passage_preview"] = "No passage set"
                
            # Include latest conversation turn if available
            if qa_system.conversation_history:
                last_turn = qa_system.conversation_history[-1]
                status["last_query"] = last_turn.get("query", "")
                status["last_resolved_query"] = last_turn.get("resolved_query", "")
                status["last_confidence"] = last_turn.get("confidence", 0)
            
            return jsonify(status)
            
        except Exception as e:
            logger.error(f"Error getting QA system status: {str(e)}")
            return jsonify({"error": str(e), "success": False}), 500 