# app.py
import time
from flask import Flask, request, jsonify
from QA import ConversationalQA
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from flask_cors import CORS 

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


app = Flask(__name__)
CORS(app)


# Create a global instance of the QA system
qa_system = ConversationalQA()

# Initialize with a default passage
default_passage = """
    Egypt is a country rich in history and culture, with many significant events shaping its past and present. 
    From ancient times, it was home to one of the world's greatest civilizations, building the famous pyramids and the Sphinx. 
    In modern history, Egypt played a key role in the Arab Spring in 2011, leading to major political changes. 
    Today, the country continues to develop its economy, tourism, and infrastructure, 
    hosting international conferences and sporting events. 
    Egypt is also known for its celebrations, such as Ramadan, Eid, and national holidays that honor its heritage and achievements.
"""
qa_system.set_passage(default_passage)


@app.route('/api/query', methods=['POST'])
def process_query():
    start_time = time.time()
    
    # Get the query from the request
    data = request.json
    user_query = data.get('query', '')
    session_id = data.get('session_id', 'default')  
    
    if not user_query:
        return jsonify({
            'error': 'No query provided',
            'status': 'error'
        }), 400
    
    # Process the query
    logger.info(f"Processing query from API: {user_query}")
    answer, need_new_passage = qa_system.process_query(user_query)

    processing_time = time.time() - start_time
    
    # Get the last turn from conversation history
    last_turn = qa_system.conversation_history[-1]
    
    # Prepare the response
    response = {
        'answer': answer,
        'processing_time': round(processing_time, 2),
        'confidence': round(last_turn.get('confidence', 0), 2),
        'status': 'success',
        'debug': {
            'original_query': last_turn['query'],
            'resolved_query': last_turn['resolved_query'],
            'current_entity': qa_system.current_entity,
            'needed_new_passage':  bool(need_new_passage) 
        }
    }
    
    return jsonify(response)


if __name__ == '__main__':
    # For development only - use a proper WSGI server in production
    app.run(host='0.0.0.0', port=5000, debug=True)