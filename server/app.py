##################################### IMPORTS START #####################################
import os
import sys
import json
from pathlib import Path
from flask_cors import CORS
from dotenv import load_dotenv
from pymongo import MongoClient
from flask_session import Session
from flask_pymongo import PyMongo
from ConversationQA.qa_initializer import initialize_qa_system # Import QA initializer
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from flask import Flask, request, jsonify, session, redirect, url_for, Request, current_app, render_template, make_response

# Import model manager
from model_manager import setup_models



##################################### IMPORTS END #####################################

##################################### MODEL SETUP START #####################################

# Setup models using the model manager
models_available = setup_models()

##################################### MODEL SETUP END #####################################

from routes.main_routes import register_main_routes
from routes.user_routes import register_user_routes
from routes.auth_routes import register_auth_routes
from routes.ai_routes import register_ai_routes
from routes.memory_aid_routes import memory_aid_routes
from routes.cognitive_game_routes import cognitive_game_routes
from routes.speech_routes import speech_bp

##################################### INIT FLASK APP START #####################################

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {
    "origins": "*",
    "supports_credentials": True,
    "allow_headers": ["Content-Type", "Authorization"],
    "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
}})

# Configure session
app.secret_key = os.getenv("SECRET_KEY", os.getenv("JWT_SECRET", "super-secret-key"))
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['PERMANENT_SESSION_LIFETIME'] = 3600
app.config['SESSION_USE_SIGNER'] = True

# Initialize Flask-Session
Session(app)

# Configure MongoDB
mongo_uri = os.getenv("MONGO_URI")
app.config["MONGO_URI"] = mongo_uri
mongo = PyMongo(app)

# Register routes
register_main_routes(app)
register_user_routes(app, mongo)
register_auth_routes(app, mongo)
register_ai_routes(app, mongo)
app.register_blueprint(memory_aid_routes)
app.register_blueprint(cognitive_game_routes)
app.register_blueprint(speech_bp, url_prefix='/api/speech')

@app.before_request
def before_request():
    app.config["DATABASE"] = mongo.db

##################################### INIT FLASK APP END #####################################

if __name__ == '__main__':    
    if models_available:
        initialize_qa_system()
    
    app.run(host="0.0.0.0", port=5001, debug=True)