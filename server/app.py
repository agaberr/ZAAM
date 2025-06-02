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

# Import route modules
from routes.main_routes import register_main_routes
from routes.user_routes import user_routes
from routes.auth_routes import authRoutes
from routes.ai_routes import ai_routes_funcitons
from routes.memory_aid_routes import memory_aid_routes
from routes.cognitive_game_routes import cognitive_game_routes
from routes.speech_routes import speech_bp

##################################### IMPORTS END #####################################

##################################### MODEL SETUP START #####################################

project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Download not available models (used for the deployment)
models_available = setup_models()

##################################### MODEL SETUP END #####################################

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

# session
app.secret_key = os.getenv("SECRET_KEY", os.getenv("JWT_SECRET", "super-secret-key"))
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['PERMANENT_SESSION_LIFETIME'] = 3600
app.config['SESSION_USE_SIGNER'] = True

Session(app)

# Configure MongoDB
mongo_uri = os.getenv("MONGO_URI")
if not mongo_uri:
    print("Error: MONGO_URI environment variable not set")
    sys.exit(1)

app.config["MONGO_URI"] = mongo_uri
mongo = PyMongo(app)

register_main_routes(app)
user_routes(app, mongo)
authRoutes(app, mongo)
ai_routes_funcitons(app, mongo)
app.register_blueprint(memory_aid_routes)
app.register_blueprint(cognitive_game_routes)
app.register_blueprint(speech_bp, url_prefix='/api/speech')

@app.before_request
def before_request():
    app.config["DATABASE"] = mongo.db

##################################### INIT FLASK APP END #####################################

if __name__ == '__main__':
    
    # wait for the qa model to be downloaded in order to init
    if models_available:
        initialize_qa_system()
    
    app.run(host="0.0.0.0", port=5005, debug=True)