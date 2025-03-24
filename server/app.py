from flask import Flask, jsonify, request, session
from flask_cors import CORS
from flask_pymongo import PyMongo
from dotenv import load_dotenv
import os
import sys
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from pymongo import MongoClient
from flask_jwt_extended import JWTManager
import pytz
from datetime import datetime, timedelta

# Import route modules
from routes.main_routes import register_main_routes
from routes.user_routes import register_user_routes, user_bp
from routes.auth_routes import register_auth_routes, auth_bp
from routes.reminder_routes import register_reminder_routes, reminder_bp
from routes.google_auth_routes import register_google_auth_routes, google_auth_bp
from routes.ai_routes import register_ai_routes
from routes.memory_aid_routes import memory_aid_routes

# Import Google related modules
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow requests from all origins

# Configure session
app.secret_key = os.getenv("SECRET_KEY", os.getenv("JWT_SECRET", "super-secret-key"))

# Configure MongoDB
try:
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/zaam_db")
    app.config["MONGO_URI"] = mongo_uri
    mongo = PyMongo(app)
    print(f"Connected to MongoDB at {mongo_uri}")
    
    # Test MongoDB connection
    mongo.db.command('ping')
    print("MongoDB connection test successful")
except Exception as e:
    print(f"MongoDB connection error: {str(e)}")
    mongo = None

# Configure JWT
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET", "dev_secret_key")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=24)
jwt = JWTManager(app)

# Google OAuth Config
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"  # For development only
app.config["GOOGLE_CLIENT_SECRETS_FILE"] = os.path.join(os.path.dirname(__file__), 'credentialsOAuth.json')
app.config["GOOGLE_OAUTH_SCOPES"] = ["https://www.googleapis.com/auth/calendar"]

# Register blueprints
app.register_blueprint(auth_bp, url_prefix='/api/auth')
app.register_blueprint(user_bp, url_prefix='/api/users')
app.register_blueprint(memory_aid_routes, url_prefix='/api/memory-aids')
app.register_blueprint(reminder_bp, url_prefix='/api/reminders')
app.register_blueprint(google_auth_bp, url_prefix='/api/google')

# Register routes using functional pattern
register_main_routes(app)
register_user_routes(app, mongo)
register_auth_routes(app, mongo)
register_reminder_routes(app, mongo)
register_google_auth_routes(app, mongo)
register_ai_routes(app, mongo)

# Configure database access for routes that use the Blueprint pattern
@app.before_request
def before_request():
    app.config["DATABASE"] = mongo.db

@app.route('/')
def index():
    return jsonify({
        "message": "Welcome to the ZAAM API",
        "version": "1.0.0",
        "status": "running"
    })

@app.route('/api/health')
def health_check():
    mongo_status = "connected" if mongo else "disconnected"
    return jsonify({
        "status": "healthy",
        "mongo": mongo_status,
        "timestamp": datetime.now().isoformat()
    })

# Google auth routes moved to routes/google_auth_routes.py or remove if needed

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0')
