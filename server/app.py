#!/usr/bin/env python3

# First apply path fix to ensure imports work correctly
import os
import sys
import importlib
import subprocess
from pathlib import Path
from flask import Flask, request, jsonify, session, redirect, url_for, Request, current_app, render_template, make_response
import json
import logging
import requests
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler
import datetime
import pytz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Add project root to sys.path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"Added {project_root} to Python path")

# ==== Model Verification and Download - RUN FIRST ====
# Ensure required packages are installed
required_packages = ["gdown", "requests"]
for package in required_packages:
    try:
        importlib.import_module(package)
    except ImportError:
        print(f"Installing required package: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Function to check for models
def verify_models():
    """Check if all required models are available"""
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ConversationQA', 'Models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)
        
    required_models = [
        "bert_seq2seq_ner.pt",
        "pronoun_resolution_model_full.pt",
        "extractiveQA.pt",
        "vectorizer.pkl",
        "classifier_model.pkl"
    ]
    
    missing_models = []
    for model_name in required_models:
        model_path = os.path.join(models_dir, model_name)
        if not os.path.exists(model_path):
            missing_models.append(model_name)
    
    if missing_models:
        print("\nWARNING: The following models are missing:")
        for model in missing_models:
            print(f"  - {model}")
        print("The ConversationQA functionality may not work correctly.\n")
        return False, missing_models
    else:
        print("\nAll required AI models are available.\n")
        return True, []

# Function to download models
def download_models(missing_models):
    """Download missing model files"""
    import tempfile
    import gdown
    import zipfile
    import requests
    
    print("\nDownloading missing models...")
    
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ConversationQA', 'Models')
    
    # Direct download URLs (not through Google Drive API)
    direct_urls = {
        "bert_seq2seq_ner.zip": "https://drive.google.com/uc?id=1lOSkIPGU4TX7L727OmkBL3f0Fh_W3_Fv&export=download",
        "pronoun_resolution_model_full.zip": "https://drive.google.com/uc?id=1DhlkILm1kzD8gPbEU0NlTUGeDA_uGisk&export=download",
        "extractiveQA.zip": "https://drive.google.com/uc?id=1ph3yhuAz7fmTv8lat8fbdwzlHYSa4y4U&export=download",
        "vectorizer.zip": "https://drive.google.com/uc?id=1QqxFX0VhLYKdgWWJBYnE9YVnEP6bZKcY&export=download"
    }
    
    # Map models to their containing archives
    model_to_archive = {
        "bert_seq2seq_ner.pt": "bert_seq2seq_ner.zip",
        "pronoun_resolution_model_full.pt": "pronoun_resolution_model_full.zip",
        "extractiveQA.pt": "extractiveQA.zip",
        "vectorizer.pkl": "vectorizer.zip",
        "classifier_model.pkl": "vectorizer.zip"
    }
    
    # Determine which archives to download
    archives_to_download = set()
    for model in missing_models:
        if model in model_to_archive:
            archives_to_download.add(model_to_archive[model])
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for archive in archives_to_download:
            archive_path = os.path.join(temp_dir, archive)
            url = direct_urls.get(archive)
            
            if not url:
                print(f"No URL found for {archive}")
                continue
            
            print(f"Downloading {archive}...")
            
            # First try gdown
            try:
                gdown.download(url, archive_path, quiet=False)
                
                # Check if downloaded successfully
                if os.path.exists(archive_path) and os.path.getsize(archive_path) > 0:
                    print(f"Successfully downloaded {archive}")
                    
                    # Extract the file
                    try:
                        print(f"Extracting {archive}...")
                        
                        # Handle ZIP extraction
                        try:
                            with zipfile.ZipFile(archive_path) as zf:
                                zf.extractall(models_dir)
                            print(f"Extracted {archive} using zipfile")
                        except Exception as e:
                            print(f"zipfile extraction failed: {e}")
                            
                            # Try with subprocess
                            try:
                                subprocess.run(['unzip', archive_path, '-d', models_dir], check=True)
                                print(f"Extracted {archive} using unzip command")
                            except Exception as e2:
                                print(f"unzip command failed: {e2}")
                                
                                # Try with 7z as last resort
                                try:
                                    subprocess.run(['7z', 'x', archive_path, f'-o{models_dir}'], check=True)
                                    print(f"Extracted {archive} using 7z")
                                except Exception as e3:
                                    print(f"7z extraction failed: {e3}")
                                    print(f"Failed to extract {archive} with any method")
                    except Exception as e:
                        print(f"Error during extraction: {e}")
                else:
                    print(f"Failed to download {archive} with gdown")
            except Exception as e:
                print(f"Error downloading with gdown: {e}")
                print("Trying direct download method...")
                
                # Try direct download with requests as fallback
                try:
                    response = requests.get(url, stream=True)
                    if response.status_code == 200:
                        with open(archive_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        print(f"Downloaded {archive} with requests")
                        
                        # Extract the file
                        try:
                            print(f"Extracting {archive}...")
                            
                            # Handle ZIP extraction
                            try:
                                with zipfile.ZipFile(archive_path) as zf:
                                    zf.extractall(models_dir)
                                print(f"Extracted {archive} using zipfile")
                            except Exception as e:
                                print(f"zipfile extraction failed: {e}")
                                
                                # Try with subprocess
                                try:
                                    subprocess.run(['unzip', archive_path, '-d', models_dir], check=True)
                                    print(f"Extracted {archive} using unzip command")
                                except Exception as e2:
                                    print(f"unzip command failed: {e2}")
                                    
                                    # Try with 7z as last resort
                                    try:
                                        subprocess.run(['7z', 'x', archive_path, f'-o{models_dir}'], check=True)
                                        print(f"Extracted {archive} using 7z")
                                    except Exception as e3:
                                        print(f"7z extraction failed: {e3}")
                                        print(f"Failed to extract {archive} with any method")
                        except Exception as e:
                            print(f"Error during extraction: {e}")
                    else:
                        print(f"Failed to download {archive} with requests: {response.status_code}")
                except Exception as e:
                    print(f"Error with direct download: {e}")
    
    # Verify models again
    models_available, still_missing = verify_models()
    if still_missing:
        print("\nSome models are still missing after download attempts:")
        for model in still_missing:
            print(f"  - {model}")
        return False
    else:
        print("\nAll models successfully downloaded and verified!")
        return True

# Function to update paths in code files
def update_model_paths():
    """Update hardcoded model paths in the code files"""
    print("Updating model paths in code files...")
    
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ConversationQA', 'Models')
    models_dir = models_dir.replace('\\', '\\\\')  # Escape backslashes for string literals
    
    # Update QA.py
    qa_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ConversationQA', 'QA.py')
    if os.path.exists(qa_path):
        try:
            with open(qa_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            content = content.replace(
                r'D:\College\Senior-2\GP\ZAAM project\ZAAM\server\ConversationQA\Models\bert_seq2seq_ner.pt',
                f'{models_dir}\\\\bert_seq2seq_ner.pt'
            )
            content = content.replace(
                r'D:\College\Senior-2\GP\ZAAM project\ZAAM\server\ConversationQA\Models\pronoun_resolution_model_full.pt',
                f'{models_dir}\\\\pronoun_resolution_model_full.pt'
            )
            content = content.replace(
                r'D:\College\Senior-2\GP\ZAAM project\ZAAM\server\ConversationQA\Models\extractiveQA.pt',
                f'{models_dir}\\\\extractiveQA.pt'
            )
            
            with open(qa_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"Updated model paths in {qa_path}")
        except Exception as e:
            print(f"Error updating QA.py: {e}")
    
    # Update TopicExtractionModel.py
    topic_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ConversationQA', 'NameEntityModel', 'TopicExtractionModel.py')
    if os.path.exists(topic_path):
        try:
            with open(topic_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            content = content.replace(
                r'D:\College\Senior-2\GP\ZAAM project\ZAAM\server\ConversationQA\Models\bert_seq2seq_ner.pt',
                f'{models_dir}\\\\bert_seq2seq_ner.pt'
            )
            
            with open(topic_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"Updated model paths in {topic_path}")
        except Exception as e:
            print(f"Error updating TopicExtractionModel.py: {e}")

# Run model checks and downloads before importing anything else
print("\n===== ZAAM Server Initialization =====")
print("Checking for AI models...")
models_available, missing_models = verify_models()

if not models_available:
    print("Missing models detected. Attempting to download...")
    if download_models(missing_models):
        update_model_paths()
        print("Model setup completed successfully!")
    else:
        print("WARNING: Failed to download all models. Some features may not work.")
else:
    print("All models are available!")
    update_model_paths()

# ===== Continue with regular imports after model checks =====
from flask import Flask
from flask_cors import CORS
from flask_pymongo import PyMongo
from flask_session import Session
from dotenv import load_dotenv
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from pymongo import MongoClient
import json

# Import route modules
from routes.main_routes import register_main_routes
from routes.user_routes import register_user_routes
from routes.auth_routes import register_auth_routes
from routes.ai_routes import register_ai_routes
from routes.conversation_qa_routes import register_conversation_qa_routes
from routes.memory_aid_routes import memory_aid_routes

# Load environment variables
load_dotenv()

# Initialize QA system early to load models at startup
def initialize_qa_system():
    """Pre-initialize the QA system to load models at startup"""
    try:
        # Add ConversationQA to the path if needed
        conversation_qa_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ConversationQA")
        if conversation_qa_path not in sys.path:
            sys.path.append(conversation_qa_path)
        
        # Import and initialize the QA singleton
        from ConversationQA.qa_singleton import get_qa_instance
        
        print("Pre-initializing QA system and loading models...")
        qa_system = get_qa_instance()
        print(f"QA system initialized on device: {qa_system.device}")
        print(f"ConversationQA models loaded successfully!")
        return True
    except Exception as e:
        print(f"Error initializing QA system: {str(e)}")
        return False

# Continue with the rest of the code...

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {
    "origins": "*",
    "supports_credentials": True,
    "allow_headers": ["Content-Type", "Authorization"],
    "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
}})  # Allow requests from all origins with credentials

# Configure session
app.secret_key = os.getenv("SECRET_KEY", os.getenv("JWT_SECRET", "super-secret-key"))
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour in seconds
app.config['SESSION_USE_SIGNER'] = True

# Initialize Flask-Session
Session(app)

# Configure MongoDB
mongo_uri = os.getenv("MONGO_URI")
if not mongo_uri:
    print("Error: MONGO_URI environment variable not set")
    sys.exit(1)

print(f"Connecting to MongoDB at: {mongo_uri}")

# Test MongoDB connection directly first
try:
    # Test connection using MongoClient
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    print("MongoDB connection test successful!")
except Exception as e:
    print(f"MongoDB connection test failed: {e}")
    sys.exit(1)

# Configure PyMongo
app.config["MONGO_URI"] = mongo_uri
app.config["MONGO_CONNECT"] = True  # Ensure we connect explicitly
mongo = PyMongo(app)

# Test to ensure database is accessible
try:
    # Try to access a collection to verify db connection
    mongo.db.list_collection_names()
    print("MongoDB collections accessible via PyMongo!")
except Exception as e:
    print(f"Error accessing MongoDB via PyMongo: {e}")
    sys.exit(1)

# Register routes
register_main_routes(app)
register_user_routes(app, mongo)
register_auth_routes(app, mongo)
register_ai_routes(app, mongo)
register_conversation_qa_routes(app, mongo)
app.register_blueprint(memory_aid_routes)

# Configure database access for routes that use the Blueprint pattern
@app.before_request
def before_request():
    app.config["DATABASE"] = mongo.db

# Add a route for the home page
@app.route('/')
def main_home():
    """Serve the home page or redirect to frontend."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ZAAM Home</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                text-align: center;
                padding: 50px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
            }
            p {
                color: #666;
                line-height: 1.6;
            }
            .btn {
                display: inline-block;
                background-color: #4CAF50;
                color: white;
                padding: 15px 32px;
                text-align: center;
                text-decoration: none;
                font-size: 16px;
                margin: 20px 10px;
                cursor: pointer;
                border-radius: 8px;
                border: none;
            }
            .token-info {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin-top: 20px;
                text-align: left;
                display: none;
            }
        </style>
        <script>
            window.onload = function() {
                // Check if we have auth data
                const authToken = localStorage.getItem('authToken');
                const userId = localStorage.getItem('userId');
                const isNewUser = localStorage.getItem('isNewUser');
                
                const statusElement = document.getElementById('auth-status');
                const tokenInfoElement = document.getElementById('token-info');
                const tokenTextElement = document.getElementById('token-text');
                
                if (authToken && userId) {
                    statusElement.textContent = 'You are authenticated!';
                    
                    // Show part of the token for verification
                    tokenInfoElement.style.display = 'block';
                    const tokenPreview = authToken.substring(0, 15) + '...' + authToken.substring(authToken.length - 10);
                    tokenTextElement.textContent = `Token: ${tokenPreview} | User ID: ${userId}`;
                    
                    if (isNewUser) {
                        document.getElementById('new-user-message').style.display = 'block';
                    }
                } else {
                    statusElement.textContent = 'You are not authenticated. Please sign in through the app.';
                }
            }
        </script>
    </head>
    <body>
        <div class="container">
            <h1>Welcome to ZAAM</h1>
            <p id="auth-status">Checking authentication status...</p>
            
            <div id="new-user-message" style="display: none; margin: 15px; padding: 10px; background-color: #fff3cd; border-radius: 5px;">
                <p>Welcome new user! You need to complete your profile setup.</p>
            </div>
            
            <div id="token-info" class="token-info">
                <p id="token-text"></p>
                <p>This token has been stored in your browser and will be used to authenticate your requests.</p>
            </div>
            
            <p>Please return to the app to continue using ZAAM.</p>
            <a href="zaam://" class="btn">Open App</a>
        </div>
    </body>
    </html>
    """

# Set up scheduler for recurring tasks
scheduler = BackgroundScheduler()

def check_upcoming_reminders():
    """Function to call the reminder check endpoint"""
    try:
        base_url = os.getenv("BASE_URL", "http://localhost:5000")
        logger.info(f"Checking upcoming reminders from {base_url}")
        response = requests.get(f"{base_url}/api/reminder/check_upcoming")
        logger.info(f"Scheduled reminder check completed: status={response.status_code}")
        if response.status_code != 200:
            logger.error(f"Error checking reminders: {response.text}")
    except Exception as e:
        logger.error(f"Failed to execute scheduled reminder check: {str(e)}")

# Start the scheduler after all routes are registered
def start_scheduler():
    # Schedule the reminder check every minute for testing (normally would be 15 minutes)
    scheduler.add_job(check_upcoming_reminders, 'interval', minutes=1)
    scheduler.start()
    logger.info("Reminder scheduler started - will check for upcoming reminders every minute")

# Start the scheduler when the app runs
with app.app_context():
    start_scheduler()

if __name__ == '__main__':
    print("\n=== ZAAM Server Initialization ===")
    
    # Initialize the QA system if models are available
    if models_available:
        try:
            initialize_qa_system()
            print("AI models initialized and ready for use.")
        except Exception as e:
            print(f"WARNING: Failed to initialize QA system: {str(e)}")
            print("ConversationQA features may not be available.")
    
    print("=== Server Initialization Complete ===\n")
    
    # from waitress import serve
    # serve(app, host="0.0.0.0", port=5000)

    app.run(host="0.0.0.0", port=5000, debug=True)