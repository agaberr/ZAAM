##################################### IMPORTS START #####################################
import os
import sys
import json
import logging
import importlib
import subprocess
from pathlib import Path
from flask_cors import CORS
from dotenv import load_dotenv
from pymongo import MongoClient
from flask_session import Session
from flask_pymongo import PyMongo
from ConversationQA.qa_initializer import initialize_qa_system # Import QA initializer
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from flask import Flask, request, jsonify, session, redirect, url_for, Request, current_app, render_template, make_response



# Import route modules
from routes.main_routes import register_main_routes
from routes.user_routes import register_user_routes
from routes.auth_routes import register_auth_routes
from routes.ai_routes import register_ai_routes
from routes.memory_aid_routes import memory_aid_routes
from routes.cognitive_game_routes import cognitive_game_routes
from routes.speech_routes import speech_bp

##################################### IMPORTS END #####################################

##################################### LOGGING START #####################################

# Configure logging for debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
##################################### LOGGING END #####################################

##################################### DOWNLOADING MODELS START #####################################

# The models are so big in size so they are downloaded directly from the drive if not found

# Add project root to sys.path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"Added {project_root} to Python path")

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
    weather_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Weather')
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)
    if not os.path.exists(weather_dir):
        os.makedirs(weather_dir, exist_ok=True)
        
    required_models = [
        "bert_seq2seq_ner.pt",
        "pronoun_resolution_model_full.pt",
        "extractiveQA.pt",
        "vectorizer.pkl",
        "classifier_model.pkl",
    ]
    
    # Add weather model check
    weather_model = "weather_model.pt"
    weather_model_path = os.path.join(weather_dir, weather_model)
    
    missing_models = []
    for model_name in required_models:
        model_path = os.path.join(models_dir, model_name)
        if not os.path.exists(model_path):
            missing_models.append(model_name)
    
    # Check weather model
    if not os.path.exists(weather_model_path):
        missing_models.append(weather_model)
    
    if missing_models:
        print("\nWARNING: The following models are missing:")
        for model in missing_models:
            print(f"  - {model}")
        print("The ConversationQA and Weather functionality may not work correctly.\n")
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
    weather_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Weather')
    
    # Direct download URLs (not through Google Drive API)
    direct_urls = {
        "bert_seq2seq_ner.zip": "https://drive.google.com/uc?id=1lOSkIPGU4TX7L727OmkBL3f0Fh_W3_Fv&export=download",
        "pronoun_resolution_model_full.zip": "https://drive.google.com/uc?id=1DhlkILm1kzD8gPbEU0NlTUGeDA_uGisk&export=download",
        "extractiveQA.zip": "https://drive.google.com/uc?id=1ph3yhuAz7fmTv8lat8fbdwzlHYSa4y4U&export=download",
        "vectorizer.zip": "https://drive.google.com/uc?id=1QqxFX0VhLYKdgWWJBYnE9YVnEP6bZKcY&export=download",
        "weather_model.zip": "https://drive.google.com/uc?id=1KBTMhfV9MaLuM2V7ID2ZEEi7ov5szG6M&export=download"
    }
    
    # Map models to their containing archives
    model_to_archive = {
        "bert_seq2seq_ner.pt": "bert_seq2seq_ner.zip",
        "pronoun_resolution_model_full.pt": "pronoun_resolution_model_full.zip",
        "extractiveQA.pt": "extractiveQA.zip",
        "vectorizer.pkl": "vectorizer.zip",
        "classifier_model.pkl": "vectorizer.zip",
        "weather_model.pt": "weather_model.zip"
    }
    
    # Map models to their target directories
    model_to_dir = {
        "weather_model.pt": weather_dir
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
                        
                        # Determine target directory for extraction
                        target_dir = models_dir
                        for model in missing_models:
                            if model_to_archive.get(model) == archive:
                                target_dir = model_to_dir.get(model, models_dir)
                                break
                        
                        # Handle ZIP extraction
                        try:
                            with zipfile.ZipFile(archive_path) as zf:
                                zf.extractall(target_dir)
                            print(f"Extracted {archive} using zipfile")
                        except Exception as e:
                            print(f"zipfile extraction failed: {e}")
                            
                            # Try with subprocess
                            try:
                                subprocess.run(['unzip', archive_path, '-d', target_dir], check=True)
                                print(f"Extracted {archive} using unzip command")
                            except Exception as e2:
                                print(f"unzip command failed: {e2}")
                                
                                # Try with 7z as last resort
                                try:
                                    subprocess.run(['7z', 'x', archive_path, f'-o{target_dir}'], check=True)
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
                            
                            # Determine target directory for extraction
                            target_dir = models_dir
                            for model in missing_models:
                                if model_to_archive.get(model) == archive:
                                    target_dir = model_to_dir.get(model, models_dir)
                                    break
                            
                            # Handle ZIP extraction
                            try:
                                with zipfile.ZipFile(archive_path) as zf:
                                    zf.extractall(target_dir)
                                print(f"Extracted {archive} using zipfile")
                            except Exception as e:
                                print(f"zipfile extraction failed: {e}")
                                
                                # Try with subprocess
                                try:
                                    subprocess.run(['unzip', archive_path, '-d', target_dir], check=True)
                                    print(f"Extracted {archive} using unzip command")
                                except Exception as e2:
                                    print(f"unzip command failed: {e2}")
                                    
                                    # Try with 7z as last resort
                                    try:
                                        subprocess.run(['7z', 'x', archive_path, f'-o{target_dir}'], check=True)
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

##################################### DOWNLOADING MODELS END #####################################

##################################### VERIFY MODELS DOWNLOADED START #####################################

# Run model checks and downloads before importing anything else
print("\n===== ZAAM Server Initialization =====")
print("Checking for AI models...")
models_available, missing_models = verify_models()

if not models_available:
    print("Missing models detected. Attempting to download...")
    if download_models(missing_models):
        print("Model setup completed successfully!")
else:
    print("All models are available!")

##################################### VERIFY MODELS DOWNLOADED END #####################################

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
if not mongo_uri:
    print("Error: MONGO_URI environment variable not set")
    sys.exit(1)

# Configure PyMongo
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
    print("\n=== ZAAM Server Initialization ===")
    
    if models_available:
        initialize_qa_system()
    
    print("=== Server Initialization Complete ===\n")

    app.run(host="0.0.0.0", port=5000, debug=True)