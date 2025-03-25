from flask import Flask
from flask_cors import CORS
from flask_pymongo import PyMongo
from flask_session import Session
from dotenv import load_dotenv
import os
import sys
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from pymongo import MongoClient
import json

# Import route modules
from routes.main_routes import register_main_routes
from routes.user_routes import register_user_routes
from routes.auth_routes import register_auth_routes
from routes.reminder_routes import register_reminder_routes
from routes.google_auth_routes import register_google_auth_routes
from routes.ai_routes import register_ai_routes
from routes.memory_aid_routes import memory_aid_routes
from routes.reminder_ai_routes import register_reminder_ai_routes
from routes.reminder_sync_routes import register_reminder_sync_routes

# Load environment variables
load_dotenv()

# Google OAuth Config - Important for authentication to work
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"  # Only for development

# Try to locate and use the credentials file from Reminder-v2
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
reminder_v2_dir = os.path.join(project_root, 'Reminder-v2')
credentials_source = os.path.join(reminder_v2_dir, 'credentialsOAuth.json')
credentials_dest = os.path.join(os.path.dirname(__file__), 'credentialsOAuth.json')

# Copy credentials file if it exists and destination doesn't
if os.path.exists(credentials_source) and not os.path.exists(credentials_dest):
    try:
        import shutil
        shutil.copy2(credentials_source, credentials_dest)
        print(f"Copied OAuth credentials from {credentials_source} to {credentials_dest}")
    except Exception as e:
        print(f"Error copying OAuth credentials: {str(e)}")

# Set the path to credentials file for the application
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_dest if os.path.exists(credentials_dest) else ""

# Extract credentials from the JSON file if it exists
if os.path.exists(credentials_dest):
    try:
        with open(credentials_dest, 'r') as f:
            creds = json.load(f)
            if 'web' in creds:
                # Set environment variables from the credentials file
                if 'client_id' in creds['web'] and not os.getenv("GOOGLE_CLIENT_ID"):
                    os.environ["GOOGLE_CLIENT_ID"] = creds['web']['client_id']
                    print(f"Set GOOGLE_CLIENT_ID from credentials file")
                
                if 'client_secret' in creds['web'] and not os.getenv("GOOGLE_CLIENT_SECRET"):
                    os.environ["GOOGLE_CLIENT_SECRET"] = creds['web']['client_secret']
                    print(f"Set GOOGLE_CLIENT_SECRET from credentials file")
                
                # Set redirect URI if not already set
                if not os.getenv("GOOGLE_REDIRECT_URI"):
                    if 'redirect_uris' in creds['web'] and creds['web']['redirect_uris']:
                        # Use the first redirect URI from the file
                        os.environ["GOOGLE_REDIRECT_URI"] = creds['web']['redirect_uris'][0]
                        print(f"Set GOOGLE_REDIRECT_URI to {os.environ['GOOGLE_REDIRECT_URI']} from credentials file")
                    else:
                        # Set default redirect URI
                        os.environ["GOOGLE_REDIRECT_URI"] = "http://192.168.1.2:5000/callback"
                        print(f"Set default GOOGLE_REDIRECT_URI")
    except Exception as e:
        print(f"Error extracting credentials from file: {str(e)}")

# Get credentials from environment or use defaults from Reminder-v2
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://192.168.1.2:5000/callback")

# Set default secret key from Reminder-v2 if not provided
if not os.getenv("JWT_SECRET"):
    os.environ["JWT_SECRET"] = "e5QD8iIgEo0iBvi2Lx2bgK89vHtcqV"  # Same as in Reminder-v2

# Enhanced debugging for OAuth
def print_oauth_debug_info():
    """Print debug info for OAuth configuration"""
    print("\n==== OAUTH DEBUG INFO ====")
    print(f"GOOGLE_CLIENT_ID: {'Set' if os.getenv('GOOGLE_CLIENT_ID') else 'Not set'}")
    print(f"GOOGLE_CLIENT_SECRET: {'Set' if os.getenv('GOOGLE_CLIENT_SECRET') else 'Not set'}")
    print(f"GOOGLE_REDIRECT_URI: {os.getenv('GOOGLE_REDIRECT_URI')}")
    print(f"GOOGLE_APPLICATION_CREDENTIALS: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}")
    print(f"JWT_SECRET: {'Set' if os.getenv('JWT_SECRET') else 'Not set'}")
    
    # Check if credentials file exists
    creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if creds_path and os.path.exists(creds_path):
        print(f"Credentials file exists at: {creds_path}")
        try:
            with open(creds_path, 'r') as f:
                creds = json.load(f)
                if 'web' in creds:
                    print("Credentials file contains 'web' configuration")
                    if 'client_id' in creds['web']:
                        print("Credentials file contains client_id")
                    if 'client_secret' in creds['web']:
                        print("Credentials file contains client_secret")
                    print(f"Redirect URIs in credentials: {creds['web'].get('redirect_uris', [])}")
                else:
                    print("WARNING: Credentials file doesn't contain 'web' configuration")
        except Exception as e:
            print(f"Error reading credentials file: {str(e)}")
    else:
        print(f"WARNING: Credentials file doesn't exist at path: {creds_path}")
    
    print("==========================\n")

# Run debug info
print_oauth_debug_info()

# Debug environment variables
jwt_secret = os.getenv("JWT_SECRET")
if not jwt_secret:
    print("WARNING: JWT_SECRET environment variable is not set!")
else:
    print("JWT_SECRET environment variable is loaded")

# Check if Google OAuth credentials are set
if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
    print("WARNING: Google OAuth credentials are not set properly!")
    print(f"GOOGLE_CLIENT_ID: {'Set' if GOOGLE_CLIENT_ID else 'Not set'}")
    print(f"GOOGLE_CLIENT_SECRET: {'Set' if GOOGLE_CLIENT_SECRET else 'Not set'}")
else:
    print("Google OAuth credentials loaded successfully")

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
register_reminder_routes(app, mongo)
register_google_auth_routes(app, mongo)
register_ai_routes(app, mongo)
register_reminder_ai_routes(app, mongo)
register_reminder_sync_routes(app, mongo)
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

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=5000, debug=True)