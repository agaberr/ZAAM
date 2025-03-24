from flask import Flask
from flask_cors import CORS
from flask_pymongo import PyMongo
from dotenv import load_dotenv
import os
import sys
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from pymongo import MongoClient

# Import route modules
from routes.main_routes import register_main_routes
from routes.user_routes import register_user_routes
from routes.auth_routes import register_auth_routes
from routes.reminder_routes import register_reminder_routes
from routes.google_auth_routes import register_google_auth_routes
from routes.ai_routes import register_ai_routes
from routes.memory_aid_routes import memory_aid_routes

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow requests from all origins

# Configure session
app.secret_key = os.getenv("SECRET_KEY", os.getenv("JWT_SECRET", "super-secret-key"))

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
app.register_blueprint(memory_aid_routes)

# Configure database access for routes that use the Blueprint pattern
@app.before_request
def before_request():
    app.config["DATABASE"] = mongo.db

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)
