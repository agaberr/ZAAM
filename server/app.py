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

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure MongoDB
mongo_uri = os.getenv("MONGO_URI")
if not mongo_uri:
    print("Error: MONGO_URI environment variable not set")
    sys.exit(1)

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
mongo = PyMongo(app)

# Register routes
register_main_routes(app)
register_user_routes(app, mongo)
register_auth_routes(app, mongo)

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)
