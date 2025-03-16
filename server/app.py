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
from routes.chat_routes import register_chat_routes
from routes.schedule_routes import register_schedule_routes

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

# Test MongoDB connection before initializing PyMongo
def test_mongo_connection():
    try:
        # Create a temporary client to test the connection
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        # The ping command is lightweight and doesn't require auth
        client.admin.command('ping')
        print("MongoDB connection successful!")
        return True
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        print(f"MongoDB connection failed: {e}")
        return False

# Only configure PyMongo if the connection test passes
if test_mongo_connection():
    app.config["MONGO_URI"] = mongo_uri
    mongo = PyMongo(app)
    
    # Register routes
    register_main_routes(app)
    register_user_routes(app, mongo)
    register_chat_routes(app, mongo)
    register_schedule_routes(app, mongo)
    
    if __name__ == '__main__':
        print("Starting Flask server...")
        app.run(debug=True)
else:
    print("Failed to connect to MongoDB. Server not started.")
    sys.exit(1) 