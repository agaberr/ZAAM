from flask import Blueprint, jsonify, request
from models.user import User
import jwt
import os
from functools import wraps
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# Get JWT secret key from environment
JWT_SECRET = os.getenv("JWT_SECRET")

# Create a Blueprint for authentication routes
auth_bp = Blueprint("auth", __name__)
# Decorator to check for JWT token in request headers
# Used to protect routes that require user login

def extract_token():
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Bearer '):
        return auth_header.split(' ')[1]
    return None
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        token = extract_token()
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        
        try:
            # Decode the token
            data = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
            current_user_id = data['user_id']
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(current_user_id, *args, **kwargs)
    
    return decorated


def authRoutes(app, mongo):




    ##################################### LOGIN ROUTE START #####################################
    @app.route('/api/auth/login', methods=['POST'])
    def login():

        reposnsedata = request.get_json()
        if not reposnsedata:
            return jsonify({"error": "Invalid request"}), 400
        
        email = reposnsedata.get('email')

        password = reposnsedata.get('password')

        if not email or not password:
            return jsonify({"error": "Please enter email and password"}), 400
        
        user = User.find_by_email(mongo.db, email)
        if not user or not user.check_password(password):
            return jsonify({"error": "email or password are incorrect"}), 401
        
        token = user.generate_auth_token()



        return jsonify({"message": "Login successful", "token": token,"user_id": str(user._id)})
        
    ##################################### LOGIN ROUTE END #####################################


    ##################################### REGISTER ROUTE START #####################################
    @app.route('/api/auth/register', methods=['POST'])
    def register():
        reposnsedata = request.get_json()
        if not reposnsedata:
            return jsonify({"error": "Invalid request"}), 400
        
        reposnsedata = User(
            full_name=reposnsedata.get('full_name'),
            age=reposnsedata.get('age'),
            gender=reposnsedata.get('gender'),
            contact_info=reposnsedata.get('contact_info', {}),
            password=reposnsedata.get('password'),
            emergency_contacts=reposnsedata.get('emergency_contacts', [])
        )
        success, errors = reposnsedata.save(mongo.db)
        if not success:
            return jsonify({"error": "Validation failed", "details": errors}), 400
        
        return jsonify({"message": "User registered successfully", "user_id": str(reposnsedata._id)}), 201
    