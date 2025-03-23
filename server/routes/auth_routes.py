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

# Middleware: Token authentication
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Check if token is in headers
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]
        
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


def register_auth_routes(app, mongo):
    @app.route('/api/auth/login', methods=['POST'])
    def login():
        try:
            data = request.json
            email = data.get('email')
            password = data.get('password')

            user = User.find_by_email(mongo.db, email)
            if not user or not user.check_password(password):
                return jsonify({"error": "Invalid email or password"}), 401

            token = user.generate_auth_token()
            return jsonify({
                "message": "Login successful", 
                "token": token,
                "user_id": str(user._id)
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
            
    @app.route('/api/auth/register', methods=['POST'])
    def register():
        try:
            user_data = request.json
            
            user = User(
                full_name=user_data.get('full_name'),
                age=user_data.get('age'),
                gender=user_data.get('gender'),
                contact_info=user_data.get('contact_info', {}),
                password=user_data.get('password'),
                emergency_contacts=user_data.get('emergency_contacts', [])
            )
            success, errors = user.save(mongo.db)
            if not success:
                return jsonify({"error": "Validation failed", "details": errors}), 400
            
            return jsonify({"message": "User registered successfully", "user_id": str(user._id)}), 201
        except Exception as e:
            return jsonify({"error": str(e)}), 500
