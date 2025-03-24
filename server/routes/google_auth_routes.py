from flask import jsonify, request, redirect, session, url_for, Blueprint
from models.google_oauth import GoogleOAuthService
import jwt
import os
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials

# Load JWT secret from environment variables
JWT_SECRET = os.getenv("JWT_SECRET")

# Create blueprint
google_auth_bp = Blueprint('google_auth_routes', __name__)

def register_google_auth_routes(app, mongo):
    oauth_service = GoogleOAuthService()
    
    def get_authenticated_user_id():
        # Get the token from the Authorization header
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return None
            
        token = auth_header.split(' ')[1]
        try:
            # Decode the token
            payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            return payload.get('user_id')
        except Exception:
            return None
    
    @app.route('/api/google/auth', methods=['GET'])
    def google_auth():
        try:
            flow = Flow.from_client_secrets_file(
                app.config["GOOGLE_CLIENT_SECRETS_FILE"],
                scopes=app.config["GOOGLE_OAUTH_SCOPES"],
                redirect_uri=request.args.get('redirect_uri', 'http://localhost:3000/auth/google/callback')
            )
            
            authorization_url, state = flow.authorization_url(
                access_type='offline',
                include_granted_scopes='true'
            )
            
            # Store the state in the session
            session['google_auth_state'] = state
            
            return jsonify({
                "authorization_url": authorization_url
            })
        except Exception as e:
            print(f"Google auth error: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/google/callback', methods=['POST'])
    def google_callback():
        try:
            data = request.get_json()
            if not data or 'code' not in data or 'redirect_uri' not in data:
                return jsonify({"error": "Missing code or redirect_uri"}), 400
            
            flow = Flow.from_client_secrets_file(
                app.config["GOOGLE_CLIENT_SECRETS_FILE"],
                scopes=app.config["GOOGLE_OAUTH_SCOPES"],
                redirect_uri=data['redirect_uri']
            )
            
            # Use the authorization code to get credentials
            flow.fetch_token(code=data['code'])
            credentials = flow.credentials
            
            # Convert credentials to a dict that can be stored and transmitted
            credentials_dict = {
                'token': credentials.token,
                'refresh_token': credentials.refresh_token,
                'token_uri': credentials.token_uri,
                'client_id': credentials.client_id,
                'client_secret': credentials.client_secret,
                'scopes': credentials.scopes
            }
            
            return jsonify({
                "credentials": credentials_dict
            })
        except Exception as e:
            print(f"Google callback error: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/auth/google/disconnect', methods=['POST'])
    def google_disconnect():
        """Disconnect Google account."""
        try:
            # Get authenticated user ID
            user_id = get_authenticated_user_id()
            if not user_id:
                return jsonify({"error": "Authentication required"}), 401
                
            # Revoke access
            success = oauth_service.revoke_access(mongo.db, user_id)
            
            if success:
                return jsonify({"message": "Google account disconnected successfully"})
            else:
                return jsonify({"error": "Failed to disconnect Google account"}), 500
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/auth/google/status', methods=['GET'])
    def google_status():
        """Check if user has connected Google account."""
        try:
            # Get authenticated user ID
            user_id = get_authenticated_user_id()
            if not user_id:
                return jsonify({"error": "Authentication required"}), 401
                
            # Check if user has Google tokens
            tokens = mongo.db.google_tokens.find_one({"user_id": user_id})
            
            return jsonify({
                "connected": tokens is not None,
                "connected_at": tokens.get("saved_at").isoformat() if tokens else None
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500 