from flask import jsonify, request, redirect, session, url_for
from models.google_oauth import GoogleOAuthService
import jwt
import os

# Load JWT secret from environment variables
JWT_SECRET = os.getenv("JWT_SECRET")

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
    
    @app.route('/api/auth/google/connect', methods=['GET'])
    def google_connect():
        """Start Google OAuth flow by redirecting to Google's auth page."""
        try:
            # Get authenticated user ID
            user_id = get_authenticated_user_id()
            if not user_id:
                return jsonify({"error": "Authentication required"}), 401
                
            # Generate authorization URL
            auth_url, state = oauth_service.get_authorization_url()
            
            # Store state and user_id in session for verification
            session['google_auth_state'] = state
            session['google_auth_user_id'] = user_id
            
            # Return the URL for frontend to redirect
            return jsonify({"authorization_url": auth_url})
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/auth/google/callback', methods=['GET'])
    def google_callback():
        """Handle callback from Google OAuth."""
        try:
            # Get state and code from query parameters
            state = request.args.get('state')
            code = request.args.get('code')
            error = request.args.get('error')
            
            # Check for errors
            if error:
                return redirect('/google-connect-failure?error=' + error)
                
            # Verify state to prevent CSRF
            if state != session.get('google_auth_state'):
                return redirect('/google-connect-failure?error=invalid_state')
                
            # Get user_id from session
            user_id = session.get('google_auth_user_id')
            if not user_id:
                return redirect('/google-connect-failure?error=missing_user_id')
                
            # Exchange code for tokens
            tokens = oauth_service.exchange_code_for_tokens(code)
            
            # Save tokens for user
            success = oauth_service.save_tokens_for_user(mongo.db, user_id, tokens)
            
            if not success:
                return redirect('/google-connect-failure?error=token_save_failed')
                
            # Clear session data
            session.pop('google_auth_state', None)
            session.pop('google_auth_user_id', None)
            
            # Redirect to success page
            return redirect('/google-connect-success')
            
        except Exception as e:
            return redirect('/google-connect-failure?error=' + str(e))
    
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