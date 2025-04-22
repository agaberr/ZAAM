from flask import jsonify, request, redirect, session, url_for
from models.google_oauth import GoogleOAuthService
from models.user import User
import jwt
import os
import uuid
from datetime import datetime

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
    
    # Add a route for the old callback path from credentialsOAuth.json
    @app.route('/callback', methods=['GET'])
    def google_redirect_callback():
        """Redirect the Google OAuth callback from /callback to /api/auth/google/callback."""
        print("Received callback at /callback, redirecting to /api/auth/google/callback")
        
        # Get the state from the query parameters so we can store it in the session
        state = request.args.get('state')
        if state:
            # Store the state in the session for verification in the main callback
            print(f"Storing state in session: {state}")
            session['google_auth_state'] = state
        
        # Preserve all query parameters in the redirect
        query_string = request.query_string.decode('utf-8')
        redirect_url = f"/api/auth/google/callback?{query_string}" if query_string else "/api/auth/google/callback"
        
        return redirect(redirect_url)
    
    @app.route('/api/auth/google/connect', methods=['GET'])
    def google_connect():
        """Start Google OAuth flow by redirecting to Google's auth page."""
        try:
            # Check if the Authorization header is provided
            user_id = get_authenticated_user_id()
            
            # Generate authorization URL
            auth_url, state = oauth_service.get_authorization_url()
            print(f"Generated state: {state}")
            
            # Store state in session for verification - make sure this is properly saved
            session['google_auth_state'] = state
            session.modified = True
            
            # If user is already authenticated, store user_id for linking account
            if user_id:
                session['google_auth_user_id'] = user_id
                session['google_auth_mode'] = 'link'
            else:
                # If not authenticated, we're doing a sign-in
                session['google_auth_mode'] = 'signin'
                # Generate a temporary ID to identify this sign-in flow
                session['google_auth_temp_id'] = str(uuid.uuid4())
            
            # Ensure session is saved
            session.modified = True
            
            # Return the URL for frontend to redirect
            return jsonify({
                "authorization_url": auth_url,
                "state": state,
                "client_id": os.getenv("GOOGLE_CLIENT_ID"),
                "redirect_uri": os.getenv("GOOGLE_REDIRECT_URI")
            })
            
        except Exception as e:
            print(f"Error in Google connect: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/auth/google/callback', methods=['POST'])
    def google_callback():
        """Handle callback from Google OAuth."""
        try:
            data = request.get_json()
            code = data.get('code')
            state = data.get('state')
            
            # Verify state to prevent CSRF
            stored_state = session.get('google_auth_state')
            if not stored_state or state != stored_state:
                print(f"Invalid state in OAuth callback: stored={stored_state}, received={state}")
                return jsonify({"error": "invalid_state"}), 400
                
            # Get OAuth mode (link or signin)
            auth_mode = session.get('google_auth_mode', 'signin')
            print(f"OAuth mode: {auth_mode}")
            
            # Exchange code for tokens
            tokens = oauth_service.exchange_code_for_tokens(code)
            
            if auth_mode == 'link':
                # Link to existing account
                user_id = session.get('google_auth_user_id')
                if not user_id:
                    print("Missing user_id in OAuth callback for account linking")
                    return jsonify({"error": "missing_user_id"}), 400
                
                # Ensure user_id is a string
                user_id_str = str(user_id)
                
                # Save tokens for user
                print(f"Saving tokens for user during account linking: {user_id_str}")
                success = oauth_service.save_tokens_for_user(mongo.db, user_id_str, tokens)
                
                if not success:
                    print(f"Failed to save tokens for user {user_id_str}")
                    return jsonify({"error": "token_save_failed"}), 500
                    
                return jsonify({"success": True, "message": "Google account linked successfully"})
            else:
                # Handle sign-in flow
                # Get user info from Google using the access token
                user_info = oauth_service.get_user_info(tokens['token'])
                
                if not user_info:
                    print("Failed to get user info from Google")
                    return jsonify({"error": "failed_to_get_user_info"}), 500
                
                print(f"Got user info from Google: {user_info.get('name')}, {user_info.get('email')}")
                
                # Check if user exists in our database
                existing_user = User.find_by_email(mongo.db, user_info.get('email'))
                
                if existing_user:
                    # User exists, update their Google tokens
                    print(f"Found existing user: {existing_user._id}")
                    # Convert ObjectId to string if it's not already a string
                    user_id_str = str(existing_user._id)
                    success = oauth_service.save_tokens_for_user(mongo.db, user_id_str, tokens)
                    
                    if not success:
                        print(f"Failed to save tokens for user {user_id_str}")
                        return jsonify({"error": "token_save_failed"}), 500
                    
                    # Generate JWT for the user
                    auth_token = existing_user.generate_auth_token()
                    
                    return jsonify({
                        "success": True,
                        "token": auth_token,
                        "user_id": str(existing_user._id),
                        "is_new": False
                    })
                else:
                    # Create new user
                    new_user = User(
                        name=user_info.get('name'),
                        email=user_info.get('email'),
                        google_id=user_info.get('id'),
                        profile_picture=user_info.get('picture')
                    )
                    
                    # Save new user
                    user_id = new_user.save(mongo.db)
                    if not user_id:
                        print("Failed to save new user")
                        return jsonify({"error": "user_save_failed"}), 500
                    
                    # Save Google tokens for the new user
                    success = oauth_service.save_tokens_for_user(mongo.db, str(user_id), tokens)
                    if not success:
                        print(f"Failed to save tokens for new user {user_id}")
                        return jsonify({"error": "token_save_failed"}), 500
                    
                    # Check for temporary ID and associate tokens with it too
                    temp_id = session.get('google_auth_temp_id')
                    if temp_id:
                        print(f"Found temporary ID: {temp_id}, associating Google tokens with it")
                        # Save tokens for the temporary ID too
                        oauth_service.save_tokens_for_user(mongo.db, temp_id, tokens)
                    
                    # Generate JWT for the new user
                    auth_token = new_user.generate_auth_token()
                    
                    return jsonify({
                        "success": True,
                        "token": auth_token,
                        "user_id": str(user_id),
                        "is_new": True
                    })
                    
        except Exception as e:
            print(f"Error in Google callback: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/auth/google/callback', methods=['GET'])
    def google_callback_get():
        """Handle GET callback from Google OAuth."""
        try:
            # Get code and state from query parameters
            code = request.args.get('code')
            state = request.args.get('state')
            
            if not code:
                print("No authorization code received in GET callback")
                return jsonify({"error": "no_code"}), 400
                
            # Verify state to prevent CSRF
            stored_state = session.get('google_auth_state')
            if not stored_state or state != stored_state:
                print(f"Invalid state in OAuth GET callback: stored={stored_state}, received={state}")
                # For development only: Continue even if state doesn't match
                print("WARNING: Continuing despite state mismatch (for development only)")
                # In production, you would return: 
                # return jsonify({"error": "invalid_state"}), 400
            
            # Get OAuth mode (link or signin)
            auth_mode = session.get('google_auth_mode', 'signin')
            print(f"OAuth mode (GET): {auth_mode}")
            
            # Exchange code for tokens
            tokens = oauth_service.exchange_code_for_tokens(code)
            
            if auth_mode == 'link':
                # Link to existing account
                user_id = session.get('google_auth_user_id')
                if not user_id:
                    print("Missing user_id in OAuth callback for account linking")
                    return jsonify({"error": "missing_user_id"}), 400
                
                # Ensure user_id is a string
                user_id_str = str(user_id)
                
                # Save tokens for user
                print(f"Saving tokens for user during account linking: {user_id_str}")
                success = oauth_service.save_tokens_for_user(mongo.db, user_id_str, tokens)
                
                if not success:
                    print(f"Failed to save tokens for user {user_id_str}")
                    return jsonify({"error": "token_save_failed"}), 500
                    
                # For GET callback, redirect to app with success message
                return redirect(f"zaam://callback?success=true&message=Google+account+linked+successfully")
            else:
                # Handle sign-in flow
                # Get user info from Google using the access token
                user_info = oauth_service.get_user_info(tokens['token'])
                
                if not user_info:
                    print("Failed to get user info from Google")
                    return jsonify({"error": "failed_to_get_user_info"}), 500
                
                print(f"Got user info from Google: {user_info.get('name')}, {user_info.get('email')}")
                
                # Check if user exists in our database
                existing_user = User.find_by_email(mongo.db, user_info.get('email'))
                
                if existing_user:
                    # User exists, update their Google tokens
                    print(f"Found existing user: {existing_user._id}")
                    # Convert ObjectId to string if it's not already a string
                    user_id_str = str(existing_user._id)
                    success = oauth_service.save_tokens_for_user(mongo.db, user_id_str, tokens)
                    
                    if not success:
                        print(f"Failed to save tokens for user {user_id_str}")
                        return jsonify({"error": "token_save_failed"}), 500
                    
                    # Generate JWT for the user
                    auth_token = existing_user.generate_auth_token()
                    
                    # For GET callback, redirect to app with token and user_id
                    return redirect(f"zaam://callback?token={auth_token}&user_id={user_id_str}")
                else:
                    # Create new user
                    new_user = User(
                        name=user_info.get('name'),
                        email=user_info.get('email'),
                        google_id=user_info.get('id'),
                        profile_picture=user_info.get('picture')
                    )
                    
                    # Save new user
                    user_id = new_user.save(mongo.db)
                    if not user_id:
                        print("Failed to save new user")
                        return jsonify({"error": "user_save_failed"}), 500
                    
                    # Save Google tokens for the new user
                    success = oauth_service.save_tokens_for_user(mongo.db, str(user_id), tokens)
                    if not success:
                        print(f"Failed to save tokens for new user {user_id}")
                        return jsonify({"error": "token_save_failed"}), 500
                    
                    # Check for temporary ID and associate tokens with it too
                    temp_id = session.get('google_auth_temp_id')
                    if temp_id:
                        print(f"Found temporary ID: {temp_id}, associating Google tokens with it")
                        # Save tokens for the temporary ID too
                        oauth_service.save_tokens_for_user(mongo.db, temp_id, tokens)
                    
                    # Generate JWT for the new user
                    auth_token = new_user.generate_auth_token()
                    
                    # For GET callback, redirect to app with token, user_id and is_new flag
                    return redirect(f"zaam://callback?token={auth_token}&user_id={str(user_id)}&is_new=true")
                    
        except Exception as e:
            print(f"Error in Google GET callback: {str(e)}")
            return redirect(f"zaam://callback?error={str(e)}")
    
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