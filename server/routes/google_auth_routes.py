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
            return jsonify({"authorization_url": auth_url, "state": state})
            
        except Exception as e:
            print(f"Error in Google connect: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/auth/google/callback', methods=['GET'])
    def google_callback():
        """Handle callback from Google OAuth."""
        try:
            # Get state and code from query parameters
            state = request.args.get('state')
            code = request.args.get('code')
            error = request.args.get('error')
            
            # Debug session state
            print(f"Session state: {session.get('google_auth_state')}")
            print(f"Received state: {state}")
            
            # Prepare HTML template for redirecting to the app
            html_redirect_template = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Authentication Completed</title>
                <script>
                    // Store auth data in localStorage for the web app to use
                    function storeAuthData(authToken, userId, isNew) {
                        if (authToken) {
                            localStorage.setItem('authToken', authToken);
                            localStorage.setItem('userId', userId);
                            if (isNew) {
                                localStorage.setItem('isNewUser', 'true');
                            }
                            console.log('Auth data stored in localStorage');
                        }
                    }
                
                    window.onload = function() {
                        // Parse URL params
                        const urlParams = new URLSearchParams(window.location.search);
                        const authToken = urlParams.get('token');
                        const userId = urlParams.get('user_id');
                        const isNew = urlParams.get('is_new') === 'true';
                        
                        // Store auth data if available
                        if (authToken && userId) {
                            storeAuthData(authToken, userId, isNew);
                        }
                    
                        // Try to use the deep link
                        window.location.replace("%s");
                        
                        // Fallback message in case deep link doesn't work
                        setTimeout(function() {
                            document.getElementById('message').style.display = 'block';
                        }, 2000);
                    }
                    
                    function goToHomePage() {
                        // Redirect to the frontend home page
                        window.location.href = "/";
                    }
                </script>
                <style>
                    body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
                    #message { display: none; margin-top: 20px; }
                    .button {
                        background-color: #4CAF50;
                        border: none;
                        color: white;
                        padding: 15px 32px;
                        text-align: center;
                        text-decoration: none;
                        display: inline-block;
                        font-size: 16px;
                        margin: 4px 2px;
                        cursor: pointer;
                        border-radius: 8px;
                    }
                </style>
            </head>
            <body>
                <h2>Authentication Completed</h2>
                <p>Redirecting you back to the ZAAM app...</p>
                <div id="message">
                    <p>If the app doesn't open automatically, you can:</p>
                    <button class="button" onclick="goToHomePage()">Go to Home Page</button>
                </div>
            </body>
            </html>
            """
            
            # Check for errors
            if error:
                print(f"OAuth error: {error}")
                # Return HTML page that redirects to app with error
                deep_link = f'zaam://google-connect-failure?error={error}'
                return html_redirect_template % deep_link, 200, {'Content-Type': 'text/html'}
                
            # Verify state to prevent CSRF
            stored_state = session.get('google_auth_state')
            if not stored_state or state != stored_state:
                print(f"Invalid state in OAuth callback: stored={stored_state}, received={state}")
                deep_link = 'zaam://google-connect-failure?error=invalid_state'
                return html_redirect_template % deep_link, 200, {'Content-Type': 'text/html'}
                
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
                    deep_link = 'zaam://google-connect-failure?error=missing_user_id'
                    return html_redirect_template % deep_link, 200, {'Content-Type': 'text/html'}
                
                # Ensure user_id is a string
                user_id_str = str(user_id)
                
                # Save tokens for user
                print(f"Saving tokens for user during account linking: {user_id_str}")
                success = oauth_service.save_tokens_for_user(mongo.db, user_id_str, tokens)
                
                if not success:
                    print(f"Failed to save tokens for user {user_id_str}")
                    deep_link = 'zaam://google-connect-failure?error=token_save_failed'
                    return html_redirect_template % deep_link, 200, {'Content-Type': 'text/html'}
                    
                # Redirect to success page for linking
                print("Successfully linked Google account")
                deep_link = 'zaam://google-connect-success'
                return html_redirect_template % deep_link, 200, {'Content-Type': 'text/html'}
            else:
                # Handle sign-in flow
                # Get user info from Google using the access token
                user_info = oauth_service.get_user_info(tokens['token'])
                
                if not user_info:
                    print("Failed to get user info from Google")
                    deep_link = 'zaam://google-connect-failure?error=failed_to_get_user_info'
                    return html_redirect_template % deep_link, 200, {'Content-Type': 'text/html'}
                
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
                        deep_link = 'zaam://google-connect-failure?error=token_save_failed'
                        return html_redirect_template % deep_link, 200, {'Content-Type': 'text/html'}
                    
                    # Generate JWT for the user
                    auth_token = existing_user.generate_auth_token()
                    
                    # Add the auth token and user_id to the URL params
                    params = f"token={auth_token}&user_id={str(existing_user._id)}"
                    
                    # Redirect to success page with token
                    print("Generated auth token for existing user")
                    deep_link = f'zaam://google-auth-success?{params}'
                    return html_redirect_template % deep_link, 200, {'Content-Type': 'text/html'}
                else:
                    # Create a new user
                    print("Creating new user from Google login")
                    # Extract name parts
                    name_parts = user_info.get('name', '').split(' ', 1)
                    first_name = name_parts[0] if name_parts else ''
                    last_name = name_parts[1] if len(name_parts) > 1 else ''
                    
                    # Create user with minimal info
                    new_user = User(
                        full_name=user_info.get('name', 'Google User'),
                        age=0,  # Default age
                        gender='', # Empty gender
                        contact_info={
                            'email': user_info.get('email', ''),
                            'phone': '',
                        },
                        password='',  # Empty password for Google users
                        emergency_contacts=[]
                    )
                    
                    # Set password hash directly to avoid hashing an empty string
                    new_user.password_hash = 'google_oauth_user'
                    new_user.created_at = datetime.utcnow()
                    new_user.updated_at = datetime.utcnow()
                    
                    # Save the user
                    success, errors = new_user.save(mongo.db)
                    
                    if not success:
                        print(f"Failed to create new user: {errors}")
                        deep_link = f'zaam://google-connect-failure?error=user_creation_failed&details={",".join(errors)}'
                        return html_redirect_template % deep_link, 200, {'Content-Type': 'text/html'}
                    
                    # Save the Google tokens for the new user
                    oauth_service.save_tokens_for_user(mongo.db, str(new_user._id), tokens)
                    
                    # Generate JWT for the new user
                    auth_token = new_user.generate_auth_token()
                    
                    # Add the auth token and user_id to the URL params
                    params = f"token={auth_token}&user_id={str(new_user._id)}&is_new=true"
                    
                    # Redirect to success page with token and indication this is a new user
                    print(f"Created new user: {new_user._id}")
                    deep_link = f'zaam://google-auth-success?{params}'
                    return html_redirect_template % deep_link, 200, {'Content-Type': 'text/html'}
            
            # Clear session data in any case
            session.pop('google_auth_state', None)
            session.pop('google_auth_user_id', None)
            session.pop('google_auth_mode', None)
            session.pop('google_auth_temp_id', None)
            
        except Exception as e:
            print(f"Exception in OAuth callback: {str(e)}")
            deep_link = f'zaam://google-connect-failure?error={str(e)}'
            return html_redirect_template % deep_link, 200, {'Content-Type': 'text/html'}
    
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