import os
import json
from datetime import datetime
import requests
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from bson import ObjectId

class GoogleOAuthService:
    """Service for Google OAuth authentication."""
    
    def __init__(self):
        """Initialize with Google OAuth settings."""
        self.client_id = os.getenv("GOOGLE_CLIENT_ID")
        self.client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
        
        # Use the first redirect URI from credentials file by default
        self.redirect_uri = os.getenv("GOOGLE_REDIRECT_URI", "https://d997-196-137-176-101.ngrok-free.app/callback")
        
        # Define paths to potential credentials files
        server_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        project_root = os.path.dirname(server_dir)
        self.credentials_candidates = [
            os.path.join(server_dir, 'credentialsOAuth.json'),  # Server directory
            os.path.join(project_root, 'Reminder-v2', 'credentialsOAuth.json')  # Original Reminder-v2 directory
        ]
        
        # Find the first available credentials file
        self.credentials_file = None
        for candidate in self.credentials_candidates:
            if os.path.exists(candidate):
                self.credentials_file = candidate
                print(f"Found OAuth credentials at: {self.credentials_file}")
                break
        
        if not self.credentials_file:
            print("WARNING: No OAuth credentials file found, will use environment variables")
            
        self.scopes = [
            "https://www.googleapis.com/auth/calendar",
            "https://www.googleapis.com/auth/calendar.events",
            "openid",
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/userinfo.profile"
        ]
    
    def get_client_config(self):
        """Get client configuration from either credentials file or environment variables."""
        if self.credentials_file and os.path.exists(self.credentials_file):
            try:
                print(f"Loading OAuth credentials from file: {self.credentials_file}")
                with open(self.credentials_file, 'r') as f:
                    config = json.load(f)
                    # Verify required fields are present
                    if 'web' not in config:
                        print("ERROR: OAuth credentials file missing 'web' section")
                        return self._fallback_config()
                    
                    web_config = config['web']
                    required_fields = ['client_id', 'client_secret', 'auth_uri', 'token_uri']
                    missing_fields = [field for field in required_fields if field not in web_config]
                    
                    if missing_fields:
                        print(f"ERROR: OAuth credentials file missing required fields: {missing_fields}")
                        return self._fallback_config()
                    
                    # Add redirect URI if not present
                    if 'redirect_uris' not in web_config or not web_config['redirect_uris']:
                        print("WARNING: Adding default redirect URI to OAuth config")
                        web_config['redirect_uris'] = [self.redirect_uri]
                    
                    print("Successfully loaded OAuth credentials from file")
                    return config
            except Exception as e:
                print(f"Error loading credentials file: {e}")
                return self._fallback_config()
        else:
            print("Using environment variables for OAuth credentials")
            return self._fallback_config()
    
    def _fallback_config(self):
        """Create a fallback configuration using environment variables."""
        if not self.client_id or not self.client_secret:
            print("ERROR: Missing required OAuth credentials in environment variables")
            print(f"client_id present: {bool(self.client_id)}")
            print(f"client_secret present: {bool(self.client_secret)}")
            
        return {
            "web": {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [self.redirect_uri]
            }
        }
    
    def get_authorization_url(self):
        """Generate the Google authorization URL."""
        flow = Flow.from_client_config(
            self.get_client_config(),
            scopes=self.scopes
        )
        
        # Use the redirect URI that was specifically set for the app
        flow.redirect_uri = self.redirect_uri
        print(f"Using redirect URI for authorization: {flow.redirect_uri}")
        
        authorization_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='consent'  # Force to show the consent screen
        )
        
        return authorization_url, state
    
    def exchange_code_for_tokens(self, code):
        """Exchange authorization code for tokens."""
        try:
            print(f"Exchanging code for tokens...")
            flow = Flow.from_client_config(
                self.get_client_config(),
                scopes=self.scopes
            )
            flow.redirect_uri = self.redirect_uri
            
            # Log details about the redirect URI
            print(f"Using redirect URI: {flow.redirect_uri}")
            
            # Exchange code for tokens
            print(f"Calling fetch_token with code length: {len(code) if code else 0}")
            flow.fetch_token(code=code)
            
            credentials = flow.credentials
            print(f"Successfully exchanged code for tokens")
            
            # Format tokens for storage
            tokens = {
                "token": credentials.token,
                "refresh_token": credentials.refresh_token,
                "token_uri": credentials.token_uri,
                "client_id": credentials.client_id,
                "client_secret": credentials.client_secret,
                "scopes": credentials.scopes,
                "expiry": credentials.expiry.isoformat() if credentials.expiry else None
            }
            
            print(f"Token exchange successful, refresh_token present: {bool(credentials.refresh_token)}")
            return tokens
        except Exception as e:
            print(f"Error exchanging code for tokens: {str(e)}")
            raise e
    
    def save_tokens_for_user(self, db, user_id, tokens):
        """Save Google OAuth tokens for a user."""
        # Check parameters
        if db is None or not user_id or not tokens:
            print(f"Cannot save tokens: db is {'None' if db is None else 'valid'}, user_id={user_id}")
            return False
            
        try:
            print(f"Attempting to save tokens for user_id: {user_id}, type: {type(user_id)}")
            
            # Check if tokens already exist for user (handle ObjectId conversion carefully)
            try:
                # Always use string type for user_id in queries
                user_id_str = str(user_id)
                print(f"Querying for existing tokens with user_id: {user_id_str}")
                existing = db.google_tokens.find_one({"user_id": user_id_str})
                print(f"Query result: {'Found' if existing is not None else 'Not found'}")
            except Exception as query_error:
                print(f"Error querying for existing tokens: {query_error}")
                return False
            
            # Add additional metadata
            tokens_to_save = tokens.copy()  # Create a copy to avoid modifying the original
            tokens_to_save["user_id"] = str(user_id)  # Ensure user_id is string
            tokens_to_save["saved_at"] = datetime.utcnow()
            
            if existing is not None:
                # Update existing tokens
                print(f"Updating existing tokens for user {user_id}")
                try:
                    result = db.google_tokens.update_one(
                        {"_id": existing["_id"]},
                        {"$set": tokens_to_save}
                    )
                    success = result.modified_count > 0
                    print(f"Update result: modified_count={result.modified_count}")
                    return success
                except Exception as update_error:
                    print(f"Error updating tokens: {update_error}")
                    return False
            else:
                # Insert new tokens
                print(f"Inserting new tokens for user {user_id}")
                try:
                    result = db.google_tokens.insert_one(tokens_to_save)
                    success = result.inserted_id is not None
                    print(f"Insert result: inserted_id={'Present' if success else 'None'}")
                    return success
                except Exception as insert_error:
                    print(f"Error inserting tokens: {insert_error}")
                    return False
        except Exception as e:
            print(f"Error saving Google tokens: {e}")
            return False
    
    def get_credentials_for_user(self, db, user_id):
        """Get valid Google credentials for a user."""
        if db is None:
            print("Cannot get credentials: db is None")
            return None
            
        try:
            # Get tokens from database
            tokens = db.google_tokens.find_one({"user_id": user_id})
            
            if tokens is None:
                print(f"No tokens found for user {user_id}")
                return None
                
            # Create credentials object
            expiry = datetime.fromisoformat(tokens["expiry"]) if tokens.get("expiry") else None
            
            credentials = Credentials(
                token=tokens["token"],
                refresh_token=tokens["refresh_token"],
                token_uri=tokens["token_uri"],
                client_id=tokens["client_id"],
                client_secret=tokens["client_secret"],
                scopes=tokens["scopes"],
                expiry=expiry
            )
            
            # Check if token needs refresh
            if not credentials.valid:
                if credentials.refresh_token:
                    credentials.refresh(None)  # Refresh the token
                    
                    # Update tokens in database
                    self.save_tokens_for_user(db, user_id, {
                        "token": credentials.token,
                        "refresh_token": credentials.refresh_token,
                        "expiry": credentials.expiry.isoformat() if credentials.expiry else None
                    })
                else:
                    # Cannot refresh, need new authorization
                    return None
                    
            return credentials
        except Exception as e:
            print(f"Error getting Google credentials: {e}")
            return None
    
    def revoke_access(self, db, user_id):
        """Revoke Google access for a user."""
        if db is None or not user_id:
            print(f"Cannot revoke access: db is {'None' if db is None else 'valid'}, user_id={user_id}")
            return False
            
        try:
            # Delete tokens from database
            result = db.google_tokens.delete_one({"user_id": user_id})
            return result.deleted_count > 0
        except Exception as e:
            print(f"Error revoking Google access: {e}")
            return False
    
    def get_user_info(self, access_token):
        """Get user information from Google."""
        try:
            # Call Google's userinfo endpoint
            response = requests.get(
                'https://www.googleapis.com/oauth2/v3/userinfo',
                headers={'Authorization': f'Bearer {access_token}'}
            )
            
            # Check if request was successful
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error getting user info: {response.status_code}, {response.text}")
                return None
        except Exception as e:
            print(f"Exception getting user info: {e}")
            return None 