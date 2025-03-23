import os
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from datetime import datetime, timedelta
from bson import ObjectId

class GoogleOAuthService:
    """Service for Google OAuth authentication."""
    
    def __init__(self):
        """Initialize with Google OAuth settings."""
        self.client_id = os.getenv("GOOGLE_CLIENT_ID")
        self.client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
        self.redirect_uri = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:5000/api/auth/google/callback")
        self.scopes = [
            "https://www.googleapis.com/auth/calendar",
            "https://www.googleapis.com/auth/calendar.events",
            "openid",
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/userinfo.profile"
        ]
    
    def get_authorization_url(self):
        """Generate the Google authorization URL."""
        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": [self.redirect_uri]
                }
            },
            scopes=self.scopes
        )
        flow.redirect_uri = self.redirect_uri
        
        authorization_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='consent'  # Force to show the consent screen
        )
        
        return authorization_url, state
    
    def exchange_code_for_tokens(self, code):
        """Exchange authorization code for tokens."""
        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": [self.redirect_uri]
                }
            },
            scopes=self.scopes
        )
        flow.redirect_uri = self.redirect_uri
        
        # Exchange code for tokens
        flow.fetch_token(code=code)
        credentials = flow.credentials
        
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
        
        return tokens
    
    def save_tokens_for_user(self, db, user_id, tokens):
        """Save Google tokens for a user in the database."""
        if not db:
            return False
            
        try:
            # Add timestamp for when tokens were saved
            tokens["saved_at"] = datetime.utcnow()
            
            # Check if user already has tokens
            existing = db.google_tokens.find_one({"user_id": user_id})
            
            if existing:
                # Update existing tokens
                db.google_tokens.update_one(
                    {"user_id": user_id},
                    {"$set": tokens}
                )
            else:
                # Insert new token record
                tokens["user_id"] = user_id
                db.google_tokens.insert_one(tokens)
                
            return True
        except Exception as e:
            print(f"Error saving Google tokens: {e}")
            return False
    
    def get_credentials_for_user(self, db, user_id):
        """Get valid Google credentials for a user."""
        if not db:
            return None
            
        try:
            # Get tokens from database
            tokens = db.google_tokens.find_one({"user_id": user_id})
            
            if not tokens:
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
        if not db:
            return False
            
        try:
            result = db.google_tokens.delete_one({"user_id": user_id})
            return result.deleted_count > 0
        except Exception as e:
            print(f"Error revoking Google access: {e}")
            return False 