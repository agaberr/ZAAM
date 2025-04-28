from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import os
from flask import url_for, session

class GoogleOAuthService:
    """Service for Google OAuth authentication and token management"""
    
    def __init__(self, client_secrets_file, scopes):
        self.client_secrets_file = client_secrets_file
        self.scopes = scopes
        
        # For development environments
        os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    
    def get_authorization_url(self, redirect_uri):
        """Generate authorization URL for OAuth flow"""
        flow = Flow.from_client_secrets_file(
            self.client_secrets_file, 
            scopes=self.scopes, 
            redirect_uri=redirect_uri
        )
        authorization_url, state = flow.authorization_url(
            access_type="offline", 
            include_granted_scopes="true"
        )
        return authorization_url, state
    
    def fetch_token(self, authorization_response, redirect_uri):
        """Fetch OAuth token from authorization response"""
        flow = Flow.from_client_secrets_file(
            self.client_secrets_file, 
            scopes=self.scopes, 
            redirect_uri=redirect_uri
        )
        flow.fetch_token(authorization_response=authorization_response)
        return flow.credentials
    
    def credentials_to_dict(self, credentials):
        """Convert credentials object to dictionary for storage"""
        return {
            "token": credentials.token,
            "refresh_token": credentials.refresh_token,
            "token_uri": credentials.token_uri,
            "client_id": credentials.client_id,
            "client_secret": credentials.client_secret,
            "scopes": credentials.scopes,
        }
    
    def dict_to_credentials(self, credentials_dict):
        """Convert dictionary back to credentials object"""
        return Credentials(**credentials_dict)
    
    def is_authenticated(self):
        """Check if user is authenticated with Google"""
        return "google_credentials" in session
    
    def get_credentials(self):
        """Get credentials from session"""
        if "google_credentials" in session:
            return self.dict_to_credentials(session["google_credentials"])
        return None
        
    def save_credentials(self, credentials):
        """Save credentials to session"""
        session["google_credentials"] = self.credentials_to_dict(credentials)
        
    def clear_credentials(self):
        """Clear credentials from session"""
        if "google_credentials" in session:
            del session["google_credentials"]
    
    def build_service(self, api_name, api_version, credentials=None):
        """Build a Google API service using credentials"""
        if not credentials:
            credentials = self.get_credentials()
            
        if not credentials:
            return None
            
        return build(api_name, api_version, credentials=credentials) 