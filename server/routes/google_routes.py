from flask import Blueprint, request, jsonify, session, redirect, url_for
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
import os

# Create blueprint
google_bp = Blueprint('google', __name__)

# Google OAuth Config
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"  # Remove in production
CLIENT_SECRETS_FILE = "reminders/credentialsOAuth.json"
SCOPES = ["https://www.googleapis.com/auth/calendar"]

def credentials_to_dict(credentials):
    return {
        "token": credentials.token,
        "refresh_token": credentials.refresh_token,
        "token_uri": credentials.token_uri,
        "client_id": credentials.client_id,
        "client_secret": credentials.client_secret,
        "scopes": credentials.scopes,
    }

@google_bp.route('/api/auth/google/login')
def google_login():
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        redirect_uri=url_for('google.google_callback', _external=True)
    )
    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true'
    )
    session['state'] = state
    return jsonify({'authorization_url': authorization_url})

@google_bp.route('/api/auth/google/callback')
def google_callback():
    state = session.get('state')
    if not state or state != request.args.get('state'):
        return jsonify({'error': 'Invalid state parameter'}), 400

    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        redirect_uri=url_for('google.google_callback', _external=True)
    )
    
    try:
        flow.fetch_token(authorization_response=request.url)
        credentials = flow.credentials
        session['credentials'] = credentials_to_dict(credentials)
        return jsonify({'status': 'success', 'message': 'Successfully authenticated with Google'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@google_bp.route('/api/auth/google/check')
def check_google_auth():
    if 'credentials' not in session:
        return jsonify({'authenticated': False})
    
    try:
        credentials = Credentials(**session['credentials'])
        if credentials.expired:
            return jsonify({'authenticated': False})
        return jsonify({'authenticated': True})
    except Exception as e:
        return jsonify({'authenticated': False, 'error': str(e)})

@google_bp.route('/api/auth/google/logout')
def google_logout():
    session.pop('credentials', None)
    return jsonify({'status': 'success', 'message': 'Successfully logged out'})

def register_google_routes(app):
    app.register_blueprint(google_bp) 