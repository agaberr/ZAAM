from flask import jsonify, request, Blueprint
from datetime import datetime, timedelta
from bson.objectid import ObjectId
from models.reminder import Reminder
from models.google_calendar import GoogleCalendarService
from models.google_oauth import GoogleOAuthService
import jwt
import os
from flask_jwt_extended import jwt_required, get_jwt_identity
import torch
import pickle
import json
import re
import pytz
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

# Load JWT secret from environment variables
JWT_SECRET = os.getenv("JWT_SECRET")

# Create blueprint
reminder_bp = Blueprint('reminder_routes', __name__)

# Load the NLP model and encoders
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, 'nlp_models', 'model.pth')
tokenizer_path = os.path.join(base_dir, 'nlp_models', 'tokenizer.pkl')
intent_encoder_path = os.path.join(base_dir, 'nlp_models', 'intent_encoder.pkl')
slot_encoder_path = os.path.join(base_dir, 'nlp_models', 'slot_encoder.pkl')

# Load model and encoders
try:
    print("Loading reminder model...")
    model = torch.load(model_path, map_location=torch.device("cpu"))
    model.eval()
    
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    with open(intent_encoder_path, 'rb') as f:
        intent_encoder = pickle.load(f)
    with open(slot_encoder_path, 'rb') as f:
        slot_encoder = pickle.load(f)
    print("Reminder model and encoders loaded successfully")
except Exception as e:
    print(f"Error loading reminder model or encoders: {str(e)}")
    model = None
    tokenizer = None
    intent_encoder = None
    slot_encoder = None

def predict(model, tokenized_text, max_seq_length=128):
    try:
        # Convert tokens to ids
        token_ids = [tokenizer.get(token, tokenizer["UNK"]) for token in tokenized_text]
        
        # Pad sequence
        token_ids = token_ids[:max_seq_length] + [tokenizer["PAD"]] * (max_seq_length - len(token_ids))
        
        # Convert to tensor
        input_tensor = torch.tensor(token_ids).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            intent_pred, slot_pred = model(input_tensor)
        
        # Get the predicted intent
        intent_label = intent_pred.argmax(dim=1).item()
        predicted_intent = intent_encoder.inverse_transform([intent_label])[0]
        
        # Get the predicted slots
        slot_preds = slot_pred.argmax(dim=2).squeeze().cpu().numpy()
        predicted_slots = slot_encoder.inverse_transform(slot_preds[:len(tokenized_text)])
        
        return predicted_intent, predicted_slots
        
    except Exception as e:
        print(f"Error in predict function: {str(e)}")
        raise e

def postprocess_ner_predictions(predicted_intent, predicted_tokens, predicted_slots):
    predicted_time, predicted_action = [], []
    current_action = []
    current_time = []
    
    # Iterate through tokens and their predicted slots
    for i, (token, slot) in enumerate(zip(predicted_tokens, predicted_slots)):
        if slot == 'B-action' or slot == 'I-action':
            current_action.append(token)
        elif slot == 'B-time' or slot == 'I-time':
            current_time.append(token)
        elif (slot == 'O' or 'B-' in slot) and (current_action or current_time):
            # We've reached the end of an entity
            if current_action:
                predicted_action = current_action
                current_action = []
            if current_time:
                predicted_time = current_time
                current_time = []
    
    # Handle any entities at the end of the sequence
    if current_action:
        predicted_action = current_action
    if current_time:
        predicted_time = current_time
    
    # Join tokens to form entity strings
    result = {
        "predicted_intent": predicted_intent,
        "predicted_time": " ".join(predicted_time) if predicted_time else None,
        "predicted_action": " ".join(predicted_action) if predicted_action else None,
    }
    
    return result

def get_timetable(start_date=None, end_date=None):
    # Default to today if no date is provided
    if not start_date:
        start_date = datetime.now(pytz.timezone('Africa/Cairo'))
        start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    
    if not end_date:
        end_date = start_date + timedelta(days=1)
    
    # Format for the API
    time_min = start_date.isoformat()
    time_max = end_date.isoformat()
    
    return {
        "calendarId": "primary",
        "timeMin": time_min,
        "timeMax": time_max,
        "singleEvents": True,
        "orderBy": "startTime",
    }

def create_event(summary, start_time):
    # Create an event that lasts 1 hour by default
    end_time = start_time + timedelta(hours=1)
    
    event = {
        "summary": summary,
        "start": {
            "dateTime": start_time.isoformat(),
            "timeZone": "Africa/Cairo",
        },
        "end": {
            "dateTime": end_time.isoformat(),
            "timeZone": "Africa/Cairo",
        },
    }
    
    return event

def register_reminder_routes(app, mongo):
    # Initialize OAuth service
    oauth_service = GoogleOAuthService()
    
    # Helper function to get authenticated user ID
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
    
    # Helper function to parse datetime from request
    def parse_datetime(date_str):
        try:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except Exception:
            return None
    
    @app.route('/api/reminders', methods=['POST'])
    def create_reminder():
        try:
            # Get authenticated user ID
            user_id = get_authenticated_user_id()
            if not user_id:
                return jsonify({"error": "Authentication required"}), 401
                
            reminder_data = request.json
            
            # Parse start_time and end_time
            start_time = parse_datetime(reminder_data.get('start_time', ''))
            end_time = parse_datetime(reminder_data.get('end_time', '')) if reminder_data.get('end_time') else None
            
            if not start_time:
                return jsonify({"error": "Valid start_time is required"}), 400
            
            # Create reminder object
            reminder = Reminder(
                user_id=user_id,
                title=reminder_data.get('title', ''),
                description=reminder_data.get('description', ''),
                start_time=start_time,
                end_time=end_time,
                recurrence=reminder_data.get('recurrence')
            )
            
            # Ensure _id is None for new records
            reminder._id = None
            
            # Validate and save reminder
            success, errors = reminder.save(mongo.db)
            
            if not success:
                error_msg = "Validation failed"
                if errors and any("duplicate key error" in str(err).lower() for err in errors):
                    error_msg = "Duplicate reminder detected. Please try again."
                return jsonify({"error": error_msg, "details": errors}), 400
            
            # Try to create Google Calendar event if credentials are available
            google_credentials = oauth_service.get_credentials_for_user(mongo.db, user_id)
            if google_credentials:
                calendar_service = GoogleCalendarService(google_credentials)
                event_id = calendar_service.create_event(reminder)
                if event_id:
                    # Update reminder with Google event ID
                    reminder.google_event_id = event_id
                    reminder.save(mongo.db)
            
            return jsonify({
                "message": "Reminder created successfully",
                "reminder_id": str(reminder._id)
            }), 201
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/reminders/<reminder_id>', methods=['GET'])
    def get_reminder(reminder_id):
        try:
            # Get authenticated user ID
            user_id = get_authenticated_user_id()
            if not user_id:
                return jsonify({"error": "Authentication required"}), 401
                
            reminder = Reminder.find_by_id(mongo.db, reminder_id)
            
            if not reminder:
                return jsonify({"error": "Reminder not found"}), 404
                
            # Verify the reminder belongs to the authenticated user
            if reminder.user_id != user_id:
                return jsonify({"error": "Unauthorized"}), 403
                
            # Convert to dictionary and prepare for JSON serialization
            reminder_dict = reminder.to_dict()
            reminder_dict['_id'] = str(reminder_dict['_id'])
            reminder_dict['created_at'] = reminder_dict['created_at'].isoformat()
            reminder_dict['updated_at'] = reminder_dict['updated_at'].isoformat()
            reminder_dict['start_time'] = reminder_dict['start_time'].isoformat()
            if reminder_dict['end_time']:
                reminder_dict['end_time'] = reminder_dict['end_time'].isoformat()
                
            return jsonify(reminder_dict)
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/reminders', methods=['GET'])
    def get_reminders():
        try:
            # Get authenticated user ID
            user_id = get_authenticated_user_id()
            if not user_id:
                return jsonify({"error": "Authentication required"}), 401
                
            # Get query parameters
            completed = request.args.get('completed')
            if completed is not None:
                completed = completed.lower() == 'true'
            
            # Get reminders for the user
            reminders = Reminder.find_by_user(mongo.db, user_id, completed)
            
            # Convert to dictionaries for JSON serialization
            reminders_dict = []
            for reminder in reminders:
                r_dict = reminder.to_dict()
                r_dict['_id'] = str(r_dict['_id'])
                r_dict['created_at'] = r_dict['created_at'].isoformat()
                r_dict['updated_at'] = r_dict['updated_at'].isoformat()
                r_dict['start_time'] = r_dict['start_time'].isoformat()
                if r_dict['end_time']:
                    r_dict['end_time'] = r_dict['end_time'].isoformat()
                reminders_dict.append(r_dict)
                
            return jsonify(reminders_dict)
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/reminders/<reminder_id>', methods=['PUT'])
    def update_reminder(reminder_id):
        try:
            # Get authenticated user ID
            user_id = get_authenticated_user_id()
            if not user_id:
                return jsonify({"error": "Authentication required"}), 401
                
            reminder = Reminder.find_by_id(mongo.db, reminder_id)
            
            if not reminder:
                return jsonify({"error": "Reminder not found"}), 404
                
            # Verify the reminder belongs to the authenticated user
            if reminder.user_id != user_id:
                return jsonify({"error": "Unauthorized"}), 403
                
            # Get updated data
            reminder_data = request.json
            
            # Update reminder fields
            if 'title' in reminder_data:
                reminder.title = reminder_data['title']
            if 'description' in reminder_data:
                reminder.description = reminder_data['description']
            if 'start_time' in reminder_data:
                start_time = parse_datetime(reminder_data['start_time'])
                if start_time:
                    reminder.start_time = start_time
            if 'end_time' in reminder_data:
                if reminder_data['end_time']:
                    end_time = parse_datetime(reminder_data['end_time'])
                    if end_time:
                        reminder.end_time = end_time
                else:
                    reminder.end_time = None
            if 'recurrence' in reminder_data:
                reminder.recurrence = reminder_data['recurrence']
            if 'completed' in reminder_data:
                reminder.completed = reminder_data['completed']
                
            # Save updated reminder
            success, errors = reminder.save(mongo.db)
            
            if not success:
                return jsonify({"error": "Validation failed", "details": errors}), 400
                
            # Update Google Calendar event if it exists
            if reminder.google_event_id:
                google_credentials = oauth_service.get_credentials_for_user(mongo.db, user_id)
                if google_credentials:
                    calendar_service = GoogleCalendarService(google_credentials)
                    calendar_service.update_event(reminder)
                
            return jsonify({"message": "Reminder updated successfully"})
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/reminders/<reminder_id>', methods=['DELETE'])
    def delete_reminder(reminder_id):
        try:
            # Get authenticated user ID
            user_id = get_authenticated_user_id()
            if not user_id:
                return jsonify({"error": "Authentication required"}), 401
                
            reminder = Reminder.find_by_id(mongo.db, reminder_id)
            
            if not reminder:
                return jsonify({"error": "Reminder not found"}), 404
                
            # Verify the reminder belongs to the authenticated user
            if reminder.user_id != user_id:
                return jsonify({"error": "Unauthorized"}), 403
                
            # Delete from Google Calendar if it exists
            if reminder.google_event_id:
                google_credentials = oauth_service.get_credentials_for_user(mongo.db, user_id)
                if google_credentials:
                    calendar_service = GoogleCalendarService(google_credentials)
                    calendar_service.delete_event(reminder.google_event_id)
                
            # Delete from database
            result = mongo.db.reminders.delete_one({"_id": ObjectId(reminder_id)})
            
            if result.deleted_count == 0:
                return jsonify({"error": "Failed to delete reminder"}), 500
                
            return jsonify({"message": "Reminder deleted successfully"})
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/reminders/today', methods=['GET'])
    def get_today_reminders():
        try:
            # Get authenticated user ID
            user_id = get_authenticated_user_id()
            if not user_id:
                return jsonify({"error": "Authentication required"}), 401
                
            reminders = Reminder.find_today_reminders(mongo.db, user_id)
            
            # Convert to dictionaries for JSON serialization
            reminders_dict = []
            for reminder in reminders:
                r_dict = reminder.to_dict()
                r_dict['_id'] = str(r_dict['_id'])
                r_dict['created_at'] = r_dict['created_at'].isoformat()
                r_dict['updated_at'] = r_dict['updated_at'].isoformat()
                r_dict['start_time'] = r_dict['start_time'].isoformat()
                if r_dict['end_time']:
                    r_dict['end_time'] = r_dict['end_time'].isoformat()
                reminders_dict.append(r_dict)
                
            return jsonify(reminders_dict)
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/reminders/upcoming', methods=['GET'])
    def get_upcoming_reminders():
        try:
            # Get authenticated user ID
            user_id = get_authenticated_user_id()
            if not user_id:
                return jsonify({"error": "Authentication required"}), 401
                
            # Get days parameter (default to 7)
            days = request.args.get('days', default=7, type=int)
            
            reminders = Reminder.find_upcoming_reminders(mongo.db, user_id, days)
            
            # Convert to dictionaries for JSON serialization
            reminders_dict = []
            for reminder in reminders:
                r_dict = reminder.to_dict()
                r_dict['_id'] = str(r_dict['_id'])
                r_dict['created_at'] = r_dict['created_at'].isoformat()
                r_dict['updated_at'] = r_dict['updated_at'].isoformat()
                r_dict['start_time'] = r_dict['start_time'].isoformat()
                if r_dict['end_time']:
                    r_dict['end_time'] = r_dict['end_time'].isoformat()
                reminders_dict.append(r_dict)
                
            return jsonify(reminders_dict)
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/reminders/stats', methods=['GET'])
    def get_reminder_stats():
        try:
            # Get authenticated user ID
            user_id = get_authenticated_user_id()
            if not user_id:
                return jsonify({"error": "Authentication required"}), 401
                
            # Get statistics
            completion_rate = Reminder.get_completion_rate(mongo.db, user_id)
            today_count = len(Reminder.find_today_reminders(mongo.db, user_id))
            upcoming_count = len(Reminder.find_upcoming_reminders(mongo.db, user_id))
            
            return jsonify({
                "completion_rate": completion_rate,
                "today_count": today_count,
                "upcoming_count": upcoming_count
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
            
    @app.route('/api/reminders/stats/detailed', methods=['GET'])
    def get_detailed_reminder_stats():
        try:
            # Get authenticated user ID
            user_id = get_authenticated_user_id()
            if not user_id:
                return jsonify({"error": "Authentication required"}), 401
                
            # Get detailed statistics
            detailed_stats = Reminder.get_detailed_stats(mongo.db, user_id)
            
            return jsonify(detailed_stats)
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@reminder_bp.route('/process-reminder', methods=['POST'])
@jwt_required()
def process_reminder():
    if not model or not tokenizer or not intent_encoder or not slot_encoder:
        return jsonify({"error": "Reminder model is not loaded properly"}), 500
    
    # Get user input
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Missing required field: text"}), 400
    
    user_input = data['text'].lower()
    
    try:
        # Process user input
        tokenized_text = user_input.split()
        predicted_intent, predicted_slots = predict(model, tokenized_text)
        result = postprocess_ner_predictions(predicted_intent, tokenized_text, predicted_slots)
        
        # Return the processed intent and slots
        return jsonify({
            "intent": result["predicted_intent"],
            "slots": {
                "time": result["predicted_time"],
                "action": result["predicted_action"]
            }
        })
    
    except Exception as e:
        print(f"Error processing reminder: {str(e)}")
        return jsonify({"error": f"Failed to process reminder: {str(e)}"}), 500

@reminder_bp.route('/google-calendar/events', methods=['GET'])
@jwt_required()
def get_calendar_events():
    # Get credentials from request
    data = request.get_json()
    if not data or 'credentials' not in data:
        return jsonify({"error": "Google credentials not provided"}), 400
    
    try:
        # Build credentials object
        credentials = Credentials(**data['credentials'])
        service = build("calendar", "v3", credentials=credentials)
        
        # Parse date parameters
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')
        
        start_date = None
        end_date = None
        
        if start_date_str:
            start_date = datetime.fromisoformat(start_date_str)
        
        if end_date_str:
            end_date = datetime.fromisoformat(end_date_str)
        
        # Get timetable parameters
        payload = get_timetable(start_date, end_date)
        
        # Fetch events
        events_result = service.events().list(**payload).execute()
        events = events_result.get("items", [])
        
        # Format events for response
        formatted_events = []
        for event in events:
            formatted_event = {
                "id": event.get("id"),
                "summary": event.get("summary"),
                "description": event.get("description"),
                "location": event.get("location"),
                "start": event.get("start"),
                "end": event.get("end"),
                "status": event.get("status")
            }
            formatted_events.append(formatted_event)
        
        return jsonify({"events": formatted_events})
    
    except Exception as e:
        print(f"Error fetching calendar events: {str(e)}")
        return jsonify({"error": f"Failed to fetch calendar events: {str(e)}"}), 500

@reminder_bp.route('/google-calendar/create-event', methods=['POST'])
@jwt_required()
def create_calendar_event():
    # Get credentials and event details from request
    data = request.get_json()
    if not data or 'credentials' not in data or 'event' not in data:
        return jsonify({"error": "Missing required fields: credentials and event"}), 400
    
    try:
        # Build credentials object
        credentials = Credentials(**data['credentials'])
        service = build("calendar", "v3", credentials=credentials)
        
        # Get event details
        event_data = data['event']
        
        if 'summary' not in event_data:
            return jsonify({"error": "Event must have a summary"}), 400
        
        # Parse start time or default to now
        start_time = datetime.now(pytz.timezone('Africa/Cairo'))
        if 'start_time' in event_data:
            try:
                start_time = datetime.fromisoformat(event_data['start_time'])
            except ValueError:
                return jsonify({"error": "Invalid start_time format. Use ISO format (YYYY-MM-DDTHH:MM:SS)"}), 400
        
        # Create event payload
        event_payload = create_event(event_data['summary'], start_time)
        
        # Add description if provided
        if 'description' in event_data:
            event_payload['description'] = event_data['description']
        
        # Add location if provided
        if 'location' in event_data:
            event_payload['location'] = event_data['location']
        
        # Add the event to Google Calendar
        event = service.events().insert(calendarId="primary", body=event_payload).execute()
        
        return jsonify({
            "success": True,
            "event": {
                "id": event.get("id"),
                "summary": event.get("summary"),
                "start": event.get("start"),
                "end": event.get("end")
            }
        })
    
    except Exception as e:
        print(f"Error creating calendar event: {str(e)}")
        return jsonify({"error": f"Failed to create calendar event: {str(e)}"}), 500 