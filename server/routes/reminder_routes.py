from flask import jsonify, request
from datetime import datetime
from bson.objectid import ObjectId
from models.reminder import Reminder
from models.google_calendar import GoogleCalendarService
from models.google_oauth import GoogleOAuthService
import jwt
import os

# Load JWT secret from environment variables
JWT_SECRET = os.getenv("JWT_SECRET")

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