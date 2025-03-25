from flask import jsonify, request
from models.reminder import Reminder
from models.google_calendar import GoogleCalendarService
from models.google_oauth import GoogleOAuthService
import jwt
import os

# Load JWT secret from environment variables
JWT_SECRET = os.getenv("JWT_SECRET")

def register_reminder_sync_routes(app, mongo):
    """Register routes for reminder synchronization with Google Calendar."""
    
    # Initialize services
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
    
    @app.route('/api/reminders/sync', methods=['POST'])
    def sync_reminders():
        """Synchronize all user's reminders with Google Calendar."""
        try:
            # Get authenticated user ID
            user_id = get_authenticated_user_id()
            if not user_id:
                return jsonify({"success": False, "message": "Authentication required"}), 401
                
            # Get Google credentials
            google_credentials = oauth_service.get_credentials_for_user(mongo.db, user_id)
            if not google_credentials:
                return jsonify({
                    "success": False, 
                    "message": "Google Calendar not connected. Please connect your Google account first."
                }), 400
                
            # Create calendar service
            calendar_service = GoogleCalendarService(google_credentials)
            
            # Get all user's reminders
            reminders = Reminder.find_by_user(mongo.db, user_id)
            
            # Track sync statistics
            created_count = 0
            updated_count = 0
            failed_count = 0
            
            # Sync each reminder
            for reminder in reminders:
                try:
                    if reminder.google_event_id:
                        # Update existing Google Calendar event
                        success = calendar_service.update_event(reminder)
                        if success:
                            updated_count += 1
                        else:
                            failed_count += 1
                    else:
                        # Create new Google Calendar event
                        event_id = calendar_service.create_event(reminder)
                        if event_id:
                            # Update reminder with Google event ID
                            reminder.google_event_id = event_id
                            reminder.save(mongo.db)
                            created_count += 1
                        else:
                            failed_count += 1
                except Exception as e:
                    print(f"Error syncing reminder {reminder._id}: {str(e)}")
                    failed_count += 1
                    
            # Generate response message
            total_count = len(reminders)
            success_count = created_count + updated_count
            
            message = f"Synced {success_count} of {total_count} reminders with Google Calendar. "
            if created_count > 0:
                message += f"Created {created_count} new events. "
            if updated_count > 0:
                message += f"Updated {updated_count} existing events. "
            if failed_count > 0:
                message += f"Failed to sync {failed_count} reminders."
                
            return jsonify({
                "success": True,
                "message": message,
                "stats": {
                    "total": total_count,
                    "created": created_count,
                    "updated": updated_count,
                    "failed": failed_count
                }
            })
            
        except Exception as e:
            print(f"Error in sync_reminders: {str(e)}")
            return jsonify({
                "success": False,
                "message": f"Error syncing reminders: {str(e)}"
            }), 500
            
    @app.route('/api/reminders/import-from-google', methods=['POST'])
    def import_from_google():
        """Import events from Google Calendar as reminders."""
        try:
            # Get authenticated user ID
            user_id = get_authenticated_user_id()
            if not user_id:
                return jsonify({"success": False, "message": "Authentication required"}), 401
                
            # Get request parameters
            data = request.json or {}
            days = data.get('days', 30)  # Default to 30 days ahead
            
            # Get Google credentials
            google_credentials = oauth_service.get_credentials_for_user(mongo.db, user_id)
            if not google_credentials:
                return jsonify({
                    "success": False, 
                    "message": "Google Calendar not connected. Please connect your Google account first."
                }), 400
                
            # Create calendar service
            calendar_service = GoogleCalendarService(google_credentials)
            
            # Get events from Google Calendar
            events = calendar_service.get_events(max_results=100)
            
            # Track import statistics
            imported_count = 0
            skipped_count = 0
            
            # Process each event
            for event in events:
                # Skip events that are already imported
                existing = mongo.db.reminders.find_one({"google_event_id": event['id'], "user_id": user_id})
                if existing:
                    skipped_count += 1
                    continue
                    
                try:
                    # Create reminder from event
                    reminder = Reminder.from_google_event(user_id, event)
                    
                    # Save reminder
                    success, errors = reminder.save(mongo.db)
                    if success:
                        imported_count += 1
                    else:
                        skipped_count += 1
                        print(f"Failed to save reminder from event {event['id']}: {str(errors)}")
                except Exception as e:
                    print(f"Error importing event {event['id']}: {str(e)}")
                    skipped_count += 1
                    
            # Generate response message
            total_count = len(events)
            
            message = f"Imported {imported_count} of {total_count} events from Google Calendar. "
            if skipped_count > 0:
                message += f"Skipped {skipped_count} events that were already imported or couldn't be processed."
                
            return jsonify({
                "success": True,
                "message": message,
                "stats": {
                    "total": total_count,
                    "imported": imported_count,
                    "skipped": skipped_count
                }
            })
            
        except Exception as e:
            print(f"Error in import_from_google: {str(e)}")
            return jsonify({
                "success": False,
                "message": f"Error importing from Google Calendar: {str(e)}"
            }), 500 