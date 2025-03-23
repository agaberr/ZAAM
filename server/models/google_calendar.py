from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from datetime import datetime, timedelta

class GoogleCalendarService:
    """Service for Google Calendar API integration."""
    
    def __init__(self, user_credentials=None):
        """Initialize with user's Google API credentials."""
        self.credentials = user_credentials
        self.service = None
        
        if self.credentials:
            self.service = build('calendar', 'v3', credentials=self.credentials)
    
    def set_credentials(self, credentials):
        """Set user credentials and build service."""
        self.credentials = credentials
        self.service = build('calendar', 'v3', credentials=self.credentials)
    
    def create_event(self, reminder):
        """Create an event in Google Calendar from a reminder."""
        if not self.service:
            return None
            
        # Format the reminder data for Google Calendar
        event = {
            'summary': reminder.title,
            'description': reminder.description,
            'start': {
                'dateTime': reminder.start_time.isoformat(),
                'timeZone': 'UTC',  # You may want to make this configurable
            },
            'end': {
                'dateTime': reminder.end_time.isoformat() if reminder.end_time else 
                           (reminder.start_time + timedelta(hours=1)).isoformat(),
                'timeZone': 'UTC',
            },
        }
        
        # Add recurrence if specified
        if reminder.recurrence:
            recurrence_map = {
                'daily': ['RRULE:FREQ=DAILY'],
                'weekly': ['RRULE:FREQ=WEEKLY'],
                'monthly': ['RRULE:FREQ=MONTHLY'],
                'yearly': ['RRULE:FREQ=YEARLY']
            }
            
            if reminder.recurrence in recurrence_map:
                event['recurrence'] = recurrence_map[reminder.recurrence]
        
        try:
            # Insert the event
            created_event = self.service.events().insert(
                calendarId='primary',
                body=event
            ).execute()
            
            return created_event['id']
        except Exception as e:
            print(f"Error creating Google Calendar event: {e}")
            return None
    
    def update_event(self, reminder):
        """Update an existing event in Google Calendar."""
        if not self.service or not reminder.google_event_id:
            return False
            
        try:
            # Get the existing event
            event = self.service.events().get(
                calendarId='primary',
                eventId=reminder.google_event_id
            ).execute()
            
            # Update fields
            event['summary'] = reminder.title
            event['description'] = reminder.description
            event['start']['dateTime'] = reminder.start_time.isoformat()
            
            if reminder.end_time:
                event['end']['dateTime'] = reminder.end_time.isoformat()
            else:
                event['end']['dateTime'] = (reminder.start_time + timedelta(hours=1)).isoformat()
            
            # Update recurrence if needed
            if reminder.recurrence:
                recurrence_map = {
                    'daily': ['RRULE:FREQ=DAILY'],
                    'weekly': ['RRULE:FREQ=WEEKLY'],
                    'monthly': ['RRULE:FREQ=MONTHLY'],
                    'yearly': ['RRULE:FREQ=YEARLY']
                }
                
                if reminder.recurrence in recurrence_map:
                    event['recurrence'] = recurrence_map[reminder.recurrence]
            
            # Update the event
            self.service.events().update(
                calendarId='primary',
                eventId=reminder.google_event_id,
                body=event
            ).execute()
            
            return True
        except Exception as e:
            print(f"Error updating Google Calendar event: {e}")
            return False
    
    def delete_event(self, event_id):
        """Delete an event from Google Calendar."""
        if not self.service:
            return False
            
        try:
            self.service.events().delete(
                calendarId='primary',
                eventId=event_id
            ).execute()
            return True
        except Exception as e:
            print(f"Error deleting Google Calendar event: {e}")
            return False
    
    def get_events(self, time_min=None, time_max=None, max_results=10):
        """Get events from Google Calendar within a time range."""
        if not self.service:
            return []
            
        if not time_min:
            time_min = datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
        if not time_max:
            # Default to events in the next 7 days
            time_max = (datetime.utcnow() + timedelta(days=7)).isoformat() + 'Z'
            
        try:
            events_result = self.service.events().list(
                calendarId='primary',
                timeMin=time_min,
                timeMax=time_max,
                maxResults=max_results,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            return events_result.get('items', [])
        except Exception as e:
            print(f"Error fetching Google Calendar events: {e}")
            return [] 