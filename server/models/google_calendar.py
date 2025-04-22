from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from datetime import datetime, timedelta
import pytz
import re

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
    
    def _format_datetime_for_google(self, dt):
        """Format a datetime object for Google Calendar API."""
        if not dt:
            return None
            
        # Make sure the datetime has timezone info
        if dt.tzinfo is None:
            # If no timezone, assume UTC
            dt = dt.replace(tzinfo=pytz.UTC)
            
        # Format according to RFC3339
        return dt.isoformat()
    
    def create_event(self, reminder):
        """Create an event in Google Calendar from a reminder."""
        if not self.service:
            return None
            
        # Format start and end times
        start_time = self._format_datetime_for_google(reminder.start_time)
        
        # Set end_time to start_time + 1 hour if not provided
        if reminder.end_time:
            end_time = self._format_datetime_for_google(reminder.end_time)
        else:
            end_time = self._format_datetime_for_google(reminder.start_time + timedelta(hours=1))
        
        if not start_time or not end_time:
            print(f"Invalid times for Google Calendar: start={reminder.start_time}, end={reminder.end_time}")
            return None
            
        # Format the reminder data for Google Calendar
        event = {
            'summary': reminder.title,
            'description': reminder.description,
            'start': {
                'dateTime': start_time,
                'timeZone': 'Africa/Cairo',  # Using explicit timezone for Egypt
            },
            'end': {
                'dateTime': end_time,
                'timeZone': 'Africa/Cairo',
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
            
            # Format start and end times
            start_time = self._format_datetime_for_google(reminder.start_time)
            
            # Set end_time to start_time + 1 hour if not provided
            if reminder.end_time:
                end_time = self._format_datetime_for_google(reminder.end_time)
            else:
                end_time = self._format_datetime_for_google(reminder.start_time + timedelta(hours=1))
            
            if not start_time or not end_time:
                print(f"Invalid times for Google Calendar update: start={reminder.start_time}, end={reminder.end_time}")
                return False
            
            # Update fields
            event['summary'] = reminder.title
            event['description'] = reminder.description
            event['start']['dateTime'] = start_time
            event['start']['timeZone'] = 'Africa/Cairo'
            event['end']['dateTime'] = end_time
            event['end']['timeZone'] = 'Africa/Cairo'
            
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