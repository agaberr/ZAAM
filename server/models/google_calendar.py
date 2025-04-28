from datetime import datetime, timedelta
import pytz
from .google_oauth import GoogleOAuthService

class GoogleCalendarService:
    """Service for interacting with Google Calendar API"""
    
    def __init__(self, google_oauth_service):
        self.google_oauth_service = google_oauth_service
    
    def create_event(self, title, start_time, end_time=None, description=None, timezone="Africa/Cairo"):
        """Create a new event on Google Calendar"""
        # Default to 1 hour event if no end time is provided
        if not end_time:
            end_time = start_time + timedelta(hours=1)
            
        # Build the calendar service
        service = self.google_oauth_service.build_service("calendar", "v3")
        if not service:
            raise Exception("Not authenticated with Google Calendar")
        
        # Create the event object
        event = {
            "summary": title,
            "start": {"dateTime": start_time.isoformat(), "timeZone": timezone},
            "end": {"dateTime": end_time.isoformat(), "timeZone": timezone}
        }
        
        # Add description if provided
        if description:
            event["description"] = description
            
        # Insert the event
        created_event = service.events().insert(calendarId="primary", body=event).execute()
        return created_event
    
    def get_events(self, time_min=None, time_max=None, max_results=10, timezone="Africa/Cairo"):
        """Get events from Google Calendar within a time range"""
        # Build the calendar service
        service = self.google_oauth_service.build_service("calendar", "v3")
        if not service:
            raise Exception("Not authenticated with Google Calendar")
            
        # Default to today if no time range is provided
        if not time_min:
            time_min = datetime.now(pytz.timezone(timezone)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            
        if not time_max:
            time_max = time_min.replace(hour=23, minute=59, second=59)
            
        # Get events
        events_result = service.events().list(
            calendarId="primary",
            timeMin=time_min.isoformat(),
            timeMax=time_max.isoformat(),
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime",
            timeZone=timezone
        ).execute()
        
        return events_result.get("items", [])
    
    def get_day_events(self, target_date=None, timezone="Africa/Cairo"):
        """Get all events for a specific day"""
        # Default to today if no date is provided
        if not target_date:
            tz = pytz.timezone(timezone)
            target_date = datetime.now(tz)
            
        # Set time range for the whole day
        time_min = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        time_max = target_date.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        return self.get_events(time_min=time_min, time_max=time_max, max_results=50)
    
    def format_events_response(self, events, date_context=None):
        """Format events into a human-readable response"""
        if not events:
            time_context = self._get_time_context(date_context)
            return f"I've checked your calendar and you don't have any events scheduled {time_context}. Your schedule is clear!"
            
        # Format events in a more conversational way
        formatted_events = []
        for event in events:
            start_time = event['start'].get('dateTime', 'All day')
            if start_time != 'All day':
                dt = datetime.fromisoformat(start_time)
                time_str = dt.strftime("%I:%M %p")
                formatted_events.append(f"At {time_str}, you have {event['summary']}")
            else:
                formatted_events.append(f"You have {event['summary']} scheduled for all day")
        
        timetable = "\n".join(formatted_events)
        
        time_context = self._get_time_context(date_context)
        return f"Let me tell you what's on your schedule {time_context}.\n\n{timetable}"
    
    def _get_time_context(self, date_context):
        """Get a human-readable time context string"""
        if not date_context:
            return "today"
            
        # If it's a datetime object
        if isinstance(date_context, datetime):
            now = datetime.now(date_context.tzinfo)
            days_diff = (date_context.date() - now.date()).days
            
            if days_diff == 0:
                return "today"
            elif days_diff == 1:
                return "tomorrow"
            else:
                return f"for {date_context.strftime('%A, %B %d')}"
                
        # If it's an integer (days offset)
        if isinstance(date_context, int):
            if date_context == 0:
                return "today"
            elif date_context == 1:
                return "tomorrow"
            else:
                tz = pytz.timezone("Africa/Cairo")
                target_date = datetime.now(tz) + timedelta(days=date_context)
                return f"for {target_date.strftime('%A, %B %d')}"
                
        return "today"  # Default fallback 