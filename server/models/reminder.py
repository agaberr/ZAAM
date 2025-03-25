from datetime import datetime, timedelta
from bson import ObjectId
import uuid

class Reminder:
    def __init__(self, user_id, title, description, start_time, end_time, 
                 recurrence=None, completed=False, google_event_id=None):
        self._id = None
        self.user_id = user_id
        self.title = title
        self.description = description
        self.start_time = start_time  # datetime object
        self.end_time = end_time      # datetime object
        self.recurrence = recurrence  # daily, weekly, monthly, or None for one-time
        self.completed = completed
        self.google_event_id = google_event_id  # ID from Google Calendar
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def to_dict(self):
        """Convert reminder to dictionary for serialization"""
        result = {
            "user_id": self.user_id,
            "title": self.title,
            "description": self.description,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "recurrence": self.recurrence,
            "completed": self.completed,
            "google_event_id": self.google_event_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
        
        # Only include _id if it's not None
        if self._id is not None:
            result["_id"] = self._id
            
        return result

    def save(self, db):
        """Save the reminder to the database."""
        if db is None:
            return False, ["Database connection is not available"]
            
        # Validate required fields
        errors = []
        if not self.user_id:
            errors.append("User ID is required")
        if not self.title:
            errors.append("Title is required")
        if not self.start_time:
            errors.append("Start time is required")
            
        if errors:
            return False, errors
            
        reminder_data = self.to_dict()
        
        # For update operations
        if self._id is not None:
            reminder_data["updated_at"] = datetime.utcnow()
            try:
                result = db.reminders.update_one(
                    {"_id": self._id},
                    {"$set": reminder_data}
                )
                return result.modified_count > 0, None
            except Exception as e:
                return False, [str(e)]
        # For insert operations
        else:
            try:
                # Remove _id field for new records since it's null and MongoDB will auto-generate it
                if "_id" in reminder_data:
                    del reminder_data["_id"]
                
                result = db.reminders.insert_one(reminder_data)
                self._id = result.inserted_id
                return True, None  # Indicate success
            except Exception as e:
                return False, [str(e)]

    @staticmethod
    def find_by_id(db, reminder_id):
        """Find a reminder by ID."""
        if db is None:
            return None
            
        try:
            data = db.reminders.find_one({"_id": ObjectId(reminder_id)})
            return Reminder.from_mongo(data) if data else None
        except Exception:
            return None

    @staticmethod
    def find_by_user(db, user_id, completed=None):
        """Find all reminders for a user, optionally filtered by completion status."""
        if db is None:
            return []
            
        query = {"user_id": user_id}
        if completed is not None:
            query["completed"] = completed
            
        try:
            reminders_data = db.reminders.find(query).sort("start_time", 1)
            return [Reminder.from_mongo(r) for r in reminders_data]
        except Exception:
            return []

    @staticmethod
    def find_today_reminders(db, user_id):
        """Find reminders for today."""
        if db is None:
            return []
            
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = datetime.utcnow().replace(hour=23, minute=59, second=59, microsecond=999999)
        
        try:
            query = {
                "user_id": user_id,
                "start_time": {"$gte": today_start, "$lte": today_end}
            }
            reminders_data = db.reminders.find(query).sort("start_time", 1)
            return [Reminder.from_mongo(r) for r in reminders_data]
        except Exception:
            return []

    @staticmethod
    def find_upcoming_reminders(db, user_id, days=7):
        """Find upcoming reminders within the specified number of days."""
        if db is None:
            return []
            
        now = datetime.utcnow()
        future_date = now.replace(hour=23, minute=59, second=59) + timedelta(days=days)
        
        try:
            query = {
                "user_id": user_id,
                "start_time": {"$gte": now, "$lte": future_date},
                "completed": False
            }
            reminders_data = db.reminders.find(query).sort("start_time", 1)
            return [Reminder.from_mongo(r) for r in reminders_data]
        except Exception:
            return []

    @staticmethod
    def from_mongo(data):
        """Convert MongoDB document to Reminder object."""
        if not data:
            return None
            
        reminder = Reminder(
            user_id=data["user_id"],
            title=data["title"],
            description=data.get("description", ""),
            start_time=data["start_time"],
            end_time=data.get("end_time"),
            recurrence=data.get("recurrence"),
            completed=data.get("completed", False),
            google_event_id=data.get("google_event_id")
        )
        reminder._id = data["_id"]
        reminder.created_at = data.get("created_at", datetime.utcnow())
        reminder.updated_at = data.get("updated_at", datetime.utcnow())
        return reminder

    @staticmethod
    def get_completion_rate(db, user_id):
        """Calculate completion rate of reminders for a user."""
        if db is None:
            return 0
            
        try:
            # Get counts for completed and total reminders
            total_count = db.reminders.count_documents({"user_id": user_id})
            if total_count == 0:
                return 0
                
            completed_count = db.reminders.count_documents({
                "user_id": user_id,
                "completed": True
            })
            
            # Calculate percentage
            return round((completed_count / total_count) * 100, 2)
        except Exception:
            return 0
            
    @staticmethod
    def get_detailed_stats(db, user_id):
        """Get detailed statistics for a user's reminders."""
        if db is None:
            return {}
            
        try:
            # Basic stats
            total_reminders = db.reminders.count_documents({"user_id": user_id})
            completed_reminders = db.reminders.count_documents({
                "user_id": user_id,
                "completed": True
            })
            
            # Calculate completion rate
            completion_rate = 0
            if total_reminders > 0:
                completion_rate = round((completed_reminders / total_reminders) * 100, 2)
                
            # Get current time
            now = datetime.utcnow()
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            today_end = now.replace(hour=23, minute=59, second=59, microsecond=999999)
            
            # Today's reminders
            today_reminders = db.reminders.count_documents({
                "user_id": user_id,
                "start_time": {"$gte": today_start, "$lte": today_end}
            })
            
            # Today's completed reminders
            today_completed = db.reminders.count_documents({
                "user_id": user_id,
                "start_time": {"$gte": today_start, "$lte": today_end},
                "completed": True
            })
            
            # Upcoming reminders (next 7 days)
            week_end = today_end + timedelta(days=7)
            upcoming_reminders = db.reminders.count_documents({
                "user_id": user_id,
                "start_time": {"$gt": today_end, "$lte": week_end}
            })
            
            # Overdue reminders
            overdue_reminders = db.reminders.count_documents({
                "user_id": user_id,
                "start_time": {"$lt": now},
                "completed": False
            })
            
            # Reminders by recurrence type
            one_time_reminders = db.reminders.count_documents({
                "user_id": user_id,
                "recurrence": None
            })
            
            daily_reminders = db.reminders.count_documents({
                "user_id": user_id,
                "recurrence": "daily"
            })
            
            weekly_reminders = db.reminders.count_documents({
                "user_id": user_id,
                "recurrence": "weekly"
            })
            
            monthly_reminders = db.reminders.count_documents({
                "user_id": user_id,
                "recurrence": "monthly"
            })
            
            yearly_reminders = db.reminders.count_documents({
                "user_id": user_id,
                "recurrence": "yearly"
            })
            
            # Reminders created in last 30 days
            thirty_days_ago = now - timedelta(days=30)
            recent_reminders = db.reminders.count_documents({
                "user_id": user_id,
                "created_at": {"$gte": thirty_days_ago}
            })
            
            return {
                "total_reminders": total_reminders,
                "completed_reminders": completed_reminders,
                "completion_rate": completion_rate,
                "today_reminders": today_reminders,
                "today_completed": today_completed,
                "today_completion_rate": round((today_completed / today_reminders * 100), 2) if today_reminders > 0 else 0,
                "upcoming_reminders": upcoming_reminders,
                "overdue_reminders": overdue_reminders,
                "reminders_by_type": {
                    "one_time": one_time_reminders,
                    "daily": daily_reminders,
                    "weekly": weekly_reminders,
                    "monthly": monthly_reminders,
                    "yearly": yearly_reminders
                },
                "recently_created": recent_reminders
            }
        except Exception as e:
            print(f"Error getting detailed stats: {str(e)}")
            return {}

    @classmethod
    def find_by_user_and_timerange(cls, db, user_id, start_time, end_time):
        """Find reminders by user within a time range."""
        if not db:
            return []
            
        try:
            query = {
                "user_id": user_id,
                "start_time": {
                    "$gte": start_time,
                    "$lte": end_time
                }
            }
                
            reminders_data = db.reminders.find(query).sort("start_time", 1)
            return [cls.from_dict(data) for data in reminders_data]
        except Exception as e:
            print(f"Error finding reminders by time range: {str(e)}")
            return []

    @classmethod
    def from_google_event(cls, user_id, event):
        """Create a Reminder object from a Google Calendar event."""
        if not event or not user_id:
            return None
            
        try:
            # Extract event details
            title = event.get('summary', 'Untitled Event')
            description = event.get('description', '')
            
            # Parse start and end times
            start_time = None
            if 'dateTime' in event.get('start', {}):
                start_time = datetime.fromisoformat(event['start']['dateTime'].replace('Z', '+00:00'))
            elif 'date' in event.get('start', {}):
                # All-day event, use start of day
                start_time = datetime.fromisoformat(event['start']['date'] + 'T00:00:00')
                
            end_time = None    
            if 'dateTime' in event.get('end', {}):
                end_time = datetime.fromisoformat(event['end']['dateTime'].replace('Z', '+00:00'))
            elif 'date' in event.get('end', {}):
                # All-day event, use end of day
                end_time = datetime.fromisoformat(event['end']['date'] + 'T23:59:59')
                
            # Determine recurrence
            recurrence = None
            if 'recurrence' in event:
                for rule in event['recurrence']:
                    if rule.startswith('RRULE:'):
                        if 'FREQ=DAILY' in rule:
                            recurrence = 'daily'
                        elif 'FREQ=WEEKLY' in rule:
                            recurrence = 'weekly'
                        elif 'FREQ=MONTHLY' in rule:
                            recurrence = 'monthly'
                        elif 'FREQ=YEARLY' in rule:
                            recurrence = 'yearly'
                        break
            
            # Create reminder
            reminder = cls(
                user_id=user_id,
                title=title,
                description=description,
                start_time=start_time,
                end_time=end_time,
                recurrence=recurrence,
                completed=False,
                google_event_id=event.get('id')
            )
            
            return reminder
        except Exception as e:
            print(f"Error creating reminder from Google event: {str(e)}")
            return None

    @classmethod
    def from_dict(cls, data):
        """Create a Reminder from a dictionary representation."""
        if not data:
            return None
            
        reminder = cls(
            user_id=data.get("user_id"),
            title=data.get("title"),
            description=data.get("description", ""),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            recurrence=data.get("recurrence"),
            completed=data.get("completed", False),
            google_event_id=data.get("google_event_id")
        )
        if "_id" in data:
            reminder._id = data["_id"]
        if "created_at" in data:
            reminder.created_at = data.get("created_at")
        if "updated_at" in data:
            reminder.updated_at = data.get("updated_at")
            
        return reminder 