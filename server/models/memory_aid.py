from datetime import datetime
from bson import ObjectId

class MemoryAid:
    def __init__(self, user_id, title, description, type, date=None, image_url=None, date_of_birth=None, date_met_patient=None, date_of_occurrence=None):
        self._id = None
        self.user_id = user_id
        self.title = title
        self.description = description
        self.type = type  # 'person', 'place', 'event', 'object'
        self.date = date or datetime.utcnow().strftime('%Y-%m-%d')
        self.image_url = image_url
        self.date_of_birth = date_of_birth  # For person type - their date of birth
        self.date_met_patient = date_met_patient  # For person type - when they met the patient
        self.date_of_occurrence = date_of_occurrence  # For event type - when the event occurred
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def to_dict(self):
        """Convert memory aid to dictionary for serialization"""
        result = {
            "user_id": self.user_id,
            "title": self.title,
            "description": self.description,
            "type": self.type,
            "date": self.date,
            "image_url": self.image_url,
            "date_of_birth": self.date_of_birth,
            "date_met_patient": self.date_met_patient,
            "date_of_occurrence": self.date_of_occurrence,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
        
        # Only include _id if it's not None
        if self._id is not None:
            result["_id"] = str(self._id)
            
        return result

    def save(self, db):
        """Save the memory aid to the database."""
        if db is None:
            return False, ["Database connection is not available"]
            
        # Validate required fields
        errors = []
        if not self.user_id:
            errors.append("User ID is required")
        if not self.title:
            errors.append("Title is required")
        if not self.type:
            errors.append("Type is required")
            
        if errors:
            return False, errors
            
        memory_aid_data = {
            "user_id": self.user_id,
            "title": self.title,
            "description": self.description,
            "type": self.type,
            "date": self.date,
            "image_url": self.image_url,
            "date_of_birth": self.date_of_birth,
            "date_met_patient": self.date_met_patient,
            "date_of_occurrence": self.date_of_occurrence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        
        # For update operations
        if self._id is not None:
            memory_aid_data["updated_at"] = datetime.utcnow()
            try:
                result = db.memory_aids.update_one(
                    {"_id": self._id},
                    {"$set": memory_aid_data}
                )
                return result.modified_count > 0, None
            except Exception as e:
                return False, [str(e)]
        # For insert operations
        else:
            try:
                result = db.memory_aids.insert_one(memory_aid_data)
                self._id = result.inserted_id
                return True, None  # Indicate success
            except Exception as e:
                return False, [str(e)]

    @staticmethod
    def findByID(db, memory_aid_id):
        """Find a memory aid by ID."""
        if db is None:
            return None
            
        try:
            data = db.memory_aids.find_one({"_id": ObjectId(memory_aid_id)})
            return MemoryAid.fromMongo(data) if data else None
        except Exception:
            return None

    @staticmethod
    def find_by_user_id(db, user_id):
        """Find all memory aids for a specific user."""
        if db is None:
            return []
            
        try:
            cursor = db.memory_aids.find({"user_id": user_id})
            memory_aids = []
            for data in cursor:
                memory_aid = MemoryAid.fromMongo(data)
                if memory_aid:
                    memory_aids.append(memory_aid)
            return memory_aids
        except Exception:
            return []

    @staticmethod
    def delete_by_id(db, memory_aid_id, user_id=None):
        """Delete a memory aid by ID, optionally check if it belongs to the specified user."""
        if db is None:
            return False
            
        try:
            query = {"_id": ObjectId(memory_aid_id)}
            if user_id:
                query["user_id"] = user_id
                
            result = db.memory_aids.delete_one(query)
            return result.deleted_count > 0
        except Exception:
            return False

    @staticmethod
    def fromMongo(data):
        """Convert MongoDB document to MemoryAid object."""
        if not data:
            return None
            
        memory_aid = MemoryAid(
            user_id=data["user_id"],
            title=data["title"],
            description=data.get("description", ""),
            type=data["type"],
            date=data.get("date"),
            image_url=data.get("image_url"),
            date_of_birth=data.get("date_of_birth"),
            date_met_patient=data.get("date_met_patient"),
            date_of_occurrence=data.get("date_of_occurrence")
        )
        memory_aid._id = data["_id"]
        memory_aid.created_at = data["created_at"]
        memory_aid.updated_at = data["updated_at"]
        return memory_aid 