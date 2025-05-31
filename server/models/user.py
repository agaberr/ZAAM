from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import jwt
import os
from bson import ObjectId
from dotenv import load_dotenv

load_dotenv()

JWT_SECRET = os.getenv("JWT_SECRET")

class User:
    def __init__(self, full_name, age, gender, contact_info, password, emergency_contacts=[]):
        self._id = None
        self.full_name = full_name
        self.age = age
        self.gender = gender
        self.contact_info = contact_info
        self.password_hash = generate_password_hash(password)
        self.emergency_contacts = emergency_contacts
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def to_dict(self):
        """Convert user to dictionary for serialization"""
        result = {
            "full_name": self.full_name,
            "age": self.age,
            "gender": self.gender,
            "contact_info": self.contact_info,
            "emergency_contacts": self.emergency_contacts,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
        
        # Only include _id if it's not None
        if self._id is not None:
            result["_id"] = self._id
            
        return result

    def save(self, db):
        """Save the user to the database."""
        if db is None:
            return False, ["Database connection is not available"]
            
        # Validate required fields
        errors = []
        if not self.full_name:
            errors.append("Full name is required")
        if not self.contact_info or not self.contact_info.get("email"):
            errors.append("Email is required")
            
        if errors:
            return False, errors
            
        # Check if email is already registered
        existing_user = User.find_by_email(db, self.contact_info.get("email"))
        if self._id is None and existing_user is not None:
            return False, ["Email is already registered"]
            
        user_data = {
            "full_name": self.full_name,
            "age": self.age,
            "gender": self.gender,
            "contact_info": self.contact_info,
            "password_hash": self.password_hash,
            "emergency_contacts": self.emergency_contacts,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        
        # For update operations
        if self._id is not None:
            user_data["updated_at"] = datetime.utcnow()
            try:
                result = db.users.update_one(
                    {"_id": self._id},
                    {"$set": user_data}
                )
                return result.modified_count > 0, None
            except Exception as e:
                return False, [str(e)]
        # For insert operations
        else:
            try:
                # Remove _id field for new records since it's null and MongoDB will auto-generate it
                if "_id" in user_data:
                    del user_data["_id"]
                
                result = db.users.insert_one(user_data)
                self._id = result.inserted_id
                return True, None  # Indicate success
            except Exception as e:
                return False, [str(e)]

    @staticmethod
    def find_by_email(db, email):
        """Find a user by email."""
        if db is None:
            return None
            
        try:
            data = db.users.find_one({"contact_info.email": email})
            # Check if data exists, not if data is truthy
            return User.from_mongo(data) if data is not None else None
        except Exception as e:
            print(f"Error finding user by email: {str(e)}")
            return None

    @staticmethod
    def find_by_id(db, user_id):
        """Find a user by ID."""
        if db is None:
            return None
            
        try:
            data = db.users.find_one({"_id": ObjectId(user_id)})
            # Check if data exists, not if data is truthy
            return User.from_mongo(data) if data is not None else None
        except Exception as e:
            print(f"Error finding user by ID: {str(e)}")
            return None

    @staticmethod
    def from_mongo(data):
        """Convert MongoDB document to User object."""
        if not data:
            return None
        user = User(
            full_name=data["full_name"],
            age=data["age"],
            gender=data["gender"],
            contact_info=data["contact_info"],
            password="dummy",  # Placeholder, since we load hash separately
            emergency_contacts=data.get("emergency_contacts", []),
        )
        user._id = data["_id"]
        user.password_hash = data["password_hash"]
        user.created_at = data["created_at"]
        user.updated_at = data["updated_at"]
        return user

    def check_password(self, password):
        """Check if the provided password matches the stored hash."""
        return check_password_hash(self.password_hash, password)

    def generate_auth_token(self):
        """Generate a JWT token for authentication."""
        payload = {"user_id": str(self._id)}
        return jwt.encode(payload, JWT_SECRET, algorithm="HS256")
