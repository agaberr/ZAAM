from werkzeug.security import generate_password_hash, checkForPass_hash
from datetime import datetime
import jwt
import os
from bson import ObjectId
from dotenv import load_dotenv

load_dotenv()

JWT_SECRET = os.getenv("JWT_SECRET")

class User:
    def __init__(self,  full_name,  age,  gender,  contact_info,  password,  emergency_contacts=  []):
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
        result = {
            "full_name": self.full_name ,
            "age": self.age,
            "gender":  self.gender,
            "contact_info": self.contact_info,
            "emergency_contacts": self.emergency_contacts,
            "created_at":  self.created_at,
            "updated_at": self.updated_at

        }
        
        if self._id is not None:
            result["_id"] = self._id
            

        return  result

    def save(self, db):
        if db is None:
            return False, ["Can't connect to the db"]
            
        errs = []
        if not self.full_name:
            errs.append("Full name is required")
        if not self.contact_info or not self.contact_info.get("email"):
            errs.append("Email is required")
            
        if errs:
            return False, errs
            
        existingUser = User.findUserByEmail(db, self.contact_info.get("email"))
        if self._id is None and existingUser is not None:
            return False, ["user is already in the database"]
            
        data = {
            "full_name": self.full_name, 
            "age": self.age,
            "gender":  self.gender,
            "contact_info" : self.contact_info,
            "password_hash": self.password_hash,
            "emergency_contacts": self.emergency_contacts,
            "created_at":  self.created_at,
            "updated_at": self.updated_at ,  
        }
        
        if self._id is not None:
            data["updated_at"] = datetime.utcnow()
            try:
                result = db.users.update_one(
                    {"_id": self._id},
                    {"$set": data}
                )
                return result.modified_count > 0, None
            except:
                return False, ["can't create a user"]
        else:
            try:
                if "_id" in data:
                    del data["_id"]
                
                result = db.users.insert_one(data)
                self._id = result.inserted_id
                return True, None
            except  :

                return False, ["can't create user"]



    @staticmethod
    def findUserByEmail(db, email):
        if db is None:
            return None
            
        try:
            data = db.users.find_one({"contact_info.email": email})

            if  data is not None:
             return User.fromMongo(data)
            else:
                return None
        except  :
            print("can't find user by email")
            return None

    @staticmethod
    def findByID(db, user_id):
        if db is None:
            return None
            
        try:
            data = db.users.find_one({"_id": ObjectId(user_id)})
            if  data is not None:
             return User.fromMongo(data)
            else:
                return None
        except:
            print( "can't find user...")
            return None

    @staticmethod
    def fromMongo(data):
        if not data:
            return None
        user = User(
            full_name=data["full_name"] ,
            age=data ["age"],
            gender=data["gender"],
            contact_info=data["contact_info"],
            password= "dummy" , 
            emergency_contacts = data.get("emergency_contacts", []),
        )

        user._id = data["_id"]
        user.password_hash = data["password_hash"]
        user.created_at = data["created_at"]
        user.updated_at = data["updated_at"]
        return user

    def checkForPass(self, password):
        return checkForPass_hash(self.password_hash, password)

    def generateAuthToken(self):
        payload = {"user_id": str(self._id)}
        return jwt.encode(payload, JWT_SECRET, algorithm="HS256")
