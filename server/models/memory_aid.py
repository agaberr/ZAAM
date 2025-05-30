from datetime import datetime
from bson import ObjectId

class MemoryAid:
    def __init__(self,  userid,  title,  description, type,  date =None):
        self._id = None
        self.userid = userid
        self.title = title
        self.description  = description
        self.type = type
        self.date = date or datetime.utcnow().strftime('%Y-%m-%d')
        self.created_at =  datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def to_dict(self):
        res = {
            "userid":   self.userid,
            "title": self.title,
            "description": self.description ,
            "type": self.type,
            "date" : self.date,
            "created_at" : self.created_at,
            "updated_at":   self.updated_at  
        }
        
        if self._id is not None:
            res["_id"] = str(self._id)
            
        return res

    def save(self, db):
        if db is None:
            return False, ["can't connect to db"]
            
        errs = []
        if not self.userid:
            errs.append("please add user id")
        if not self.title:
            errs.append("please add title")
        if not self.type:
            errs.append("please add type")
            
        if errs:
            return False, errs
            
        memaid_data = {
            "userid": self.userid,
            "title": self.title,
            "description":   self.description,
            "type": self.type,
            "date": self.date,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        
        if self._id is not None:
            memaid_data["updated_at"] = datetime.utcnow()
            try:
                result = db.memory_aids.update_one(
                    {"_id": self._id},
                    {"$set": memaid_data}
                )
                return result.modified_count > 0, None
            except:
                return False, ["error creating memory aid"]
        else:
            try:
                result = db.memory_aids.insert_one(memaid_data)
                self._id = result.inserted_id
                return True, None
            except:
                return False, ["error creating memory aid"]


    @staticmethod
    def findByID(db, memory_aid_id):
        if db is None:
            return None
        try:
            memaid =  db.memory_aids.find_one({"_id": ObjectId(memory_aid_id)})

            if memaid:
                return   MemoryAid.fromMongo(memaid)
            else:
                return None
        except:
            return None

    @staticmethod
    def findByUserID(db, userid):
        if db is None:
            return []
            
        try:
            memaid = db.memory_aids.find({"userid": userid})
            allMemaids = []
            for data in memaid:
                memory_aid = MemoryAid.fromMongo(data)
                if memory_aid:
                    allMemaids.append(memory_aid)
            return allMemaids
        except:
            return []

    @staticmethod
    def deleteByID(db, memaidID, userid=None):
        if db is None:
            return False
            
        try:
            query = {"_id": ObjectId(memaidID)}
            if userid:
                query["userid"] = userid
                
            result = db.memory_aids.delete_one(query)
            return result.deleted_count > 0
        except Exception:
            return False

    @staticmethod
    def fromMongo(data):
        if not data:
            return None
            
        memaid = MemoryAid(
            userid=data["userid"],
            title=data["title"],
            description=data.get("description", ""),
            type=data["type"],
            date=data.get("date")
        )
        memaid._id = data["_id"]
        memaid.created_at = data["created_at"]
        memaid.updated_at = data["updated_at"]
        return memaid 