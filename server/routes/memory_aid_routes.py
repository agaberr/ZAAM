from flask import Blueprint, request, jsonify, current_app
from bson import ObjectId
import jwt
import os
from functools import wraps
from dotenv import load_dotenv
from models.memory_aid import MemoryAid

load_dotenv()

JWT_SECRET = os.getenv("JWT_SECRET")

memory_aid_routes = Blueprint("memory_aid_routes", __name__)

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # try to see the autherization to access the memoryaid routes
        if "Authorization" in request.headers:
            auth_header = request.headers["Authorization"]
            if auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
        
        if not token:
            return jsonify({"error": "there is no token, try to sign in or smthing"}), 401
        
        try:
            data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            user_id = data["user_id"]
            
            kwargs["user_id"] = user_id
            
            return f(*args, **kwargs)
        except :
            return jsonify({"error": "try to sign in again, the token has something wrong with it.."}), 401
        
    return decorated




@memory_aid_routes.route("/api/memory-aids", methods=["POST"])
@token_required
def createMemAid(userid):
    db = current_app.config["DATABASE"]

    data =  request.json


    if db is None:
        return jsonify({"error": "Database connection failed"}), 500

    if not  data:
        return jsonify({"error": "Please add some data!"}), 400

    memFields = ["title", "type"]
    for f in memFields:

        if  f not in data:
                return jsonify({"error": "please add all fields"}), 400
   

    memAid = MemoryAid(
        user_id=userid,
        title=data["title"],
        description=data.get("description", ""),
        type=data["type"],
        date=data.get("date")
    )

    success, _ = memAid.save(db)
    if not success:
        return jsonify({"error": "Failed to create memory aid"}), 400

    dictmemAid = memAid.to_dict()

    return jsonify(dictmemAid), 201


@memory_aid_routes.route("/api/memory-aids", methods=["GET"])
@token_required
def get_memory_aids(user_id):
    db = current_app.config["DATABASE"]
    if db is None:
        return jsonify({"error": "Database connection failed"}), 500
        
    allMemAids = MemoryAid.find_by_user_id(db, user_id)
    
    allMemaidsData = [memory_aid.to_dict() for memory_aid in allMemAids]
    
    return jsonify(allMemaidsData), 200


@memory_aid_routes.route("/api/memory-aids/<memory_aid_id>", methods=["GET"])
@token_required
def get_memory_aid(user_id, memory_aid_id):
    db = current_app.config["DATABASE"]
    if db is None:
        return jsonify({"error": "database failed to connect"}), 500
        
    memAid = MemoryAid.findByID(db, memory_aid_id)
    
    if not memAid:
        return jsonify({"error": "there is no memory aids there"}), 404
        
    if memAid.user_id != user_id:
        return jsonify({"error": "you have no permission to get the data"}), 403
        
    return jsonify(memAid.to_dict()), 200


@memory_aid_routes.route("/api/memory-aids/<memory_aid_id>", methods=["PUT"])
@token_required
def update_memory_aid(user_id, memory_aid_id):
    db = current_app.config["DATABASE"]
    if db is None:
        return jsonify({"error": "Failed to conn with database"}), 500
        
    memAid = MemoryAid.findByID(db, memory_aid_id)
    
    if not memAid:
        return jsonify({"error": "No memory aids are to be found"}), 404
        
    if memAid.user_id != user_id:
        return jsonify({"error": "permision required to update in this"}), 403
        
    data = request.json
    
    if not data:
        return jsonify({"error": "You didn't add data to update with"}), 400
        
    if "title" in data:
        memAid.title = data["title"]
    if "description" in data:
        memAid.description = data["description"]
    if "type" in data:
        memAid.type = data["type"]
    if "date" in data:
        memAid.date = data["date"]
        
    success, _ = memAid.save(db)
    
    if not success:
        return jsonify({"error": "can't update memory aid"}), 400
        
    return jsonify(memAid.to_dict()), 200


@memory_aid_routes.route("/api/memory-aids/<memory_aid_id>", methods=["DELETE"])
@token_required
def delete_memory_aid(user_id, memory_aid_id):
    db = current_app.config["DATABASE"]
    if db is None:
        return jsonify({"error": "failed to connect with db"}), 500
        
    success = MemoryAid.deleteByID(db, memory_aid_id, user_id)
    
    if not success:
        return jsonify({"error": "error occured while deleting"}), 404
        
    return jsonify({"message": "Memory aid deleted successfully"}), 200 

