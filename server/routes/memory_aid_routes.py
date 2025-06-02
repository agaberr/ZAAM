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
        
        # Check if token is in headers
        if "Authorization" in request.headers:
            auth_header = request.headers["Authorization"]
            if auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
        
        if not token:
            return jsonify({"error": "Authentication token is missing"}), 401
        
        try:
            # Decode the token
            data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            user_id = data["user_id"]
            
            # Pass user_id to the route function
            kwargs["user_id"] = user_id
            
            return f(*args, **kwargs)
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token has expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
        
    return decorated

    ###### MEMORY AID ROUTES ######
    # POST: /api/memory-aids -> Create a new memory aid
    # GET: /api/memory-aids -> Get all memory aids
    # GET: /api/memory-aids/<memory_aid_id> -> Get a memory aid by ID
    # PUT: /api/memory-aids/<memory_aid_id> -> Update a memory aid by ID
    # DELETE: /api/memory-aids/<memory_aid_id> -> Delete a memory aid by ID


##################################### CREATE MEMORY AID ROUTE START #####################################

@memory_aid_routes.route("/api/memory-aids", methods=["POST"])
@token_required
def create_memory_aid(user_id):
    db = current_app.config["DATABASE"]
    if db is None:
        return jsonify({"error": "Database connection failed"}), 500
        
    data = request.json
    
    # Validate input
    if not data:
        return jsonify({"error": "No data provided"}), 400
        
    required_fields = ["title", "type"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400
            
    # Create memory aid
    memory_aid = MemoryAid(
        user_id=user_id,
        title=data["title"],
        description=data.get("description", ""),
        type=data["type"],
        date=data.get("date"),
        image_url=data.get("image_url"),
        date_of_birth=data.get("date_of_birth"),
        date_met_patient=data.get("date_met_patient"),
        date_of_occurrence=data.get("date_of_occurrence")
    )
    
    # Save to database
    success, errors = memory_aid.save(db)
    
    if not success:
        return jsonify({"error": errors[0] if errors else "Failed to create memory aid"}), 400
        
    # Return the created memory aid
    return jsonify(memory_aid.to_dict()), 201
##################################### CREATE MEMORY AID ROUTE END #####################################

##################################### GET MEMORY AID ROUTE START #####################################  

@memory_aid_routes.route("/api/memory-aids", methods=["GET"])
@token_required
def get_memory_aids(user_id):
    db = current_app.config["DATABASE"]
    if db is None:
        return jsonify({"error": "Database connection failed"}), 500
        
    # Get memory aids for the user
    memory_aids = MemoryAid.find_by_user_id(db, user_id)
    
    # Convert to dictionaries for JSON response
    memory_aids_data = [memory_aid.to_dict() for memory_aid in memory_aids]
    
    return jsonify(memory_aids_data), 200
##################################### GET MEMORY AID ROUTE END #####################################  

##################################### GET MEMORY AID ROUTE BY ID START #####################################  

@memory_aid_routes.route("/api/memory-aids/<memory_aid_id>", methods=["GET"])
@token_required
def get_memory_aid(user_id, memory_aid_id):
    db = current_app.config["DATABASE"]
    if db is None:
        return jsonify({"error": "Database connection failed"}), 500
        
    # Get the memory aid
    memory_aid = MemoryAid.findByID(db, memory_aid_id)
    
    if not memory_aid:
        return jsonify({"error": "Memory aid not found"}), 404
        
    # Check if the memory aid belongs to the user
    if memory_aid.user_id != user_id:
        return jsonify({"error": "You don't have permission to access this memory aid"}), 403
        
    return jsonify(memory_aid.to_dict()), 200
##################################### GET MEMORY AID ROUTE BY ID END #####################################  

##################################### UPDATE MEMORY AID ROUTE BY ID START #####################################  

@memory_aid_routes.route("/api/memory-aids/<memory_aid_id>", methods=["PUT"])
@token_required
def update_memory_aid(user_id, memory_aid_id):
    db = current_app.config["DATABASE"]
    if db is None:
        return jsonify({"error": "Database connection failed"}), 500
        
    # Get the memory aid
    memory_aid = MemoryAid.findByID(db, memory_aid_id)
    
    if not memory_aid:
        return jsonify({"error": "Memory aid not found"}), 404
        
    # Check if the memory aid belongs to the user
    if memory_aid.user_id != user_id:
        return jsonify({"error": "You don't have permission to update this memory aid"}), 403
        
    # Update the memory aid
    data = request.json
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
        
    # Update fields
    if "title" in data:
        memory_aid.title = data["title"]
    if "description" in data:
        memory_aid.description = data["description"]
    if "type" in data:
        memory_aid.type = data["type"]
    if "date" in data:
        memory_aid.date = data["date"]
    if "image_url" in data:
        memory_aid.image_url = data["image_url"]
    if "date_of_birth" in data:
        memory_aid.date_of_birth = data["date_of_birth"]
    if "date_met_patient" in data:
        memory_aid.date_met_patient = data["date_met_patient"]
    if "date_of_occurrence" in data:
        memory_aid.date_of_occurrence = data["date_of_occurrence"]
        
    # Save changes
    success, errors = memory_aid.save(db)
    
    if not success:
        return jsonify({"error": errors[0] if errors else "Failed to update memory aid"}), 400
        
    return jsonify(memory_aid.to_dict()), 200
##################################### UPDATE MEMORY AID ROUTE BY ID END #####################################  

##################################### DELETE MEMORY AID ROUTE BY ID START #####################################  


@memory_aid_routes.route("/api/memory-aids/<memory_aid_id>", methods=["DELETE"])
@token_required
def delete_memory_aid(user_id, memory_aid_id):
    db = current_app.config["DATABASE"]
    if db is None:
        return jsonify({"error": "Database connection failed"}), 500
        
    # Delete the memory aid
    success = MemoryAid.delete_by_id(db, memory_aid_id, user_id)
    
    if not success:
        return jsonify({"error": "Memory aid not found or you don't have permission to delete it"}), 404
        
    return jsonify({"message": "Memory aid deleted successfully"}), 200 
##################################### DELETE MEMORY AID ROUTE BY ID END #####################################  

