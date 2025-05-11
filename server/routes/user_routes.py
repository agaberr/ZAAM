from flask import jsonify, request
from bson.objectid import ObjectId
from models.user import User
from datetime import datetime


def register_user_routes(app, mongo):

    ###### USER ROUTES ######
    # POST: /api/users -> Create a new user
    # GET: /api/users/<user_id> -> Get a user by ID
    # PUT: /api/users/<user_id> -> Update a user by ID
    # DELETE: /api/users/<user_id> -> Delete a user by ID

    ##################################### CREATE USER ROUTE START #####################################
    @app.route('/api/users', methods=['POST'])
    def create_user():
        try:
            user_data = request.json
            
            # Create user object
            user = User(
                full_name=user_data.get('full_name'),
                age=user_data.get('age'),
                gender=user_data.get('gender'),
                contact_info=user_data.get('contact_info', {}),
                emergency_contacts=user_data.get('emergency_contacts', [])
            )
            
            # Validate and save user
            success, errors = user.save(mongo.db)
            
            if not success:
                return jsonify({"error": "Validation failed", "details": errors}), 400
            
            return jsonify({
                "message": "User registered successfully",
                "user_id": str(user._id)
            }), 201
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    ##################################### CREATE USER ROUTE END #####################################

    ##################################### GET USER ROUTE START #####################################
    @app.route('/api/users/<user_id>', methods=['GET'])
    def get_user(user_id):
        try:
            user = User.find_by_id(mongo.db, user_id)
            
            if not user:
                return jsonify({"error": "User not found"}), 404
            
            # Convert to dictionary and convert ObjectId to string
            user_dict = user.to_dict()
            user_dict['_id'] = str(user_dict['_id'])
            
            # Convert datetime objects to ISO format strings
            user_dict['created_at'] = user_dict['created_at'].isoformat()
            user_dict['updated_at'] = user_dict['updated_at'].isoformat()
            
            return jsonify(user_dict)
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    ##################################### GET USER ROUTE END #####################################

    ##################################### UPDATE USER ROUTE START #####################################
    @app.route('/api/users/<user_id>', methods=['PUT'])
    def update_user(user_id):
        try:
            user_data = request.json
            user = User.find_by_id(mongo.db, user_id)
            
            if not user:
                return jsonify({"error": "User not found"}), 404
            
            # Update user fields
            if 'full_name' in user_data:
                user.full_name = user_data['full_name']
            if 'age' in user_data:
                user.age = user_data['age']
            if 'gender' in user_data:
                user.gender = user_data['gender']
            if 'contact_info' in user_data:
                user.contact_info = user_data['contact_info']
            if 'emergency_contacts' in user_data:
                user.emergency_contacts = user_data['emergency_contacts']
            
            # Save updated user
            success, errors = user.save(mongo.db)
            
            if not success:
                return jsonify({"error": "Validation failed", "details": errors}), 400
            
            return jsonify({"message": "User updated successfully"})
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    ##################################### UPDATE USER ROUTE END #####################################

    ##################################### DELETE USER ROUTE START #####################################
    @app.route('/api/users/<user_id>', methods=['DELETE'])
    def delete_user(user_id):
        try:
            result = mongo.db.users.delete_one({"_id": ObjectId(user_id)})
            
            if result.deleted_count == 0:
                return jsonify({"error": "User not found"}), 404
            
            return jsonify({"message": "User deleted successfully"})
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500 
    ##################################### DELETE USER ROUTE END #####################################
