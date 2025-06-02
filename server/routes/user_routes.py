from flask import jsonify, request
from bson.objectid import ObjectId
from models.user import User
from datetime import datetime


def user_routes(app, mongo):

    @app.route('/api/users', methods=['POST'])
    def createUser():
        
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        user = User(
            full_name=data.get('full_name'),
            age=data.get('age'),
            gender=data.get('gender'),
            contact_info=data.get('contact_info', {}),
            emergency_contacts=data.get('emergency_contacts', [])
        )
        
        success, _ = user.save(mongo.db)
        
        if not success:
            return jsonify({"error": "can't create a new user"}), 400
        
        return jsonify({
            "message": "user is created",
            "user_id": str(user._id)
        }), 201
            
        

    @app.route('/api/users/<user_id>', methods=['GET'])
    def getUser(user_id):
        try:
            user = User.findByID(mongo.db, user_id)
            
            if not user:
                return jsonify({"error": "User not found"}), 404
            
            # bakhaleh keda 3shan a return data correctly
            usertodictionary  = user.to_dict()
            usertodictionary['_id'] = str(usertodictionary['_id'])
            
            return jsonify(usertodictionary)
            
        except:
            return jsonify({"err occured in server"}), 500

    @app.route('/api/users/<user_id>', methods=['PUT'])
    def updateUser(user_id):
        try:
            data = request.json
            user = User.findByID(mongo.db, user_id)
            
            if not user:
                return jsonify({"error": "User not found"}), 404
            
            if 'full_name' in data:
                user.full_name = data['full_name']
            if 'age' in data:
                user.age = data['age']
            if 'gender' in data:
                user.gender = data['gender']
            if 'contact_info' in data:
                user.contact_info = data['contact_info']
            if 'emergency_contacts' in data:
                user.emergency_contacts = data['emergency_contacts']
            
            success, _ = user.save(mongo.db)
            
            if not success:
                return jsonify({"error": "err updating data to user"}), 400
            
            return jsonify({"message": "user updated successfully.."})
            
        except:
            return jsonify({"error": "some server errors happened"}), 500

    @app.route('/api/users/<user_id>', methods=['DELETE'])
    def deleteUsr(user_id):
        try:
            usertodelete = mongo.db.users.delete_one({"_id": ObjectId(user_id)})
            
            if usertodelete.deleted_count == 0:
                return jsonify({"error": "User not found"}), 404
            
            return jsonify({"message": "User deleted successfully"})
            
        except:
            return jsonify({"error": "error deleting user in the dbb.."}), 500 
