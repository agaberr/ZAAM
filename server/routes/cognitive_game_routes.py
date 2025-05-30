from flask import Blueprint, request, jsonify, current_app, session
from bson import ObjectId
import jwt
import os
from functools import wraps
from dotenv import load_dotenv
from cognitive_game import CognitiveGame

load_dotenv()

JWT_SECRET = os.getenv("JWT_SECRET")

cognitive_game_routes = Blueprint("cognitive_game_routes", __name__)

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

###### COGNITIVE GAME ROUTES ######
# POST: /api/game/start -> Start a new game session
# POST: /api/game/question -> Get a new question
# POST: /api/game/answer -> Submit an answer and get validation
# POST: /api/game/stop -> Stop the current game session

@cognitive_game_routes.route("/api/game/start", methods=["POST"])
@token_required
def start_game(user_id):
    """Start a new cognitive game session"""
    db = current_app.config["DATABASE"]
    if db is None:
        return jsonify({"error": "Database connection failed"}), 500
    
    try:
        # Initialize the game for the user
        game = CognitiveGame(db, user_id)
        
        # Check if user has any people or events
        people = game.get_all_people()
        events = game.get_all_events()
        
        if not people and not events:
            return jsonify({
                "success": False,
                "message": "You don't have any people or events yet. Please add some people or events before playing the game!"
            }), 400
        
        # Store game session in session storage
        session['game_active'] = True
        session['game_user_id'] = user_id
        session['game_score'] = 0
        session['game_questions_asked'] = 0
        
        # Generate first question
        question = game.generate_random_question()
        session['current_question'] = question
        
        return jsonify({
            "success": True,
            "message": "Cognitive game started! Let's test your memory!",
            "question": question,
            "people_count": len(people),
            "events_count": len(events),
            "score": 0,
            "questions_asked": 0
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": f"Failed to start game: {str(e)}",
            "success": False
        }), 500

@cognitive_game_routes.route("/api/game/question", methods=["POST"])
@token_required
def get_question(user_id):
    """Get a new question for the current game session"""
    db = current_app.config["DATABASE"]
    if db is None:
        return jsonify({"error": "Database connection failed"}), 500
    
    # Check if game is active
    if not session.get('game_active') or session.get('game_user_id') != user_id:
        return jsonify({
            "error": "No active game session. Please start a new game first.",
            "success": False
        }), 400
    
    try:
        # Initialize the game for the user
        game = CognitiveGame(db, user_id)
        
        # Generate a new question
        question = game.generate_random_question()
        session['current_question'] = question
        
        return jsonify({
            "success": True,
            "question": question,
            "score": session.get('game_score', 0),
            "questions_asked": session.get('game_questions_asked', 0)
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": f"Failed to generate question: {str(e)}",
            "success": False
        }), 500

@cognitive_game_routes.route("/api/game/answer", methods=["POST"])
@token_required
def submit_answer(user_id):
    """Submit an answer and get validation"""
    db = current_app.config["DATABASE"]
    if db is None:
        return jsonify({"error": "Database connection failed"}), 500
    
    # Check if game is active
    if not session.get('game_active') or session.get('game_user_id') != user_id:
        return jsonify({
            "error": "No active game session. Please start a new game first.",
            "success": False
        }), 400
    
    data = request.json
    if not data or 'answer' not in data:
        return jsonify({"error": "No answer provided"}), 400
    
    user_answer = data['answer']
    current_question = session.get('current_question')
    
    if not current_question:
        return jsonify({
            "error": "No current question to answer. Please get a new question first.",
            "success": False
        }), 400
    
    try:
        # Initialize the game for the user
        game = CognitiveGame(db, user_id)
        
        # Check the answer
        result = game.check_answer(current_question, user_answer)
        
        # Update session data
        session['game_questions_asked'] = session.get('game_questions_asked', 0) + 1
        
        if result.get('correct', False):
            session['game_score'] = session.get('game_score', 0) + 1
        
        # Generate next question
        next_question = game.generate_random_question()
        session['current_question'] = next_question
        
        return jsonify({
            "success": True,
            "validation_result": result,
            "next_question": next_question,
            "score": session.get('game_score', 0),
            "questions_asked": session.get('game_questions_asked', 0),
            "accuracy": round((session.get('game_score', 0) / session.get('game_questions_asked', 1)) * 100, 2)
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": f"Failed to validate answer: {str(e)}",
            "success": False
        }), 500

@cognitive_game_routes.route("/api/game/stop", methods=["POST"])
@token_required
def stop_game(user_id):
    """Stop the current game session"""
    # Check if game is active
    if not session.get('game_active') or session.get('game_user_id') != user_id:
        return jsonify({
            "error": "No active game session to stop.",
            "success": False
        }), 400
    
    try:
        # Get final stats
        final_score = session.get('game_score', 0)
        total_questions = session.get('game_questions_asked', 0)
        accuracy = round((final_score / total_questions) * 100, 2) if total_questions > 0 else 0
        
        # Clear game session
        session.pop('game_active', None)
        session.pop('game_user_id', None)
        session.pop('game_score', None)
        session.pop('game_questions_asked', None)
        session.pop('current_question', None)
        
        return jsonify({
            "success": True,
            "message": "Game session ended. Great job exercising your memory!",
            "final_stats": {
                "score": final_score,
                "total_questions": total_questions,
                "accuracy": accuracy
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": f"Failed to stop game: {str(e)}",
            "success": False
        }), 500

@cognitive_game_routes.route("/api/game/status", methods=["GET"])
@token_required
def get_game_status(user_id):
    """Get current game session status"""
    if not session.get('game_active') or session.get('game_user_id') != user_id:
        return jsonify({
            "game_active": False,
            "message": "No active game session"
        }), 200
    
    return jsonify({
        "game_active": True,
        "score": session.get('game_score', 0),
        "questions_asked": session.get('game_questions_asked', 0),
        "current_question": session.get('current_question'),
        "accuracy": round((session.get('game_score', 0) / max(session.get('game_questions_asked', 1), 1)) * 100, 2)
    }), 200 