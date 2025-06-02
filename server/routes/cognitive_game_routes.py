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
        
        if "Authorization" in request.headers:
            auth_header = request.headers["Authorization"]
            if auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
        
        if not token:
            return jsonify({"error": "Authentication token is missing"}), 401
        
        try:
            data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            user_id = data["user_id"]
            
            kwargs["user_id"] = user_id
            
            return f(*args, **kwargs)
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token has expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
        
    return decorated


@cognitive_game_routes.route("/api/game/start", methods=["POST"])
@token_required
def start_game(user_id):
    db = current_app.config["DATABASE"]
    if db is None:
        return jsonify({"error": "Database connection failed"}), 500
    
    try:
        game = CognitiveGame(db, user_id)
        
        people = game.get_all_people()
        events = game.get_all_events()
        
        if not people and not events:
            return jsonify({"message": "Add some memory aids to play game!!!"}), 400
        
        session['game_active'] = True
        session['game_user_id'] = user_id
        session['game_score'] = 0
        session['game_questions_asked'] = 0
        
        question = game.generate_random_question()
        session['current_question'] = question
        
        return jsonify({
            "message": "Cognitive game started! Let's test your memory!",
            "question": question,
            "people_count": len(people),
            "events_count": len(events),
            "score": 0,
            "questions_asked": 0
        }), 200
        
    except :
        return jsonify({"error": "can;t start game..."}), 500

@cognitive_game_routes.route("/api/game/question", methods=["POST"])
@token_required
def get_question(user_id):
    db = current_app.config["DATABASE"]
    if db is None:
        return jsonify({"error": "Database connection failed"}), 500
    
    if not session.get('game_active') or session.get('game_user_id') != user_id:
        return jsonify({"error": "Game is not active so say start game to start..."}), 400
    
    try:
        game = CognitiveGame(db, user_id)
        
        question = game.generate_random_question()
        session['current_question'] = question
        
        return jsonify({
            "question": question,
            "score": session.get('game_score', 0),
            "questions_asked": session.get('game_questions_asked', 0) }), 200
        
    except :
        return jsonify({"error": "There is something wrong with generating questions"}), 500

@cognitive_game_routes.route("/api/game/answer", methods=["POST"])
@token_required
def submit_answer(user_id):
    db = current_app.config["DATABASE"]
    if db is None:
        return jsonify({"error": "can't connect to the current database...."}), 500
    
    if not session.get('game_active') or session.get('game_user_id') != user_id:
        return jsonify({"error": "Say start game to start a new game"}), 400
    
    data = request.json
    if not data or 'answer' not in data:
        return jsonify({"error": "No answer provided"}), 400
    
    user_answer = data['answer']
    current_question = session.get('current_question')
    
    if not current_question:
        return jsonify({"error": "No questions provided for now,,,"}), 400
    
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
            "validation_result": result,
            "next_question": next_question,
            "score": session.get('game_score', 0),
            "questions_asked": session.get('game_questions_asked', 0),
            "accuracy": round((session.get('game_score', 0) / session.get('game_questions_asked', 1)) * 100, 2) }), 200
        
    except :
        return jsonify({"error": "can;t get the correct answer"}), 500

@cognitive_game_routes.route("/api/game/stop", methods=["POST"])
@token_required
def stop_game(user_id):
    if not session.get('game_active') or session.get('game_user_id') != user_id:
        return jsonify({"error": "There is no game to stop"}), 400
    
    try:
        final_score = session.get('game_score', 0)
        total_questions = session.get('game_questions_asked', 0)
        accuracy = round((final_score / total_questions) * 100, 2) if total_questions > 0 else 0
        
        session.pop('game_active', None)
        session.pop('game_user_id', None)
        session.pop('game_score', None)
        session.pop('game_questions_asked', None)
        session.pop('current_question', None)
        
        return jsonify({
            "message": "Game session ended. Great job exercising your memory!",
            "final_stats": {
                "score": final_score,
                "total_questions": total_questions,
                "accuracy": accuracy
            } }), 200
        
    except:
        return jsonify({"error": "error and game cannot be stopped!!" }), 500

@cognitive_game_routes.route("/api/game/status", methods=["GET"])
@token_required
def get_game_status(user_id):
    if not session.get('game_active') or session.get('game_user_id') != user_id:
        return jsonify({
            "game_active": False,
            "message": "game session is not active right now"}), 200
    
    return jsonify({
        "game_active": True,
        "score": session.get('game_score', 0),
        "questions_asked": session.get('game_questions_asked', 0),
        "current_question": session.get('current_question'),
        "accuracy": round((session.get('game_score', 0) / max(session.get('game_questions_asked', 1), 1)) * 100, 2)}), 200 