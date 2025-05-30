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


@cognitive_game_routes.route("/api/game/start", methods=["POST"])
@token_required
def startgame(user_id):

    db = current_app.config["DATABASE"]
    if db is None:
        return jsonify({"error": "Database connection failed"}), 500
    
    try:
        game = CognitiveGame(db, user_id)
        
        people = game.get_all_people()
        events = game.get_all_events()
        
        if not people and not events:
            return jsonify({"message": "Add some memory aids with people or events to play this game"}), 400
        
        session['game_active'] = True
        session['game_user_id'] = user_id
        session['game_score'] = 0
        session['game_questions_asked'] = 0
        
        question = game.generate_random_question()
        session['current_question'] = question
        
        return jsonify({"message": "Cognitive game started! Let's test your memory!"}), 200
        
    except:
        return jsonify({"error": "error can't start this game.."}), 500


@cognitive_game_routes.route("/api/game/question", methods=["POST"])
@token_required
def getQ(user_id):
    db = current_app.config["DATABASE"]
    if db is None:
        return jsonify({"error": "Can't connect to dbb"}), 500
    
    if not session.get('game_active') or session.get('game_user_id') != user_id:
        return jsonify({"error": "there is no game, start game to play"}), 400
    
    try:
        game = CognitiveGame(db, user_id)
        
        q = game.generate_random_question()
        session['current_question'] = q
        
        return jsonify({"question": q}), 200
        
    except:
        return jsonify({"error": "can't create a question"}), 500

@cognitive_game_routes.route("/api/game/answer", methods=["POST"])
@token_required
def answerQ(user_id):
    db = current_app.config["DATABASE"]
    if db is None:
        return jsonify({"error": "failed to conn to database"}), 500
    
    # lw ana msh f session f there is no game aslun
    if not session.get('game_active') or session.get('game_user_id') != user_id:
        return jsonify({"error": "No active game session. Please start a new game first."}), 400
    
    data = request.json
    if not data or 'answer' not in data:
        return jsonify({"error": "you didn't put an answer in the code.."}), 400
    
    user_answer = data['answer']
    current_question = session.get('current_question')
    
    if not current_question:
        return jsonify({"error": "there os no questions to answer them!"}), 400
    
    try:
        game = CognitiveGame(db, user_id)
        
        qAnswer = game.check_answer(current_question, user_answer)
        
        session['game_questions_asked'] = session.get('game_questions_asked', 0) + 1
        
        if qAnswer.get('correct', False):
            session['game_score'] = session.get('game_score', 0) + 1
        
        nextQ = game.generate_random_question()
        session['current_question'] = nextQ
        
        return jsonify({"true answer": qAnswer,"next question": nextQ}), 200
        
    except:
        return jsonify({"error": "error in answering the question..."}), 500

@cognitive_game_routes.route("/api/game/stop", methods=["POST"])
@token_required
def stopgame(user_id):

    # lw el user makansh mawgood f session
    if not session.get('game_active') or session.get('game_user_id') != user_id:
        return jsonify({"error": "There is no game to stop.."}), 400
    
    try:
        totalScore = session.get('game_score', 0)
        allQ = session.get('game_questions_asked', 0)

        if allQ == 0:
            acc = 0
        else:
            acc = round((totalScore / allQ) * 100, 2)
        
        # remova all sessions 3shan a2dr a play game another time
        for x in ['game_active', 'game_user_id', 'game_score', 'game_questions_asked', 'current_question']:
            session.pop(x, None)

        
        return jsonify({
            "message": "Game ended. You did great job!",
            "final_stats": {
                "score": totalScore,
                "accuracy": acc
            }
        }), 200
        
    except:
        return jsonify({"error": "error in finishing game"}), 500

@cognitive_game_routes.route("/api/game/status", methods=["GET"])
@token_required
def gamestats(user_id):
# i get all the user stats that occured
    
    if not session.get('game_active') or session.get('game_user_id') != user_id:
        return jsonify({"message": "There is no game in active..."}), 200
    

    score = session.get('game_score', 0)
    qAsked = max(session.get('game_questions_asked', 1), 1)
    acc = round(( score  / qAsked) * 100, 2)



    return jsonify({
        "score": session.get('game_score', 0),
        "accuracy": acc
    }), 200 