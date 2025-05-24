# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS

import os
import requests
import json
import base64
import logging
import subprocess
import time
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Environment variables
ELEVEN_LABS_API_KEY = "sk_86a7241af089833ec81e7ac8eeb01fa8e39a99c2c8e5d5db"
VOICE_ID = "Mu5jxyqZOLIGltFpfalg"


def exec_command(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Command failed: {result.stderr}")
    return result.stdout

def lip_sync_message(message):
    start_time = time.time()
    print(f"Starting conversion for message {message}")

    # Generate timestamp-based unique file names
    timestamp = int(time.time() * 1000)
    mp3_path = f"audios/message_{timestamp}.mp3"
    wav_path = f"audios/message_{timestamp}.wav"
    json_path = f"audios/message_{timestamp}.json"

    # Convert MP3 to WAV
    exec_command(f"ffmpeg -y -i {mp3_path} {wav_path}")
    print(f"Conversion done in {int((time.time() - start_time) * 1000)}ms")

    # Generate lip sync JSON using Rhubarb
    exec_command(f"./bin/rhubarb -f json -o {json_path} {wav_path} -r phonetic")
    print(f"Lip sync done in {int((time.time() - start_time) * 1000)}ms")
    
    return wav_path, json_path

# Helper functions
def audio_file_to_base64(file_path):
    try:
        with open(file_path, 'rb') as audio_file:
            encoded_string = base64.b64encode(audio_file.read()).decode('utf-8')
            return encoded_string
    except Exception as e:
        logger.error(f"Error reading audio file {file_path}: {e}")
        return None

def read_json_transcript(file_path):
    try:
        with open(file_path, 'r') as json_file:
            return json.load(json_file)
    except Exception as e:
        logger.error(f"Error reading JSON file {file_path}: {e}")
        return None

def generate_audio_from_eleven_labs(text, voice_id=VOICE_ID):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVEN_LABS_API_KEY
    }
    
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    
    try:
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            # Return audio data as base64
            return base64.b64encode(response.content).decode('utf-8')
        else:
            logger.error(f"Error with Eleven Labs API: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        logger.error(f"Exception when calling Eleven Labs API: {e}")
        return None

# Default responses for empty messages
DEFAULT_RESPONSES = [
    {
        "text": "Hey dear... How was your day?",
        "facialExpression": "smile",
        "animation": "Talking_1",
    },
    {
        "text": "I missed you so much... Please don't go for so long!",
        "facialExpression": "sad",
        "animation": "Crying",
    }
]

# Root endpoint
@app.route('/')
def home():
    return jsonify({
        "message": "Welcome to the Voice API Server",
        "status": "online"
    })

# Chat endpoint
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    
    # Process user message and generate response
    # This is a simplified example - you would typically connect to an AI service here
    response_text = f"{user_message}"
    
    # Generate audio from Eleven Labs
    audio_base64 = generate_audio_from_eleven_labs(response_text)
    
    if not audio_base64:
        return jsonify({"error": "Failed to generate audio"}), 500
    
    # Return response with audio data directly - no need to save to disk
    response = {
        "messages": [
            {
                "text": response_text,
                "audio": audio_base64,
                "animation": "Talking_1"
            }
        ]
    }
    
    return jsonify(response)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)