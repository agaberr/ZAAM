from flask import Blueprint, request, jsonify, current_app
import openai
import base64
import io
import tempfile
import os
import requests
from datetime import datetime
import logging

# Create blueprint
speech_bp = Blueprint('speech', __name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@speech_bp.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """
    Transcribe audio using AssemblyAI API (recommended)
    Expects: { "audio": "base64_audio_data", "format": "webm" }
    """
    try:
        data = request.get_json()
        
        if not data or 'audio' not in data:
            return jsonify({
                'success': False,
                'error': 'No audio data provided'
            }), 400
        
        audio_data = data['audio']
        audio_format = data.get('format', 'webm')
        
        # Decode base64 audio
        try:
            audio_bytes = base64.b64decode(audio_data)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Invalid base64 audio data: {str(e)}'
            }), 400
        
        # Use AssemblyAI for transcription
        transcript = transcribe_with_assemblyai(audio_bytes, audio_format)
        
        if transcript:
            logger.info(f"AssemblyAI transcription successful: {len(transcript)} characters")
            return jsonify({
                'success': True,
                'transcript': transcript,
                'provider': 'AssemblyAI'
            })
        else:
            # Fallback to OpenAI Whisper
            transcript = transcribe_with_whisper(audio_bytes, audio_format)
            
            if transcript:
                logger.info(f"Whisper transcription successful: {len(transcript)} characters")
                return jsonify({
                    'success': True,
                    'transcript': transcript,
                    'provider': 'OpenAI Whisper'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'All transcription services failed'
                }), 500
    
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Transcription failed: {str(e)}'
        }), 500

@speech_bp.route('/whisper', methods=['POST'])
def whisper_transcribe():
    """
    Transcribe audio using OpenAI Whisper API
    Expects: multipart/form-data with 'file' and 'model'
    """
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No audio file provided'
            }), 400
        
        audio_file = request.files['file']
        model = request.form.get('model', 'whisper-1')
        
        if audio_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Save file temporarily
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            audio_file.save(temp_file.name)
            
            try:
                # Transcribe with OpenAI Whisper
                transcript = transcribe_file_with_whisper(temp_file.name, model)
                
                if transcript:
                    logger.info(f"Whisper file transcription successful: {len(transcript)} characters")
                    return jsonify({
                        'success': True,
                        'text': transcript,
                        'provider': 'OpenAI Whisper'
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': 'Whisper transcription failed'
                    }), 500
            
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
    
    except Exception as e:
        logger.error(f"Whisper transcription error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Whisper transcription failed: {str(e)}'
        }), 500

def transcribe_with_assemblyai(audio_bytes, audio_format='webm'):
    """
    Transcribe audio using AssemblyAI API
    """
    try:
        # AssemblyAI API configuration
        ASSEMBLYAI_API_KEY = os.getenv('ASSEMBLYAI_API_KEY')
        
        if not ASSEMBLYAI_API_KEY:
            logger.warning("AssemblyAI API key not found")
            return None
        
        # First, upload the audio file
        headers = {
            'authorization': ASSEMBLYAI_API_KEY,
            'content-type': 'application/octet-stream'
        }
        
        upload_response = requests.post(
            'https://api.assemblyai.com/v2/upload',
            headers=headers,
            data=audio_bytes
        )
        
        if upload_response.status_code != 200:
            logger.error(f"AssemblyAI upload failed: {upload_response.text}")
            return None
        
        audio_url = upload_response.json()['upload_url']
        
        # Create transcription job
        transcript_data = {
            'audio_url': audio_url,
            'language_code': 'en_us',
            'punctuate': True,
            'format_text': True,
        }
        
        headers = {
            'authorization': ASSEMBLYAI_API_KEY,
            'content-type': 'application/json'
        }
        
        transcript_response = requests.post(
            'https://api.assemblyai.com/v2/transcript',
            headers=headers,
            json=transcript_data
        )
        
        if transcript_response.status_code != 200:
            logger.error(f"AssemblyAI transcription failed: {transcript_response.text}")
            return None
        
        transcript_id = transcript_response.json()['id']
        
        # Poll for completion
        max_attempts = 60  # Wait up to 60 seconds
        attempt = 0
        
        while attempt < max_attempts:
            status_response = requests.get(
                f'https://api.assemblyai.com/v2/transcript/{transcript_id}',
                headers=headers
            )
            
            if status_response.status_code == 200:
                result = status_response.json()
                status = result['status']
                
                if status == 'completed':
                    return result.get('text', '')
                elif status == 'error':
                    logger.error(f"AssemblyAI transcription error: {result.get('error')}")
                    return None
            
            attempt += 1
            import time
            time.sleep(1)
        
        logger.error("AssemblyAI transcription timeout")
        return None
    
    except Exception as e:
        logger.error(f"AssemblyAI error: {str(e)}")
        return None

def transcribe_with_whisper(audio_bytes, audio_format='webm'):
    """
    Transcribe audio using OpenAI Whisper API
    """
    try:
        # Get OpenAI API key
        openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not openai_api_key:
            logger.warning("OpenAI API key not found")
            return None
        
        # Set up OpenAI client
        client = openai.OpenAI(api_key=openai_api_key)
        
        # Create a temporary file for the audio
        with tempfile.NamedTemporaryFile(suffix=f'.{audio_format}', delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_file.flush()
            
            try:
                # Transcribe with Whisper
                with open(temp_file.name, 'rb') as audio_file:
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language="en"
                    )
                
                return transcript.text
            
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
    
    except Exception as e:
        logger.error(f"Whisper error: {str(e)}")
        return None

def transcribe_file_with_whisper(file_path, model='whisper-1'):
    """
    Transcribe an audio file using OpenAI Whisper API
    """
    try:
        # Get OpenAI API key
        openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not openai_api_key:
            logger.warning("OpenAI API key not found")
            return None
        
        # Set up OpenAI client
        client = openai.OpenAI(api_key=openai_api_key)
        
        # Transcribe with Whisper
        with open(file_path, 'rb') as audio_file:
            transcript = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                language="en"
            )
        
        return transcript.text
    
    except Exception as e:
        logger.error(f"Whisper file transcription error: {str(e)}")
        return None

@speech_bp.route('/providers', methods=['GET'])
def get_available_providers():
    """
    Get list of available speech-to-text providers
    """
    try:
        providers = []
        
        # Check AssemblyAI
        if os.getenv('ASSEMBLYAI_API_KEY'):
            providers.append({
                'name': 'AssemblyAI Real-time',
                'available': True,
                'recommended': True
            })
        
        # Check OpenAI Whisper
        if os.getenv('OPENAI_API_KEY'):
            providers.append({
                'name': 'OpenAI Whisper',
                'available': True,
                'recommended': False
            })
        
        # Browser API is always available (fallback)
        providers.append({
            'name': 'Browser Web Speech API',
            'available': True,
            'recommended': False,
            'note': 'Fallback option, may be unreliable'
        })
        
        return jsonify({
            'success': True,
            'providers': providers
        })
    
    except Exception as e:
        logger.error(f"Error getting providers: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500 