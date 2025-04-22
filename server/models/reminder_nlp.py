import torch
import re
import pickle
from datetime import datetime, timedelta
import pytz
import os
import sys
from pathlib import Path
import numpy as np

#####################################################
import torch.nn as nn

class NERIntentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_intents, num_slots):
        super(NERIntentModel, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # BiLSTM layer for sequence labeling (NER)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # Linear layer for intent classification
        self.intent_fc = nn.Linear(hidden_dim * 2, num_intents)
        
        # Linear layer for slot tagging (NER)
        self.slot_fc = nn.Linear(hidden_dim * 2, num_slots)

    def forward(self, x):
        # Get token embeddings
        embedded = self.embedding(x)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(embedded)
        
        # Predict intent (classification for entire sentence)
        intent_pred = self.intent_fc(lstm_out[:, 0, :])  # Use the first token (CLS) for intent prediction
        
        # Predict slots (token-level classification)
        slot_pred = self.slot_fc(lstm_out)
        
        return intent_pred, slot_pred 
######################################################

class ReminderNLP:
    """Class to handle NLP models for reminder processing."""

    def __init__(self):
        """Initialize with loaded models and encoders."""
        # Paths to model files
        model_dir = os.path.join(os.path.dirname(__file__), 'reminder_ml')
        model_path = os.path.join(model_dir, 'reminder_model.pth')
        tokenizer_path = os.path.join(model_dir, 'reminder_tokenizer.pkl')
        intent_encoder_path = os.path.join(model_dir, 'reminder_intent_encoder.pkl')
        slot_encoder_path = os.path.join(model_dir, 'reminder_slot_encoder.pkl')
        
        # Ensure model directory exists
        self._ensure_model_directory(model_dir)
        
        # Load model and initialize encoders
        try:
            print("Loading reminder NLP model...")
            # First load the encoders to determine model parameters
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            with open(intent_encoder_path, 'rb') as f:
                self.intent_encoder = pickle.load(f)
            with open(slot_encoder_path, 'rb') as f:
                self.slot_encoder = pickle.load(f)
            
            # Get necessary dimensions for the model
            vocab_size = len(self.tokenizer)
            num_intents = len(self.intent_encoder.classes_)
            num_slots = len(self.slot_encoder.classes_)
            
            # Create model architecture
            print(f"Initializing model with vocabulary size {vocab_size}, {num_intents} intents, {num_slots} slots")
            
            # Load the model state dictionary
            if os.path.exists(model_path):
                # Either load the full model if it was saved as a model
                try:
                    # First try loading the full saved model
                    self.model = torch.load(model_path, map_location=torch.device("cpu"))
                    print("Loaded full model object")
                except Exception as e:
                    print(f"Could not load as full model: {e}")
                    print("Trying to load as state dict...")
                    
                    # Or create a new model and load the state dict
                    embedding_dim = 64  # Default embedding dimension
                    hidden_dim = 128     # Default hidden dimension
                    self.model = NERIntentModel(vocab_size, embedding_dim, hidden_dim, num_intents, num_slots)
                    
                    try:
                        self.model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
                        print("Loaded model from state dict")
                    except Exception as e:
                        print(f"Error loading state dict: {e}")
                        print("Using a new model instance with default weights")
                        # Continue with the newly created model
                
                self.model.eval()
                print("Reminder NLP model loaded successfully")
                self.is_loaded = True
            else:
                print(f"Model file not found at {model_path}")
                self.is_loaded = False
                
        except Exception as e:
            print(f"Error loading reminder NLP model: {str(e)}")
            self.is_loaded = False
    
    def predict(self, user_input):
        """Process user input and return structured prediction."""
        if not self.is_loaded:
            print(f"[ERROR] NLP model not loaded. Cannot process input: '{user_input}'")
            return {
                "success": False,
                "error": "NLP model not loaded. Please ensure the model files exist and are valid."
            }
        
        try:
            print(f"[DEBUG] Using NLP model to process: '{user_input}'")
            # Preprocess the input
            user_input = user_input.lower()
            
            # Force certain patterns to be recognized as create_event
            force_create_event = False
            if user_input.startswith('remind me') or user_input.startswith('add reminder') or 'reminder for' in user_input:
                print(f"[DEBUG] Detected reminder creation pattern in: '{user_input}'")
                force_create_event = True
                
            tokenized_text = user_input.split()
            
            # Get predictions
            predicted_intent, predicted_slots = self._predict_intent_slots(tokenized_text)
            
            # Override intent if necessary
            if force_create_event:
                print(f"[DEBUG] Overriding predicted intent: {predicted_intent} -> create_event")
                predicted_intent = "create_event"
            
            # Postprocess predictions
            result = self._postprocess_predictions(predicted_intent, tokenized_text, predicted_slots)
            
            # Generate response based on intent
            response = self._generate_response(result)
            
            return {
                "success": True,
                "intent": predicted_intent,
                "parsed_data": result,
                "response": response
            }
        
        except Exception as e:
            print(f"[ERROR] Error in NLP prediction: {str(e)}")
            return {
                "success": False,
                "error": f"Error processing input: {str(e)}"
            }
    
    def _predict_intent_slots(self, tokenized_text, max_seq_length=128):
        """Make prediction using the model."""
        # Convert tokens to ids
        token_ids = [self.tokenizer.get(token, self.tokenizer["UNK"]) for token in tokenized_text]
        
        # Pad sequence
        token_ids = token_ids[:max_seq_length] + [self.tokenizer["PAD"]] * (max_seq_length - len(token_ids))
        
        # Convert to tensor
        input_tensor = torch.tensor(token_ids).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            intent_pred, slot_pred = self.model(input_tensor)
        
        # Get the predicted intent
        intent_label = intent_pred.argmax(dim=1).item()
        predicted_intent = self.intent_encoder.inverse_transform([intent_label])[0]
        
        # Get the predicted slots
        slot_preds = slot_pred.argmax(dim=2).squeeze().cpu().numpy()
        predicted_slots = self.slot_encoder.inverse_transform(slot_preds[:len(tokenized_text)])
        
        return predicted_intent, predicted_slots
    
    def _postprocess_predictions(self, predicted_intent, predicted_tokens, predicted_slots):
        """Process NER predictions to extract structured data."""
        predicted_time, predicted_action = [], []
        current_action = []
        current_time = []
        
        for token, slot in zip(predicted_tokens, predicted_slots):
            # Handle time slots
            if slot == 'B-TIME':
                if current_time:  # If we have a previous time phrase, save it
                    predicted_time.append(' '.join(current_time))
                    current_time = []
                current_time.append(token)
            elif slot == 'I-TIME' and current_time:  # Continue current time phrase
                current_time.append(token)
                
            # Handle action slots
            if slot == 'B-ACTION':
                if current_action:  # If we have a previous action phrase, save it
                    predicted_action.append(' '.join(current_action))
                    current_action = []
                current_action.append(token)
            elif slot == 'I-ACTION' and current_action:  # Continue current action phrase
                current_action.append(token)
        
        # Add any remaining phrases
        if current_time:
            predicted_time.append(' '.join(current_time))
        if current_action:
            predicted_action.append(' '.join(current_action))
        
        result = {
            "text": " ".join(predicted_tokens),
            "predicted_intent": predicted_intent,
            "predicted_time": " ".join(predicted_time) if predicted_time else None,
            "predicted_action": " ".join(predicted_action) if predicted_action else None
        }
        
        return result
    
    def _generate_response(self, result):
        """Generate a user-friendly response based on the parsed intent."""
        intent = result.get("predicted_intent")
        action = result.get("predicted_action")
        time_str = result.get("predicted_time")
        
        # Use Egypt timezone
        egypt_tz = pytz.timezone('Africa/Cairo')
        now = datetime.now(egypt_tz)
        
        # Format time for response if provided
        formatted_time = "right now"
        if time_str:
            try:
                # Parse time from string
                time_parts = time_str.split()
                
                if len(time_parts) == 1:  # Only number provided
                    hour = int(time_parts[0])
                    meridian = 'am'  # Default to AM
                elif len(time_parts) == 2:  # Format: "3 pm" or "11 am"
                    hour = int(time_parts[0])
                    meridian = time_parts[1].lower()
                else:
                    # Use current time if can't parse
                    return f"I'll set that reminder for you, but I'm not sure about the time you mentioned."
                
                # Convert to 24-hour format
                if meridian == 'pm' and hour != 12:
                    hour += 12
                elif meridian == 'am' and hour == 12:
                    hour = 0
                    
                # Format time for display
                formatted_time = f"{hour:02d}:00" if hour < 12 else f"{hour-12 if hour > 12 else hour}:00 PM"
                
            except ValueError:
                # Use current time if can't parse
                formatted_time = "right now"
        
        # Generate appropriate response based on intent
        if intent == "create_event":
            if not action:
                return "I'd be happy to create a reminder for you, but what would you like me to remind you about?"
            
            if time_str:
                return f"I've created a reminder for '{action}' at {formatted_time}."
            else:
                return f"I've created a reminder for '{action}'."
                
        elif intent == "get_timetable":
            # This would be handled by the calendar service
            return "Let me check your schedule for today. I'll show you all your reminders."
            
        else:
            return "I'm not sure what you'd like me to do with your reminders. Could you please rephrase that?"
            
    def parse_time_to_datetime(self, time_str):
        """Convert a time string to a datetime object."""
        print(f"\n===== PARSING TIME STRING: '{time_str}' =====")
        # Get current time in Egypt timezone
        egypt_tz = pytz.timezone('Africa/Cairo')
        now = datetime.now(egypt_tz)
        print(f"[TIME_PARSE] Current time in Egypt: {now}")
        
        if not time_str:
            print("[TIME_PARSE] No time string provided, returning current time")
            return now
            
        try:
            # Parse time parts
            print(f"[TIME_PARSE] Splitting time string: '{time_str}'")
            time_parts = time_str.split()
            print(f"[TIME_PARSE] Time parts after split: {time_parts}")
            
            if len(time_parts) == 1:  # Only number provided
                try:
                    print(f"[TIME_PARSE] Parsing single number: {time_parts[0]}")
                    hour = int(time_parts[0])
                    meridian = 'am'  # Default to AM
                    print(f"[TIME_PARSE] Parsed hour: {hour}, default meridian: {meridian}")
                except ValueError as e:
                    print(f"[TIME_PARSE] Error parsing as number: {str(e)}")
                    # If can't parse as a number, return current time
                    return now
            elif len(time_parts) == 2:  # Format: "3 pm" or "11 am"
                try:
                    print(f"[TIME_PARSE] Parsing time with meridian: {time_parts}")
                    hour = int(time_parts[0])
                    meridian = time_parts[1].lower()
                    print(f"[TIME_PARSE] Parsed hour: {hour}, meridian: {meridian}")
                except ValueError as e:
                    print(f"[TIME_PARSE] Error parsing time with meridian: {str(e)}")
                    # If can't parse as a number, return current time
                    return now
            else:
                # Can't parse, return current time
                print(f"[TIME_PARSE] Too many parts ({len(time_parts)}), can't parse")
                return now
            
            # Convert to 24-hour format
            print(f"[TIME_PARSE] Converting to 24-hour format. Hour: {hour}, Meridian: {meridian}")
            if meridian in ['pm', 'p.m.', 'p.m'] and hour != 12:
                hour += 12
                print(f"[TIME_PARSE] PM time, adjusted hour: {hour}")
            elif meridian in ['am', 'a.m.', 'a.m'] and hour == 12:
                hour = 0
                print(f"[TIME_PARSE] 12 AM, adjusted hour to 0")
                
            # Create datetime in Egypt timezone
            result_time = now.replace(
                hour=hour,
                minute=0,
                second=0,
                microsecond=0
            )
            print(f"[TIME_PARSE] Initial result time: {result_time}")
            
            # If the time is in the past, assume it's for tomorrow
            if result_time < now:
                print("[TIME_PARSE] Time is in the past, adding one day")
                result_time = result_time + timedelta(days=1)
            
            # Make sure the timezone is correctly set
            if result_time.tzinfo is None:
                print("[TIME_PARSE] Setting timezone to Egypt")
                result_time = egypt_tz.localize(result_time)
                
            print(f"[TIME_PARSE] Final result time: {result_time.isoformat()}")
            return result_time
            
        except Exception as e:
            print(f"[TIME_PARSE] Unhandled error parsing time: {str(e)}")
            import traceback
            print(f"[TIME_PARSE] Traceback: {traceback.format_exc()}")
            # Return current time if parsing fails
            return now
    
    def _ensure_model_directory(self, model_dir):
        """Ensure the model directory exists."""
        # Create directory if it doesn't exist
        if not os.path.exists(model_dir):
            print(f"Creating model directory: {model_dir}")
            os.makedirs(model_dir, exist_ok=True)
            
            # Check if we need to create stub files
            required_files = ['reminder_model.pth', 'reminder_tokenizer.pkl', 
                           'reminder_intent_encoder.pkl', 'reminder_slot_encoder.pkl']
            
            missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
            
            if missing_files:
                print("\n" + "="*80)
                print("MISSING MODEL FILES DETECTED")
                print("="*80)
                print(f"The following required model files are missing from {model_dir}:")
                for f in missing_files:
                    print(f"  - {f}")
                print("\nYou must provide trained model files to use the reminder NLP functionality.")
                print("="*80 + "\n") 