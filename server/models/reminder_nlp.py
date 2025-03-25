import torch
import re
import pickle
from datetime import datetime, timedelta
import pytz
import os

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
        
        # Load model and initialize encoders
        try:
            print("Loading reminder NLP model...")
            self.model = torch.load(model_path, map_location=torch.device("cpu"))
            self.model.eval()
            
            # Load tokenizer and encoders
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            with open(intent_encoder_path, 'rb') as f:
                self.intent_encoder = pickle.load(f)
            with open(slot_encoder_path, 'rb') as f:
                self.slot_encoder = pickle.load(f)
                
            print("Reminder NLP model loaded successfully")
            self.is_loaded = True
        except Exception as e:
            print(f"Error loading reminder NLP model: {str(e)}")
            self.is_loaded = False
    
    def predict(self, user_input):
        """Process user input and return structured prediction."""
        if not self.is_loaded:
            return {"error": "NLP model not loaded"}
        
        try:
            # Preprocess the input
            user_input = user_input.lower()
            tokenized_text = user_input.split()
            
            # Get predictions
            predicted_intent, predicted_slots = self._predict_intent_slots(tokenized_text)
            
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
            print(f"Error in reminder NLP prediction: {str(e)}")
            return {
                "success": False,
                "error": str(e)
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
        if not time_str:
            return datetime.now(pytz.timezone('Africa/Cairo'))
            
        try:
            # Parse time parts
            time_parts = time_str.split()
            
            if len(time_parts) == 1:  # Only number provided
                hour = int(time_parts[0])
                meridian = 'am'  # Default to AM
            elif len(time_parts) == 2:  # Format: "3 pm" or "11 am"
                hour = int(time_parts[0])
                meridian = time_parts[1].lower()
            else:
                # Can't parse, return current time
                return datetime.now(pytz.timezone('Africa/Cairo'))
            
            # Convert to 24-hour format
            if meridian == 'pm' and hour != 12:
                hour += 12
            elif meridian == 'am' and hour == 12:
                hour = 0
                
            # Create datetime in Egypt timezone
            egypt_tz = pytz.timezone('Africa/Cairo')
            now = datetime.now(egypt_tz)
            
            # Set the parsed time
            result_time = now.replace(
                hour=hour,
                minute=0,
                second=0,
                microsecond=0
            )
            
            return result_time
            
        except ValueError:
            # Return current time if parsing fails
            return datetime.now(pytz.timezone('Africa/Cairo')) 