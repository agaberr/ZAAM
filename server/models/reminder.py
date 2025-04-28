import torch
import torch.nn as nn
import pickle
import os
import re
from datetime import datetime, timedelta
import pytz
# from reminders.model import NERIntentModel
# from reminders.model import NERIntentModel

################################################################

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
################################################################

# def load_model_safe(path, device='cpu'):
#     """Custom function to load a model with class fixing"""
#     model_data = torch.load(path, map_location=device)
#     print("[DEBUG] ###############IN LOAD_MODEL_SAFE")
#     # Check if we got a full state dict or just the model
#     if isinstance(model_data, dict) and 'state_dict' in model_data:
#         # Create a new model instance
#         vocab_size = model_data.get('vocab_size', 10000)  # Default if not found
#         embedding_dim = model_data.get('embedding_dim', 128)
#         hidden_dim = model_data.get('hidden_dim', 64)
#         num_intents = model_data.get('num_intents', 2)
#         num_slots = model_data.get('num_slots', 10)
#         print("[DEBUG] ###############IN LOAD_MODEL_SAFE GOA AWY")
        
#         model = NERIntentModel(
#             vocab_size=vocab_size,
#             embedding_dim=embedding_dim,
#             hidden_dim=hidden_dim,
#             num_intents=num_intents,
#             num_slots=num_slots
#         )
        
#         # Load the state dict
#         model.load_state_dict(model_data['state_dict'])
#     else:
#         # If it's already the model object, ensure it's the right type
#         model = model_data
        
#     return model

class ReminderNLP:
    """NLP model for processing reminder text and extracting intent and entities"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.intent_encoder = None
        self.slot_encoder = None
        self.initialize_model()
        
        # Time expressions mapping
        self.TIME_EXPRESSIONS = {
            "today": 0,
            "tomorrow": 1,
            "next week": 7,
            "next month": 30,
            "next day": 1,
            "next": 1,
            "day after tomorrow": 2,
            "day after": 2,
            "in two days": 2,
            "in a week": 7,
            "in 2 days": 2,
            "in 3 days": 3,
            "in 4 days": 4,
            "in 5 days": 5,
            "in a month": 30,
        }
    
    def initialize_model(self):
        """Load the model and tokenizers from files"""
        try:
            reminders_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reminders')
            print(f"Reminders directory: {reminders_dir}")
            
            # Load tokenizer and encoders first
            tokenizer_path = os.path.join(reminders_dir, 'reminder_tokenizer.pkl')
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
                print("Tokenizer loaded")
                
            intent_encoder_path = os.path.join(reminders_dir, 'reminder_intent_encoder.pkl')
            with open(intent_encoder_path, 'rb') as f:
                self.intent_encoder = pickle.load(f)
                print("Intent encoder loaded")
                
            slot_encoder_path = os.path.join(reminders_dir, 'reminder_slot_encoder.pkl')
            with open(slot_encoder_path, 'rb') as f:
                self.slot_encoder = pickle.load(f)
                print("Slot encoder loaded")
            
            # Create a new model instance with default parameters
            # These should match the parameters of the saved model
            vocab_size = len(self.tokenizer)
            embedding_dim = 100  # standard dimension
            hidden_dim = 128      # standard dimension
            num_intents = len(self.intent_encoder.classes_)
            num_slots = len(self.slot_encoder.classes_)
            
            print(f"Creating model with parameters:")
            print(f"  vocab_size: {vocab_size}")
            print(f"  embedding_dim: {embedding_dim}")
            print(f"  hidden_dim: {hidden_dim}")
            print(f"  num_intents: {num_intents}")
            print(f"  num_slots: {num_slots}")
            
            # Create new model with these parameters
            self.model = NERIntentModel(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_intents=num_intents,
                num_slots=num_slots
            )
            
            # Load the weights from the saved model
            print("Loading model weights...")
            model_path = os.path.join(reminders_dir, 'reminder_model_checkpoint.pth')
            saved_model = torch.load(model_path, map_location=torch.device("cpu"))
            print("Model loaded successfully")
            print("Model structure:", saved_model)
            
            # If the saved model is a state dict, load it directly
            if isinstance(saved_model, dict) and 'state_dict' in saved_model:
                self.model.load_state_dict(saved_model['state_dict'])
            # If it's the full model, extract the state dict
            elif hasattr(saved_model, 'state_dict'):
                self.model.load_state_dict(saved_model.state_dict())
            # If it's already a state dict
            elif isinstance(saved_model, dict):
                self.model.load_state_dict(saved_model)
            else:
                raise ValueError(f"Unexpected model format: {type(saved_model)}")
                
            print("Model loaded successfully")
            self.model.eval()
                
        except Exception as e:
            print(f"Error initializing reminder model: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e
    
    def extract_time_expressions(self, text):
        """Extract time expressions from text and return modified text and days offset"""
        text_lower = text.lower()
        days_offset = 0
        found_expression = None
        
        # Check for time expressions
        for expression, offset in self.TIME_EXPRESSIONS.items():
            if expression in text_lower:
                days_offset = offset
                found_expression = expression
                break
        
        # Remove the expression from text if found
        if found_expression:
            print(f"Found time expression: '{found_expression}', days offset: {days_offset}")
            # Use regex to remove the expression while preserving word boundaries
            text = re.sub(r'\b' + re.escape(found_expression) + r'\b', '', text_lower, flags=re.IGNORECASE)
            # Clean up any double spaces created
            text = re.sub(r'\s+', ' ', text).strip()
        
        return text, days_offset
    
    def predict(self, tokenized_text, max_seq_length=128):
        """Run prediction on tokenized text"""
        try:
            # Convert tokens to ids
            token_ids = [self.tokenizer.get(token, self.tokenizer["UNK"]) for token in tokenized_text]
            
            # Pad sequence
            token_ids = token_ids[:max_seq_length] + [self.tokenizer["PAD"]] * (max_seq_length - len(token_ids))
            
            # Convert to tensor
            input_tensor = torch.tensor(token_ids).unsqueeze(0)
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                intent_pred, slot_pred = self.model(input_tensor)
            
            # Get the predicted intent
            intent_label = intent_pred.argmax(dim=1).item()
            predicted_intent = self.intent_encoder.inverse_transform([intent_label])[0]
            
            # Get the predicted slots
            slot_preds = slot_pred.argmax(dim=2).squeeze().cpu().numpy()
            predicted_slots = self.slot_encoder.inverse_transform(slot_preds[:len(tokenized_text)])
            
            return predicted_intent, predicted_slots
            
        except Exception as e:
            print(f"Error in predict function: {str(e)}")
            raise e
    
    def postprocess_ner_predictions(self, predicted_intent, predicted_tokens, predicted_slots):
        """Process the model predictions and extract structured data"""
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
    
    def process_text(self, text):
        """Process user input text and extract reminder information"""
        # Extract time expressions
        user_input, days_offset = self.extract_time_expressions(text)
        
        # Tokenize the text
        tokenized_text = user_input.lower().split()
        
        # Get predictions from the model
        predicted_intent, predicted_slots = self.predict(tokenized_text)
        
        # Process the predictions
        result = self.postprocess_ner_predictions(predicted_intent, tokenized_text, predicted_slots)
        
        # Add days offset to the result
        result['days_offset'] = days_offset
        
        return result

class Reminder:
    """Reminder utility for creating and managing calendar events"""
    
    @staticmethod
    def create_event(title, start_time):
        """Create a calendar event object"""
        event = {
            "summary": title,
            "start": {"dateTime": start_time.isoformat(), "timeZone": "Africa/Cairo"},
            "end": {"dateTime": (start_time + timedelta(hours=1)).isoformat(), "timeZone": "Africa/Cairo"}
        }
        return event
    
    @staticmethod
    def get_timetable(days_offset=0):
        """Get the time range for fetching calendar events"""
        egypt_tz = pytz.timezone('Africa/Cairo')
        target_date = datetime.now(egypt_tz) + timedelta(days=days_offset)
        
        return {
            "calendarId": "primary",
            "timeMin": target_date.replace(hour=0, minute=0, second=0, microsecond=0).isoformat(),
            "timeMax": target_date.replace(hour=23, minute=59, second=59, microsecond=999999).isoformat(),
            "singleEvents": True,
            "orderBy": "startTime",
            "timeZone": "Africa/Cairo"
        }
    
    @staticmethod
    def parse_time(time_str, target_date):
        """Parse time string like '3 pm' into datetime object"""
        if not time_str:
            return target_date
            
        try:
            # Handle different time formats
            time_parts = time_str.split()
            
            if len(time_parts) == 1:  # Only number provided, default to AM
                hour = int(time_parts[0])
                meridian = 'am'  # Default to AM
            elif len(time_parts) == 2:  # Format: "3 pm" or "11 am"
                hour = int(time_parts[0])
                meridian = time_parts[1].lower()
            else:
                raise ValueError(f"Could not parse time format: {time_str}")
            
            # Convert to 24-hour format
            if meridian == 'pm' and hour != 12:
                hour += 12
            elif meridian == 'am' and hour == 12:
                hour = 0
            
            # Create new time
            return target_date.replace(
                hour=hour,
                minute=0,
                second=0,
                microsecond=0
            )
        except ValueError as e:
            raise ValueError(f"Invalid time format. Please use format like '3 pm' or '11 am'. Error: {str(e)}") 