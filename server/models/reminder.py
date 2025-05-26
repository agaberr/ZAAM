import torch
import torch.nn as nn
import pickle
import os
import re
from datetime import datetime, timedelta
import pytz
from bson import ObjectId
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
        time_match = None
        
        # Extract time expressions like "3 pm", "5:30 pm", etc.
        time_pattern = r'\b(\d{1,2})(?::(\d{1,2}))?\s*(am|pm|a\.m\.|p\.m\.)\b'
        time_match = re.search(time_pattern, text_lower)
        
        # Check for date expressions (today, tomorrow, etc.)
        for expression, offset in self.TIME_EXPRESSIONS.items():
            if expression in text_lower:
                days_offset = offset
                found_expression = expression
                break
        
        # Process text to keep any extracted time but remove date expressions
        processed_text = text_lower
        if found_expression:
            print(f"Found time expression: '{found_expression}', days offset: {days_offset}")
            # Use regex to remove the expression while preserving word boundaries
            processed_text = re.sub(r'\b' + re.escape(found_expression) + r'\b', '', processed_text, flags=re.IGNORECASE)
        
        # Clean up any double spaces created
        processed_text = re.sub(r'\s+', ' ', processed_text).strip()
        
        return processed_text, days_offset
    
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
        
        # Debug: Print token-slot pairs to understand what the model is predicting
        print(f"[DEBUG] Token-Slot pairs:")
        for token, slot in zip(predicted_tokens, predicted_slots):
            print(f"  '{token}' -> '{slot}'")
        
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
        
        # Post-process action to fix common ML tagging errors
        final_action = None
        if predicted_action:
            raw_action = " ".join(predicted_action)
            print(f"[DEBUG] Raw ML action: '{raw_action}'")
            
            # Fix common patterns where the model incorrectly tags action boundaries
            # Pattern 1: "me to [actual_action]" -> extract just the actual action
            if raw_action.startswith('me to '):
                # Look for the real action after "me to"
                remaining_tokens = predicted_tokens[:]
                start_idx = -1
                
                # Find where "me to" sequence ends in the original tokens
                for i in range(len(predicted_tokens) - 1):
                    if predicted_tokens[i] == 'me' and predicted_tokens[i + 1] == 'to':
                        start_idx = i + 2  # Start after "me to"
                        break
                
                if start_idx >= 0 and start_idx < len(predicted_tokens):
                    # Extract remaining meaningful words (skip pronouns and articles at the start)
                    meaningful_tokens = []
                    for token in predicted_tokens[start_idx:]:
                        # Skip common stopwords that shouldn't be part of action
                        if token not in ['at', 'on', 'for', 'by', 'in', 'the', 'a', 'an']:
                            meaningful_tokens.append(token)
                        else:
                            break  # Stop at time/date prepositions
                    
                    if meaningful_tokens:
                        final_action = ' '.join(meaningful_tokens)
                        print(f"[DEBUG] Fixed 'me to' pattern: '{raw_action}' -> '{final_action}'")
            
            # Pattern 2: Action ends with "me" or other incorrect endings
            elif raw_action.endswith(' me') or raw_action.endswith(' to'):
                # Remove the incorrect ending
                final_action = raw_action.rsplit(' ', 1)[0]
                print(f"[DEBUG] Fixed incorrect ending: '{raw_action}' -> '{final_action}'")
            
            # If no fixes applied, use the raw action
            if not final_action:
                final_action = raw_action
        else:
            final_action = None
        
        result = {
            "text": " ".join(predicted_tokens),
            "predicted_intent": predicted_intent,
            "predicted_time": " ".join(predicted_time) if predicted_time else None,
            "predicted_action": final_action
        }
        
        print(f"[DEBUG] Final processed result: {result}")
        return result
    
    def process_text(self, text):
        """Process user input text and extract reminder information"""
        # Extract time expressions
        user_input, days_offset = self.extract_time_expressions(text)
        
        # Check for time expressions directly in the text
        time_str = None
        time_pattern = r'\b(\d{1,2})(?::(\d{1,2}))?\s*(am|pm|a\.m\.|p\.m\.)\b'
        time_match = re.search(time_pattern, text.lower())
        
        if time_match:
            time_str = time_match.group(0)
            print(f"Found time expression: '{time_str}'")
        
        # Tokenize the text
        tokenized_text = user_input.lower().split()
        
        # Get predictions from the model
        predicted_intent, predicted_slots = self.predict(tokenized_text)
        
        # Process the predictions
        result = self.postprocess_ner_predictions(predicted_intent, tokenized_text, predicted_slots)
        
        # Add days offset to the result
        result['days_offset'] = days_offset
        
        # Add time string if found directly in the text but not by the model
        if time_str and not result.get('predicted_time'):
            result['predicted_time'] = time_str
            
        return result

class ReminderDB:
    """Database interface for storing and retrieving reminders"""
    
    @staticmethod
    def create_reminder(user_id, title, start_time, end_time=None, description=None, db=None):
        """Create a new reminder in the database"""
        if db is None:
            raise ValueError("Database connection required")
        
        # Ensure timezone is set to Egypt time
        egypt_tz = pytz.timezone("Africa/Cairo")
        
        # Make sure the start_time is timezone-aware in Egypt time
        if start_time.tzinfo is None:
            start_time = egypt_tz.localize(start_time)
        elif start_time.tzinfo != egypt_tz:
            start_time = start_time.astimezone(egypt_tz)
            
        # Calculate end_time properly
        if not end_time:
            end_time = start_time + timedelta(hours=1)
        else:
            # Make sure end_time has the same timezone handling
            if end_time.tzinfo is None:
                end_time = egypt_tz.localize(end_time)
            elif end_time.tzinfo != egypt_tz:
                end_time = end_time.astimezone(egypt_tz)
            
        # Create reminder object
        reminder = {
            "user_id": user_id,
            "title": title,
            "start_time": start_time,
            "end_time": end_time,
            "description": description,
            "completed": False,  # Default to not completed
            "created_at": datetime.now(egypt_tz),
            "status": "active"
        }
        
        # Insert into database
        result = db.reminders.insert_one(reminder)
        
        # Add the ID to the reminder object
        reminder["_id"] = result.inserted_id
        
        return reminder
    
    @staticmethod
    def get_reminders(user_id, time_min=None, time_max=None, max_results=10, db=None):
        """Get reminders from database within a time range"""
        if db is None:
            raise ValueError("Database connection required")
            
        # Default to today if no time range is provided
        egypt_tz = pytz.timezone("Africa/Cairo")
        
        # Set default time range if not provided
        if not time_min:
            time_min = datetime.now(egypt_tz).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            
        if not time_max:
            time_max = time_min.replace(hour=23, minute=59, second=59)
        
        # Ensure timezone awareness for query parameters
        if time_min.tzinfo is None:
            time_min = egypt_tz.localize(time_min)
        elif time_min.tzinfo != egypt_tz:
            time_min = time_min.astimezone(egypt_tz)
            
        if time_max.tzinfo is None:
            time_max = egypt_tz.localize(time_max)
        elif time_max.tzinfo != egypt_tz:
            time_max = time_max.astimezone(egypt_tz)
            
        # Print debug info
        print(f"[DEBUG] Querying reminders from {time_min} to {time_max}")
        
        # Build query
        query = {
            "user_id": user_id,
            "start_time": {"$gte": time_min, "$lte": time_max},
            "status": "active"
        }
        
        # Get reminders
        reminders = list(db.reminders.find(query).sort("start_time", 1).limit(max_results))
        
        # Ensure all returned times have consistent timezone
        for reminder in reminders:
            for field in ['start_time', 'end_time', 'created_at']:
                if field in reminder and reminder[field]:
                    dt = reminder[field]
                    if dt.tzinfo is None:
                        reminder[field] = egypt_tz.localize(dt)
                    elif dt.tzinfo != egypt_tz:
                        reminder[field] = dt.astimezone(egypt_tz)
        
        return reminders
    
    @staticmethod
    def get_day_reminders(user_id, target_date=None, db=None):
        """Get all reminders for a specific day"""
        if db is None:
            raise ValueError("Database connection required")
            
        # Default to today if no date is provided
        egypt_tz = pytz.timezone('Africa/Cairo')
        if not target_date:
            target_date = datetime.now(egypt_tz)
            
        # Set time range for the whole day
        time_min = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        time_max = target_date.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        return ReminderDB.get_reminders(user_id, time_min=time_min, time_max=time_max, max_results=50, db=db)
    
    @staticmethod
    def format_reminders_response(reminders, date_context=None):
        """Format reminders into a human-readable response"""
        if not reminders:
            time_context = ReminderDB._get_time_context(date_context)
            return f"I've checked your reminders and you don't have any scheduled {time_context}. Your schedule is clear!"
            
        # Format reminders in a more conversational way
        formatted_reminders = []
        for reminder in reminders:
            start_time = reminder['start_time']
            time_str = start_time.strftime("%I:%M %p")
            formatted_reminders.append(f"At {time_str}, you have {reminder['title']}")
        
        timetable = "\n".join(formatted_reminders)
        
        time_context = ReminderDB._get_time_context(date_context)
        return f"Let me tell you what's on your schedule {time_context}.\n\n{timetable}"
    
    @staticmethod
    def _get_time_context(date_context):
        """Get a human-readable time context string"""
        if not date_context:
            return "today"
            
        # If it's a datetime object
        if isinstance(date_context, datetime):
            now = datetime.now(date_context.tzinfo)
            days_diff = (date_context.date() - now.date()).days
            
            if days_diff == 0:
                return "today"
            elif days_diff == 1:
                return "tomorrow"
            else:
                return f"for {date_context.strftime('%A, %B %d')}"
                
        # If it's an integer (days offset)
        if isinstance(date_context, int):
            if date_context == 0:
                return "today"
            elif date_context == 1:
                return "tomorrow"
            else:
                tz = pytz.timezone("Africa/Cairo")
                target_date = datetime.now(tz) + timedelta(days=date_context)
                return f"for {target_date.strftime('%A, %B %d')}"
                
        return "today"  # Default fallback
    
    @staticmethod
    def update_reminder(reminder_id, updates, db=None):
        """Update a reminder in the database"""
        if db is None:
            raise ValueError("Database connection required")
            
        result = db.reminders.update_one(
            {"_id": ObjectId(reminder_id)},
            {"$set": updates}
        )
        
        return result.modified_count > 0
    
    @staticmethod
    def delete_reminder(reminder_id, db=None):
        """Delete a reminder from the database"""
        if db is None:
            raise ValueError("Database connection required")
            
        print(f"[DEBUG] ReminderDB.delete_reminder called with ID: {reminder_id}")
        
        # We don't actually delete, just mark as inactive
        try:
            result = db.reminders.update_one(
                {"_id": ObjectId(reminder_id)},
                {"$set": {"status": "deleted"}}
            )
            
            print(f"[DEBUG] Update result - matched: {result.matched_count}, modified: {result.modified_count}")
            
            return result.modified_count > 0
        except Exception as e:
            print(f"[DEBUG] Error in delete_reminder: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e

class Reminder:
    """Reminder utility for creating and managing reminders"""
    
    @staticmethod
    def create_event(title, start_time):
        """Create a reminder event object"""
        reminder = {
            "title": title,
            "start_time": start_time,
            "end_time": start_time + timedelta(hours=1)
        }
        return reminder
    
    @staticmethod
    def get_timetable(days_offset=0):
        """Get the time range for fetching reminders"""
        egypt_tz = pytz.timezone('Africa/Cairo')
        target_date = datetime.now(egypt_tz) + timedelta(days=days_offset)
        
        return {
            "time_min": target_date.replace(hour=0, minute=0, second=0, microsecond=0),
            "time_max": target_date.replace(hour=23, minute=59, second=59, microsecond=999999),
            "days_offset": days_offset
        }
    
    @staticmethod
    def parse_time(time_str, target_date):
        """Parse time string into datetime object with timezone"""
        if not time_str:
            return target_date
            
        try:
            # Extract hours, minutes, and am/pm
            time_pattern = r'(\d{1,2})(?::(\d{1,2}))?\s*(am|pm|a\.m\.|p\.m\.)'
            match = re.match(time_pattern, time_str.lower())
            
            if match:
                hour = int(match.group(1))
                minute = int(match.group(2)) if match.group(2) else 0
                meridian = match.group(3)
                
                # Convert to 24-hour format
                if meridian in ['pm', 'p.m.'] and hour != 12:
                    hour += 12
                elif meridian in ['am', 'a.m.'] and hour == 12:
                    hour = 0
                
                # Ensure target_date has Egypt timezone
                egypt_tz = pytz.timezone('Africa/Cairo')
                if target_date.tzinfo is None:
                    target_date = egypt_tz.localize(target_date)
                elif target_date.tzinfo != egypt_tz:
                    target_date = target_date.astimezone(egypt_tz)
                
                # Print debug info for timezone validation
                print(f"[DEBUG] Parsing time: {hour}:{minute} ({meridian})")
                print(f"[DEBUG] Target date before: {target_date}")
                
                # Create new time with the target date
                new_time = target_date.replace(
                    hour=hour,
                    minute=minute,
                    second=0,
                    microsecond=0
                )
                
                print(f"[DEBUG] New time after: {new_time}")
                return new_time
            else:
                raise ValueError(f"Could not parse time format: {time_str}")
                
        except ValueError as e:
            raise ValueError(f"Invalid time format. Please use format like '3 pm' or '11:30 am'. Error: {str(e)}") 