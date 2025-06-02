import torch
import torch.nn as nn
import pickle
import os
import re
from datetime import datetime, timedelta
import pytz
from bson import ObjectId


class NERIntentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_intents, num_slots):
        super(NERIntentModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        self.intent_fc = nn.Linear(hidden_dim * 2, num_intents)
        self.slot_fc = nn.Linear(hidden_dim * 2, num_slots)

    def forward(self, x):
        embedded =  self.embedding (x)
        
        lstm_out, _ =  self.lstm(embedded)
        
        intentPrediction = self.intent_fc (lstm_out[:, 0, :])
        
        slotPrediction =  self.slot_fc (lstm_out)
        
        return intentPrediction,  slotPrediction 

class ReminderNLP:
    """NLP model for processing reminder text and extracting intent and entities"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.intent_encoder = None
        self.slot_encoder = None
        self.intiModel()
        
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
    
    def intiModel(self):
        try:
            reminders_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reminders')
            
            tokenizer_path = os.path.join(reminders_dir, 'reminder_tokenizer.pkl')
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
                
            intent_encoder_path = os.path.join(reminders_dir, 'reminder_intent_encoder.pkl')
            with open(intent_encoder_path, 'rb') as f:
                self.intent_encoder = pickle.load(f)
                
            slot_encoder_path = os.path.join(reminders_dir, 'reminder_slot_encoder.pkl')
            with open(slot_encoder_path, 'rb') as f:
                self.slot_encoder = pickle.load(f)
            
            vocab_size = len(self.tokenizer)
            embedding_dim = 100 
            hidden_dim = 128 
            num_intents = len(self.intent_encoder.classes_)
            num_slots = len(self.slot_encoder.classes_)
       
            self.model = NERIntentModel(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_intents=num_intents,
                num_slots=num_slots
            )
            
            model_path = os.path.join(reminders_dir, 'reminder_model_checkpoint.pth')
            saved_model = torch.load(model_path, map_location=torch.device("cpu"))
         
            
            if isinstance(saved_model, dict) and 'state_dict' in saved_model:
                self.model.load_state_dict(saved_model['state_dict'])
            elif hasattr(saved_model, 'state_dict'):
                self.model.load_state_dict(saved_model.state_dict())
            elif isinstance(saved_model, dict):
                self.model.load_state_dict(saved_model)
            else:
                raise ValueError(f"Unexpected model format: {type(saved_model)}")
                
            self.model.eval()
                
        except:            
            raise ("Can't init reminder model")
    


    def getTimeExpression(self, txt):
        text =  txt.lower()
        days_offset = 0
        finalExprr = None
                
        for expr, offset in self.TIME_EXPRESSIONS.items():
            if expr in text:
                days_offset = offset
                finalExprr = expr
                break
        
        final_txt = text
        if finalExprr:
            final_txt = re.sub(r'\b' + re.escape(finalExprr) + r'\b', '', final_txt, flags=re.IGNORECASE)
        
        final_txt = re.sub(r'\s+', ' ', final_txt).strip()
        
        return final_txt, days_offset
    
    def predict(self,  tokenized_text,  max_seq_length=128):
        try:
            tokenIds = [self.tokenizer.get(token, self.tokenizer["UNK"]) for token in tokenized_text]
            
            tokenIds = tokenIds[:max_seq_length] + [self.tokenizer["PAD"]] * (max_seq_length - len(tokenIds))
            
            INtensor = torch.tensor(tokenIds).unsqueeze(0)
            
            self.model.eval()
            with torch.no_grad():
                intentPrediction, slotPrediction = self.model(INtensor)
            
            intentLabell = intentPrediction.argmax(dim=1).item()
            predIndent = self.intent_encoder.inverse_transform([intentLabell])[0]
            
            slotPredictions = slotPrediction.argmax(dim=2).squeeze().cpu().numpy()
            predslots = self.slot_encoder.inverse_transform(slotPredictions[:len(tokenized_text)])
            
            return predIndent, predslots
            
        except:
            raise ("there is an error in prediction")
    


    def postprocessNER(self,  predicted_intent,  predicted_tokens,  predicted_slots):
        predicted_time, predicted_action = [], []
        current_action = []
        current_time = []
        
      
        
        for token,  slot in zip(predicted_tokens, predicted_slots):
            if slot ==  'B-TIME':
                if current_time: 
                    predicted_time.append(' '.join(current_time))
                    current_time = []
                current_time.append(token)
            elif slot ==  'I-TIME' and current_time:
                current_time.append(token)
                 
            if slot ==  'B-ACTION':
                if current_action: 
                    predicted_action.append(' '.join(current_action))
                    current_action = []
                current_action.append(token)
            elif slot ==  'I-ACTION' and current_action: 
                current_action.append(token)
        
        if current_time:
            predicted_time.append(' '.join(current_time))
        if current_action:
            predicted_action.append(' '.join(current_action))
        
        final_action = None
        if predicted_action:
            raw_action =  " ".join(predicted_action)
            
            if raw_action.startswith('me to '):
                remaining_tokens = predicted_tokens[:]
                start_idx = -1
                
                for i in  range(len(predicted_tokens) - 1):
                    if predicted_tokens[i] == 'me' and predicted_tokens[i + 1] == 'to':
                        start_idx = i + 2 
                        break
                
                if start_idx >= 0 and start_idx  < len(predicted_tokens):
                    meaningful_tokens = []
                    for token in predicted_tokens[start_idx:]:
                        if token not in ['at', 'on', 'for', 'by', 'in', 'the', 'a', 'an']:
                            meaningful_tokens.append(token)
                        else:
                            break
                    
                    if meaningful_tokens:
                        final_action = ' '.join(meaningful_tokens)
            
            elif raw_action.endswith(' me')  or raw_action.endswith(' to'):
                final_action = raw_action.rsplit(' ', 1)[0]
            
            if not final_action:
                final_action = raw_action
        else:
            final_action = None
        
        result = {
            "text": " " .join(predicted_tokens),
            "predicted_intent": predicted_intent,
            "predicted_time": " ".join(predicted_time) if predicted_time else None,
            "predicted_action":  final_action
        }
        
        return result
    
    def process_text(self, text):
        """Process user input text and extract reminder information"""
        
        # Extract time expressions directly from original text as fallback
        time_str = None
        time_pattern = r'\b(\d{1,2})(?::(\d{1,2}))?\s*(am|pm|a\.m\.|p\.m\.)\b'
        time_match = re.search(time_pattern, text.lower())
        
        if time_match:
            time_str = time_match.group(0)
        
        # Extract time expressions and process text
        user_input, days_offset = self.getTimeExpression(text)
        
        # Tokenize the text
        tokenized_text = user_input.lower().split()
        
        # Get predictions from the model
        predicted_intent, predicted_slots = self.predict(tokenized_text)
        
     
        original_intent = predicted_intent
        
        # Check if time information is present in the original text
        has_time_info = time_str is not None
        
        time_patterns = [
            r'\b\d{1,2}:\d{2}\s*(am|pm|a\.m\.|p\.m\.)\b',  # 3:30 pm, 10:15 am
            r'\b\d{1,2}\s*(am|pm|a\.m\.|p\.m\.)\b',        # 3 pm, 10 am
            r'\b\d{1,2}\s*o\'?clock\b',                      # 3 o'clock, 3 oclock
            r'\bat\s+\d',                                     # at 3, at 10
            r'\b(morning|afternoon|evening|night)\b',
            r'\b(noon|midnight)\b'
        ]
        
        has_time_pattern = any(re.search(pattern, text.lower()) for pattern in time_patterns)
        
        scheduling_keywords = ['remind', 'schedule', 'set', 'appointment', 'meeting', 'at']
        has_scheduling_keywords = any(keyword in text.lower() for keyword in scheduling_keywords)
       
        if has_time_info or has_time_pattern:
            if predicted_intent != "create_event":
                predicted_intent = "create_event"
        else:
            if predicted_intent != "get_timetable":
                predicted_intent = "get_timetable"
        
      
        # Process the predictions
        result = self.postprocessNER(predicted_intent, tokenized_text, predicted_slots)
        
        # Add days offset to the result
        result['days_offset'] = days_offset
        
        if time_str:
            result['predicted_time'] = time_str
     
            
        return result








####### REMINDER DATABSASE



class ReminderDB:
    
    @staticmethod
    def createReminder(userID, title, startTime, endTime=None, description=None, db=None):
        if db is None:
            raise  ValueError("Can't connect to the db..")

        # de bt7l almost kol mashakel el time zone as we need cairo zone
        if startTime.tzinfo is None:
            startTime =  pytz.timezone("Africa/Cairo").localize(startTime)
        elif startTime.tzinfo != pytz.timezone("Africa/Cairo"):
            startTime = startTime.astimezone(pytz.timezone("Africa/Cairo"))
            

        # by default el reminder byb2a sa3a wahda
        if not  endTime:
            endTime = startTime + timedelta(hours=1)
        else:
            if  endTime.tzinfo is None:
                endTime  = pytz.timezone("Africa/Cairo").localize(endTime)
            elif endTime.tzinfo != pytz.timezone("Africa/Cairo"):
                endTime = endTime.astimezone(pytz.timezone("Africa/Cairo"))
            
        rem = {
            "user_id":  userID,
            "title":  title,
            "start_time":  startTime,
            "end_time": endTime,
            "description": description,
            "completed": False, # byb2a pending in the frontend
            "created_at":  datetime.now(pytz.timezone("Africa/Cairo")),
            "status": "active"
        }
        
        result =  db.reminders.insert_one(rem)
        
        rem["_id"] = result.inserted_id
        
        return rem
    


    @staticmethod
    def getReminders(userID,  timeMin=None,  timeMax=None,  maxRes=10,  db=None):

        if db is None:
            raise ValueError("can't connecto to database..")
            
        
        if not timeMin:
            timeMin = datetime.now(pytz.timezone("Africa/Cairo")).replace(hour=0, minute=0, second=0, microsecond=0)
            
        if not timeMax:
            timeMax = timeMin.replace(hour=23,  minute=59,  second=59)
        
        if timeMin.tzinfo  is None:
            timeMin =  pytz.timezone("Africa/Cairo").localize(timeMin)
        elif timeMin.tzinfo != pytz.timezone("Africa/Cairo"):
            timeMin =  timeMin.astimezone(pytz.timezone("Africa/Cairo"))
            
        if timeMax.tzinfo is  None:
            timeMax = pytz.timezone("Africa/Cairo").localize(timeMax)
        elif timeMax.tzinfo != pytz.timezone("Africa/Cairo"):
            timeMax = timeMax.astimezone(pytz.timezone("Africa/Cairo"))
            
        
        query =  {
        "user_id": userID,
        "start_time": {"$gte": timeMin, "$lte": timeMax},
        "status": "active"}
        
        reminders =  list(db.reminders.find(query).sort("start_time", 1) )
        
        for rem in reminders:
            for field in ['start_time', 'end_time', 'created_at']:
                if field in rem and rem[field]:
                    dt  = rem[field]
                    if  dt.tzinfo is  None:
                        rem[field] =  pytz.timezone("Africa/Cairo").localize(dt)
                    elif  dt.tzinfo !=  pytz.timezone("Africa/Cairo"):
                        rem[field] =  dt.astimezone(pytz.timezone("Africa/Cairo"))
        
        return reminders
    




    @staticmethod
    def getDayReminders(userID, targetDate=None, db=None):
        if db is None:
            raise ValueError("can't connect to the database..")
            
        # lw mafeesh date provided hakhaly el date elnaharda
        if not targetDate:
            targetDate = datetime.now(pytz.timezone('Africa/Cairo'))
            
        timeMin = targetDate.replace(hour=0, minute=0, second=0, microsecond=0)
        timeMax = targetDate.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        return ReminderDB.getReminders(userID, timeMin=timeMin, timeMax=timeMax, maxRes=50, db=db)
    

    
    @staticmethod
    def formatReminderResponse(reminders, dateContext=None):
        if not reminders:
            timeContext = ReminderDB._get_time_context(dateContext)
            return f"I've checked your reminders and you don't have any scheduled {timeContext}. Your schedule is clear!"
            
        remindersFormatted = []
        for reminder in reminders:
            startTime = reminder['start_time']
            timeString = startTime.strftime("%I:%M %p")
            remindersFormatted.append(f"At {timeString}, you have {reminder['title']}")
        
        timetable = "".join(remindersFormatted)
        
        timeContext = ReminderDB._get_time_context(dateContext)
        return f"Let me tell you what's on your schedule {timeContext}.\n{timetable}"
    




    @staticmethod
    def _get_time_context(dateContext):
        if not  dateContext:
            return "today"
            
        if isinstance(dateContext,  datetime):
            now =  datetime.now(dateContext.tzinfo)
            daysDiff = (dateContext.date() - now.date()).days
            
            if  daysDiff == 0:
                return "today"
            elif  daysDiff == 1:
                return "tomorrow"
            else:
                return f"for {dateContext.strftime('%A, %B %d')}"
                
        if isinstance(dateContext, int):
            if dateContext == 0:
                return "today"
            elif dateContext == 1:
                return "tomorrow"
            else:
                target_date = datetime.now(pytz.timezone("Africa/Cairo")) + timedelta(days=dateContext)
                return f"for {target_date.strftime('%A, %B %d')}"
                
        return "today"
    
    @staticmethod
    def update_reminder(reminder_id, updates, db=None):
        if db is None:
            raise ValueError("err in db connection")
            
        res = db.reminders.update_one(
            {"_id": ObjectId(reminder_id)},
            {"$set": updates}
        )
        
        return res.modified_count > 0
    
    @staticmethod
    def delete_reminder(remiID, db=None):
        if db  is None:
            raise ValueError("there is error in database connection..")
            
        try:
            res =  db.reminders.update_one(
                {"_id": ObjectId(remiID)},
                {"$set": {"status": "deleted"}})
                        
            return  res.modified_count > 0
        except:
            raise  ("can't delete reminder..")

class Reminder:
    
    @staticmethod
    def create_event(title, startTime):
        rem = {
            "title": title,
            "start_time": startTime,
            "end_time": startTime + timedelta(hours=1)
        }
        return rem
    
    @staticmethod
    def get_timetable(days_offset=0):
        egypt_tz = pytz.timezone('Africa/Cairo')
        target_date = datetime.now(egypt_tz) + timedelta(days=days_offset)
        
        return {
            "time_min": target_date.replace(hour=0, minute=0, second=0, microsecond=0),
            "time_max": target_date.replace(hour=23, minute=59, second=59, microsecond=999999),
            "days_offset": days_offset
        }
    
    @staticmethod
    def parse_time(timeString,  targetDate):
        
        if not timeString:
            return  targetDate
            
        try:
            match = re.match(r'(\d{1,2})(?::(\d{1,2}))?\s*(am|pm|a\.m\.|p\.m\.)', timeString.lower())
                        
            if match:
                hr =  int(match.group(1))
                minute =  int(match.group(2)) if match.group(2) else 0
                amorpm = match.group(3)
                
                
                if amorpm ==  "pm" and hr != 12:
                    hr +=  12
                elif amorpm ==  "am" and hr == 12:
                    hr = 0
                
                if targetDate.tzinfo  is None:
                    targetDate =  pytz.timezone('Africa/Cairo').localize(targetDate)
                elif targetDate.tzinfo != pytz.timezone('Africa/Cairo'):
                    targetDate =  targetDate.astimezone(pytz.timezone('Africa/Cairo'))
                
                
                newTime =  targetDate.replace(
                    hour=hr,
                    minute=minute,
                    second =0 ,
                    microsecond=0
                )
                
                return newTime
            else:
                raise  ValueError(f"Could not parse time format")
                
        except:
            raise ValueError("there is error in format.") 