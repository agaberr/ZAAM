from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
import os
import torch
import json
import re
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
import pickle
import pytz  # Make sure to import pytz
from server.model import NERIntentModel


app = Flask(__name__)
app.secret_key = "e5QD8iIgEo0iBvi2Lx2bgK89vHtcqV"

# Google OAuth Config
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
CLIENT_SECRETS_FILE = "credentialsOAuth.json"
SCOPES = ["https://www.googleapis.com/auth/calendar"]

# Time expressions mapping
TIME_EXPRESSIONS = {
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

# Load model and initialize encoders
try:
    print("Loading model...")
    model = torch.load("reminder_model.pth", map_location=torch.device("cpu"))
    print("Model loaded successfully")
    print("Model structure:", model)
    model.eval()
except Exception as e:
    print("Error loading model:", str(e))

# Load tokenizer and encoders
try:
    print("Loading tokenizer and encoders...")
    with open('reminder_tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('reminder_intent_encoder.pkl', 'rb') as f:
        intent_encoder = pickle.load(f)
    with open('reminder_slot_encoder.pkl', 'rb') as f:
        slot_encoder = pickle.load(f)
    print("Tokenizer and encoders loaded successfully")
except Exception as e:
    print("Error loading tokenizer and encoders:", str(e))


def credentials_to_dict(credentials):
    return {
        "token": credentials.token,
        "refresh_token": credentials.refresh_token,
        "token_uri": credentials.token_uri,
        "client_id": credentials.client_id,
        "client_secret": credentials.client_secret,
        "scopes": credentials.scopes,
    }

def extract_time_expressions(text):
    """Extract time expressions from text and return modified text and days offset"""
    text_lower = text.lower()
    days_offset = 0
    found_expression = None
    
    # Check for time expressions
    for expression, offset in TIME_EXPRESSIONS.items():
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

@app.route("/login")
def login():
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE, scopes=SCOPES, redirect_uri=url_for("callback", _external=True)
    )
    authorization_url, state = flow.authorization_url(access_type="offline", include_granted_scopes="true")
    session["state"] = state
    return redirect(authorization_url)

@app.route("/callback")
def callback():
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE, scopes=SCOPES, redirect_uri=url_for("callback", _external=True)
    )
    flow.fetch_token(authorization_response=request.url)
    session["credentials"] = credentials_to_dict(flow.credentials)
    return redirect(url_for("home"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        if "credentials" not in session:
            return jsonify({"response": "You'll need to log in first before I can help you with your calendar."}), 401
        
        credentials = Credentials(**session["credentials"])
        service = build("calendar", "v3", credentials=credentials)
        
        original_input = request.json["text"]
        user_input, days_offset = extract_time_expressions(original_input)
        
        print(f"\nProcessing request:")
        print(f"Original input: {original_input}")
        print(f"Modified input: {user_input}")
        print(f"Days offset: {days_offset}")
        
        tokenized_text = user_input.lower().split()
        print(f"Tokenized text: {tokenized_text}")
        
        predicted_intent, predicted_slots = predict(model, tokenized_text)
        print(f"Prediction results:")
        print(f"Intent: {predicted_intent}")
        print(f"Slots: {predicted_slots}")
        
        result = postprocess_ner_predictions(predicted_intent, tokenized_text, predicted_slots)
        print(f"Postprocessed result: {result}")
        
        # Add null checks for action and time
        action = (result.get("predicted_action") or "").lower()
        time_str = result.get("predicted_time") or ""
        event_text = result.get("predicted_action", "")  # Use predicted_action as event text

    
        if predicted_intent == "get_timetable":
            payload = get_timetable(days_offset)
            events_result = service.events().list(**payload).execute()
            events = events_result.get("items", [])
            
            # Determine which date we're showing based on days_offset
            egypt_tz = pytz.timezone('Africa/Cairo')
            target_date = datetime.now(egypt_tz) + timedelta(days=days_offset)
            target_date_str = target_date.strftime("%A, %B %d")
            
            if not events:
                time_context = "today" if days_offset == 0 else (
                    "tomorrow" if days_offset == 1 else f"on {target_date_str}")
                return jsonify({"response": f"I've checked your calendar and you don't have any events scheduled {time_context}. Your schedule is clear!"})
            
            # Format events in a more conversational way
            formatted_events = []
            for event in events:
                start_time = event['start'].get('dateTime', 'All day')
                if start_time != 'All day':
                    dt = datetime.fromisoformat(start_time)
                    time_str = dt.strftime("%I:%M %p")
                    formatted_events.append(f"At {time_str}, you have {event['summary']}")
                else:
                    formatted_events.append(f"You have {event['summary']} scheduled for all day")
            
            timetable = "\n".join(formatted_events)
            
            time_context = "today" if days_offset == 0 else (
                "tomorrow" if days_offset == 1 else f"for {target_date_str}")
            return jsonify({"response": f"Let me tell you what's on your schedule {time_context}.\n\n{timetable}"})
        
        elif predicted_intent == "create_event":
            if not event_text:
                return jsonify({"response": "Could not understand what event to create. Please try again."})

            if not action:
                return jsonify({"response": "I couldn't understand the action. Please try again."})
            
            # Use Egypt timezone
            egypt_tz = pytz.timezone('Africa/Cairo')
            start_time = datetime.now(egypt_tz) + timedelta(days=days_offset)
            
            if time_str:
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
                        return jsonify({"response": f"Could not parse time format: {time_str}. Please use format like '3 pm' or '11 am'."})
                    
                    print(f"Parsed time - Hour: {hour}, Meridian: {meridian}")
                    
                    # Convert to 24-hour format
                    if meridian == 'pm' and hour != 12:
                        hour += 12
                    elif meridian == 'am' and hour == 12:
                        hour = 0
                    
                    print(f"24-hour format - Hour: {hour}")
                    
                    # Create new time in Egypt timezone
                    start_time = start_time.replace(
                        hour=hour,
                        minute=0,
                        second=0,
                        microsecond=0
                    )
                    
                    print(f"Final start_time: {start_time}")
                    
                except ValueError as e:
                    return jsonify({"response": f"Invalid time format. Please use format like '3 pm' or '11 am'. Error: {str(e)}"})
        
            payload = create_event(event_text, start_time)
            print(f"Event payload: {payload}")
            event = service.events().insert(calendarId="primary", body=payload).execute()
            
            # Format the time for the response
            formatted_time = start_time.strftime("%I:%M %p")
            formatted_date = start_time.strftime("%A, %B %d")
            
            # Choose appropriate time context based on days_offset
            time_context = "today" if days_offset == 0 else (
                "tomorrow" if days_offset == 1 else f"on {formatted_date}")
            
            return jsonify({
                "response": f"Perfect! I've added {event_text} to your calendar for {formatted_time} {time_context}. It's scheduled for one hour."
            })
        
        return jsonify({"response": "I'm not quite sure what you'd like me to do with your calendar. Could you please rephrase that?"})
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({"response": "I'm sorry, but I ran into a problem while trying to help you with that. Could you try again?"}), 500

def tokenize_text(texts):
    return [re.findall(r"\w+|[.,!?;]", text.lower()) for text in texts]

def predict(model, tokenized_text, max_seq_length=128):
    try:
        print(f"\nPrediction debug:")
        print(f"Input text: {tokenized_text}")
        
        # Convert tokens to ids
        token_ids = [tokenizer.get(token, tokenizer["UNK"]) for token in tokenized_text]
        print(f"Token IDs: {token_ids}")
        
        # Pad sequence
        token_ids = token_ids[:max_seq_length] + [tokenizer["PAD"]] * (max_seq_length - len(token_ids))
        print(f"Padded token IDs: {token_ids}")
        
        # Convert to tensor
        input_tensor = torch.tensor(token_ids).unsqueeze(0)
        print(f"Input tensor shape: {input_tensor.shape}")
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            intent_pred, slot_pred = model(input_tensor)
            print(f"Intent prediction shape: {intent_pred.shape}")
            print(f"Slot prediction shape: {slot_pred.shape}")
        
        # Get the predicted intent
        intent_label = intent_pred.argmax(dim=1).item()
        predicted_intent = intent_encoder.inverse_transform([intent_label])[0]
        print(f"Predicted intent: {predicted_intent}")
        
        # Get the predicted slots
        slot_preds = slot_pred.argmax(dim=2).squeeze().cpu().numpy()
        predicted_slots = slot_encoder.inverse_transform(slot_preds[:len(tokenized_text)])
        print(f"Predicted slots: {predicted_slots}")
        
        return predicted_intent, predicted_slots
        
    except Exception as e:
        print(f"Error in predict function: {str(e)}")
        raise e

def postprocess_ner_predictions(predicted_intent, predicted_tokens, predicted_slots):
    print(f"Processing - Intent: {predicted_intent}")
    print(f"Processing - Tokens: {predicted_tokens}")
    print(f"Processing - Slots: {predicted_slots}")
    
    predicted_time, predicted_action = [], []
    current_action = []
    current_time = []
    
    for token, slot in zip(predicted_tokens, predicted_slots):
        print(f"Token: {token}, Slot: {slot}")
        
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
    
    print(f"Final result: {result}")
    return result

def create_event(title, start_time):
    event = {
        "summary": title,
        "start": {"dateTime": start_time.isoformat(), "timeZone": "Africa/Cairo"},
        "end": {"dateTime": (start_time + timedelta(hours=1)).isoformat(), "timeZone": "Africa/Cairo"}
    }
    print(f"Creating event: {event}")
    return event

def get_timetable(days_offset=0):
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

if __name__ == "__main__":
    app.run(debug=True)