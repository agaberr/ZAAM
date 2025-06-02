import os
import re
from datetime import datetime

# External utility imports
from data_utils import intent2idx, entity2idx, tokenizer
from model import load_model, predict_intent_and_entities
from weather_api import (
    get_location_and_timezone,
    get_forecast_weather,
    display_forecast_weather,
    fetch_forecast_weather
)
from entity_extraction import extract_entities, parse_date
from response_generation import generate_response

# Set up some globals
WEATHER_DIR = os.path.dirname(os.path.abspath(__file__))

# Note: Hardcoding this for now – should move to env vars later
API_KEY = "6598d1fbebc24041896182152251402"

# Load model stuff
best_model_path = os.path.join(WEATHER_DIR, "weather_model.pt")
model, optimizer, checkpoint = load_model(best_model_path)

print(f"✅ Loaded model from epoch {checkpoint['epoch']}, Intent F1: {checkpoint['intent_f1']:.4f}, NER F1: {checkpoint['ner_f1']:.4f}")

def get_weather_response(user_query):
    # First, try to get the user's location
    timezone, lat, lon = get_location_and_timezone()
    location = f"{lat},{lon}"  # just a simple "lat,lon" format

    # Extract whatever entities we can from the query
    entities = extract_entities(user_query)
    print("👉 Extracted Entities:", entities)

    clothing_pref = entities.get('clothing_items')
    
    # Date and time parsing
    if entities.get('dates'):
        forecast_date, delta_days = parse_date(entities['dates'][0])
    else:
        forecast_date, delta_days = None, 0  # fallback values
    
    forecast_time = None
    if entities.get('times'):
        # Yeah, we have to import this only here due to circular import issues (ugh)
        from weather_api import parse_time
        forecast_time = parse_time(entities['times'][0], timezone)

    # Fallbacks if date/time parsing didn't work
    if not forecast_date:
        forecast_date = datetime.now().strftime("%Y-%m-%d")
    if not forecast_time:
        forecast_time = datetime.now().strftime("%H")

    print(f"📅 Forecast Date: {forecast_date} (Δ {delta_days} days)")
    print(f"🕒 Forecast Time: {forecast_time}")
    print(f"📍 Location: {location}")
    print("-" * 60)

    # Get weather data
    forecast_raw = get_forecast_weather(API_KEY, forecast_date, forecast_time, location, delta_days)
    display_forecast_weather(forecast_raw)
    print("-" * 60)

    weather_info = fetch_forecast_weather(forecast_raw)
    if not weather_info:
        print("❌ No forecast data retrieved.")
        return "Sorry, I couldn't retrieve the weather data."

    # Classify weather for smarter responses
    from data_utils import classify_weather
    summary = classify_weather(
        temp=weather_info['Current Temperature (°C)'],
        humidity=weather_info['Current Humidity (%)'],
        wind_kph=weather_info['Current Wind Speed (kph)'],
        chance_of_rain=weather_info['Chance of Rain (%)'],
        is_raining=weather_info['Will it Rain?'],
        condition=weather_info['Condition']
    )
    print("🌤️ Weather Classification:", summary)
    print("-" * 60)

    # Get predicted intents and entities (again? slightly redundant but ok for now)
    prediction_result = predict_intent_and_entities(model, user_query, tokenizer, intent2idx, entity2idx)
    intents = prediction_result['predicted_intents']
    print("🎯 Predicted Intents:", intents)

    if not intents:
        return "Sorry, I couldn't understand your query clearly."

    # Start forming a response
    full_response = ""
    for intent in intents:
        if intent == "yes_no_clothing":
            resp = generate_response(
                intent,
                summary,
                user_preferred_clothing=clothing_pref[0] if clothing_pref else None
            )
        elif intent == "yes_no_weather":
            resp = generate_response(
                intent,
                summary,
                user_query=user_query
            )
        else:
            resp = generate_response(intent, summary)

        full_response += f"{resp}\n"

    # Clean up formatting
    final = full_response.strip()
    if final:
        final = final[0].upper() + final[1:]
        final = re.sub(r'(?<=\.\s)([a-z])', lambda m: m.group(1).upper(), final)

    return final

