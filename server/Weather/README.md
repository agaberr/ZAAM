# Weather Chat Flask Application

This Flask application provides a web interface to interact with a weather prediction model.

## Features

- Chat-based interface for asking weather-related questions
- Supports queries about:
  - Weather forecasts
  - Clothing recommendations
  - Yes/no weather questions
  - Whether certain clothing is appropriate

## Requirements

- Python 3.7+
- Required packages (see requirements.txt)

## Getting Started

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Make sure you have the model file `best_model.pt` in the directory.

3. Run the application:

```bash
python run.py
```

4. Open your web browser and navigate to http://localhost:5002

## API Usage

The application exposes an API endpoint that can be used to communicate with the model directly:

### POST /api/chat

**Request:**
```json
{
  "message": "Will it rain tomorrow in Cairo?"
}
```

**Response:**
```json
{
  "response": "Based on the forecast, it will be partly cloudy in Cairo on Sunday, July 10."
}
```

## Example Queries

- "Will it rain tomorrow in Cairo?"
- "Should I wear a jacket tonight?"
- "What's the weather forecast for next Monday?"
- "Is it going to be cold in New York next week?"
- "Do I need an umbrella today?"

## Model Details

The application uses a fine-tuned BERT model (`best_model.pt`) to determine the intent of user queries and extract relevant entities. The model was trained to identify four main intents:

1. Weather forecast queries
2. Yes/no weather questions
3. Clothing recommendation queries
4. Yes/no clothing questions

## Note

This application uses the Weather API (weatherapi.com) to fetch real-time weather data. For demonstration purposes, the app is configured with a default location (Cairo, Egypt) but can be modified to use geolocation or user-provided locations.

## License

This project is for educational purposes only.
