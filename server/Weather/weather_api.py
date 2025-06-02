import requests
import pytz
import dateparser

BASE_URL = "https://api.weatherapi.com/v1"

# gets timezone info based on IP address
def get_location_and_timezone():
    try:
        res = requests.get('https://ipinfo.io')
        data = res.json()
        coords = data['loc'].split(',')  # got "lat,lon"
        lat = float(coords[0])
        lon = float(coords[1])
        tz = data['timezone']
        return tz, lat, lon
    except Exception as e:
        print("uh-oh: couldn't get location/timezone ->", e)
        return None  # maybe return default lat/lon?

# parse vague time input like "evening" into a 24h hour string
def parse_time(q, tz="Africa/Cairo"):
    # TODO: make this smarter (maybe use ML model someday?)
    if q.lower() == "night":
        q = "midnight"
    elif q.lower() == "morning":
        q = "8 AM"
    elif q.lower() == "afternoon":
        q = "3 PM"
    elif q.lower() == "evening":
        q = "8 PM"

    dt = dateparser.parse(q, settings={'TIMEZONE': tz})
    if dt:
        return dt.strftime("%H")
    return None

# get current weather for a location, returns raw JSON
def get_current_weather(api_key, loc="Giza"):
    url = f"{BASE_URL}/current.json?key={api_key}&q={loc}"
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()
    print("API error:", r.status_code)
    return None

# just grabs the main pieces of the data we care about
def fetch_current_weather(data):
    if not data:
        return None
    return {
        "Location": data['location']['name'],
        "Country": data['location']['country'],
        "Temperature (°C)": data['current']['temp_c'],
        "Humidity (%)": data['current']['humidity'],
        "Wind Speed (kph)": data['current']['wind_kph'],
        "Condition": data['current']['condition']['text']
    }

# cli-style printout
def display_current_weather(d):
    w = fetch_current_weather(d)
    if w:
        print(f"{w['Location']}, {w['Country']}")
        print(f"Temp: {w['Temperature (°C)']}°C")
        print(f"Humidity: {w['Humidity (%)']}%")
        print(f"Wind: {w['Wind Speed (kph)']} kph")
        print(f"Condition: {w['Condition']}")
    else:
        print("no data available.")

# get forecast from the API, for a specific date and time
def get_forecast_weather(api_key, date, hour, location="Giza", diff=0):
    url = f"{BASE_URL}/forecast.json?key={api_key}&q={location}&days={diff}&dt={date}&hour={hour}"
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()
    else:
        print("forecast error:", r.status_code)
        return None

# not sure if this should return hourly or daily, mixing for now
def fetch_forecast_weather(d):
    if not d:
        return None
    try:
        loc = d['location']['name']
        ctry = d['location']['country']
        day = d['forecast']['forecastday'][0]['day']
        hr = d['forecast']['forecastday'][0]['hour'][0]

        return {
            "Location": loc,
            "Country": ctry,
            "Current Time": hr['time'],
            "Current Temperature (°C)": hr['temp_c'],
            "Current Humidity (%)": hr['humidity'],
            "Current Wind Speed (kph)": hr['wind_kph'],
            "Current Condition": hr['condition']['text'],
            "Current Chance of Rain (%)": hr['chance_of_rain'],
            "Current Chance of Snow (%)": hr['chance_of_snow'],
            "Max Temperature (°C)": day['maxtemp_c'],
            "Min Temperature (°C)": day['mintemp_c'],
            "Avg Temperature (°C)": day['avgtemp_c'],
            "Avg Humidity (%)": day['avghumidity'],
            "Wind Speed (kph)": day['maxwind_kph'],
            "Condition": day['condition']['text'],
            "Will it Rain?": day['daily_will_it_rain'],
            "Will it Snow?": day['daily_will_it_snow'],
            "Chance of Rain (%)": day['daily_chance_of_rain'],
            "Chance of Snow (%)": day['daily_chance_of_snow']
        }
    except Exception as e:
        print("couldn't parse forecast data:", e)
        return None

# print forecast info (rough output for now)
def display_forecast_weather(d):
    w = fetch_forecast_weather(d)
    if w:
        print(f"{w['Location']}, {w['Country']}")
        print("Time:", w['Current Time'])
        print("-" * 40)
        print("Now:")
        print(f"  Temp: {w['Current Temperature (°C)']}°C")
        print(f"  Humidity: {w['Current Humidity (%)']}%")
        print(f"  Wind: {w['Current Wind Speed (kph)']} kph")
        print(f"  Condition: {w['Current Condition']}")
        print("-" * 40)
        print("Daily Summary:")
        print(f"  Max Temp: {w['Max Temperature (°C)']}°C")
        print(f"  Min Temp: {w['Min Temperature (°C)']}°C")
        print(f"  Avg Temp: {w['Avg Temperature (°C)']}°C")
        print(f"  Avg Humidity: {w['Avg Humidity (%)']}%")
        print(f"  Wind: {w['Wind Speed (kph)']} kph")
        print(f"  Condition: {w['Condition']}")
        print(f"  Rain? {'Yes' if w['Will it Rain?'] else 'No'} ({w['Chance of Rain (%)']}%)")
        print(f"  Snow? {'Yes' if w['Will it Snow?'] else 'No'} ({w['Chance of Snow (%)']}%)")
    else:
        print("no forecast info available :(")
