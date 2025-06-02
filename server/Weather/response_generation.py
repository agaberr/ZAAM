import os
import json
import random

WEATHER_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(WEATHER_DIR, "classified_clothing_items.json"), "r", encoding="utf-8") as f:
    CLOTHING_ITEMS = json.load(f)

TEMP_LEVELS = {"cold": 0, "moderate": 1, "hot": 2}
HUMIDITY_LEVELS = {"not humid": 0, "humid": 1}
WIND_LEVELS = {"not windy": 0, "windy": 1}
RAIN_LEVELS = {"not rain": 0, "rain": 1}

templates = {
    "weather_forecast": [
        "The forecast shows {condition} skies and {temp_phrase} temperatures. It's expected to feel {humidity_phrase} with {wind_phrase} winds. Rain chances are {rain_phrase}.",
        "It will be {condition}, with weather feeling {temp_phrase} and {humidity_phrase}. It is predicted to be {wind_phrase}, and it should {rain_phrase} rain.",
        "Expect a {temp_phrase} day under {condition} skies. It's {humidity_phrase} and {wind_phrase}. Rain is {rain_phrase}.",
        "Skies will be {condition}, temperature is {temp_phrase}, and it's {humidity_phrase}. It is also going to be {wind_phrase}. As for rain, it is {rain_phrase}."
    ],
    "clothing_recommendation": [
        "It's {temp_condition}, so consider wearing {temp_items}.",
        "For {temp_condition} weather, {temp_items} would be ideal.",
        "Since it's {humidity_condition}, go for {humidity_items}.",
        "{wind_condition} conditions suggest wearing {wind_items}.",
        "With {rain_condition} conditions, {rain_items} are recommended.",
        "Dress for {temp_condition} temperatures, {temp_items} would work well.",
        "Given it's supposed to {rain_condition}, {rain_items} are a good idea.",
        "It's likely {wind_condition}, {wind_items} would be great.",
        "{humidity_condition} conditions are predicted, {humidity_items} will keep you comfortable."
    ],
    "yes_no_clothing": [
        "Yes, wearing {item} would be a smart choice given the forecast.",
        "No, wearing {item} isn't necessary for this weather.",
        "Yes, you should definitely wear {item} in this weather.",
        "No, putting on {item} isn't needed for this weather.",
        "Yes, putting on {item} is advisable for this weather.",
        "No, you can skip the {item}.",
        "Yes, sporting {item} is recommended given the conditions.",
        "No, you won't need {item} I believe."
    ],
    "yes_no_weather": [
        "Yes, it will be {temp_phrase}.",
        "No, it won't be {temp_phrase}.",
        "Yes, expect {condition} skies.",
        "No, it won't be {condition}.",
        "Yes, it will likely {rain_phrase} rain.",
        "No, it won't {rain_phrase} rain.",
        "Yes, it will feel {humidity_phrase}.",
        "No, it won't feel {humidity_phrase}.",
        "Yes, winds will be {wind_phrase}.",
        "No, winds won't be {wind_phrase}."
    ]
}

def assess_clothing(item_choice, weather_list):
    """
    Return two flags (temp_ok, rain_ok) indicating if the chosen item makes sense for the weather.
    """
    try:
        with open("classified_clothing_items.json", encoding="utf-8") as f:
            clothes_map = json.load(f)
    except Exception:
        # If something goes wrong reading the file, assume nothing is suitable
        return 0, 0

    temp_ok = 0
    rain_ok = 0

    # weather_list is like [temp_label, ..., ..., rain_label]
    temp_label = weather_list[0]
    if temp_label in clothes_map.get("temperature", {}):
        allowed = clothes_map["temperature"][temp_label]
        if item_choice in allowed:
            temp_ok = 1

    rain_label = weather_list[3]
    if rain_label == "rain":
        rain_items = clothes_map.get("rain", [])
        if item_choice in rain_items:
            rain_ok = 1

    return temp_ok, rain_ok


def respond_yes_no(item_choice, weather_list, phrases):
    """
    Pick a yes/no template and fill in any placeholders to reply about the item.
    """
    if not item_choice:
        return "Hey, I need to know which item you want me to check."

    t_ok, r_ok = assess_clothing(item_choice, weather_list)
    pool = templates.get("yes_no_clothing", [])

    if t_ok or r_ok:
        # look for templates that start with "Yes"
        candidates = [t for t in pool if t.lower().startswith("yes")]
    else:
        candidates = [t for t in pool if t.lower().startswith("no")]

    if not candidates:
        candidates = pool[:]  # fallback if no "Yes"/"No" subset found

    choice_tpl = random.choice(candidates)

    # If template mentions any of these markers, fill in all of them
    if any(marker in choice_tpl for marker in ["{temp_", "{humidity_", "{wind_", "{rain_"]):
        return choice_tpl.format(
            item=item_choice,
            temp_condition=phrases["temp_phrase"],
            humidity_condition=phrases["humidity_phrase"],
            wind_condition=phrases["wind_phrase"],
            rain_condition=phrases["rain_phrase"]
        )

    return choice_tpl.format(item=item_choice)


def pick_relevant_items(cond_type, cond_value, max_items=3):
    """
    Return a small, natural list of recommended clothing based on one condition.
    """
    if cond_type == "temperature":
        # Find the matching label for this temp code
        temp_label = next(k for k, v in TEMP_LEVELS.items() if v == cond_value)
        pool = CLOTHING_ITEMS.get("temperature", {}).get(temp_label, [])
    elif cond_type == "rain" and cond_value == 1:
        pool = CLOTHING_ITEMS.get("rain", [])
    else:
        return ""

    # Remove duplicates
    unique = list(set(pool))
    if not unique:
        return ""

    count = min(max(2, len(unique)), max_items)
    picked = random.sample(unique, min(count, len(unique)))

    if len(picked) == 1:
        return picked[0]
    return ", ".join(picked[:-1]) + " or " + picked[-1]


def suggest_outfit(temp_level, hum_level, wind_level, rain_level):
    """
    Build up to three sentences suggesting what to wear for given weather codes.
    """
    ideas = []

    # Turn numeric codes back into words
    temp_label = next(k for k, v in TEMP_LEVELS.items() if v == temp_level)
    humidity_label = next(k for k, v in HUMIDITY_LEVELS.items() if v == hum_level)
    wind_label = next(k for k, v in WIND_LEVELS.items() if v == wind_level)
    rain_label = next(k for k, v in RAIN_LEVELS.items() if v == rain_level)

    temp_choices = pick_relevant_items("temperature", temp_level)
    rain_choices = pick_relevant_items("rain", rain_level) if rain_level == 1 else ""

    humidity_tip = "light, breathable fabrics" if hum_level == 1 else ""
    wind_tip = "a windbreaker or protective layer" if wind_level == 1 else ""

    active = []
    if temp_choices:
        active.append("temp")
    if rain_choices:
        active.append("rain")
    if hum_level == 1:
        active.append("humidity")
    if wind_level == 1:
        active.append("wind")
    active = active[:3]

    for cond in active:
        # Filter for templates containing the appropriate placeholder
        choices = [
            t for t in templates.get("clothing_recommendation", [])
            if f"{{{cond}_" in t
        ]
        if not choices:
            continue

        tpl = random.choice(choices)
        sentence = tpl.format(
            temp_condition=temp_label,
            temp_items=temp_choices,
            humidity_condition=humidity_label,
            humidity_items=humidity_tip,
            wind_condition=wind_label,
            wind_items=wind_tip,
            rain_condition=rain_label,
            rain_items=rain_choices
        )
        ideas.append(sentence)

    if not ideas:
        return "Looks pretty mild out—just wear whatever feels good."

    full = " ".join(ideas).strip()
    return full[0].upper() + full[1:]


def rephrase_weather(temp_w, hum_w, wind_w, rain_w, summary):
    """
    Convert raw weather words into more natural phrases for templates.
    """
    t_map = {"hot": "warm and sunny", "cold": "chilly", "moderate": "mild"}
    h_map = {"humid": "humid", "not humid": "dry"}
    w_map = {"windy": "breezy", "not windy": "calm"}
    r_map = {"rain": "likely", "not rain": "unlikely"}

    return {
        "condition": summary.lower(),
        "temp_phrase": t_map.get(temp_w, temp_w),
        "humidity_phrase": h_map.get(hum_w, hum_w),
        "wind_phrase": w_map.get(wind_w, wind_w),
        "rain_phrase": r_map.get(rain_w, rain_w)
    }

def generate_weather_yes_no(q, data) -> str:
    # unpack, not proud of this
    temp, humid, wind_cond, rain_status, desc = data

    q = q.lower()  # lowercase just in case

    wx_words = rephrase_weather(temp, humid, wind_cond, rain_status, desc)

    # rain first — because people ask this the most
    if any(word in q for word in ["rain", "wet", "shower", "precipitation"]):
        if rain_status == "rain":
            return random.choice([
                "Yeah, it's likely to rain.",
                "Probably some rain on the way.",
                "Rain is expected, yep."
            ])
        return random.choice([
            "Nope, doesn't look rainy.",
            "Dry skies today probably.",
            "No sign of rain."
        ])
    
    # wind? maybe
    if "wind" in q or "breeze" in q or "gust" in q:
        if wind_cond == "windy":
            return random.choice([
                "Yes, it'll be windy.",
                "Winds are up today.",
                "Expect breezy conditions."
            ])
        return "Nope, calm winds."  # short one here

    # checking temp-related stuff
    if any(x in q for x in ["cold", "hot", "warm", "freez", "chill"]):
        # could definitely optimize this mess
        if "cold" in q:
            if temp == "cold":
                return random.choice([
                    "Yup, it's going to be cold.",
                    "Cold day ahead.",
                    "Yep, chilly for sure."
                ])
            return "Nah, not really cold."

        if "hot" in q:
            if temp == "hot":
                return "Yeah, hot one coming up."
            else:
                return "Not hot-hot. Kinda moderate."

        # fallback for warm/chill etc.
        if temp in ["hot", "cold"]:
            return f"Yes, it's gonna be {wx_words['temp_phrase']}."
        return "Nope, temps are mild."

    # let's talk humidity now
    if "humid" in q or "muggy" in q or "sticky" in q:
        if humid == "humid":
            return random.choice([
                "Yep, muggy air today.",
                "Humidity's up — it's sticky.",
                "Definitely humid."
            ])
        return "No, air should be dry-ish."

    # vague sky questions
    if any(x in q for x in ["sun", "cloud", "clear", "fog", "storm"]):
        if any(term in desc.lower() for term in ["sun", "cloud", "clear", "fog", "storm"]):
            return f"Yes, should be {desc.lower()}."
        else:
            try:
                guess = q.split()[-1].strip("?")
            except:
                guess = "that"
            return f"No, not {guess}."

    # no clue what's being asked
    return "No idea, could you rephrase that?"

def generate_response(intent, weather_conditions, user_query=None, user_preferred_clothing=None):
    # unpacked like a weather burrito 🌯
    temp, humidity, wind, rain, condition = weather_conditions
    phrased = rephrase_weather(temp, humidity, wind, rain, condition)

    # Forecast intent — give the people what they want 🗣️
    if intent == "weather_forecast":
        template = random.choice(templates["weather_forecast"])
        return template.format(**phrased)

    # Clothing recs 👕🧥 — what's the vibe?
    elif intent == "clothing_recommendation":
        levels = (
            TEMP_LEVELS.get(temp, 1),  # defaulting to 1 like a lazy dev 😅
            HUMIDITY_LEVELS.get(humidity, 0),
            WIND_LEVELS.get(wind, 0),
            RAIN_LEVELS.get(rain, 0)
        )
        return suggest_outfit(*levels)

    # Yes/no about what to wear 🤔
    elif intent == "yes_no_clothing":
        return respond_yes_no(user_preferred_clothing, weather_conditions, phrased)

    # Yes/no weather answer 🌦️
    elif intent == "yes_no_weather":
        if user_query is None:
            return "Could you rephrase your weather question?"  # be nice 💬
        print("User query:", user_query)  # lol forgot to remove this 🐛
        return generate_weather_yes_no(user_query, weather_conditions)

    # fallback case — shrug 🧠
    return "I'm not sure how to respond to that."