import os
import re
import json
from datetime import datetime
from dateutil import parser, relativedelta
import dateparser

# Just loading stuff — yeah I hardcoded the paths
BASE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE, "classified_weather_conditions.json"), encoding="utf-8") as f:
    weather_map = json.load(f)

with open(os.path.join(BASE, "classified_clothing_items.json"), encoding="utf-8") as f:
    clothing_map = json.load(f)

# probably should be a class later, but whatever for now
date_words = {"today", "tomorrow", "yesterday", "tonight", "morning", "afternoon", "evening", "night"}
weekdays = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
rel_prefixes = {"next", "last", "this"}
time_re = r"\b\d{1,2}(:\d{2})?\s?(am|pm)?\b"

# Flatten weird nested json structure
weather_terms = set(sum(weather_map.values(), []))

clothes = set()
for k in clothing_map:
    val = clothing_map[k]
    if isinstance(val, dict):
        for sub in val.values():
            clothes |= set(sub)
    elif isinstance(val, list):
        clothes |= set(val)

# This is not robust but it mostly works...
def extract_entities(sentence):
    sentence = sentence.lower().strip()
    sentence = re.sub(r"[^\w\s]", "", sentence)
    tokens = sentence.split()

    weather = [w for w in tokens if w in weather_terms]
    outfits = [w for w in tokens if w in clothes]

    dates = []
    for i, word in enumerate(tokens):
        if word in date_words or word in weekdays:
            if i > 0 and tokens[i - 1] in rel_prefixes:
                dates.append(tokens[i - 1] + " " + word)
            else:
                dates.append(word)

    # naive time regex (will miss edge cases)
    times = []
    try:
        times = [m.group().strip() for m in re.finditer(time_re, sentence)]
    except Exception as e:
        pass  # not worth crashing over

    return {
        "weather_conditions": weather,
        "clothing_items": outfits,
        "dates": dates,
        "times": times
    }

# Parses stuff like "next monday", "last friday", etc. or just uses dateparser
def parse_date(text):
    now = datetime.now()
    text = text.strip().lower()
    parts = text.split()

    if "next" in text and len(parts) > 1:
        try:
            wd = parser.parse(parts[1]).weekday()
            offset = wd - now.weekday()
            if offset <= 0:
                offset += 7
            dt = now + relativedelta.relativedelta(days=offset)
        except:
            dt = None

    elif "last" in text and len(parts) > 1:
        try:
            wd = parser.parse(parts[1]).weekday()
            offset = now.weekday() - wd
            if offset <= 0:
                offset += 7
            dt = now - relativedelta.relativedelta(days=offset)
        except:
            dt = None

    elif "this" in text and len(parts) > 1:
        try:
            wd = parser.parse(parts[1]).weekday()
            offset = wd - now.weekday()
            dt = now + relativedelta.relativedelta(days=offset)
        except:
            dt = None

    elif "tonight" in text:
        dt = now
    else:
        try:
            dt = dateparser.parse(text, settings={'RELATIVE_BASE': now})
        except:
            dt = None

    if dt:
        diff = (dt - now).days
        return dt.strftime("%Y-%m-%d"), diff
    return None, None
