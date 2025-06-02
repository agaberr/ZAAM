import json
import os
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Set up root directory (ugh, relative paths... but fine for now)
WEATHER_DIR = os.path.dirname(os.path.abspath(__file__))

# Load cleaned dataset (fixed typos manually last night)
with open(os.path.join(WEATHER_DIR, "weather_fixed.json"), "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Using standard BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# All the intents our bot is trained on
intent_labels = ["weather_forecast", "yes_no_weather", "clothing_recommendation", "yes_no_clothing"]
intent2idx = {intent: i for i, intent in enumerate(intent_labels)}

# BIO tagging for entity recognition — a bit verbose but needed
entity_labels = [
    "O", "B-condition", "I-condition", "B-outfit", "I-outfit",
    "B-address", "I-address", "B-temperature", "I-temperature",
    "B-date-time", "I-date-time"
]
entity2idx = {label: idx for idx, label in enumerate(entity_labels)}

# Function to generate BIO tags from entity dict (can get messy with token mismatches)
def get_bio_tags(tokens, entities=None):
    tags = ["O"] * len(tokens)
    if entities:
        for ent_type, ent_val in entities.items():
            if isinstance(ent_val, list):
                for item in ent_val:
                    ent_tokens = tokenizer.tokenize(item)
                    for i, tok in enumerate(ent_tokens):
                        tag = f"B-{ent_type}" if i == 0 else f"I-{ent_type}"
                        if tag in entity2idx:
                            try:
                                idx = tokens.index(tok)
                                tags[idx] = tag
                            except ValueError:
                                # Happens if tokenization mismatch; just skip it
                                pass
            else:
                ent_tokens = tokenizer.tokenize(ent_val)
                for i, tok in enumerate(ent_tokens):
                    tag = f"B-{ent_type}" if i == 0 else f"I-{ent_type}"
                    if tag in entity2idx:
                        try:
                            idx = tokens.index(tok)
                            tags[idx] = tag
                        except ValueError:
                            pass  # yeah, not every token matches perfectly
    return [entity2idx[tag] for tag in tags]

# Preparing the data — tokenizing, tagging, labeling
sentences = []
intent_targets = []
entity_targets = []

for entry in tqdm(raw_data):
    toks = tokenizer.tokenize(entry["query"])
    intent_ids = [intent2idx[i] for i in entry["intents"]]  # assuming multiple intents possible
    entity_ids = get_bio_tags(toks, entry.get("entities"))
    sentences.append(toks)
    intent_targets.append(intent_ids)
    entity_targets.append(entity_ids)

# Split into train/test (fixed random seed for reproducibility)
train_x, test_x, train_y, test_y, train_ents, test_ents = train_test_split(
    sentences, intent_targets, entity_targets, test_size=0.1, random_state=42
)

# Dataset class for PyTorch training — tried to keep it modular
class WeatherDataset(Dataset):
    def __init__(self, texts, intents, entities, intent2idx, entity2idx, max_length=75):
        self.texts = texts
        self.intents = intents
        self.entities = entities
        self.intent2idx = intent2idx
        self.entity2idx = entity2idx
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.max_length = max_length  # could maybe increase for longer questions?

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.texts[idx]
        intent_labels = self.intents[idx]
        intent_tensor = torch.zeros(len(self.intent2idx), dtype=torch.float)

        for i in intent_labels:
            intent_tensor[i] = 1  # one-hot-ish multi-label setup

        # fallback to "O" if no entities exist (rare)
        ent_ids = self.entities[idx] if self.entities[idx] else [self.entity2idx["O"]] * len(tokens)

        # Encode text into input tensors
        encoded = self.tokenizer.encode_plus(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Pad or trim entity labels to match input length
        padded_entities = ent_ids + [self.entity2idx["O"]] * (self.max_length - len(ent_ids))
        padded_entities = padded_entities[:self.max_length]

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "intent": intent_tensor,
            "entity_labels": torch.tensor(padded_entities, dtype=torch.long)
        }
    

def classify_weather(temp, humidity, wind_kph, chance_of_rain, is_raining, condition):
    results = {}

    # TEMP 🌡️
    if temp >= 31:
        results["t"] = "hot"
    elif temp < 15:
        results["t"] = "cold"
    else:
        results["t"] = "moderate"

    # HUMIDITY 💧
    if humidity > 69:
        results["humid"] = "humid"
    else:
        results["humid"] = "not humid"

    # WIND 🍃 — no breeze category this time
    if wind_kph > 20:
        results["wind"] = "windy"
    else:
        results["wind"] = "not windy"

    # RAIN ☔ — simplified logic
    if is_raining or chance_of_rain > 50:
        results["rain"] = "rain"
    else:
        results["rain"] = "not rain"

    results["condition"] = condition  # match the caller’s param

    return (
        results["t"],
        results["humid"],
        results["wind"],
        results["rain"],
        results["condition"]
    )
