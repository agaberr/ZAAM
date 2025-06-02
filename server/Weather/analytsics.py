import json
from collections import Counter
import matplotlib.pyplot as plt
import os

# Get current directory path for loading files
WEATHER_DIR = os.path.dirname(os.path.abspath(__file__))

# Load dataset
with open(os.path.join(WEATHER_DIR, "weather_fixed.json"), "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Count intent and entity occurrences
intent_counter = Counter()
entity_counter = Counter()
multi_intent_combinations = Counter()

for sample in dataset:
    # Count intents
    intents = sample.get("intents", [])
    for intent in intents:
        intent_counter[intent] += 1

    # Count multi-intent combinations
    if len(intents) > 1:
        sorted_intents = tuple(sorted(intents))  # Sort to avoid duplicates
        multi_intent_combinations[sorted_intents] += 1

    # Count entities
    entities = sample.get("entities", {})
    if isinstance(entities, dict):  
        for entity_type in entities.keys():
            entity_counter[entity_type] += 1

# Function to plot intent and entity distributions
def plot_intent_entity_distributions(intent_counter, entity_counter):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot intent distribution
    axes[0].bar(intent_counter.keys(), intent_counter.values(), color='skyblue')
    axes[0].set_title("Intent Distribution")
    axes[0].set_xlabel("Intents")
    axes[0].set_ylabel("Frequency")
    axes[0].tick_params(axis='x', rotation=45)

    # Plot entity distribution
    axes[1].bar(entity_counter.keys(), entity_counter.values(), color='lightcoral')
    axes[1].set_title("Entity Distribution")
    axes[1].set_xlabel("Entities")
    axes[1].set_ylabel("Frequency")
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

# Function to plot multi-intent combinations
def plot_multi_intent_combinations(multi_intent_combinations):
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot multi-intent distribution (top 10 most common)
    top_multi_intents = multi_intent_combinations.most_common(10)
    multi_intent_labels = ["\n".join(list(combo)) for combo, _ in top_multi_intents]  # Format labels nicely
    multi_intent_values = [count for _, count in top_multi_intents]

    ax.bar(multi_intent_labels, multi_intent_values, color='mediumseagreen')
    ax.set_title("Top Multi-Intent Combinations")
    ax.set_xlabel("Intent Pairs")
    ax.set_ylabel("Frequency")
    ax.tick_params(axis='x', rotation=0)  # Keep labels readable

    plt.tight_layout()
    plt.show()

# Run functions
plot_intent_entity_distributions(intent_counter, entity_counter)
plot_multi_intent_combinations(multi_intent_combinations)

# Return multi-intent combinations for further analysis
multi_intent_combinations