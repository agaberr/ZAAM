import datetime
from typing import List, Tuple, Dict, Any
import openai


def ExtractTopic(predictions, important_keywords):
    topics = []
    important_words = []
    current_topic = []
    current_tag = None
    
    for word, tag in predictions:
        # Check if the word is in the important keywords list
        if word.lower() in important_keywords:
            important_words.append(word)
        
        # Handle topic extraction
        if tag.startswith("B-"):
            # If a new topic starts, save the previous one (if any)
            if current_topic:
                topics.append(" ".join(current_topic))
                current_topic = []
            current_topic.append(word)
            current_tag = tag
        elif tag.startswith("I-"):
            # If it's part of the current topic, add the word
            if current_topic:
                current_topic.append(word)
            else:
                # If there's no current topic, treat it as a new topic
                current_topic = [word]
                current_tag = tag
        else:
            # If it's "O", save the current topic (if any)
            if current_topic:
                topics.append(" ".join(current_topic))
                current_topic = []
            current_tag = None

    if current_topic:
        topics.append(" ".join(current_topic))
    
    return topics, important_words

