from typing import List, Tuple, Dict, Any
import openai

# Set up your API key
openai.api_key = "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"


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

def create_generic_prompt(entities: List[str], topics: List[str],userQuery:str) -> str:
    # If we have both entities and topics
    if entities and topics:
        topic_phrase = f"about {', '.join(topics)} related to"
    # If we only have entities but no topics
    elif entities and not topics:
        topic_phrase = "about"
    # If we only have topics but no entities (unlikely but handled)
    elif topics and not entities:
        topic_phrase = f"about {', '.join(topics)}"
    # Fallback for empty inputs
    else:
        return "Please provide a comprehensive overview of the requested subject."
    
    entities_phrase = " and ".join(entities)
    
    prompt = f"""Write a comprehensive, well-organized passage {topic_phrase} {entities_phrase}. The passage should be based on recent news and reliable sources. The output should be raw text with well-structured sentences, avoiding any special formatting, bullet points, or comments. 
            The passage should include these information from user query: {userQuery}.besides to this:
            1. Provide background information and context.
            2. Include key facts, dates, and relevant details from the latest news.
            3. Outline major developments, events, or achievements related to the topic.
            4. Discuss the significance and impact of these developments.
            5. Ensure clarity and logical flow between sentences.

            The response should be detailed and educational, approximately 500 words in length. Do not include citations, URLs, or references—only structured sentences with factual accuracy. 

            Ensure the passage is in plain text format, suitable for natural language processing tasks like cosine similarity analysis.
        """
    return prompt

def detect_question_type(entities: List[str], topics: List[str]) -> str:
    # No specific topics provided - biographical information likely wanted
    if not topics:
        return "general"
    
    # Check for common question topics
    common_topics = {
        "born": "biographical",
        "birth": "biographical",
        "died": "biographical",
        "death": "biographical",
        "war": "conflict",
        "battle": "conflict",
        "history": "historical",
        "discovery": "scientific",
        "invention": "technological",
        "founded": "organizational",
        "created": "creation"
    }
    
    for topic in topics:
        if topic.lower() in common_topics:
            return common_topics[topic.lower()]
    
    # Default to general information
    return "general"

def create_specific_prompt(entities: List[str], topics: List[str], question_type: str,userQuery: str) -> str:

    entities_phrase = " and ".join(entities)
    
    if question_type == "biography" and not topics:
        return f"""Write a comprehensive biographical profile about {entities_phrase}.The passage should be based on recent news and reliable sources. The output should be raw text with well-structured sentences, avoiding any special formatting, bullet points, or comments. 
        The passage should include these information from user query: {userQuery}.besides to this:

        1. Provide birth information, early life, and background
        2. Outline education and formative experiences
        3. Detail major career achievements and milestones
        4. Discuss personal life where appropriate
        5. Explain their significance and legacy
        6. Include accurate dates and key facts

        Format the biography with clear sections, chronological organization, and professional tone.
        The response should be detailed and educational, approximately 500 words in length. Do not include citations, URLs, or references—only structured sentences with factual accuracy. 

        Ensure the passage is in plain text format, suitable for natural language processing tasks like cosine similarity analysis.
        """
    
    elif question_type == "biographical" and topics:
        topics_phrase = ", ".join(topics)
        return f"""Write a detailed response about {topics_phrase} related to {entities_phrase}.The passage should be based on recent news and reliable sources. The output should be raw text with well-structured sentences, avoiding any special formatting, bullet points, or comments. 
          The passage should include these information from user query: {userQuery}.besides to this:

        1. Provide specific information about the {topics_phrase} of {entities_phrase}
        2. Include dates, locations, and context
        3. Explain the significance of this information
        4. Reference reliable sources where appropriate

        Format the response in a clear, factual manner with appropriate detail.
        The response should be detailed and educational, approximately 500 words in length. Do not include citations, URLs, or references—only structured sentences with factual accuracy. 

        Ensure the passage is in plain text format, suitable for natural language processing tasks like cosine similarity analysis.
        """
    
    elif question_type == "conflict":
        return f"""Write a comprehensive analysis of the conflict involving {entities_phrase}.The passage should be based on recent news and reliable sources. The output should be raw text with well-structured sentences, avoiding any special formatting, bullet points, or comments. 
          The passage should include these information from user query: {userQuery}.besides to this:

        1. Provide historical context and background of the conflict
        2. Explain major events and turning points
        3. Discuss the humanitarian impact on all sides
        4. Outline international responses and peace efforts
        5. Address the current status of the situation
        6. Include factual details including key dates, casualties, and diplomatic developments

        Format the passage with clear sections, subheadings, and chronological organization.
        The response should be detailed and educational, approximately 500 words in length. Do not include citations, URLs, or references—only structured sentences with factual accuracy. 

        Ensure the passage is in plain text format, suitable for natural language processing tasks like cosine similarity analysis.
        """
    
    # Default to the generic prompt for other types
    return create_generic_prompt(entities, topics,userQuery)

def process_entity_tuple(entity_tuple: Tuple[List[str], List[str]]) -> Tuple[List[str], List[str]]:
    entities = []
    for entity in entity_tuple[0]:
        # Check if the entity contains multiple words that should be separate entities
        if ' ' in entity and not any(char.isdigit() for char in entity):
            # For names like "mohamed salah" keep them as one entity
            entities.append(entity)
        else:
            entities.append(entity)
    print("entities: ",entity_tuple)
    return entities, entity_tuple[1]

def generate_passage_from_entity_tuple(entity_tuple: Tuple[List[str], List[str]],userQuery:str) -> str:

    # Process the entity tuple
    entities, topics = process_entity_tuple(entity_tuple)
    
    # Detect the question type
    question_type = detect_question_type(entities, topics)
    
    # Create a specific prompt
    prompt = create_specific_prompt(entities, topics, question_type,userQuery)
    
    return prompt

def generate_passage (prompt) -> str:
    response = openai.chat.completions.create(
    model="gpt-4-turbo",  # or "gpt-3.5-turbo"
    messages=[
        {"role": "system", "content": "You are an AI that generates a well-organized passage from a given text. The output should be raw text without any special formatting, styling, or comments."},
        {"role": "user", "content": prompt}
    ],
    )

    output = response.choices[0].message.content

    return output