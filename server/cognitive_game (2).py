
    

import pymongo
import random
from datetime import datetime
import re
import difflib
from collections import Counter
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize

class CognitiveGame:
    def __init__(self, mongo_uri="mongodb://localhost:27017/", db_name="game"):
        """
        Initialize the cognitive game with MongoDB connection
        
        Args:
            mongo_uri: MongoDB connection string
            db_name: Database name
        """
        self.client = pymongo.MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.people_collection = self.db["people"]
        self.events_collection = self.db["events"]
        
    def get_all_people(self):
        """Get all people from the database"""
        return list(self.people_collection.find({}))
    
    def get_all_events(self):
        """Get all events from the database"""
        return list(self.events_collection.find({}))
    
    def generate_people_question(self):
        """Generate a question related to people"""
        people = self.get_all_people()
        
        if not people:
            return "No people data available to generate questions."
        
        # Choose a random person
        person = random.choice(people)
        
        # Different question types
        question_types = [
            # Questions about relationships
            lambda p: f"What is your relationship with {p.get('name')}?",
            

            
            # Questions combining people
            lambda p: self._generate_multi_people_question(p, people)
        ]
        
        # Select a question type
        question_func = random.choice(question_types)
        return question_func(person)
    
    def _generate_multi_people_question(self, target_person, all_people):
        """Generate a question that involves multiple people"""
        if len(all_people) < 2:
            return f"How long have you known {target_person.get('name')}?"
        
        # Get another random person different from target
        other_people = [p for p in all_people if p["_id"] != target_person["_id"]]
        if other_people:
            other_person = random.choice(other_people)
            
            question_types = [
                f"Who did you meet first, {target_person.get('name')} or {other_person.get('name')}?",
                f"Do {target_person.get('name')} and {other_person.get('name')} know each other?"
            ]
            
            return random.choice(question_types)
        else:
            return f"How long have you known {target_person.get('name')}?"
    
    def generate_event_question(self):
        """Generate a question related to events"""
        events = self.get_all_events()
        
        if not events:
            return "No event data available to generate questions."
        
        # Choose a random event
        event = random.choice(events)
        
        # Different question types
        question_types = [
            # Basic recall
            lambda e: f"What happened during the event: {e.get('name')}?",
            lambda e: f"When did the event '{e.get('name')}' occur?",
            
            # Detail-oriented questions
            lambda e: self._generate_detail_question(e),
            
            # Comparing events
            lambda e: self._generate_multi_event_question(e, events)
        ]
        
        # Select a question type
        question_func = random.choice(question_types)
        return question_func(event)
    
    def _generate_detail_question(self, event):
        """Generate a detail-oriented question about an event"""
        description = event.get('description', '')
        
        # Extract potential detail words (nouns) from description
        words = description.split()
        potential_details = [word for word in words if len(word) > 4]
        
        if potential_details:
            detail = random.choice(potential_details)
            return f"What was the {detail} mentioned in the event '{event.get('name')}'?"
        else:
            return f"Can you describe what happened during '{event.get('name')}'?"
    
    def _generate_multi_event_question(self, target_event, all_events):
        """Generate a question that involves multiple events"""
        if len(all_events) < 2:
            return f"What do you remember most about '{target_event.get('name')}'?"
        
        # Get another random event different from target
        other_events = [e for e in all_events if e["_id"] != target_event["_id"]]
        if other_events:
            other_event = random.choice(other_events)
            
            question_types = [
                f"Which happened first: '{target_event.get('name')}' or '{other_event.get('name')}'?",
                f"What common elements were there between '{target_event.get('name')}' and '{other_event.get('name')}'?"
            ]
            
            return random.choice(question_types)
        else:
            return f"What do you remember most about '{target_event.get('name')}'?"
    
    def generate_mixed_question(self):
        """Generate a question that combines people and events"""
        people = self.get_all_people()
        events = self.get_all_events()
        
        if not people or not events:
            return "Not enough data to generate mixed questions."
        
        person = random.choice(people)
        event = random.choice(events)
        
        question_types = [
            f"Was {person.get('name')} present during the event '{event.get('name')}'?",
            f"What was {person.get('name')}'s reaction to '{event.get('name')}'?",
            f"Did you talk to {person.get('name')} about '{event.get('name')}'?",
            f"Did any memorable interaction happen between you and {person.get('name')} during '{event.get('name')}'?"
        ]
        
        return random.choice(question_types)
    
    def generate_random_question(self):
        """Generate a random question selecting from all types"""
        question_generators = [
            self.generate_people_question,
            self.generate_event_question,
            # self.generate_mixed_question
        ]
        
        # Select a random question type
        generator = random.choice(question_generators)
        return generator()
    
    def check_answer(self, question, user_answer):
        """
        Validates user's answer against data from MongoDB
        
        Args:
            question: The question that was asked
            user_answer: The user's response
            
        Returns:
            dict: Result with correctness, feedback, and correct answer if available
        """
        # Parse the question to determine the type and extract key information
        if not question or not user_answer:
            return {"correct": False, "feedback": "Missing question or answer"}
        
        result = {
            "correct": False,
            "feedback": "Couldn't determine the correct answer for this question type.",
            "correct_answer": None,
            "similarity_score": None
        }
        print(f"Questionsnowwww: {question}")
        # Handle people-related questions
        print("relationship in question", "relationship" in question)
        if "relationship" in question:
            print("enter relationship")
            # Extract person name from question
            for person in self.get_all_people():
                if person["name"] in question:
                    expected_relation = person.get("relation", "").lower()
                    print("calling")
                    similarity = self.calculate_text_similarity(expected_relation, user_answer,relationshipQuestion=True)
                    result["similarity_score"] = similarity
                    
                    # We'll consider it correct if similarity is above 0.7
                    if similarity > 0.7:
                        result["correct"] = True
                        result["feedback"] = f"Correct! Your relationship with {person['name']} is '{expected_relation}'."
                    elif similarity > 0.4:
                        result["correct"] = True
                        result["feedback"] = f"Partially correct. Your relationship with {person['name']} is '{expected_relation}'."
                    else:
                        result["correct"] = False
                        result["feedback"] = f"Not quite. Your relationship with {person['name']} is '{expected_relation}'."
                    result["correct_answer"] = expected_relation
                    break
                    
        elif "phone number" in question:
            # Extract person name from question
            for person in self.get_all_people():
                if person["name"] in question:
                    expected_phone = person.get("phone", "")
                    # Allow for variations in format (e.g., with or without dashes)
                    user_digits = ''.join(c for c in user_answer if c.isdigit())
                    expected_digits = ''.join(c for c in expected_phone if c.isdigit())
                    
                    if user_digits == expected_digits or user_answer.strip() == expected_phone:
                        result["correct"] = True
                        result["feedback"] = f"Correct! {person['name']}'s phone number is {expected_phone}."
                    else:
                        result["correct"] = False
                        result["feedback"] = f"Not quite. {person['name']}'s phone number is {expected_phone}."
                    result["correct_answer"] = expected_phone
                    break
                    
        # Handle event-related questions
        elif "What happened during the event" in question or "Can you describe what happened during" in question or "about the event" in question:
            # Extract event name from question
            for event in self.get_all_events():
                event_name = event.get("name", "")
                if event_name in question:
                    expected_desc = event.get("description", "")
                    print("event_name", event_name)
                    # Calculate similarity between user answer and stored description
                    similarity = self.calculate_text_similarity(expected_desc, user_answer,relationshipQuestion=False)
                    result["similarity_score"] = similarity
                    
                    # Different thresholds for different levels of correctness
                    if similarity > 0.7:
                        result["correct"] = True
                        result["feedback"] = f"Excellent! Your description of '{event_name}' matches closely with the recorded details."
                    elif similarity > 0.5:
                        result["correct"] = True
                        result["feedback"] = f"Good! Your description of '{event_name}' contains many key elements, though some details differ."
                    elif similarity > 0.3:
                        result["correct"] = True
                        result["feedback"] = f"Partially correct. Your description of '{event_name}' includes some elements but misses others."
                    else:
                        result["correct"] = False
                        result["feedback"] = f"Your description of '{event_name}' doesn't match the recorded details very well."
                    
                    result["correct_answer"] = expected_desc
                    break
                    
        elif "What was the" in question and "mentioned in the event" in question:
            # Detail-oriented question about an event
            detail_word = None
            event_name = None
            
            # Extract the detail word and event name
            for event in self.get_all_events():
                if event.get("name", "") in question:
                    event_name = event.get("name", "")
                    description = event.get("description", "")
                    
                    # Try to find which detail is being asked about
                    question_parts = question.split("What was the ")[1].split(" mentioned")[0]
                    if question_parts in description:
                        detail_word = question_parts
                        
                    if detail_word:
                        # Find context around the detail word
                        if detail_word.lower() in description.lower():
                            # Calculate similarity between user answer and context around the detail
                            # First, extract the context (sentence or phrase containing the detail)
                            pattern = r"[^.!?]*\b" + re.escape(detail_word) + r"\b[^.!?]*[.!?]"
                            matches = re.findall(pattern, description, re.IGNORECASE)
                            if matches:
                                context = matches[0]
                            else:
                                context = description
                                
                            similarity = self.calculate_text_similarity(context, user_answer,relationshipQuestion=False)
                            result["similarity_score"] = similarity
                            
                            if similarity > 0.5:
                                result["correct"] = True
                                result["feedback"] = f"Correct! You remembered the detail about '{detail_word}' from '{event_name}'."
                            elif similarity > 0.4:
                                result["correct"] = True
                                result["feedback"] = f"Partially correct. Your answer contains some elements about the '{detail_word}' from '{event_name}'."
                            else:
                                result["correct"] = False
                                result["feedback"] = f"Not quite. The detail about '{detail_word}' was different in the description of '{event_name}'."
                            
                            result["correct_answer"] = context
                            break
                    
        # Handle mixed questions (people + events)
        elif "present during the event" in question:
            # This requires information that may not be explicitly stored in the database
            # The system would need a way to connect people to events
            person_name = None
            event_name = None
            
            # Extract person and event names
            for person in self.get_all_people():
                if person["name"] in question:
                    person_name = person["name"]
                    break
                    
            for event in self.get_all_events():
                if event.get("name", "") in question:
                    event_name = event.get("name", "")
                    break
            
            if person_name and event_name:
                # We don't have actual attendance data, so we can only provide feedback
                result["feedback"] = f"For questions about whether {person_name} was present at {event_name}, " \
                                    f"we don't have that information stored directly. " \
                                    f"This is a memory question for you to recall."
                                    
                # We could potentially check if both person and event are mentioned together
                # in some other data source, but that's beyond the current implementation
        
        # For other question types that we don't handle yet
        if not result["correct_answer"]:
            result["feedback"] = "This is a question about your personal memory and experience. " \
                                "The system doesn't have a way to validate the correctness of your answer."
                                
        return result
    
    def calculate_text_similarity(self, text1, text2,relationshipQuestion=False):
        """
        Calculate similarity between two text descriptions
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            float: Similarity score between 0 and 1
        """
        #special handle for relationships so no need to use any of the methods
        relation_ships=['friend', 'buddy', 'pal', 'mate', 'companion', 'partner', 'associate', 'comrade', 'confidant', 'chum'
                        
                        'family', 'brother', 'sister', 'sibling', 'father', 'mother', 'dad', 'mom', 'parent',
                      'son', 'daughter', 'child', 'uncle', 'aunt', 'cousin', 'nephew', 'niece',
                      
                      'colleague', 'coworker', 'co-worker', 'workmate', 'associate', 'partner'
                      'spouse', 'husband', 'wife', 'partner', 'significant other']
        
        if relationshipQuestion:
            # Check if any of the relationship words are in the text
            if not text1 or not text2:
                return 0.0
            for word in relation_ships:
                if word in text1.lower() and word in text2.lower():
                    return 1.0
            return 0.0




        # Preprocess texts: lowercase, remove punctuation, split into words
        def preprocess(text):
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
            words = text.split()
            stop_words = set(stopwords.words('english'))
            words = [word for word in words if word not in stop_words]  # Remove stopwords

            return words
            
        # Process both texts
        print(f"Text1: {text1}")    
        words1 = preprocess(text1)
        words2 = preprocess(text2)
        
        print(f"Words1: {words1}")
        print(f"Words2: {words2}")
        # Method 1: Sequence matching using difflib
        seq_similarity = difflib.SequenceMatcher(None, words1, words2).ratio()
        print(f"Sequence similarity: {seq_similarity:.2f}")
        # Method 2: Word overlap similarity (Jaccard similarity)
        set1 = set(words1)
        set2 = set(words2)
        
        if not set1 or not set2:
            jaccard = 0
        else:
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            jaccard = intersection / union if union > 0 else 0

        print(f"Jaccard similarity: {jaccard:.2f}") 
        # Method 3: Semantic similarity using embeddings
        try:
            
            # Load a pre-trained model (only done once and cached)
            if not hasattr(self, 'embedding_model'):
                # Choose a model that balances performance and speed
                # 'all-MiniLM-L6-v2' is a good lightweight option
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Generate embeddings for both texts
            embedding1 = self.embedding_model.encode(text1)
            embedding2 = self.embedding_model.encode(text2)
            
            # Calculate cosine similarity between embeddings
            embedding_sim = self.cosine_similarity(embedding1, embedding2)
            
        except (ImportError, Exception) as e:
            # Fall back to the original weighted word frequency method if embeddings fail
            print(f"Warning: Could not use embeddings ({str(e)}). Falling back to word frequency method.")
            
            # Original Method 3: Weighted word frequency similarity
            def get_important_words(words):
                return [w for w in words if len(w) > 3]  # Important words are longer than 3 chars
            
            important1 = get_important_words(words1)
            important2 = get_important_words(words2)
            
            # Count frequencies
            counter1 = Counter(important1)
            counter2 = Counter(important2)
            
            # Get all unique words
            all_words = set(important1).union(set(important2))
            
            if not all_words:
                embedding_sim = 0
            else:
                # Sum of products of frequencies
                dot_product = sum(counter1.get(word, 0) * counter2.get(word, 0) for word in all_words)
                
                # Magnitudes
                mag1 = sum(counter1.get(word, 0) ** 2 for word in all_words) ** 0.5
                mag2 = sum(counter2.get(word, 0) ** 2 for word in all_words) ** 0.5
                
                # Cosine similarity
                embedding_sim = dot_product / (mag1 * mag2) if mag1 * mag2 > 0 else 0
        
        print(f"Embedding similarity: {embedding_sim:.2f}")
        # Combine the similarities with weights - give embedding similarity higher weight
        # Sequence matching is good for order, jaccard for overall content, embedding_sim for semantic meaning
        combined_similarity = (0.15 * seq_similarity) + (0.15 * jaccard) + (0.7 * embedding_sim)
        
        return combined_similarity
    
    def cosine_similarity(self, vec1, vec2):
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First vector (numpy array)
            vec2: Second vector (numpy array)
            
        Returns:
            float: Cosine similarity between 0 and 1
        """
        # Dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Magnitudes
        magnitude1 = sum(val ** 2 for val in vec1) ** 0.5
        magnitude2 = sum(val ** 2 for val in vec2) ** 0.5
        
        # Avoid division by zero
        if magnitude1 * magnitude2 == 0:
            return 0
            
        return dot_product / (magnitude1 * magnitude2)




# Example usage
if __name__ == "__main__":
    # Sample data insertion
    def insert_sample_data(db):
        # # Clear existing data
        db["people"].delete_many({})
        db["events"].delete_many({})
        
        # Insert sample people
        people = [
            {"name": "John Smith", "phone": "555-1234", "relation": "friend"},
            {"name": "Alice Johnson", "phone": "555-5678", "relation": "coworker"},
            {"name": "Michael Brown", "phone": "555-9012", "relation": "brother"}
        ]
        db["people"].insert_many(people)
        
        # Insert sample events
        events = [
            {
                "name": "A visit to Paris",
                "description": "I visited Paris in 2020 and bought a souvenir Eiffel Tower"
            },
            {
                "name": "Birthday party",
                "description": "We celebrated at the beach restaurant last summer"
            },
            {
                "name": "Conference presentation",
                "description": "I presented my research on cognitive psychology to a room of experts"
            }
        ]
        db["events"].insert_many(events)
    
    # Initialize the database connection
    mongo_uri = "mongodb://localhost:27017/"
    db_name = "game"
    client = pymongo.MongoClient(mongo_uri)
    db = client[db_name]
    
    insert_sample_data(db)


    
    # Initialize the game
    game = CognitiveGame()
    
    # Uncomment to insert sample data
    # insert_sample_data(game.db)
    
    # Example usage
    print("\nDemonstrating text similarity and answer validation:")
    
    # Relationship question example
    test_question = "What is your relationship with John Smith?"
    test_answer = "He is a good friend of mine"
    expected_relation = "friend"
    
    print(f"Question: {test_question}")
    print(f"User answer: {test_answer}")
    print(f"Expected relation: {expected_relation}")
    
    # Calculate similarity manually for demonstration
    validation_result1 = game.check_answer(test_question, test_answer)
    print("out1 :")
    print(validation_result1)
    print("\n")
    
    print("\nEvent description question example:")
    test_question = "What happened during the event: A visit to Paris?"
    test_answer = "We went to Paris and I remember buying a small Eiffel Tower"
    expected_description = "I visited Paris in 2020 and bought a souvenir Eiffel Tower"
    
    print(f"Question: {test_question}")
    print(f"User answer: {test_answer}")
    print(f"Expected description: {expected_description}")
    
    
    # Use the validation function
    validation_result2 = game.check_answer(test_question, test_answer)
    print("out2 :")
    print(validation_result2)
    print("\n")

    
    print("\nExample with less similar description:")
    test_answer2 = "I think we visited somewhere in Europe last year"
    validation_result2_1 = game.check_answer(expected_description, test_answer2)
    print("out2_1 :")
    print(validation_result2_1)
    print("\n")
    

