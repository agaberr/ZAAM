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
    def __init__(self, db, user_id):
        """
        Initialize the cognitive game with database connection and user ID
        
        Args:
            db: MongoDB database instance
            user_id: The current user's ID to filter memory aids
        """
        self.db = db
        self.user_id = user_id
        self.memory_aids_collection = self.db["memory_aids"]
        
    def get_user_people(self):
        """Get all people memory aids for the current user"""
        return list(self.memory_aids_collection.find({
            "user_id": self.user_id, 
            "type": "person"
        }))
    
    def get_user_places(self):
        """Get all place memory aids for the current user"""
        return list(self.memory_aids_collection.find({
            "user_id": self.user_id, 
            "type": "place"
        }))
    
    def get_user_objects(self):
        """Get all object memory aids for the current user"""
        return list(self.memory_aids_collection.find({
            "user_id": self.user_id, 
            "type": "object"
        }))
    
    def get_user_events(self):
        """Get all event memory aids for the current user"""
        return list(self.memory_aids_collection.find({
            "user_id": self.user_id, 
            "type": "event"
        }))
    
    def get_all_memory_aids(self):
        """Get all memory aids for the current user"""
        return list(self.memory_aids_collection.find({
            "user_id": self.user_id
        }))
    
    def generate_people_question(self):
        """Generate a question related to people"""
        people = self.get_user_people()
        
        if not people:
            return "No people in your memory aids to generate questions about."
        
        # Choose a random person
        person = random.choice(people)
        
        # Different question types
        question_types = [
            # Questions about the person
            lambda p: f"Who is {p.get('title')}?",
            lambda p: f"Tell me about {p.get('title')}.",
            lambda p: f"What do you remember about {p.get('title')}?",
            lambda p: f"Describe your relationship with {p.get('title')}.",
            
            # Questions combining people
            lambda p: self._generate_multi_people_question(p, people)
        ]
        
        # Select a question type
        question_func = random.choice(question_types)
        return question_func(person)
    
    def _generate_multi_people_question(self, target_person, all_people):
        """Generate a question that involves multiple people"""
        if len(all_people) < 2:
            return f"What do you remember about {target_person.get('title')}?"
        
        # Get another random person different from target
        other_people = [p for p in all_people if p["_id"] != target_person["_id"]]
        if other_people:
            other_person = random.choice(other_people)
            
            question_types = [
                f"Who did you meet first, {target_person.get('title')} or {other_person.get('title')}?",
                f"Do {target_person.get('title')} and {other_person.get('title')} know each other?",
                f"Compare your relationships with {target_person.get('title')} and {other_person.get('title')}."
            ]
            
            return random.choice(question_types)
        else:
            return f"What do you remember about {target_person.get('title')}?"
    
    def generate_place_question(self):
        """Generate a question related to places"""
        places = self.get_user_places()
        
        if not places:
            return "No places in your memory aids to generate questions about."
        
        # Choose a random place
        place = random.choice(places)
        
        # Different question types
        question_types = [
            # Basic recall
            lambda p: f"Tell me about {p.get('title')}.",
            lambda p: f"What happened at {p.get('title')}?",
            lambda p: f"Describe your experience at {p.get('title')}.",
            lambda p: f"What do you remember about visiting {p.get('title')}?",
            
            # Detail-oriented questions
            lambda p: self._generate_place_detail_question(p),
            
            # Comparing places
            lambda p: self._generate_multi_place_question(p, places)
        ]
        
        # Select a question type
        question_func = random.choice(question_types)
        return question_func(place)
    
    def _generate_place_detail_question(self, place):
        """Generate a detail-oriented question about a place"""
        description = place.get('description', '')
        
        # Extract potential detail words from description
        words = description.split()
        potential_details = [word for word in words if len(word) > 4]
        
        if potential_details:
            detail = random.choice(potential_details)
            return f"What was the {detail} you mentioned about {place.get('title')}?"
        else:
            return f"Can you describe what you did at {place.get('title')}?"
    
    def _generate_multi_place_question(self, target_place, all_places):
        """Generate a question that involves multiple places"""
        if len(all_places) < 2:
            return f"What do you remember most about {target_place.get('title')}?"
        
        # Get another random place different from target
        other_places = [p for p in all_places if p["_id"] != target_place["_id"]]
        if other_places:
            other_place = random.choice(other_places)
            
            question_types = [
                f"Which did you visit first: {target_place.get('title')} or {other_place.get('title')}?",
                f"Compare your experiences at {target_place.get('title')} and {other_place.get('title')}.",
                f"What's the difference between {target_place.get('title')} and {other_place.get('title')}?"
            ]
            
            return random.choice(question_types)
        else:
            return f"What do you remember most about {target_place.get('title')}?"
    
    def generate_object_question(self):
        """Generate a question related to objects"""
        objects = self.get_user_objects()
        
        if not objects:
            return "No objects in your memory aids to generate questions about."
        
        # Choose a random object
        obj = random.choice(objects)
        
        # Different question types
        question_types = [
            # Basic recall
            lambda o: f"Tell me about {o.get('title')}.",
            lambda o: f"What is {o.get('title')}?",
            lambda o: f"Describe {o.get('title')}.",
            lambda o: f"What do you remember about {o.get('title')}?",
            lambda o: f"Why is {o.get('title')} important to you?",
            
            # Detail-oriented questions
            lambda o: self._generate_object_detail_question(o),
            
            # Comparing objects
            lambda o: self._generate_multi_object_question(o, objects)
        ]
        
        # Select a question type
        question_func = random.choice(question_types)
        return question_func(obj)
    
    def _generate_object_detail_question(self, obj):
        """Generate a detail-oriented question about an object"""
        description = obj.get('description', '')
        
        # Extract potential detail words from description
        words = description.split()
        potential_details = [word for word in words if len(word) > 4]
        
        if potential_details:
            detail = random.choice(potential_details)
            return f"What was the {detail} you mentioned about {obj.get('title')}?"
        else:
            return f"Can you describe the details of {obj.get('title')}?"
    
    def _generate_multi_object_question(self, target_obj, all_objects):
        """Generate a question that involves multiple objects"""
        if len(all_objects) < 2:
            return f"What do you remember most about {target_obj.get('title')}?"
        
        # Get another random object different from target
        other_objects = [o for o in all_objects if o["_id"] != target_obj["_id"]]
        if other_objects:
            other_obj = random.choice(other_objects)
            
            question_types = [
                f"Which do you use more often: {target_obj.get('title')} or {other_obj.get('title')}?",
                f"Compare {target_obj.get('title')} and {other_obj.get('title')}.",
                f"What's the difference between {target_obj.get('title')} and {other_obj.get('title')}?"
            ]
            
            return random.choice(question_types)
        else:
            return f"What do you remember most about {target_obj.get('title')}?"
    
    def generate_event_question(self):
        """Generate a question related to events"""
        events = self.get_user_events()
        
        if not events:
            return "No events in your memory aids to generate questions about."
        
        # Choose a random event
        event = random.choice(events)
        
        # Different question types
        question_types = [
            # Basic recall
            lambda e: f"Tell me about {e.get('title')}.",
            lambda e: f"What happened at {e.get('title')}?",
            lambda e: f"Describe your experience at {e.get('title')}.",
            lambda e: f"What do you remember about {e.get('title')}?",
            lambda e: f"How did you feel during {e.get('title')}?",
            
            # Detail-oriented questions
            lambda e: self._generate_event_detail_question(e),
            
            # Comparing events
            lambda e: self._generate_multi_event_question(e, events)
        ]
        
        # Select a question type
        question_func = random.choice(question_types)
        return question_func(event)
    
    def _generate_event_detail_question(self, event):
        """Generate a detail-oriented question about an event"""
        description = event.get('description', '')
        
        # Extract potential detail words from description
        words = description.split()
        potential_details = [word for word in words if len(word) > 4]
        
        if potential_details:
            detail = random.choice(potential_details)
            return f"What was the {detail} you mentioned about {event.get('title')}?"
        else:
            return f"Can you describe what happened during {event.get('title')}?"
    
    def _generate_multi_event_question(self, target_event, all_events):
        """Generate a question that involves multiple events"""
        if len(all_events) < 2:
            return f"What do you remember most about {target_event.get('title')}?"
        
        # Get another random event different from target
        other_events = [e for e in all_events if e["_id"] != target_event["_id"]]
        if other_events:
            other_event = random.choice(other_events)
            
            question_types = [
                f"Which happened first: {target_event.get('title')} or {other_event.get('title')}?",
                f"Compare your experiences at {target_event.get('title')} and {other_event.get('title')}.",
                f"What's the difference between {target_event.get('title')} and {other_event.get('title')}?",
                f"Which was more memorable: {target_event.get('title')} or {other_event.get('title')}?"
            ]
            
            return random.choice(question_types)
        else:
            return f"What do you remember most about {target_event.get('title')}?"
    
    def generate_mixed_question(self):
        """Generate a question that combines different types of memory aids"""
        people = self.get_user_people()
        places = self.get_user_places()
        objects = self.get_user_objects()
        events = self.get_user_events()
        
        # Create a list of available memory aid types
        available_types = []
        if people:
            available_types.append(('people', people))
        if places:
            available_types.append(('places', places))
        if objects:
            available_types.append(('objects', objects))
        if events:
            available_types.append(('events', events))
        
        # Need at least 2 types for mixed questions
        if len(available_types) < 2:
            return "Not enough different types of memory aids to generate mixed questions."
        
        # Select two random types
        type1, aids1 = random.choice(available_types)
        remaining_types = [t for t in available_types if t[0] != type1]
        type2, aids2 = random.choice(remaining_types)
        
        # Select random items from each type
        aid1 = random.choice(aids1)
        aid2 = random.choice(aids2)
        
        # Generate different question combinations
        question_types = []
        
        # People + Places
        if type1 == 'people' and type2 == 'places':
            question_types = [
                f"Have you been to {aid2.get('title')} with {aid1.get('title')}?",
                f"What did {aid1.get('title')} think about {aid2.get('title')}?",
                f"Did you talk to {aid1.get('title')} about {aid2.get('title')}?",
                f"What memories do you have of {aid1.get('title')} and {aid2.get('title')} together?"
            ]
        elif type1 == 'places' and type2 == 'people':
            question_types = [
                f"Have you been to {aid1.get('title')} with {aid2.get('title')}?",
                f"What did {aid2.get('title')} think about {aid1.get('title')}?",
                f"Did you talk to {aid2.get('title')} about {aid1.get('title')}?",
                f"What memories do you have of {aid2.get('title')} and {aid1.get('title')} together?"
            ]
        
        # People + Objects
        elif type1 == 'people' and type2 == 'objects':
            question_types = [
                f"Did {aid1.get('title')} give you {aid2.get('title')}?",
                f"Have you used {aid2.get('title')} with {aid1.get('title')}?",
                f"What does {aid1.get('title')} think about {aid2.get('title')}?",
                f"Did you share {aid2.get('title')} with {aid1.get('title')}?"
            ]
        elif type1 == 'objects' and type2 == 'people':
            question_types = [
                f"Did {aid2.get('title')} give you {aid1.get('title')}?",
                f"Have you used {aid1.get('title')} with {aid2.get('title')}?",
                f"What does {aid2.get('title')} think about {aid1.get('title')}?",
                f"Did you share {aid1.get('title')} with {aid2.get('title')}?"
            ]
        
        # People + Events
        elif type1 == 'people' and type2 == 'events':
            question_types = [
                f"Was {aid1.get('title')} at {aid2.get('title')}?",
                f"Did you attend {aid2.get('title')} with {aid1.get('title')}?",
                f"What did {aid1.get('title')} think about {aid2.get('title')}?",
                f"How did {aid1.get('title')} react during {aid2.get('title')}?"
            ]
        elif type1 == 'events' and type2 == 'people':
            question_types = [
                f"Was {aid2.get('title')} at {aid1.get('title')}?",
                f"Did you attend {aid1.get('title')} with {aid2.get('title')}?",
                f"What did {aid2.get('title')} think about {aid1.get('title')}?",
                f"How did {aid2.get('title')} react during {aid1.get('title')}?"
            ]
        
        # Places + Objects
        elif type1 == 'places' and type2 == 'objects':
            question_types = [
                f"Did you use {aid2.get('title')} at {aid1.get('title')}?",
                f"Did you see {aid2.get('title')} at {aid1.get('title')}?",
                f"Did you bring {aid2.get('title')} to {aid1.get('title')}?",
                f"Is {aid2.get('title')} from {aid1.get('title')}?"
            ]
        elif type1 == 'objects' and type2 == 'places':
            question_types = [
                f"Did you use {aid1.get('title')} at {aid2.get('title')}?",
                f"Did you see {aid1.get('title')} at {aid2.get('title')}?",
                f"Did you bring {aid1.get('title')} to {aid2.get('title')}?",
                f"Is {aid1.get('title')} from {aid2.get('title')}?"
            ]
        
        # Places + Events
        elif type1 == 'places' and type2 == 'events':
            question_types = [
                f"Did {aid2.get('title')} happen at {aid1.get('title')}?",
                f"Were you at {aid1.get('title')} during {aid2.get('title')}?",
                f"Did you visit {aid1.get('title')} for {aid2.get('title')}?"
            ]
        elif type1 == 'events' and type2 == 'places':
            question_types = [
                f"Did {aid1.get('title')} happen at {aid2.get('title')}?",
                f"Were you at {aid2.get('title')} during {aid1.get('title')}?",
                f"Did you visit {aid2.get('title')} for {aid1.get('title')}?"
            ]
        
        # Objects + Events
        elif type1 == 'objects' and type2 == 'events':
            question_types = [
                f"Did you use {aid1.get('title')} during {aid2.get('title')}?",
                f"Did you get {aid1.get('title')} at {aid2.get('title')}?",
                f"Was {aid1.get('title')} involved in {aid2.get('title')}?"
            ]
        elif type1 == 'events' and type2 == 'objects':
            question_types = [
                f"Did you use {aid2.get('title')} during {aid1.get('title')}?",
                f"Did you get {aid2.get('title')} at {aid1.get('title')}?",
                f"Was {aid2.get('title')} involved in {aid1.get('title')}?"
            ]
        
        if question_types:
            return random.choice(question_types)
        else:
            return f"How are {aid1.get('title')} and {aid2.get('title')} related in your memory?"
    
    def generate_random_question(self):
        """Generate a random question selecting from all types"""
        question_generators = []
        
        # Add generators based on available data
        if self.get_user_people():
            question_generators.append(self.generate_people_question)
        
        if self.get_user_places():
            question_generators.append(self.generate_place_question)
        
        if self.get_user_objects():
            question_generators.append(self.generate_object_question)
        
        if self.get_user_events():
            question_generators.append(self.generate_event_question)
        
        # Add mixed questions if we have at least 2 different types
        available_types = 0
        if self.get_user_people():
            available_types += 1
        if self.get_user_places():
            available_types += 1
        if self.get_user_objects():
            available_types += 1
        if self.get_user_events():
            available_types += 1
            
        if available_types >= 2:
            question_generators.append(self.generate_mixed_question)
        
        if not question_generators:
            return "You don't have any memory aids yet. Add some people, places, objects, or events to your memory aids to play the cognitive game!"
        
        # Select a random question type
        generator = random.choice(question_generators)
        return generator()
    
    def check_answer(self, question, user_answer):
        """
        Validates user's answer against data from memory aids
        
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
        
        print(f"Question: {question}")
        print(f"User answer: {user_answer}")
        
        # Handle people-related questions
        if any(phrase in question.lower() for phrase in ["who is", "tell me about", "remember about", "relationship with"]):
            # Extract person name from question
            for person in self.get_user_people():
                person_name = person.get('title', '')
                if person_name.lower() in question.lower():
                    expected_description = person.get('description', '')
                    print(f"Found person: {person_name}, expected: {expected_description}")
                    
                    similarity = self.calculate_text_similarity(expected_description, user_answer, relationshipQuestion=False)
                    result["similarity_score"] = similarity
                    
                    # Provide feedback based on similarity
                    if similarity > 0.7:
                        result["correct"] = True
                        result["feedback"] = f"Excellent! Your description of {person_name} matches well with what you recorded."
                    elif similarity > 0.5:
                        result["correct"] = True
                        result["feedback"] = f"Good! Your description of {person_name} contains many key elements."
                    elif similarity > 0.3:
                        result["correct"] = True
                        result["feedback"] = f"Partially correct. Your description of {person_name} includes some elements."
                    else:
                        result["correct"] = False
                        result["feedback"] = f"Your description of {person_name} doesn't match what you recorded very well."
                    
                    result["correct_answer"] = expected_description
                    break
                    
        # Handle place-related questions
        elif any(phrase in question.lower() for phrase in ["tell me about", "what happened at", "experience at", "visiting", "remember about"]):
            # Extract place name from question
            for place in self.get_user_places():
                place_name = place.get('title', '')
                if place_name.lower() in question.lower():
                    expected_description = place.get('description', '')
                    print(f"Found place: {place_name}, expected: {expected_description}")
                    
                    similarity = self.calculate_text_similarity(expected_description, user_answer, relationshipQuestion=False)
                    result["similarity_score"] = similarity
                    
                    # Provide feedback based on similarity
                    if similarity > 0.7:
                        result["correct"] = True
                        result["feedback"] = f"Excellent! Your description of {place_name} matches well with what you recorded."
                    elif similarity > 0.5:
                        result["correct"] = True
                        result["feedback"] = f"Good! Your description of {place_name} contains many key elements."
                    elif similarity > 0.3:
                        result["correct"] = True
                        result["feedback"] = f"Partially correct. Your description of {place_name} includes some elements."
                    else:
                        result["correct"] = False
                        result["feedback"] = f"Your description of {place_name} doesn't match what you recorded very well."
                    
                    result["correct_answer"] = expected_description
                    break
        
        # Handle object-related questions
        elif any(phrase in question.lower() for phrase in ["tell me about", "what is", "remember about"]):
            # Extract object name from question
            for object in self.get_user_objects():
                object_name = object.get('title', '')
                if object_name.lower() in question.lower():
                    expected_description = object.get('description', '')
                    print(f"Found object: {object_name}, expected: {expected_description}")
                    
                    similarity = self.calculate_text_similarity(expected_description, user_answer, relationshipQuestion=False)
                    result["similarity_score"] = similarity
                    
                    # Provide feedback based on similarity
                    if similarity > 0.7:
                        result["correct"] = True
                        result["feedback"] = f"Excellent! Your description of {object_name} matches well with what you recorded."
                    elif similarity > 0.5:
                        result["correct"] = True
                        result["feedback"] = f"Good! Your description of {object_name} contains many key elements."
                    elif similarity > 0.3:
                        result["correct"] = True
                        result["feedback"] = f"Partially correct. Your description of {object_name} includes some elements."
                    else:
                        result["correct"] = False
                        result["feedback"] = f"Your description of {object_name} doesn't match what you recorded very well."
                    
                    result["correct_answer"] = expected_description
                    break
        
        # Handle event-related questions
        elif any(phrase in question.lower() for phrase in ["tell me about", "what happened at", "experience at", "remember about"]):
            # Extract event name from question
            for event in self.get_user_events():
                event_name = event.get('title', '')
                if event_name.lower() in question.lower():
                    expected_description = event.get('description', '')
                    print(f"Found event: {event_name}, expected: {expected_description}")
                    
                    similarity = self.calculate_text_similarity(expected_description, user_answer, relationshipQuestion=False)
                    result["similarity_score"] = similarity
                    
                    # Provide feedback based on similarity
                    if similarity > 0.7:
                        result["correct"] = True
                        result["feedback"] = f"Excellent! Your description of {event_name} matches well with what you recorded."
                    elif similarity > 0.5:
                        result["correct"] = True
                        result["feedback"] = f"Good! Your description of {event_name} contains many key elements."
                    elif similarity > 0.3:
                        result["correct"] = True
                        result["feedback"] = f"Partially correct. Your description of {event_name} includes some elements."
                    else:
                        result["correct"] = False
                        result["feedback"] = f"Your description of {event_name} doesn't match what you recorded very well."
                    
                    result["correct_answer"] = expected_description
                    break
        
        # Handle mixed questions or comparative questions
        elif any(phrase in question.lower() for phrase in ["who did you meet first", "which did you visit first", "compare", "difference between"]):
            result["feedback"] = "This is a question about your personal memory and experience. " \
                                "The system doesn't have temporal data to validate the correctness of your answer, " \
                                "but thank you for sharing your memory!"
            result["correct"] = True  # We consider these always correct since we can't validate
                                
        # For other question types that we don't handle yet
        if not result["correct_answer"] and result["correct"] == False:
            result["feedback"] = "This is a question about your personal memory and experience. " \
                                "The system doesn't have a way to validate the correctness of your answer, " \
                                "but thank you for participating!"
            result["correct"] = True  # Consider it correct since we can't validate
                                
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
        db["memory_aids"].delete_many({})
        
        # Insert sample people
        people = [
            {"title": "John Smith", "description": "He is a good friend of mine"},
            {"title": "Alice Johnson", "description": "She is a coworker"},
            {"title": "Michael Brown", "description": "He is a brother"}
        ]
        db["memory_aids"].insert_many(people)
        
        # Insert sample places
        places = [
            {"title": "Paris", "description": "I visited Paris in 2020 and bought a souvenir Eiffel Tower"},
            {"title": "Beach Restaurant", "description": "We celebrated at the beach restaurant last summer"},
            {"title": "Conference", "description": "I presented my research on cognitive psychology to a room of experts"}
        ]
        db["memory_aids"].insert_many(places)
    
    # Initialize the database connection
    mongo_uri = "mongodb://localhost:27017/"
    db_name = "game"
    client = pymongo.MongoClient(mongo_uri)
    db = client[db_name]
    
    insert_sample_data(db)


    
    # Initialize the game
    game = CognitiveGame(db, "user1")
    
    # Uncomment to insert sample data
    # insert_sample_data(game.db)
    
    # Example usage
    print("\nDemonstrating text similarity and answer validation:")
    
    # Relationship question example
    test_question = "Who is John Smith?"
    test_answer = "He is a good friend of mine"
    expected_relation = "He is a good friend of mine"
    
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
    

