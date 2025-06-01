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
        Initialize the cognitive game with MongoDB connection
        
        Args:
            mongo_uri: MongoDB connection string
            db_name: Database name
        """
        self.db = db
        self.user_id = user_id
        self.memory_aids_collection = self.db["memory_aids"]
        
    def get_all_people(self):
        """Get all people from the database for the specific user"""
        return list(self.memory_aids_collection.find({"user_id": self.user_id, "type": "person"}))
    
    def get_all_events(self):
        """Get all events from the database for the specific user"""
        return list(self.memory_aids_collection.find({"user_id": self.user_id, "type": "event"}))
    
    def get_all_memory_aids(self):
        """Get all memory aids (people and events) from the database for the specific user"""
        people = self.get_all_people()
        events = self.get_all_events()
        return people + events
    
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
            lambda p: f"What is your relationship with {p.get('title')}?",
            

            
            # Questions combining people
            lambda p: self._generate_multi_people_question(p, people)
        ]
        
        # Select a question type
        question_func = random.choice(question_types)
        return question_func(person)
    
    def _generate_multi_people_question(self, target_person, all_people):
        """Generate a question that involves multiple people"""
        if len(all_people) < 2:
            return f"How long have you known {target_person.get('title')}?"
        
        # Get another random person different from target
        other_people = [p for p in all_people if p["user_id"] != target_person["user_id"]]
        if other_people:
            other_person = random.choice(other_people)
            
            question_types = [
                f"Who did you meet first, {target_person.get('title')} or {other_person.get('title')}?",
                f"who is older, {target_person.get('title')} or {other_person.get('title')}?",
                

            ]
            
            return random.choice(question_types)
        else:
            return f"How long have you known {target_person.get('title')}?"
    
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
            lambda e: f"What happened during the event: {e.get('title')}?",
            lambda e: f"When did the event '{e.get('title')}' occur?",
            
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
            return f"What was the {detail} mentioned in the event '{event.get('title')}'?"
        else:
            return f"Can you describe what happened during '{event.get('title')}'?"
    
    def _generate_multi_event_question(self, target_event, all_events):
        """Generate a question that involves multiple events"""
        if len(all_events) < 2:
            return f"What do you remember most about '{target_event.get('title')}'?"
        
        # Get another random event different from target
        other_events = [e for e in all_events if e["_id"] != target_event["_id"]]
        if other_events:
            other_event = random.choice(other_events)
            
            question_types = [
                f"Which happened first: '{target_event.get('title')}' or '{other_event.get('title')}'?",
                # f"What common elements were there between '{target_event.get('title')}' and '{other_event.get('title')}'?"
            ]
            
            return random.choice(question_types)
        else:
            return f"What do you remember most about '{target_event.get('title')}'?"
    
    # def generate_mixed_question(self):
    #     """Generate a question that combines people and events"""
    #     people = self.get_all_people()
    #     events = self.get_all_events()
        
    #     if not people or not events:
    #         return "Not enough data to generate mixed questions."
        
    #     person = random.choice(people)
    #     event = random.choice(events)
        
    #     question_types = [
    #         f"Was {person.get('title')} present during the event '{event.get('title')}'?",
    #         f"What was {person.get('title')}'s reaction to '{event.get('title')}'?",
    #         f"Did you talk to {person.get('title')} about '{event.get('title')}'?",
    #         f"Did any memorable interaction happen between you and {person.get('title')} during '{event.get('title')}'?"
    #     ]
        
    #     return random.choice(question_types)
    
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
                if person["title"] in question:
                    expected_relation = person.get("description", "").lower()
                    print("calling")
                    similarity = self.calculate_text_similarity(expected_relation, user_answer,relationshipQuestion=True)
                    result["similarity_score"] = similarity
                    
                    # We'll consider it correct if similarity is above 0.7
                    if similarity > 0.6:
                        result["correct"] = True
                        result["feedback"] = f"Correct! Your relationship with {person['title']} is '{expected_relation}'."
                    elif similarity > 0.4:
                        result["correct"] = True
                        result["feedback"] = f"Partially correct. Your relationship with {person['title']} is '{expected_relation}'."
                    else:
                        result["correct"] = False
                        result["feedback"] = f"Not quite. Your relationship with {person['title']} is '{expected_relation}'."
                    result["correct_answer"] = expected_relation
                    break
                    


        #Handle how long have you known 
        elif "how long have you known" in question.lower():
            print("enter how long have you known")
            for person in self.get_all_people():
                if person["title"].lower() in question.lower():
                    expected_duration = person.get("date_met_patient", "")
                    result["person"] = person["title"]
                    today = datetime.today()

                    try:
                        expected_date = datetime.strptime(expected_duration, "%Y-%d-%m")
                        years_known = today.year - expected_date.year
                        if (today.month, today.day) < (expected_date.month, expected_date.day):
                            years_known -= 1  # Adjust if anniversary hasn't occurred yet
                    except Exception as e:
                        result["score"] = 0.0
                        result["feedback"] = f"Invalid stored date format: {e}"
                        result["correct_answer"] = expected_duration
                        break

                    # Try extracting a year, full date, or year duration from user_answer
                    year_match = re.search(r"since\s+(\d{4})", user_answer.lower())
                    full_date_match = re.search(r"since\s+(\d{4}-\d{2}-\d{2})", user_answer.lower())
                    number_match = re.search(r"\b(\d{1,3})\s*(years?|yrs?)\b", user_answer.lower())

                    score = 0.0
                    tolerance = 2  # Allow ±2 years

                    if full_date_match:
                        try:
                            user_date = datetime.strptime(full_date_match.group(1), "%Y-%d-%m")
                            delta_years = abs(user_date.year - expected_date.year)
                            if delta_years == 0:
                                score = 1.0
                                result["feedback"] = f"Correct! You met {person['title']} on {expected_duration}."
                            elif delta_years <= tolerance:
                                score = 1.0 - (0.1 * delta_years)
                                result["feedback"] = f"Almost correct. You met {person['title']} in {expected_date.year}."
                            else:
                                score = max(0.0, 1.0 - (0.1 * delta_years))
                                result["feedback"] = f"You were a bit off. You met {person['title']} in {expected_date.year}."
                            result["score"] = score
                            result["correct_answer"] = expected_duration
                        except ValueError:
                            result["score"] = 0.0
                            result["feedback"] = "Invalid date format. Expected format: YYYY-MM-DD."
                            result["correct_answer"] = expected_duration

                    elif year_match:
                        user_year = int(year_match.group(1))
                        delta_years = abs(user_year - expected_date.year)
                        if delta_years == 0:
                            score = 1.0
                            result["feedback"] = f"Correct! You met {person['title']} in {expected_date.year}."
                        elif delta_years <= tolerance:
                            score = 1.0 - (0.1 * delta_years)
                            result["feedback"] = f"Close. You met {person['title']} in {expected_date.year}."
                        else:
                            score = max(0.0, 1.0 - (0.1 * delta_years))
                            result["feedback"] = f"Not quite. The correct year was {expected_date.year}."
                        result["score"] = score
                        result["correct_answer"] = str(expected_date.year)

                    elif number_match:
                        user_years = int(number_match.group(1))
                        delta_years = abs(user_years - years_known)
                        if delta_years == 0:
                            score = 1.0
                            result["feedback"] = f"Correct! You've known {person['title']} for {years_known} years."
                        elif delta_years <= tolerance:
                            score = 1.0 - (0.1 * delta_years)
                            result["feedback"] = f"Almost right. You've known {person['title']} for {years_known} years."
                        else:
                            score = max(0.0, 1.0 - (0.1 * delta_years))
                            result["feedback"] = f"You were off by a few years. It's actually {years_known} years."
                        result["score"] =score
                        result["correct_answer"] = str(years_known)

                    else:
                        result["score"] = 0.0
                        result["feedback"] = "Sorry, I couldn't understand your answer. Try giving a year or a number of years."
                        result["correct_answer"] = str(years_known)

                    break  # End after first matched person
        # handle Who did you meet first jhon or snow 
        elif "who did you meet first" in question.lower():
            # Extract person names from question
            all_people = self.get_all_people()
            all_names = [person["title"] for person in all_people]
            names_in_question = [name for name in all_names if name in question]
            if len(names_in_question) < 2:
                result["feedback"] = "Please provide two names to compare."
                return result
            
            # Get the first and second person from the database
            first_person = next((p for p in all_people if p["title"] == names_in_question[0]), None)
            second_person = next((p for p in all_people if p["title"] == names_in_question[1]), None)

            if not first_person or not second_person:
                result["feedback"] = "One or both of the people are not found in the database."
                return result
            
            first_meet_date = datetime.strptime(first_person["date_met_patient"], "%Y-%d-%m")
            second_meet_date = datetime.strptime(second_person["date_met_patient"], "%Y-%d-%m")
            
            correct_name = None
            if first_meet_date < second_meet_date:
                correct_name = first_person["title"]
            else:
                correct_name = second_person["title"]

            # User answer can be "John met snow first" or just "John"
            user_answer = user_answer.strip().lower()
            if correct_name.lower() in user_answer:
                result["correct"] = True
                result["feedback"] = f"Correct! You met {correct_name} first."
            else:
                result["correct"] = False
                result["feedback"] = f"Not quite. You met {correct_name} first."
                
            result["correct_answer"] = correct_name
        # Handle who is older questions
        elif "is older" in question:
            # Extract person names from question
            all_people = self.get_all_people()
            all_names = [person["title"] for person in all_people]
            names_in_question = [name for name in all_names if name in question]
            if len(names_in_question) < 2:
                result["feedback"] = "Please provide two names to compare."
                return result
            
            # Get the first and second person from the database
            first_person = next((p for p in all_people if p["title"] == names_in_question[0]), None)
            second_person = next((p for p in all_people if p["title"] == names_in_question[1]), None)

            if not first_person or not second_person:
                result["feedback"] = "One or both of the people are not found in the database."
                return result
            
            first_birth_date = datetime.strptime(first_person["date_of_birth"], "%Y-%d-%m")
            second_birth_date = datetime.strptime(second_person["date_of_birth"], "%Y-%d-%m")
            
            correct_name = None
            if first_birth_date < second_birth_date:
                correct_name = first_person["title"]
            else:
                correct_name = second_person["title"]

            # User answer can be "John is older" or just "John"
            user_answer = user_answer.strip().lower()
            if correct_name.lower() in user_answer:
                result["correct"] = True
                result["feedback"] = f"Correct! {correct_name} is older."
            else:
                result["correct"] = False
                result["feedback"] = f"Not quite. {correct_name} is older."
                
            result["correct_answer"] = correct_name

            


        # Handle event-related questions
        elif "What happened during the event" in question or "Can you describe what happened during" in question or "about the event" in question:
            # Extract event name from question
            for event in self.get_all_events():
                event_name = event.get("title", "")
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
                if event.get("title", "") in question:
                    event_name = event.get("title", "")
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
                    
        elif "when did the event" in question.lower():
            print("enter when did the event")
            for event in self.get_all_events():
                event_name = event.get("title", "")
                if event_name.lower() in question.lower():
                    expected_date_str = event.get("date_of_occurrence", "")
                    result["correct_answer"] = expected_date_str

                    try:
                        expected_date = datetime.strptime(expected_date_str, "%Y-%d-%m")
                    except ValueError:
                        result["correct"] = False
                        result["feedback"] = f"Stored date for '{event_name}' is invalid."
                        break

                    # Try to extract full date or just the year from the user input
                    match_full_date = re.search(r"\d{4}(-\d{2})?(-\d{2})?", user_answer)
                    match_year = re.search(r"\b\d{4}\b", user_answer)
                    
                    if match_full_date:
                        date_str = match_full_date.group()
                        parts = date_str.split("-")

                        while len(parts) < 3:
                            parts.append("01")

                        formatted_date_str = "-".join(parts)
                        user_date = datetime.strptime(formatted_date_str, "%Y-%d-%m")


                    score = 0.0
                    feedback = ""

                    if match_full_date:
                        try:
                            # user_date = datetime.strptime(match_full_date.group(), "%Y-%d-%m")
                            delta_days = abs((user_date - expected_date).days)
                            print("delta_days", delta_days)
                            if delta_days <= 30:
                                score = 1.0
                                feedback = f"Excellent! You remembered the date of '{event_name}' almost exactly."
                            elif delta_days <= 90:
                                score = 0.8
                                feedback = f"Good! You were close on the date of '{event_name}'."
                            else:
                                score = 0.5
                                feedback = f"Your answer is a bit off, but you remembered something about '{event_name}'."

                        except ValueError:
                            feedback = "Couldn't understand the date format you gave."

                    elif match_year:
                        try:
                            user_year = int(match_year.group())
                            year_diff = abs(user_year - expected_date.year)

                            if year_diff == 0:
                                score = 0.9
                                feedback = f"Great! You got the right year for '{event_name}'."
                            elif year_diff == 1:
                                score = 0.7
                                feedback = f"Close! You were just a year off for '{event_name}'."
                            elif year_diff == 2:
                                score = 0.5
                                feedback = f"A bit far off, but at least you remembered roughly when '{event_name}' happened."
                            else:
                                score = 0.0
                                feedback = f"Your answer about '{event_name}' is quite far from the actual date."

                        except Exception:
                            feedback = "Couldn't understand the year format you gave."

                    else:
                        score = 0.0
                        feedback = f"Could not find a date in your answer for '{event_name}'."

                    result["correct"] = score >= 0.5
                    result["feedback"] = feedback
                    result["score"] = round(score, 2)
                    break

            
        
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
        db["memory_aids"].delete_many({})
        
        # Insert sample people
        people = [
            {"title": "John Smith", "phone": "555-1234", "relation": "friend", "user_id": "user1", "type": "person"},
            {"title": "Alice Johnson", "phone": "555-5678", "relation": "coworker", "user_id": "user1", "type": "person"},
            {"title": "Michael Brown", "phone": "555-9012", "relation": "brother", "user_id": "user1", "type": "person"}
        ]
        db["memory_aids"].insert_many(people)
        
        # Insert sample events
        events = [
            {
                "title": "A visit to Paris",
                "description": "I visited Paris in 2020 and bought a souvenir Eiffel Tower",
                "user_id": "user1",
                "type": "event"
            },
            {
                "title": "Birthday party",
                "description": "We celebrated at the beach restaurant last summer",
                "user_id": "user1",
                "type": "event"
            },
            {
                "title": "Conference presentation",
                "description": "I presented my research on cognitive psychology to a room of experts",
                "user_id": "user1",
                "type": "event"
            }
        ]
        db["memory_aids"].insert_many(events)
    
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


