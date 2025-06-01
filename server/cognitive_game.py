# not deployed 

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

class CognitiveGame:
    def __init__(self, db, userid):

        self.db = db
        # self.db = self.client[dbName]
        self.userid = userid
        self.people_collection = self.db["people"] 
        self.memory_aids_collection = self.db["memory_aids"]
        
    def getAllpeople(self):
        people = []
        for doc in self.memory_aids_collection.find({"userid": self.userid, "type": "person"}):
            people.append(doc)
        return people

    
    def getAllevents(self):
      
        return list(self.memory_aids_collection.find({"userid": self.userid, "type": "event"}))

    
    def generatePeoplequestion(self):
        peoplesss = self.getAllpeople()
        
        if not peoplesss:
            return "No people data availabl"
        
        person = random.choice(peoplesss)
        

        def question1(p):
            title = p.get('title')
            title ="What is your relationship with " + title + "?"
            return title


        def question2(p):
            return self.generateManiesPeoplesQuestion(p, peoplesss)

        QT = [question1, question2]


        
        aaaa = random.choice(QT)
        return aaaa(person)
    
    def generateManiesPeoplesQuestion(self, tperson, people):
        if len(people) < 2:
            return f"How long have you known {tperson.get('title')}?" 
        
        notuser = []
        for p in people:
            if p["userid"] != tperson["userid"]:
                notuser.append(p)

        if notuser:
            notperson = random.choice(notuser)
            
            QTT = [
                f"Who did you meet first {tperson.get('title')} or {notperson.get('title')}?",
                f"who is older {tperson.get('title')} or {notperson.get('title')}?",
                

            ]
            
            return random.choice(QTT)
        else:
            aaa= "How long have you known "
            title = tperson.get('title')
            aaa = aaa + title + "?"
            return aaa
    
    def generateEventSquestion(self):
        events = self.getAllevents()
        
        if not events:
            return "No event data available to generate questions."
        
        event = random.choice(events)
        
        def eventsummquestion(e):
            aaa = "What happened during the event: "
            aaa = aaa + e.get('title') + "?"

            return aaa

        def eventdatequestion(e):
            aaa = "When did the event '"
            aaa = aaa + e.get('title') + "' occur?"
            return aaa

        def eventdtailsquestion(e):
            return self.generateDetailsquestion(e)

        def manyevntquestion(e):
            return self.generatemanyevntquestion(e, events)

        QTS = [
            eventsummquestion,
            eventdatequestion,
            eventdtailsquestion,
            manyevntquestion
        ]

        
        questionSFunc = random.choice(QTS)
        return questionSFunc(event)
    ##################################### IMPORTS START #####################################

    def generateDetailsquestion(self, event):
        description = event.get('description', '').lower()
        evntname = event.get('title', 'this event')
        
        evaluativewords = {
            'pos': ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'awesome', 
                        'beautiful', 'nice', 'enjoyable', 'fun', 'pleasant', 'lovely', 'perfect',
                        'satisfying', 'impressive', 'outstanding', 'brilliant', 'marvelous'],
            'neg': ['bad', 'terrible', 'awful', 'horrible', 'disappointing', 'frustrating', 
                        'annoying', 'boring', 'difficult', 'challenging', 'hard', 'tough', 
                        'unpleasant', 'uncomfortable', 'stressful', 'exhausting', 'disappointing'],
            'neut': ['surprising', 'unexpected', 'interesting', 'different', 'strange', 'weird',
                    'unusual', 'unique', 'memorable', 'notable', 'remarkable', 'curious']
        }
        
        foundwords = []
        wordcategories = []
        
        for category, words in evaluativewords.items():
            for word in words:
                if re.search(r'\b' + re.escape(word) + r'\b', description):
                    foundwords.append(word)
                    wordcategories.append(category)
        
        if foundwords:
            chosen_word = random.choice(foundwords)
            chosen_category = wordcategories[foundwords.index(chosen_word)]
            
            if chosen_category == 'pos':
                questions = [
                    f"What specifically was {chosen_word} about '{evntname}'?ll",
                    f"Why was '{evntname}' {chosen_word}?ll",
                    f"What made '{evntname}' {chosen_word} for you?ll",
                    f"Can you elaborate on what was {chosen_word} about '{evntname}'?ll",
                    f"What aspects of '{evntname}' were {chosen_word}?ll"
                ]
            elif chosen_category == 'neg':
                questions = [
                    f"What specifically was {chosen_word} about '{evntname}'?ll",
                    f"Why was '{evntname}' {chosen_word}?ll",
                    f"What made '{evntname}' {chosen_word} for you?ll",
                    f"Can you explain what was {chosen_word} about '{evntname}'?ll",
                    f"What aspects of '{evntname}' were {chosen_word}?ll"
                ]
            else: 
                questions = [
                    f"What was {chosen_word} about '{evntname}'?ll",
                    f"In what way was '{evntname}' {chosen_word}?ll",
                    f"What made '{evntname}' {chosen_word}?ll",
                    f"Can you elaborate on how '{evntname}' was {chosen_word}?ll",
                    f"What specifically was {chosen_word} about '{evntname}'?ll"
                ]
            
            return random.choice(questions)
        
        else:
            description_questions = [
                f"Can you describe what happened during '{evntname}'?ll",
                f"Tell me more about '{evntname}'.ll",
                f"What can you tell me about '{evntname}'?ll",
                f"How did '{evntname}' go?ll",
                f"What was '{evntname}' like?ll",
                f"Can you give me more details about '{evntname}'?ll"
            ]
            
            return random.choice(description_questions)
    ##################################### IMPORTS START #####################################

    def generatemanyevntquestion(self, tevent, events):
        if len(events) < 2:
            aaa="What do you remember most about "
            title=tevent.get('title')
            aaa=aaa+ title + "'?"
            return aaa
        
        notthisevent = []
        for e in events:
            if e["_id"] != tevent["_id"]:
                notthisevent.append(e)

        if notthisevent:
            nothisevent = random.choice(notthisevent)
            
            question_types = [
                f"Which happened first: '{tevent.get('title')}' or '{nothisevent.get('title')}'?",
            ]
            
            return random.choice(question_types)
        else:
            aaa = "What do you remember most about "
            title = tevent.get('title')
            aaa = aaa + title + "'?"
            return aaa
    
   
    
##################################### IMPORTS START #####################################
    ##################################### MODEL SETUP start question checking   #####################################

    
    def checkans(self, question, user_answer):

        if not question or not user_answer:
            return {"feedback": "Missing question or answer", 'correct_answer': 'none', 'similarity_score':None}
        
        result = {
            
            "feedback": "Couldn't determine the correct answer for this question type.",
            "correct_answer": None,
            "similarity_score": None
        }
        print(f"Questionsnowwww: {question}")
        print("relationship in question", "relationship" in question)
        if "relationship" in question:
            print("enter relationship")
            for person in self.getAllpeople():
                if person["title"] in question:
                    expected_relation = person.get("description", "").lower()
                    print("calling")
                    similarity = self.calculate_text_similarity(expected_relation, user_answer,relationshipQuestion=True)
                    result["similarity_score"] = similarity
                    
                    # We'll consider it correct if similarity is above 0.7
                    if similarity == 1:
                        result["feedback"] = f"Correct! Your relationship with {person['title']} is '{expected_relation}'."
                    else:
                        result["feedback"] = f"Not quite. Your relationship with {person['title']} is '{expected_relation}'."
                    result["correct_answer"] = expected_relation
                    break
                    


        elif "how long have you known" in question.lower():
            print("enter how long have you known")
            for person in self.getAllpeople():
                if person["title"].lower() in question.lower():
                    expected_duration = person.get("date_met_patient", "")
                    
                    today = datetime.today()

                    try:
                        expected_date = datetime.strptime(expected_duration, "%Y-%m-%d")
                        years_known = today.year - expected_date.year
                        if (today.month, today.day) < (expected_date.month, expected_date.day):
                            years_known -= 1  
                    except Exception as e:
                        result["similarity_score"] =None
                        result["feedback"] = f"Invalid stored date format: {e}"
                        result["correct_answer"] = expected_duration
                        break

                    year_match = re.search(r"since\s+(\d{4})", user_answer.lower())
                    full_date_match = re.search(r"since\s+(\d{4}-\d{2}-\d{2})", user_answer.lower())
                    number_match = re.search(r"\b(\d{1,3})\s*(years?|yrs?)\b", user_answer.lower())

                    score = 0.0
                    tolerance = 2 

                    if full_date_match:
                        try:
                            user_date = datetime.strptime(full_date_match.group(1), "%Y-%m-%d")
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
                            result["similarity_score"] = score
                            result["correct_answer"] = expected_duration
                        except ValueError:
                            result["similarity_score"] = 0.0
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
                        result["similarity_score"] = score
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
                        result["similarity_score"] =score
                        result["correct_answer"] = str(years_known)

                    else:
                        result["similarity_score"] = 0.0
                        result["feedback"] = "Sorry, I couldn't understand your answer. Try giving a year or a number of years."
                        result["correct_answer"] = str(years_known)

                    break  
        elif "who did you meet first" in question.lower():
            all_people = self.getAllpeople()
            all_names = [person["title"] for person in all_people]
            names_in_question = [name for name in all_names if name in question]
            result["similarity_score"] = None
            if len(names_in_question) < 2:
                result["feedback"] = "Please provide two names to compare"
                return result
            
            def findPerson(name):
                for person in all_people:
                    if person["title"] == name:
                        return person
                return None
            
            first_person = findPerson(names_in_question[0])
            second_person = findPerson(names_in_question[1])

            if not first_person or not second_person:
                result["feedback"] = "One or both of the people are not found in the database."
                return result
            
            first_meet_date = datetime.strptime(first_person["date_met_patient"], "%Y-%m-%d")
            second_meet_date = datetime.strptime(second_person["date_met_patient"], "%Y-%m-%d")
            
            correct_name = None
            if first_meet_date < second_meet_date:
                correct_name = first_person["title"]
            else:
                correct_name = second_person["title"]

            user_answer = user_answer.strip().lower()
            if correct_name.lower() in user_answer:
                result["feedback"] = f"Correct! You met {correct_name} first."
                result["similarity_score"] = 1
            else:
                result["feedback"] = f"Not quite. You met {correct_name} first."
                result["similarity_score"] =0
                
            result["correct_answer"] = correct_name
        elif "is older" in question:
            all_people = self.getAllpeople()
            all_names = [person["title"] for person in all_people]
            names_in_question = [name for name in all_names if name in question]
            result["similarity_score"] = None
            if len(names_in_question) < 2:
                result["feedback"] = "Please provide two names to compare."
                return result
            
            def findPerson(name):
                for person in all_people:
                    if person["title"] ==name:
                        return person
                return None
            
            first_person = findPerson(names_in_question[0])
            second_person = findPerson(names_in_question[1])

            if not first_person or not second_person:
                result["feedback"] = "One or both of the people are not found in the database."
                return result
            
            first_birth_date = datetime.strptime(first_person["date_of_birth"], "%Y-%m-%d")
            second_birth_date = datetime.strptime(second_person["date_of_birth"], "%Y-%m-%d")
            
            correct_name = None
            if first_birth_date < second_birth_date:
                correct_name = first_person["title"]
            else:
                correct_name = second_person["title"]

            user_answer = user_answer.strip().lower()
            if correct_name.lower() in user_answer:
                result["feedback"] = f"Correct! {correct_name} is older."
                result["similarity_score"] = 1
            else:
                result["feedback"] = f"Not quite. {correct_name} is older."
                result["similarity_score"] = 0
                
            result["correct_answer"] = correct_name

            


        elif "What happened during" in question or "Can you describe what" in question or "about the" in question or "in the" in question or "?ll" in question or ".ll" in question or "remember most about" in question:
            print("enter what happened during the event")
            result["similarity_score"] = None
            for event in self.getAllevents():
                evntname = event.get("title", "")
                if evntname in question:
                    expected_desc = event.get("description", "")
                    similarity = self.calculate_text_similarity(expected_desc, user_answer,relationshipQuestion=False)
                    result["similarity_score"] = similarity
                
                    
                    if similarity >0.4:
                        result["feedback"] = f"Excellent! Your description of '{evntname}' matches closely with the recorded details."
                    elif similarity >0.25:
                        result["feedback"] = f"Good! Your description of '{evntname}' contains many key elements, though some details differ."
                    elif similarity >0.1:
                        result["feedback"] = f"Partially correct. Your description of '{evntname}' includes some elements but misses others."
                    else:
                        result["feedback"] = f"Your description of '{evntname}' doesn't match the recorded details very well."
                    
                    result["correct_answer"] = expected_desc
                    break
       
                    
        elif "when did the event" in question.lower():
            print("enter when did the event")
            for event in self.getAllevents():
                evntname = event.get("title", "")
                if evntname.lower() in question.lower():
                    expected_date_str = event.get("date_of_occurrence", "")
                    result["correct_answer"] = expected_date_str

                    try:
                        expected_date = datetime.strptime(expected_date_str, "%Y-%m-%d")
                    except ValueError:
                        result["feedback"] = f"Stored date for '{evntname}' is invalid."
                        break

                    match_full_date = re.search(r"\d{4}(-\d{2})?(-\d{2})?", user_answer)
                    match_year = re.search(r"\b\d{4}\b", user_answer)
                    
                    if match_full_date:
                        date_str = match_full_date.group()
                        parts = date_str.split("-")

                        while len(parts) < 3:
                            parts.append("01")

                        formatted_date_str = "-".join(parts)
                        user_date = datetime.strptime(formatted_date_str, "%Y-%m-%d")


                    score = 0.0
                    feedback = ""

                    if match_full_date:
                        try:
                            # user_date = datetime.strptime(match_full_date.group(), "%Y-%m-%d")
                            deltaDays = abs((user_date - expected_date).days)
                            print("deltaDays", deltaDays)
                            if deltaDays <= 30:
                                score = 1.0
                                feedback = f"Excellent! You remembered the date of '{evntname}' almost exactly."
                            elif deltaDays <= 90:
                                score = 0.8
                                feedback = f"Good! You were close on the date of '{evntname}'."
                            else:
                                score = 0.5
                                feedback = f"Your answer is a bit off, but you remembered something about '{evntname}'."

                        except ValueError:
                            feedback = "Couldn't understand the date format you gave."

                    elif match_year:
                        try:
                            user_year = int(match_year.group())
                            year_diff = abs(user_year - expected_date.year)

                            if year_diff == 0:
                                score = 0.9
                                feedback = f"Great! You got the right year for '{evntname}'."
                            elif year_diff == 1:
                                score = 0.7
                                feedback = f"Close! You were just a year off for '{evntname}'."
                            elif year_diff == 2:
                                score = 0.5
                                feedback = f"A bit far off, but at least you remembered roughly when '{evntname}' happened."
                            else:
                                score = 0.0
                                feedback = f"Your answer about '{evntname}' is quite far from the actual date."

                        except Exception:
                            feedback = "Couldn't understand the year format you gave."

                    else:
                        score = 0.0
                        feedback = f"Could not find a date in your answer for '{evntname}'."

                    result["feedback"] = feedback
                    result["similarity_score"] = round(score, 2)
                    break
        
        elif "which happened first" in question.lower():
            eventss = self.getAllevents()
            all_names = [event["title"] for event in eventss]
            nameInquestion = [name for name in all_names if name in question]
            result["similarity_score"] = None
            
            if len(nameInquestion) < 2:
                result["feedback"] = "Please provide two events to compare."
                return result
            
            def geteventbyTitle(tt):
                for event in eventss:
                    if event["title"] == tt:
                        return event
                return None

            first_event = geteventbyTitle(nameInquestion[0])
            second_event = geteventbyTitle(nameInquestion[1])

            if not first_event or not second_event:
                result["feedback"] = "One or both of the events are not found in the database."
                return result
            
            first_date = datetime.strptime(first_event["date_of_occurrence"], "%Y-%m-%d")
            second_date = datetime.strptime(second_event["date_of_occurrence"], "%Y-%m-%d")
            
            correctName = None
            if first_date < second_date:
                correctName = first_event["title"]
            else:
                correctName = second_event["title"]

            user_answer = user_answer.strip().lower()
            if correctName.lower() in user_answer:
                result["feedback"] = f"Correct! '{correctName}' happened first."
                result["similarity_score"] = 1
            else:
                result["feedback"] = f"Not quite. '{correctName}' happened first."
                result["similarity_score"] = 0
                
            result["correct_answer"] = correctName

            
        
        if not result["correct_answer"]:
            result["feedback"] = "This is a question about your personal memory and experience. " \
                                "The system doesn't have a way to validate the correctness of your answer."
                                
        return result
    ##################################### MODEL SETUP start simlarty  #####################################

    def calculate_text_similarity(self, text1, text2,relationshipQuestion=False):
        """
        a7seb  similarity been  descriptions
        
        variable:
            text1
            text2
            
        rag3:
            float bean 0 and 1 
        """
        relation_ships=['friend', 'buddy', 'pal', 'mate', 'companion', 'partner', 'associate', 'comrade', 'confidant', 'chum'
                        
                        'family', 'brother', 'sister', 'sibling', 'father', 'mother', 'dad', 'mom', 'parent',
                      'son', 'daughter', 'child', 'uncle', 'aunt', 'cousin', 'nephew', 'niece',
                      
                      'colleague', 'coworker', 'co-worker', 'workmate', 'associate', 'partner'
                      'spouse', 'husband', 'wife', 'partner', 'significant other']
        
        if relationshipQuestion:
            if not text1 or not text2:
                return 0.0
            for word in relation_ships:
                if word in text1.lower() and word in text2.lower():
                    return 1.0
            return 0.0




        def preprocess(text):
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            words = text.split()
            stop_words = set(stopwords.words('english'))
            words = [word for word in words if word not in stop_words]

            return words
            
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
            
            if not hasattr(self, 'embedding_model'):

                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            embedding1 = self.embedding_model.encode(text1)
            embedding2 = self.embedding_model.encode(text2)
            
            embedding_sim = self.cosine_similarity(embedding1, embedding2)
            
        except (ImportError, Exception) as e:
            print(f"Warning: Could not use embeddings ({str(e)}). Falling back to word frequency method.")
            
            def getimportantWord(words):
                important = []
                for w in words:
                    if len(w) > 3:
                        important.append(w)
                return important
            
            important1 = getimportantWord(words1)
            important2 = getimportantWord(words2)
            
            counter1 = Counter(important1)
            counter2 = Counter(important2)
            
            all_words = set(important1).union(set(important2))
            
            if not all_words:
                embedding_sim = 0
            else:
                dot_product = sum(counter1.get(word, 0) * counter2.get(word, 0) for word in all_words)
                
                mag1 = sum(counter1.get(word, 0) ** 2 for word in all_words) ** 0.5
                mag2 = sum(counter2.get(word, 0) ** 2 for word in all_words) ** 0.5
                
                embedding_sim = dot_product / (mag1 * mag2) if mag1 * mag2 > 0 else 0
        
        print(f"Embedding similarity: {embedding_sim:.2f}")

        combined_similarity = (0.1 * seq_similarity) + (0.2 * jaccard) + (0.7 * embedding_sim)
        
        return combined_similarity
    
    def cosine_similarity(self, vec1, vec2):

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        magnitude1 = sum(val ** 2 for val in vec1) ** 0.5
        magnitude2 = sum(val ** 2 for val in vec2) ** 0.5
        
        if magnitude1 * magnitude2 == 0:
            return 0
            
        return dot_product / (magnitude1 * magnitude2)


##################################### MODEL SETUP END #####################################

# Example usage
if __name__ == "__main__":
    # Sample data insertion
    def insert_sample_data(db):
        # # Clear existing data
        db["people"].delete_many({})
        db["events"].delete_many({})
        
        # Insert sample people
        people = [
            {"title": "John Smith", "phone": "555-1234", "description": "friend","date_met_patient": "2015-06-01","date_of_birth":"1993-01-30"},
            {"title": "Alice Johnson", "phone": "555-5678", "description": "coworker","date_met_patient": "2016-06-01", "date_of_birth": "1990-05-15"},
            {"title": "Michael Brown", "phone": "555-9012", "description": "brother","date_met_patient": "2015-07-01","date_of_birth": "1992-08-20"},
            {"title": "Emily Davis", "phone": "555-3456", "description": "sister","date_met_patient": "2017-06-01","date_of_birth":  "1995-09-10"},
        ]
        db["people"].insert_many(people)
        
        # Insert sample events
        events = [
            {
                "title": "A visit to Paris",
                "description": "I visited Paris in 2020 and bought a souvenir Eiffel Tower",
                "date_of_occurrence": "2020-06-01"

            },
            {
                "title": "Birthday party",
                "description": "We celebrated at the beach restaurant last summer , it was a great time with friends",
                "date_of_occurrence": "2022-07-15"
            },
            {
                "title": "Conference presentation",
                "description": "I presented my research on cognitive psychology to a room of experts",
                "date_of_occurrence": "2023-03-10"
            }
        ]
        db["events"].insert_many(events)
    
    # Initialize the database connection
    mongo_uri = "mongodb://localhost:27017/"
    dbName = "game"
    client = pymongo.MongoClient(mongo_uri)
    db = client[dbName]
    
    insert_sample_data(db)


    
    # Initialize the game
    userid = "test_user" 

    game = CognitiveGame(db, userid)

    
    # Uncomment to insert sample data
    # insert_sample_data(game.db)
    
    # Example usage
    print("\nDemonstrating text similarity and answer validation:")
    
    # Relationship question example
    test_question1 = "What is your relationship with John Smith?"
    test_answer1 = "He is a good friend of mine"
    expected_relation1 = "friend"
    
    print(f"Question: {test_question1}")
    print(f"User answer: {test_answer1}")
    print(f"Expected relation: {expected_relation1}")
    
    # Calculate similarity manually for demonstration
    validation_result1 = game.checkans(test_question1, test_answer1)
    print("out1 :")
    print(validation_result1)
    print("\n")

  

    
    
    print("\nEvent description question example:")
    test_question2 = "What happened during the event: A visit to Paris?"
    test_answer2 = "We went to Paris and I remember buying a small Eiffel Tower"
    expected_description2 = "I visited Paris in 2020 and bought a souvenir Eiffel Tower"
    
    print(f"Question: {test_question2}")
    print(f"User answer: {test_answer2}")
    print(f"Expected description: {expected_description2}")
    
    
    # Use the validation function
    validation_result2 = game.checkans(test_question2, test_answer2)
    print("out2 :")
    print(validation_result2)
    print("\n")

    
    print("\nExample with less similar description:")
    test_answer2 = "I think we visited somewhere in Europe last year"
    validation_result2_1 = game.checkans(test_question2, test_answer2)
    print("out2_1 :")
    print(validation_result2_1)
    print("\n")
    

    #How long have you known john
    print("\nHow long have you known question example:")
    test_question3 = "How long have you known John Smith?"
    test_answer3 = "I have known him since 2014-06-01"
    expected_duration = "2015-06-01"
    print(f"User answer: {test_answer3}")
    print(f"Expected duration: {expected_duration}")
    validation_result1_1 = game.checkans(test_question3, test_answer3)
    print("out1_1 :")
    print(validation_result1_1)
    print("\n")


    #How long have you known john
    print("\nHow long have you known question example:")
    test_question3 = "How long have you known John Smith?"
    test_answer3 = "I have known him for 8 years"
    expected_duration = "10"
    print(f"User answer: {test_answer3}")
    print(f"Expected duration: {expected_duration}")
    validation_result1_1 = game.checkans(test_question3, test_answer3)
    print("out1_1 :")
    print(validation_result1_1)
    print("\n")


    # Who did you meet first question example
    print("\nWho did you meet first question example:")
    test_question4 = "Who did you meet first, John Smith or Alice Johnson?"
    test_answer4 = "I met John Smith first"
    expected_first_person = "John Smith"
    print(f"Question: {test_question4}")
    print(f"User answer: {test_answer4}")
    validation_result3 = game.checkans(test_question4, test_answer4)
    print("out3 :")
    print(validation_result3)
    print("\n")

    # Who is older question example
    print("\nWho is older question example:")
    test_question5 = "Who is older, John Smith or Alice Johnson?"
    test_answer5 = "John Smith is older"
    expected_older_person = "John Smith"
    print(f"Question: {test_question5}")
    print(f"User answer: {test_answer5}")
    validation_result4 = game.checkans(test_question5, test_answer5)
    print("out4 :")
    print(validation_result4)
    print("\n")


    #When did the event '{e.get('name')}' occur?
    print("\nEvent occurrence question example:")
    test_question6 = "When did the event A visit to Paris occur?"
    test_answer6 = "It happened in 2020-06-01"
    expected_event_date = "2020"
    print(f"Question: {test_question6}")
    print(f"User answer: {test_answer6}")   
    validation_result5 = game.checkans(test_question6, test_answer6)
    print("out5 :")
    print(validation_result5)
    print("\n")
    # When did the event 'A visit to Paris' occur?
    test_answer6_1 = "I think it was in 2019"
    validation_result5_1 = game.checkans(test_question6, test_answer6_1)
    print("out5_1 :")
    print(validation_result5_1)
    print("\n")

    #what was the detail mentioned in the event
    print("\nEvent detail question example:")
    test_question7 = "What was good about Birthday party?"
    test_answer7 = "It was a great time with friends"
    expected_event_detail = "We celebrated at the beach restaurant last summer , it was a great time with friends"
    print(f"Question: {test_question7}")
    print(f"User answer: {test_answer7}")
    validation_result6 = game.checkans(test_question7, test_answer7)
    print("out6 :")
    print(validation_result6)
    print("\n")
    # What was the detail mentioned in the event
    test_answer7_1 = "I think it was a nice party"
    validation_result6_1 = game.checkans(test_question7, test_answer7_1)
    print("out6_1 :")
    print(validation_result6_1)
    print("\n")