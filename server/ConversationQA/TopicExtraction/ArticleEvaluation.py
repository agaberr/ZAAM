import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

'''
Scoring Article Criteria:
    Base score from TF-IDF cosine similarity
    Keyword matching bonus for question terms appearing in potential answers
    Entity recognition bonus for significant words
    Question type analysis with specialized pattern matching (for when/where/who/how many questions)
    
'''

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)


def article_contains_answer(article, question, threshold=0.30, debug=False):
    # Clean and preprocess the text
    article = re.sub(r'\s+', ' ', article).strip()
    question = re.sub(r'\s+', ' ', question).strip()
    
    # Check if article or question is empty
    if not article or not question:
        return False, "", 0.0
    
    # Split article into sentences
    sentences = sent_tokenize(article)
    if not sentences:
        return False, "", 0.0

    # Preprocess question
    preprocessed_question = preprocess_text(question)
    
    # Preprocess sentences
    preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]
    
    # Create a list containing both question and sentences for vectorization
    all_texts = [preprocessed_question] + preprocessed_sentences
    
    # Create TF-IDF vectorizer and transform texts
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Get question and sentence vectors
    question_vector = tfidf_matrix[0]
    sentence_vectors = tfidf_matrix[1:]
    
    # Extract question keywords for additional matching
    question_words = set([word.lower() for word in re.findall(r'\b\w+\b', question) 
                      if word.lower() not in stopwords.words('english')])
    
    # Calculate similarity for each sentence in the article
    similarities = []
    for i, sentence in enumerate(sentences):
        # Calculate cosine similarity between question and sentence vectors
        similarity = cosine_similarity(question_vector, sentence_vectors[i])[0][0]
        
        # Bonus for question keyword matches
        sentence_words = set([word.lower() for word in re.findall(r'\b\w+\b', sentence)])
        keyword_matches = sentence_words.intersection(question_words)
        keyword_bonus = min(0.15, len(keyword_matches) * 0.05)  # Cap the bonus at 0.15
        
        # Named entity bonus
        entity_bonus = 0
        for word in question_words:
            if len(word) > 4 and word in sentence.lower():  # Focus on significant words
                entity_bonus += 0.03
        entity_bonus = min(0.10, entity_bonus)  # Cap the bonus
        
        # Question type analysis
        # Enhance score for sentences more likely to contain answers based on question type
        question_type_bonus = 0
        question_lower = question.lower()
        if any(q in question_lower for q in ['when', 'what time', 'what date']):
            # Date/time questions - look for date/time patterns
            if re.search(r'\b(in|on|at)\s+\d{4}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{1,2}(st|nd|rd|th)\b|\b(january|february|march|april|may|june|july|august|september|october|november|december)\b|\b(jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b', sentence.lower()):
                question_type_bonus = 0.05
        elif any(q in question_lower for q in ['where', 'location', 'place']):
            # Location questions - look for location indicators
            if re.search(r'\b(in|at|on|near|from)\s+[A-Z][a-z]+\b|\b[A-Z][a-z]+\s+(city|state|country|continent|region|area)\b', sentence):
                question_type_bonus = 0.05
        elif any(q in question_lower for q in ['who', 'person', 'people']):
            # Person questions - look for proper nouns that might be names
            if re.search(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', sentence):
                question_type_bonus = 0.05
        elif any(q in question_lower for q in ['how many', 'count', 'number']):
            # Numerical questions - look for numbers
            if re.search(r'\b\d+\b', sentence):
                question_type_bonus = 0.05
                
        # Adjust similarity score
        adjusted_similarity = similarity + keyword_bonus + entity_bonus + question_type_bonus
        
        similarities.append((sentence, adjusted_similarity))
        
        if debug:
            print(f"Sentence: {sentence}")
            print(f"Raw TF-IDF similarity: {similarity:.4f}")
            print(f"Keyword bonus: {keyword_bonus:.4f}")
            print(f"Entity bonus: {entity_bonus:.4f}")
            print(f"Question type bonus: {question_type_bonus:.4f}")
            print(f"Adjusted similarity: {adjusted_similarity:.4f}")
            print("-" * 80)
    
    best_match = max(similarities, key=lambda x: x[1])
    best_sentence, best_score = best_match
    
    # Check if any sentence exceeds the threshold
    contains_answer = best_score >= threshold
    
    if len(sentences) > 1:
        # Create adjacent sentence pairs
        paired_sentences = []
        paired_indices = []
        
        for i in range(len(sentences) - 1):
            paired_sentences.append(sentences[i] + " " + sentences[i+1])
            paired_indices.append((i, i+1))
        
        # Preprocess paired sentences
        preprocessed_pairs = [preprocess_text(pair) for pair in paired_sentences]
        
        # Add to the existing corpus for vectorization
        all_texts_with_pairs = all_texts + preprocessed_pairs
        
        # Re-vectorize including the pairs
        pair_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
        pair_tfidf_matrix = pair_vectorizer.fit_transform(all_texts_with_pairs)
        
        # Get question and paired sentence vectors
        pair_question_vector = pair_tfidf_matrix[0]
        pair_sentence_vectors = pair_tfidf_matrix[len(all_texts):]
        
        # Calculate similarities for paired sentences
        pair_similarities = []
        for i, pair_text in enumerate(paired_sentences):
            # Calculate cosine similarity
            pair_similarity = cosine_similarity(pair_question_vector, pair_sentence_vectors[i])[0][0]
            
            # Apply similar bonuses as for individual sentences
            pair_words = set([word.lower() for word in re.findall(r'\b\w+\b', pair_text)])
            pair_keyword_matches = pair_words.intersection(question_words)
            pair_keyword_bonus = min(0.15, len(pair_keyword_matches) * 0.05)
            
            pair_entity_bonus = 0
            for word in question_words:
                if len(word) > 4 and word in pair_text.lower():
                    pair_entity_bonus += 0.03
            pair_entity_bonus = min(0.10, pair_entity_bonus)
            
            # Question type analysis for pairs
            question_type_bonus = 0
            question_lower = question.lower()
            if any(q in question_lower for q in ['when', 'what time', 'what date']):
                if re.search(r'\b(in|on|at)\s+\d{4}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{1,2}(st|nd|rd|th)\b|\b(january|february|march|april|may|june|july|august|september|october|november|december)\b|\b(jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b', pair_text.lower()):
                    question_type_bonus = 0.05
            elif any(q in question_lower for q in ['where', 'location', 'place']):
                if re.search(r'\b(in|at|on|near|from)\s+[A-Z][a-z]+\b|\b[A-Z][a-z]+\s+(city|state|country|continent|region|area)\b', pair_text):
                    question_type_bonus = 0.05
            elif any(q in question_lower for q in ['who', 'person', 'people']):
                if re.search(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', pair_text):
                    question_type_bonus = 0.05
            elif any(q in question_lower for q in ['how many', 'count', 'number']):
                if re.search(r'\b\d+\b', pair_text):
                    question_type_bonus = 0.05
                    
            # Adjust similarity score for the pair
            adjusted_pair_similarity = pair_similarity + pair_keyword_bonus + pair_entity_bonus + question_type_bonus
            
            pair_similarities.append((pair_text, adjusted_pair_similarity))
            
            if debug and adjusted_pair_similarity > best_score:
                print(f"Paired sentences: {pair_text}")
                print(f"Raw TF-IDF similarity: {pair_similarity:.4f}")
                print(f"Keyword bonus: {pair_keyword_bonus:.4f}")
                print(f"Entity bonus: {pair_entity_bonus:.4f}")
                print(f"Question type bonus: {question_type_bonus:.4f}")
                print(f"Adjusted similarity: {adjusted_pair_similarity:.4f}")
                print("-" * 80)
            
            # Update best match if this pair is better
            if adjusted_pair_similarity > best_score:
                best_score = adjusted_pair_similarity
                best_sentence = pair_text
                contains_answer = best_score >= threshold
    
    return contains_answer, best_sentence, best_score
