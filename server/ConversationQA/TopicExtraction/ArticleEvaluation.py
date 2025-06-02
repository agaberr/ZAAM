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
    stopWords = set(stopwords.words('english'))
    tk =[]
    for word in tokens:
        if word not in stopWords:
            tk.append(word)
   
    lemmatizer = WordNetLemmatizer()
    tkon=[]
    for word in tk:
        tkon.append(lemmatizer.lemmatize(word))
  
    return ' '.join(tkon)


def article_contains_answer(article, question, threshold=0.30):

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
    
    ################################## APPLYING TF-IDF VECTORIZATION ##################################
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    

            ## bageb matrices
    questionVector = tfidf_matrix[0]
    sentence_vectors = tfidf_matrix[1:]
    

    ########## bageb words fi user query ######
    ##### b3d kda hashouf similarities ######
    words = []
    for w in question.split(" "):
        if w.lower() not in stopwords.words("english"):
            if w.lower().isalpha():
                w = w.lower()
                if w not in words:
                    words.append(w)
    words = set(words)
    
    # Calculate similarity for each sentence in the article
    similarities = []
    for i, sentence in enumerate(sentences):
            ### 3ayez ashoud ad eh similarity ben el question w el sentence

        similarity = cosine_similarity(questionVector, sentence_vectors[i])[0][0]
        
            ## lw fi similarity hyakhoud bonus lw kda laa
        sentence_words = set([word.lower() for word in re.findall(r'\b\w+\b', sentence)])
        Keyword = sentence_words.intersection(words)
        Kbonus = min(0.15, len(Keyword) * 0.05)  # Cap the bonus at 0.15
        
   
        Ebonus = 0
        for word in words:
            if len(word) > 4 and word in sentence.lower():
                Ebonus += 0.03
        Ebonus = min(0.10, Ebonus)  
        
                
        ### sum bonus scoress

        res = similarity + Kbonus + Ebonus 
        
        similarities.append((sentence, res))
        
     
    
    best_match = max(similarities, key=lambda x: x[1])
    best_sentence, best_score = best_match
    
    
    HaveAnswer = best_score >= threshold
    
    
    return HaveAnswer, best_sentence, best_score
