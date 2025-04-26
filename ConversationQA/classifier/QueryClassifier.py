import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import joblib 


def extract_features(text_series,nlp):
    features = pd.DataFrame(index=text_series.index)  # Ensure indices match
    
    # 1. Basic text features
    features['text_length'] = text_series.apply(len)
    features['word_count'] = text_series.apply(lambda x: len(x.split()))
    features['avg_word_length'] = text_series.apply(lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) > 0 else 0)
    
    # 2. Question-related features
    features['has_question_mark'] = text_series.apply(lambda x: 1 if '?' in x else 0)
    
    # 3. Check for question words
    question_words = ['what', 'who', 'when', 'where', 'why', 'how', 'which', 'whose']
    for word in question_words:
        features[f'starts_with_{word}'] = text_series.apply(
            lambda x: 1 if x.lower().strip().startswith(word) else 0
        )
    
    features['starts_with_question_word'] = features[[f'starts_with_{word}' for word in question_words]].max(axis=1)
    
    # 4. Check for question-asking verbs
    question_verbs = ['can', 'could', 'would', 'will', 'should', 'is', 'are', 'do', 'does', 'did']
    for verb in question_verbs:
        features[f'starts_with_{verb}'] = text_series.apply(
            lambda x: 1 if x.lower().strip().startswith(verb) else 0
        )
    
    features['starts_with_question_verb'] = features[[f'starts_with_{verb}' for verb in question_verbs]].max(axis=1)
    
    # 5. Check for summarization keywords
    summarization_words = ['summarize', 'summary', 'summarization', 'condense', 'shorten', 
                          'brief', 'overview', 'digest', 'recap', 'synopsis', 'tldr', 
                          'key points', 'main points', 'highlight', 'gist', 'bullet']
    
    features['contains_summarization_word'] = text_series.apply(
        lambda x: 1 if any(word in x.lower() for word in summarization_words) else 0
    )
    
    # 6. NLP-based features using spaCy
    # Initialize columns for NLP features to ensure consistent shape
    nlp_feature_columns = ['VERB', 'NOUN', 'ADJ', 'ADV', 'PRON', 'DET', 'ADP', 'NUM', 
                         'first_token_is_verb', 'has_imperative', 'sentence_count']
    for col in nlp_feature_columns:
        features[col] = 0
    
    # Process each text with spaCy
    for idx, text in text_series.items():
        try:
            doc = nlp(text[:5000])  # Limit to 5000 chars to avoid memory issues
            
            # Count different parts of speech
            pos_counts = {
                'VERB': 0, 'NOUN': 0, 'ADJ': 0, 'ADV': 0, 
                'PRON': 0, 'DET': 0, 'ADP': 0, 'NUM': 0
            }
            
            for token in doc:
                if token.pos_ in pos_counts:
                    pos_counts[token.pos_] += 1
            
            # Update feature dataframe with POS counts
            for pos, count in pos_counts.items():
                features.at[idx, pos] = count
            
            # Check if the first token is a verb (common in questions)
            features.at[idx, 'first_token_is_verb'] = 1 if len(doc) > 0 and doc[0].pos_ == 'VERB' else 0
            
            # Check if there's an imperative verb (command) - common in summarization requests
            has_imperative = 0
            if len(doc) > 0 and doc[0].pos_ == 'VERB':
                has_subject = any(token.dep_ == 'nsubj' for token in doc)
                if not has_subject:
                    has_imperative = 1
            
            features.at[idx, 'has_imperative'] = has_imperative
            features.at[idx, 'sentence_count'] = len(list(doc.sents))
            
        except Exception as e:
            print(f"Error processing text at index {idx}: {str(e)}")
            # Keep default values (0) for this document
    
    return features

def predict_query_type(text, nlp,classifier_model,classifier_vectorizer):
    # Extract features
    text_series = pd.Series([text])
    features = extract_features(text_series,nlp)
    
    # Vectorize
    text_vectorized = classifier_vectorizer.fit_transform([text])

    
    # Combine features
    X_combined = np.hstack([
        features.values,
        text_vectorized.toarray()
    ])
    
    # load model classifier_model
    prediction = classifier_model.predict(X_combined)[0]

    # confidence = None
    # if hasattr(classifier_model, 'predict_proba'):
    #     proba = classifier_model.predict_proba(X_combined)[0]
    #     confidence = max(proba)
    
    return prediction
