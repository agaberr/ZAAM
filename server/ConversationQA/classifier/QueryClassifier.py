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


def extract_features(textSeries,nlp):
    features = pd.DataFrame(index=textSeries.index)  # Ensure indices match
    
    # 1. Basic text features
    features['text_length'] = textSeries.apply(len)
    WORDcount = []
    for text in textSeries:
        words = text.split()
        WORDcount.append(len(words))
    features['word_count'] = WORDcount

    awordLength = []
    for text in textSeries:
        words = text.split()
        if len(words) > 0:
            # lengths = [len(word) for word in words]
            avg=sum([len(word) for word in words])/len(words)
        else:
            avg=0
        awordLength.append(avg)
    features['avg_word_length']=awordLength
    
    features['has_question_mark']=textSeries.apply(lambda x: 1 if '?' in x else 0)
    
    questionWords = ['what', 'who', 'when', 'where', 'why', 'how', 'which', 'whose']
    for word in questionWords:
        # print(word)
        features[f'starts_with_{word}'] =textSeries.apply(
            lambda x: 1 if x.lower().strip().startswith(word) else 0
        )
    
    features['starts_with_question_word'] =features[[f'starts_with_{word}' for word in questionWords]].max(axis=1)
    
    questionVerbs = ['can','could','would','will','should',  'is','are','do','does','did']
    for verb in questionVerbs:
        features[f'starts_with_{verb}']=textSeries.apply(
            lambda x: 1 if x.lower().strip().startswith(verb) else 0
        )
        # print(verb)
    
    features['starts_with_question_verb']= features[[f'starts_with_{verb}' for verb in questionVerbs]].max(axis=1)
    
  
    summarization_words=['summarize','summary','summarization','condense','shorten'
                ,'brief','overview','digest','recap','synopsis','tldr', 
            'key points','main points','highlight','gist','bullet']
    
    features['contains_summarization_word'] =textSeries.apply(
        lambda x: 1 if any(word in x.lower()for word in summarization_words) else 0
    )
    
    
    featuresGrammer = ['VERB'
                       ,'NOUN','ADJ','ADV',  'PRON','DET','ADP','NUM', 
                         'first_token_is_verb',  'has_imperative', 'sentence_count']
    for col in featuresGrammer:
        features[col]=0
    
    # Process each text with spaCy
    for idx, text in textSeries.items():
        try:
            doc= nlp(text[:5000])  # Limit to 5000 chars to avoid memory issues
            
            # Count different parts of speech
            mpCounts = {
                'VERB':0,
            'NOUN':0,'ADJ':0
            ,'ADV':0, 
                'PRON':0,'DET':0,
                  'ADP': 0, 'NUM': 0
            }
            
            for token in doc:
                if token.pos_ in mpCounts:
                    mpCounts[token.pos_]+=1
            
       
            for pos, count in mpCounts.items():
                features.at[idx, pos]= count
            
      
            features.at[idx, 'first_token_is_verb'] = 1 if len(doc) > 0 and doc[0].pos_ == 'VERB' else 0
            
            hasImperative = 0
            if len(doc) > 0 and doc[0].pos_=='VERB':
              
                if not any(token.dep_ == 'nsubj' for token in doc):
                    
                    hasImperative = 1
            
            features.at[idx, 'has_imperative']=hasImperative
            features.at[idx, 'sentence_count']=len(list(doc.sents))
            
        except :
            print("errorrrrrrrrrrrrrr")
    
    return features

def predict_query_type(text, nlp,classifier_model,classifier_vectorizer):
    # Extract features
    text_series =pd.Series([text])
    features =extract_features(text_series,nlp)
    
    # Vectorize
    text_vectorized=classifier_vectorizer.transform([text])

    
    # Combine features
    X_combined = np.hstack([
        features.values,
        text_vectorized.toarray()
    ])
    
    # print("=== DEBUG ===")
    # print("Extracted features shape:", features.shape)
    # print("Text vectorized shape:", text_vectorized.shape)
    # print("Combined shape:", X_combined.shape)
    # print("Model expects:", classifier_model.n_features_in_)
    # print("================")
    # load model classifier_model
    prediction = classifier_model.predict(X_combined)[0]
    
    return prediction
