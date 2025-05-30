import numpy as np
import re
from collections import defaultdict
import os
import pickle
from pathlib import Path
from gensim.models import Word2Vec

class AIProcessor:
    """the main router"""
    
    def __init__(self, model_path=None):
        self.model = None
        self.category_embeddings = None
        self.similarity_threshold = 0.3
        self.model_loaded = False
        
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "word2vec_model.model")
        
        self.model_path = model_path
        self.load_model()
        
        if not self.model_loaded:
            self.setup_categories_model_not_loaded()

        
    def load_model(self):
            
        try:
            if Path(self.model_path).exists():
                
                if self.model_path.endswith('.model'):
                    self.model = Word2Vec.load(self.model_path)
                    self.create_embeddings()
                    self.model_loaded = True
                    
                # there is an error loading the model
                else:
                    self.model = Word2Vec.load(self.model_path)
                    self.create_embeddings()
                    self.model_loaded = True
                                
            else:
                self.model_loaded = False
                
        except Exception as e:
            self.model_loaded = False
    
    def create_embeddings(self):
        
        seeds = {
            "news": ["news", "report", "headline", "breaking", "article", "story", "media", 
                     "cook", "cooking", "recipe", "food", "kitchen", "chef", "bake", "ingredient",
                     "football", "soccer", "match", "team", "player", "goal", "league", "cup"],
            "weather": ["weather", "temperature", "rain", "snow", "sunny", "cloudy", "storm"],
            "reminder": ["remind", "calendar", "schedule", "event", "appointment", "meeting"]
        }
        
        self.category_embeddings = {}
        
        for category, seed_words in seeds.items():
            valid_seeds = [word for word in seed_words if word in self.model.wv.key_to_index]
            
            if valid_seeds:
                seed_vectors = [self.model.wv[word] for word in valid_seeds]
                category_embedding = np.mean(seed_vectors, axis=0)
                self.category_embeddings[category] = category_embedding
            else:
                # cread random embedding if no valid seed
                vector_size = self.model.wv.vector_size
                self.category_embeddings[category] = np.random.normal(0, 0.1, vector_size)
        
    
    def setup_categories_model_not_loaded(self):
        vector_size = 100
        self.category_embeddings = {
            "news": np.ones(vector_size) / np.sqrt(vector_size),
            "weather": np.ones(vector_size) / np.sqrt(vector_size) * 0.6,
            "reminder": np.ones(vector_size) / np.sqrt(vector_size) * 0.8
        }
            
    def cosine_similarity(self, vec1, vec2):
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def tokenize_words(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9'\s]", " ", text)
        tokens = text.split()
        return tokens
    
    def segment_all_texts(self, text):
      
        #splitters to slit text
        splitters = ['and', 'or', 'but', 'so', 'yet']
        
        pattern = r'(?<=[.!?])\s+(?!(' + '|'.join(splitters) + r')\b)'
        
        sentences = re.split(pattern, text)
        
        splitter_splits = []
        for sentence in sentences:
            parts = re.split(r'\s+and\s+', sentence)
            splitter_splits.extend(parts)
        sentences = splitter_splits
        
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Initialize segments dictionary
        segments = defaultdict(list)
        
        # If no model is loaded, use simple keyword matching
        if not self.model_loaded:
            keys = {
                "greeting": ["hey", "hi", "hello"],
                "news": ["news", "report", "headline", "breaking", "article", "story", "journalist", 
                         "media", "press", "announce", "publish", "who is", "what is",
                         "cook", "cooking", "recipe", "food", "kitchen", "chef", "bake", "ingredient", "meal",
                         "football", "soccer", "match", "team", "player", "goal", "league", "cup", "score"],
                "weather": ["weather", "temperature", "forecast", "rain", "snow", "sunny", "cloudy", 
                           "storm", "wind", "humidity", "cold", "hot", "degrees", "wear"],
                "reminder": ["remind", "calendar", "schedule", "event", "appointment", 
                             "meeting", "reminder", "don't forget", "remember", "plan", 
                             "tomorrow", "next week", "later", "day after tomorrow",
                             "what's on", "what do i have", "what's scheduled", "am i free", 
                             "do i have any", "when is", "at what time"]
            }
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                matched = False
                
                # Check for exact greeting matches to be more specific
                greeting_words = ["hey", "hi", "hello"]
                if any(sentence_lower.strip().startswith(greeting) for greeting in greeting_words):
                    segments["greeting"].append(sentence)
                    matched = True
                
                # Check other categories if not a greeting
                if not matched:
                    # Check weather first for weather-specific queries
                    if any(word in sentence_lower for word in keys["weather"]):
                        segments["weather"].append(sentence)
                        matched = True
                    # Check news (including food and football)
                    elif any(word in sentence_lower for word in keys["news"]):
                        segments["news"].append(sentence)
                        matched = True
                    # Check reminder last
                    elif any(word in sentence_lower for word in keys["reminder"]):
                        segments["reminder"].append(sentence)
                        matched = True
                
                # If no match found, route to news (assuming uncategorized is news)
                if not matched:
                    segments["news"].append(sentence)
                
            return segments
        

        for sentence in sentences:
            words = self.tokenize_words(sentence)
            
            # Check for greetings first with more specific matching
            sentence_lower = sentence.lower()
            greeting_words = ["hey", "hi", "hello"]
            if any(sentence_lower.strip().startswith(greeting) for greeting in greeting_words):
                segments["greeting"].append(sentence)
                continue
            
            avail_tokens = [word for word in words if hasattr(self.model, 'wv') and word in self.model.wv]
            
            if not avail_tokens:
                # Route uncategorized to news
                segments["news"].append(sentence)
                continue
            
            sen_vectors = [self.model.wv[word] for word in avail_tokens]
            sentence_embedding = np.mean(sen_vectors, axis=0)
            
            chosen_cat = None
            chosen_sim = 0
            
            for category, cat_vec in self.category_embeddings.items():
                sim = self.cosine_similarity(sentence_embedding, cat_vec)
                if sim > chosen_sim:
                    chosen_sim = sim
                    chosen_cat = category
            
            if chosen_sim >= self.similarity_threshold:
                segments[chosen_cat].append(sentence)
            else:
                # Route uncategorized to news
                segments["news"].append(sentence)
        
        return segments
    
    def process_text(self, text):

        segments = self.segment_all_texts(text)
        
        responses = {}

        if segments.get("news"):
            responses["news"] = self.process_news(segments["news"])
        
        if segments.get("weather"):
            responses["weather"] = self.process_weather(segments["weather"])
            
        if segments.get("reminder"):
            responses["reminder"] = self.process_reminder(segments["reminder"])

        
        all_model_responses = ""
        for c, r in responses.items():
            if r:
                all_model_responses += r

        if not all_model_responses:
            return "I couldn't understand your request."
        else:
            return all_model_responses.strip()
    
    def process_news(self, sentences):
      
        if isinstance(sentences, list):
            all_texts = " ".join(sentences)
            return all_texts
    
    def process_weather(self, sentences):
        return f"".join(sentences)
    
    def process_reminder(self, sentences):
        all_texts = " ".join(sentences)
        return all_texts