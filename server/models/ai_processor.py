import numpy as np
import re
from collections import defaultdict
import os
import pickle
from pathlib import Path

class AIProcessor:
    """Service for processing natural language input and categorizing it."""
    
    def __init__(self, model_path=None):
        """Initialize with optional model path."""
        self.model = None
        self.category_embeddings = None
        self.similarity_threshold = 0.3
        self.model_loaded = False
        
        # Set default model path if not provided
        if model_path is None:
            # Default to models directory in the project
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                     "models", "word2vec_model.pkl")
        
        self.model_path = model_path
        self.load_model()
        
        # Initialize category seeds
        if not self.model_loaded:
            self.setup_default_categories()
        
    def load_model(self):
        """Load the Word2Vec model and category embeddings from file."""
        try:
            if Path(self.model_path).exists():
                with open(self.model_path, 'rb') as file:
                    model_data = pickle.load(file)
                    self.model = model_data.get('model')
                    self.category_embeddings = model_data.get('category_embeddings')
                    self.model_loaded = True
                    print(f"Model loaded successfully from {self.model_path}")
            else:
                print(f"Model file not found at {self.model_path}. Using default embeddings.")
                self.model_loaded = False
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_loaded = False
    
    def setup_default_categories(self):
        """Set up default category embeddings when model is not available."""
        # Create simple placeholder embeddings for categories
        vector_size = 100
        self.category_embeddings = {
            "news": np.ones(vector_size) / np.sqrt(vector_size),
            "reminders": np.ones(vector_size) / np.sqrt(vector_size) * 0.8,
            "weather": np.ones(vector_size) / np.sqrt(vector_size) * 0.6
        }
        
        print("Using default category embeddings")
    
    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def tokenize(self, text):
        """Tokenize the input text."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9'\s]", " ", text)
        tokens = text.split()
        return tokens
    
    def segment_sentences(self, text):
        """Segment text into sentences and categorize them."""
        sentences = re.split(r'[.;!?]', text)  # Split using punctuation
        sentences = [s.strip() for s in sentences if s.strip()]  # Remove empty sentences
        
        segments = defaultdict(list)
        
        # If no model is loaded, use simple keyword matching
        if not self.model_loaded:
            keywords = {
                "news": ["news", "report", "headline", "breaking", "article", "story", "journalist", 
                         "media", "press", "announce", "publish"],
                "reminders": ["remind", "remember", "appointment", "schedule", "meeting", "event", 
                             "task", "deadline", "todo", "don't forget", "call", "email", "calendar"],
                "weather": ["weather", "temperature", "forecast", "rain", "snow", "sunny", "cloudy", 
                           "storm", "wind", "humidity", "cold", "hot", "degrees"]
            }
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                matched = False
                
                for category, words in keywords.items():
                    if any(word in sentence_lower for word in words):
                        segments[category].append(sentence)
                        matched = True
                        break
                
                if not matched:
                    segments["uncategorized"].append(sentence)
                    
            return segments
        
        # Use vectorization for categorization if model is loaded
        for sentence in sentences:
            words = self.tokenize(sentence)
            
            # Create an embedding from valid words
            valid_words = [word for word in words if hasattr(self.model, 'wv') and word in self.model.wv]
            
            if not valid_words:
                segments["uncategorized"].append(sentence)
                continue
            
            sentence_vecs = [self.model.wv[word] for word in valid_words]
            sentence_embedding = np.mean(sentence_vecs, axis=0)  # Average word vectors
            
            best_category = None
            best_similarity = 0
            
            for category, cat_vec in self.category_embeddings.items():
                sim = self.cosine_similarity(sentence_embedding, cat_vec)
                if sim > best_similarity:
                    best_similarity = sim
                    best_category = category
            
            if best_similarity >= self.similarity_threshold:
                segments[best_category].append(sentence)
            else:
                segments["uncategorized"].append(sentence)
        
        return segments
    
    def process_text(self, text):
        """Process input text and return categorized responses."""
        segments = self.segment_sentences(text)
        
        responses = {}
        
        # Process each category of sentences
        if segments.get("news"):
            responses["news"] = self.process_news(segments["news"])
        
        if segments.get("reminders"):
            responses["reminders"] = self.process_reminders(segments["reminders"])
        
        if segments.get("weather"):
            responses["weather"] = self.process_weather(segments["weather"])
        
        if segments.get("uncategorized"):
            responses["uncategorized"] = self.process_uncategorized(segments["uncategorized"])
        
        # Combine responses
        combined_response = ""
        for category, response in responses.items():
            if response:
                combined_response += f"{response}\n"
        
        return combined_response.strip() if combined_response else "I couldn't understand your request."
    
    def process_news(self, sentences):
        """Process news-related sentences."""
        # This would be replaced with a model call in production
        newline = '\n'
        return f"NEWS: I found news information in your request:{newline}- {newline}- ".join(sentences)
    
    def process_reminders(self, sentences):
        """Process reminder-related sentences."""
        # This would be replaced with a model call in production
        newline = '\n'
        return f"REMINDER: I'll help you with these reminders:{newline}- {newline}- ".join(sentences)
    
    def process_weather(self, sentences):
        """Process weather-related sentences."""
        # This would be replaced with a model call in production
        newline = '\n'
        return f"WEATHER: Here's the weather information you asked about:{newline}- {newline}- ".join(sentences)
    
    def process_uncategorized(self, sentences):
        """Process uncategorized sentences."""
        # This would be replaced with a more sophisticated fallback in production
        newline = '\n'
        return f"I'm not sure how to categorize these parts of your request:{newline}- {newline}- ".join(sentences) 