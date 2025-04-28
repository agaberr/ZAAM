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
            "weather": np.ones(vector_size) / np.sqrt(vector_size) * 0.6,
            "reminder": np.ones(vector_size) / np.sqrt(vector_size) * 0.8
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
        # if not self.model_loaded:
        ##########################TODO:
        if True :
            keywords = {
                "news": ["news", "report", "headline", "breaking", "article", "story", "journalist", 
                         "media", "press", "announce", "publish", "tell me", "who is", "what is"],
                "weather": ["weather", "temperature", "forecast", "rain", "snow", "sunny", "cloudy", 
                           "storm", "wind", "humidity", "cold", "hot", "degrees"],
                "reminder": ["remind", "calendar", "schedule", "event", "appointment", 
                             "meeting", "reminder", "don't forget", "remember", "plan", 
                             "tomorrow", "today", "next week", "later", "day after tomorrow",
                             "what's on", "what do i have", "what's scheduled", "am i free", 
                             "do i have any", "when is", "at what time"]
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
        
        if segments.get("weather"):
            responses["weather"] = self.process_weather(segments["weather"])
            
        if segments.get("reminder"):
            responses["reminder"] = self.process_reminder(segments["reminder"])
        
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
        # If we receive a list of sentences, join them
        if isinstance(sentences, list):
            # Join the sentences for a coherent paragraph
            combined_text = " ".join(sentences)
            
            # For longer news text, we might want to add some formatting
            if len(combined_text) > 100:
                newline = '\n'
                formatted = f"NEWS: Here's what I found about your news query:{newline}{combined_text}"
                return formatted
            else:
                return f"NEWS: {combined_text}"
        
        # If we already have a processed string (e.g., from ConversationQA)
        elif isinstance(sentences, str):
            # Return the pre-processed response
            if not sentences.upper().startswith("NEWS:"):
                return f"NEWS: {sentences}"
            return sentences
        
        # Fallback for unexpected input
        newline = '\n'
        return f"NEWS: I found news information in your request:{newline}- {newline}- ".join(str(s) for s in sentences if s)
    
    def process_weather(self, sentences):
        """Process weather-related sentences."""
        # This would be replaced with a model call in production
        newline = '\n'
        return f"WEATHER: Here's the weather information you asked about:{newline}- {newline}- ".join(sentences)
    
    def process_reminder(self, sentences):
        """Process reminder-related sentences."""
        # This would be handled by the ReminderNLP model
        combined_text = " ".join(sentences)
        return f"REMINDER: {combined_text}"
    
    def process_uncategorized(self, sentences):
        """Process uncategorized sentences."""
        # Default processing for uncategorized sentences
        if not sentences:
            return ""
            
        # Join sentences and return as general response
        combined_text = " ".join(sentences)
        return f"I found this general information: {combined_text}" 