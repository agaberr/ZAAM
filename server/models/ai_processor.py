import numpy as np
import re
from collections import defaultdict
import os
import pickle
from pathlib import Path

try:
    from gensim.models import Word2Vec
    GENSIM_AVAILABLE = True
    print("DEBUG: Gensim imported successfully")
except ImportError:
    GENSIM_AVAILABLE = False
    print("DEBUG: Gensim not available, using fallback methods")

class AIProcessor:
    """Service for processing natural language input and categorizing it."""
    
    def __init__(self, model_path=None):
        """Initialize with optional model path."""
        print("DEBUG: Initializing AIProcessor...")
        self.model = None
        self.category_embeddings = None
        self.similarity_threshold = 0.3
        self.model_loaded = False
        
        # Set default model path if not provided
        if model_path is None:
            # Default to models directory in the project
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                     "models", "word2vec_model.model")
        
        self.model_path = model_path
        print(f"DEBUG: Model path set to: {self.model_path}")
        self.load_model()
        
        # Initialize category seeds
        if not self.model_loaded:
            print("DEBUG: Model not loaded, setting up default categories...")
            self.setup_default_categories()
        else:
            print("DEBUG: Model loaded successfully, using AI-based categorization")
        
    def load_model(self):
        """Load the Word2Vec model and category embeddings from file."""
        print(f"DEBUG: Attempting to load model from: {self.model_path}")
        
        if not GENSIM_AVAILABLE:
            print("DEBUG: Gensim not available, cannot load Word2Vec models")
            self.model_loaded = False
            return
            
        try:
            if Path(self.model_path).exists():
                print("DEBUG: Model file exists, loading...")
                
                # Try to load as Gensim Word2Vec model first
                if self.model_path.endswith('.model'):
                    print("DEBUG: Loading as Gensim Word2Vec model...")
                    self.model = Word2Vec.load(self.model_path)
                    print(f"DEBUG: Gensim model loaded successfully")
                    print(f"DEBUG: Model vocabulary size: {len(self.model.wv.key_to_index)}")
                    
                    # Create default category embeddings for the loaded model
                    self.create_category_embeddings()
                    self.model_loaded = True
                    
                elif self.model_path.endswith('.pkl'):
                    print("DEBUG: Loading as pickle file...")
                    with open(self.model_path, 'rb') as file:
                        model_data = pickle.load(file)
                        self.model = model_data.get('model')
                        self.category_embeddings = model_data.get('category_embeddings')
                        self.model_loaded = True
                        print(f"DEBUG: Pickle model loaded successfully")
                        
                else:
                    print("DEBUG: Unknown model file format, trying as Gensim model...")
                    self.model = Word2Vec.load(self.model_path)
                    self.create_category_embeddings()
                    self.model_loaded = True
                
                print(f"DEBUG: Model type: {type(self.model)}")
                print(f"DEBUG: Category embeddings available: {list(self.category_embeddings.keys()) if self.category_embeddings else 'None'}")
                
            else:
                print(f"DEBUG: Model file not found at {self.model_path}. Using default embeddings.")
                self.model_loaded = False
                
        except Exception as e:
            print(f"DEBUG: Error loading model: {e}")
            print(f"DEBUG: Error type: {type(e).__name__}")
            import traceback
            print(f"DEBUG: Full traceback: {traceback.format_exc()}")
            self.model_loaded = False
    
    def create_category_embeddings(self):
        """Create category embeddings from the loaded Word2Vec model."""
        print("DEBUG: Creating category embeddings from Word2Vec model...")
        
        if not self.model or not hasattr(self.model, 'wv'):
            print("DEBUG: No valid Word2Vec model available for creating embeddings")
            return
            
        # Define seed words for each category
        category_seeds = {
            "news": ["news", "report", "headline", "breaking", "article", "story", "media"],
            "weather": ["weather", "temperature", "rain", "snow", "sunny", "cloudy", "storm"],
            "reminder": ["remind", "calendar", "schedule", "event", "appointment", "meeting"]
        }
        
        self.category_embeddings = {}
        
        for category, seed_words in category_seeds.items():
            # Find seed words that exist in the model vocabulary
            valid_seeds = [word for word in seed_words if word in self.model.wv.key_to_index]
            print(f"DEBUG: Valid seed words for '{category}': {valid_seeds}")
            
            if valid_seeds:
                # Average the embeddings of valid seed words
                seed_vectors = [self.model.wv[word] for word in valid_seeds]
                category_embedding = np.mean(seed_vectors, axis=0)
                self.category_embeddings[category] = category_embedding
                print(f"DEBUG: Created embedding for '{category}' using {len(valid_seeds)} seed words")
            else:
                print(f"DEBUG: No valid seed words found for '{category}', using random embedding")
                # Create a random embedding if no seed words are found
                vector_size = self.model.wv.vector_size
                self.category_embeddings[category] = np.random.normal(0, 0.1, vector_size)
        
        print(f"DEBUG: Category embeddings created: {list(self.category_embeddings.keys())}")
    
    def setup_default_categories(self):
        """Set up default category embeddings when model is not available."""
        print("DEBUG: Setting up default category embeddings...")
        # Create simple placeholder embeddings for categories
        vector_size = 100
        self.category_embeddings = {
            "news": np.ones(vector_size) / np.sqrt(vector_size),
            "weather": np.ones(vector_size) / np.sqrt(vector_size) * 0.6,
            "reminder": np.ones(vector_size) / np.sqrt(vector_size) * 0.8
        }
        
        print("DEBUG: Using default category embeddings")
        print(f"DEBUG: Default categories: {list(self.category_embeddings.keys())}")
    
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
        print(f"\nDEBUG: Input text received: {text}")
        print(f"DEBUG: Model loaded status: {self.model_loaded}")
        
        # Define conjunctions that shouldn't trigger a split
        conjunctions = ['and', 'or', 'but', 'so', 'yet']
        
        # Regex pattern: split on '.', '?', '!', but NOT if followed by a conjunction
        pattern = r'(?<=[.!?])\s+(?!(' + '|'.join(conjunctions) + r')\b)'
        
        # Split the text
        sentences = re.split(pattern, text)
        
        # Also split by 'and' when not part of a larger word
        additional_splits = []
        for sentence in sentences:
            parts = re.split(r'\s+and\s+', sentence)
            additional_splits.extend(parts)
        sentences = additional_splits
        
        # Clean up the results
        sentences = [s.strip() for s in sentences if s.strip()]
        print(f"DEBUG: Sentences after splitting: {sentences}")
        
        # Initialize segments dictionary
        segments = defaultdict(list)
        
        # If no model is loaded, use simple keyword matching
        if not self.model_loaded:
            print("DEBUG: Using keyword-based categorization (no AI model)")
            keywords = {
                "greeting": ["hey zaam", "hi zaam", "hello zaam"],
                "news": ["news", "report", "headline", "breaking", "article", "story", "journalist", 
                         "media", "press", "announce", "publish", "who is", "what is"],
                "weather": ["weather", "temperature", "forecast", "rain", "snow", "sunny", "cloudy", 
                           "storm", "wind", "humidity", "cold", "hot", "degrees", "wear"],
                "reminder": ["remind", "calendar", "schedule", "event", "appointment", 
                             "meeting", "reminder", "don't forget", "remember", "plan", 
                             "tomorrow", "today", "next week", "later", "day after tomorrow",
                             "what's on", "what do i have", "what's scheduled", "am i free", 
                             "do i have any", "when is", "at what time"]
            }
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                print(f"\nDEBUG: Processing sentence: {sentence}")
                matched_categories = []
                
                # Check for greetings first
                if any(greeting in sentence_lower for greeting in keywords["greeting"]):
                    segments["greeting"].append(sentence)
                    matched_categories.append("greeting")
                
                # Check other categories
                for category, words in keywords.items():
                    if any(word in sentence_lower for word in words):
                        segments[category].append(sentence)
                        matched_categories.append(category)
                        print(f"DEBUG: Matched category '{category}' for sentence: {sentence}")
                
                if not matched_categories:
                    segments["uncategorized"].append(sentence)
                    print(f"DEBUG: No category match found for: {sentence}")
                
            print(f"\nDEBUG: Final categorized segments: {dict(segments)}")
            return segments
        
        # Use vectorization for categorization if model is loaded
        print("DEBUG: Using AI model-based categorization")
        print(f"DEBUG: Model type: {type(self.model)}")
        print(f"DEBUG: Available categories: {list(self.category_embeddings.keys()) if self.category_embeddings else 'None'}")
        
        for sentence in sentences:
            print(f"\nDEBUG: Processing sentence with AI model: {sentence}")
            words = self.tokenize(sentence)
            print(f"DEBUG: Tokenized words: {words}")
            
            # Create an embedding from valid words
            valid_words = [word for word in words if hasattr(self.model, 'wv') and word in self.model.wv]
            print(f"DEBUG: Valid words found in model vocabulary: {valid_words}")
            
            if not valid_words:
                print("DEBUG: No valid words found, categorizing as uncategorized")
                segments["uncategorized"].append(sentence)
                continue
            
            sentence_vecs = [self.model.wv[word] for word in valid_words]
            sentence_embedding = np.mean(sentence_vecs, axis=0)  # Average word vectors
            print(f"DEBUG: Created sentence embedding with shape: {sentence_embedding.shape}")
            
            best_category = None
            best_similarity = 0
            
            for category, cat_vec in self.category_embeddings.items():
                sim = self.cosine_similarity(sentence_embedding, cat_vec)
                print(f"DEBUG: Similarity with '{category}': {sim}")
                if sim > best_similarity:
                    best_similarity = sim
                    best_category = category
            
            print(f"DEBUG: Best category: {best_category} with similarity: {best_similarity}")
            print(f"DEBUG: Similarity threshold: {self.similarity_threshold}")
            
            if best_similarity >= self.similarity_threshold:
                segments[best_category].append(sentence)
                print(f"DEBUG: Assigned to category: {best_category}")
            else:
                segments["uncategorized"].append(sentence)
                print("DEBUG: Below threshold, assigned to uncategorized")
        
        print(f"\nDEBUG: Final AI-based categorized segments: {dict(segments)}")
        return segments
    
    def process_text(self, text):
        """Process input text and return categorized responses."""
        segments = self.segment_sentences(text)
        
        responses = {}
        
        # Process greetings first
        if segments.get("greeting"):
            responses["greeting"] = "Hello! How may I help you today?"
        
        # Process each category of sentences
        if segments.get("news"):
            responses["news"] = self.process_news(segments["news"])
        
        if segments.get("weather"):
            responses["weather"] = self.process_weather(segments["weather"])
            
        if segments.get("reminder"):
            responses["reminder"] = self.process_reminder(segments["reminder"])
        
        if segments.get("uncategorized"):
            responses["uncategorized"] = self.process_uncategorized(segments["uncategorized"])
        
        # If only greeting is present, return just the greeting
        if len(responses) == 1 and "greeting" in responses:
            return responses["greeting"]
        
        # Combine responses
        combined_response = ""
        for category, response in responses.items():
            if response:
                if category == "greeting":
                    combined_response = response + "\n\n" + combined_response
                else:
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