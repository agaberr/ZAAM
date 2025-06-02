import numpy as np
import re
from collections import defaultdict
import os
import pickle
from pathlib import Path
from gensim.models import Word2Vec

class AIProcessor:
    def __init__(self, model_path=None):
        self.model = None
        self.category_embeddings = None
        self.similarity_threshold = 0.3
        self.modelLoaded = False
        
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "word2vec_model.model")
        
        self.model_path = model_path
        self.loadModel()
        
        if not self.modelLoaded:
            print("")

        
    def loadModel(self):
            
        try:
            if Path(self.model_path).exists():
                
                if self.model_path.endswith('.model'):
                    self.model = Word2Vec.load(self.model_path)
                    self.createEmbeddings()
                    self.modelLoaded = True
                    
                # there is an error loading the model
                else:
                    self.model = Word2Vec.load(self.model_path)
                    self.createEmbeddings()
                    self.modelLoaded = True
                                
            else:
                self.modelLoaded = False
                
        except Exception as e:
            self.modelLoaded = False
    
    def createEmbeddings(self):
        
        seeds = {
            "news": ["news", "report", "headline", "breaking", "article",  "story",  "media"],
            "weather": ["weather",  "temperature",  "rain",  "snow", "sunny", "cloudy", "storm"],
            "reminder": ["remind", "calendar", "schedule",  "event",  "appointment",  "meeting"]
        }
        
        self.category_embeddings =  {}
        
        for category, seedWords in seeds.items():
            validseeds =  [word for word in seedWords if word in self.model.wv.key_to_index]
            
            if validseeds:
                seedVec =  [self.model.wv[word] for word in validseeds]
                category_embedding = np.mean(seedVec, axis=0)
                self.category_embeddings[category] = category_embedding
            else:
                vecSize  = self.model.wv.vector_size
                self.category_embeddings[category] = np.random.normal(0, 0.1, vecSize)
        
            
    def cosine_similarity(self,vector1,vector2):
        
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        if norm1 == 0 :
            return  0.0
        elif norm2 == 0 :
            return  0.0
        
        else:
            return np.dot(vector1, vector2)/  (np.linalg.norm(vector1) * np.linalg.norm(vector2))


    
    def tokenizeWords(self,  txt):
        txtLowered = txt.lower()
        txt = re.sub(r"[^a-z0-9'\s]", " ", txtLowered)
        tokens = txt.split()
        return tokens
    
    def segmentAllTxt(self, text):
      
        splitters = ['and', 'or', 'but', 'so', 'yet']
        
        pattern = r'(?<=[.!?])\s+(?!(' + '|'.join(splitters) + r')\b)'
        
        sentences = re.split(pattern, text)
        
        splittersSplits = []
        for sentence in sentences:
            parts = re.split(r'\s+and\s+', sentence)
            splittersSplits.extend(parts)
        sentences = splittersSplits
        
        sentences = [s.strip() for s in sentences if s.strip()]
        
        segs = defaultdict(list)
        
        if not self.modelLoaded:
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
            
            for sentence  in sentences:
                sentence_lower = sentence.lower()
                matched =  False
                
                greetWords = ["hey", "hi", "hello"]
                if any(sentence_lower.strip().startswith(greeting) for greeting in greetWords):
                    segs["greeting"].append(sentence)
                    matched = True
                
                if not  matched:
                    if any(word in sentence_lower for word in keys["weather"]):
                        segs["weather"].append(sentence)
                        matched = True
                    elif any(word in sentence_lower for word in keys["news"]):
                        segs["news"].append(sentence)
                        matched = True
                    elif any(word in sentence_lower for word in keys["reminder"]):
                        segs["reminder"].append(sentence)
                        matched = True
                
                if not matched:
                    segs["news" ].append(sentence)
                
            return segs
        

        for sentence in sentences:
            words = self.tokenizeWords(sentence)
            
            sentence_lower = sentence.lower()
            greetWords = ["hey", "hi",  "hello"]
            if any(sentence_lower.strip().startswith(greeting) for greeting in greetWords):
                segs["greeting"].append(sentence)
                continue
            
            availTokens = [word  for  word in words if hasattr(self.model, 'wv') and word in self.model.wv]
            
            if not availTokens:
                segs["news"] .append(sentence)
                continue
            
            sen_vectors =  [self.model.wv[word] for word in availTokens]
            sentence_embedding = np.mean(sen_vectors, axis=0)
            
            chosen_cat =  None
            chosen_sim =  0
            
            for category, cat_vec in self.category_embeddings.items():
                sim = self.cosine_similarity(sentence_embedding, cat_vec)
                if sim > chosen_sim:
                    chosen_sim =  sim
                    chosen_cat =  category
            
            if chosen_sim >= self.similarity_threshold:
                segs[chosen_cat] .append(sentence)
            else:
                segs["news"].append(sentence)
        
        return segs
    
    def process_text(self, text):

        seg = self.segmentAllTxt(text)
        
        res = {}

        if seg.get("news"):
            res["news"] = self.processNews(seg["news"])
        
        if seg.get("weather"):
            res["weather"] = self.processWeather(seg["weather"])
            
        if seg.get("reminder"):
            res["reminder"] = self.processReminder(seg["reminder"])

        
        allRes = ""
        for c, r in res.items():
            if r:
                allRes += r

        if not allRes:
            return "I couldn't understand your request."
        else:
            return allRes.strip()
    
    def processNews(self, sentences):
      
        if isinstance(sentences, list):
            all_texts = " ".join(sentences)
            return all_texts
    
    def processWeather(self,  sentences):
        return  f"".join(sentences )
    
    def processReminder (self,  sentences):
        alltxtt = " " .join(sentences)
        return  alltxtt