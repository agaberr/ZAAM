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
            print("el model didn't load... panic")

        
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
    


    def segmentAllTxt(self,  txt):
      
        splitters = ['and', 'or', 'but', 'so', 'yet']
        
        pattern = r'(?<=[.!?])\s+(?!(' + '|'.join(splitters) + r')\b)'
        
        sentences = re.split(pattern, txt)
        
        splitterSplits = []
        for sen in sentences:
            parts = re.split(r'\s+and\s+', sen)
            splitterSplits.extend(parts)
        sentences = splitterSplits
        
        sentences = [s.strip() for s in sentences if s.strip()]
        
        segments = defaultdict(list)
        
        # Lw el model darab, make a backup plan
        if not self.modelLoaded:
            keys = {
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
                sentence = sentence.lower()
                match = []
                
                # prioritize grretings first 
                if any(greeting in sentence for greeting in keys["greeting"]):
                    segments["greeting"].append(sentence)
                    match.append("greeting")
                
                for category, words in keys.items():
                    if any(word in sentence for word in words):
                        segments[category].append(sentence)
                        match.append(category)
                
                if not match:
                    segments["uncategorized"].append(sentence)
                
            return segments
        

        for sentence in sentences:
            words = self.tokenizeWords(sentence)
            
            availTokens = [word for word in words if hasattr(self.model, 'wv') and word in self.model.wv]
            
            if not availTokens:
                segments["uncategorized"].append(sentence)
                continue
            
            sentenceVectors = [self.model.wv[word] for word in availTokens]
            sentenceEmbed = np.mean(sentenceVectors, axis=0)
            
            chosenCat = None
            chosenSimilarity = 0
            
            for category, cat_vec in self.category_embeddings.items():
                similarity = self.cosine_similarity(sentenceEmbed, cat_vec)
                if similarity > chosenSimilarity:
                    chosenSimilarity = similarity
                    chosenCat = category
            
            if chosenSimilarity >= self.similarity_threshold:
                segments[chosenCat].append(sentence)
            else:
                segments["uncategorized"].append(sentence)
        
        return segments
    
    def process_text(self, text):

        seg = self.segmentAllTxt(text)
        
        responses = {}

        if seg.get("news") :
            responses["news"] = self.process_news(seg["news"])
        
        if seg.get("weather") :
            responses["weather"] = self.processWeather(seg["weather"])
            
        if seg.get("reminder") :
            responses["reminder"] = self.processReminder(seg["reminder"])

        
        allModelResponses = ""
        for c, r in responses.items():
            if r:
                allModelResponses += r

        if not allModelResponses:
            uncategorized = seg.get("uncategorized", [])
            if uncategorized:
                allModelResponses += self.process_uncategorized(uncategorized)
        
        if not allModelResponses:
            return "I couldn't understand your request."
        else:
            return allModelResponses.strip()
    


    def process_news(self, sentences):
      
        if isinstance(sentences, list):
            all_texts = " ".join(sentences)
            return all_texts
    
    def processWeather(self, sen):
        return "".join(sen)
    
    def processReminder(self, sen):
        return " ".join(sen)