import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from NameEntityModel.TopicExtractionModel import NERPredictor
from transformers import BertModel, BertTokenizerFast, AdamW, get_linear_schedule_with_warmup
from TopicExtraction.ExtractTopic import ExtractTopic, generate_passage, generate_passage_from_entity_tuple
import spacy
import functools
import time
from PronounResolution.PronounResolutionModel import PronounResolutionModel
from QuestionAnswer.ExtractiveQABertModel import BertForQA
import torch
import re
from transformers import BertTokenizer
from text_summarization import article_summarize
from classifier.QueryClassifier import predict_query_type
import os

base_path = os.path.dirname(__file__)

# Performance decorator for timing functions
def timing_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

class ConversationalQA:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.conversation_history = []
        self.current_passage = None
        # Load a smaller model for better performance
        self.nlp = spacy.load("en_core_web_sm")  # Using smaller model
        self.current_entity = None
        self.entity_info = {}
        
        # Initialize sentence transformer
        self.model = SentenceTransformer(model_name)
        
        # Add caching for embeddings
        self.embedding_cache = {}
        self.entity_cache = {}
        self.sentence_cache = {}
        
        # Precompute common question words
        self.question_words = set(['who', 'what', 'where', 'when', 'why', 'how'])
        
        # Process passage into sentences just once
        self.processed_sentences = []
        
        # Batch processing settings
        self.batch_size = 32  # Optimal batch size for embedding computation
        # load NER model
        self.ner_predictor = NERPredictor()
        # load Pronoun resolution model
        self._load_pronoun_resolution_model()
        # Initialize QA model
        self._load_qa_model()

        classifyQuery_model_path = os.path.join(base_path, "Models", "classifier_model.pkl")
        classifyVectorizar_path = os.path.join(base_path, "Models", "vectorizer.pkl")

        self.classifyQuery_model =  joblib.load(classifyQuery_model_path)
        self.classifyVectorizar = joblib.load(classifyVectorizar_path)
        
    
    def _load_pronoun_resolution_model(self):
        # model_path = 'Models/pronoun_resolution_model_full.pt'
        model_path = os.path.join(base_path, "Models", "pronoun_resolution_model_full.pt")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=self.device)
        bert_model_name = checkpoint.get('bert_model_name', 'bert-base-uncased')

        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.pronoun_model = PronounResolutionModel(bert_model_name=bert_model_name)
        self.pronoun_model.load_state_dict(checkpoint['model_state_dict'])
        self.pronoun_model.to(self.device)
        self.pronoun_model.eval()

    def _load_qa_model(self):
        self.model_qa = BertForQA()
        model_path_qa = os.path.join(base_path, "Models", "extractiveQA.pt")

        self.model_qa.load_state_dict(torch.load(model_path_qa, 
                                             map_location=self.device))
        self.model_qa.to(self.device)
        self.model_qa.eval()
        self.model_qa = self.model_qa.to(self.device)
        self.tokenizer_qa = BertTokenizerFast.from_pretrained("bert-base-uncased")
        print(f"Model loaded successfully on {self.device}")

    def answer_question(self, question, context, max_length=384):

        # # Ensure model is in evaluation mode
        # self.model_qa.eval()
        
        # Tokenize input
        inputs = self.tokenizer_qa(
            question, 
            context, 
            max_length=max_length,
            truncation="only_second",
            stride=128,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Get sequence IDs and offset mapping
        sequence_ids = inputs.sequence_ids(0)
        offset_mapping = inputs.pop("offset_mapping").tolist()[0]
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model_qa(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs.get("token_type_ids", None)
            )
    
        # Get predictions
        start_logits = outputs["start_logits"]
        end_logits = outputs["end_logits"]
        
        # Convert to Python lists
        start_logits = start_logits[0].cpu().numpy()
        end_logits = end_logits[0].cpu().numpy()
        
        # Get best answer (consider only context tokens)
        context_tokens = []
        for i, seq_id in enumerate(sequence_ids):
            if seq_id == 1:  # 1 refers to context (not question or special tokens)
                context_tokens.append(i)
        
        # Only consider answers in the context
        start_logits = [float('-inf') if i not in context_tokens else score for i, score in enumerate(start_logits)]
        end_logits = [float('-inf') if i not in context_tokens else score for i, score in enumerate(end_logits)]
        
        # Find best answer
        start_idx = np.argmax(start_logits)
        end_idx = np.argmax(end_logits[start_idx:]) + start_idx
        
        # Convert token indices to character spans
        token_start, token_end = start_idx, end_idx
        
        # Get character span from token indices
        char_start = offset_mapping[token_start][0]
        char_end = offset_mapping[token_end][1]
        
        # Extract answer text
        answer = context[char_start:char_end]
        
        return answer
        
    def _get_embedding(self, text):
        """Cache embeddings to avoid recomputation"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        # Get embedding using sentence transformer
        embedding = self.model.encode(text, show_progress_bar=False)
        
        # Store in cache
        self.embedding_cache[text] = embedding
        return embedding
    
    def _get_embeddings_batch(self, texts):
        """Process embeddings in batches for efficiency"""
        # Filter out texts that are already cached
        new_texts = []
        new_texts_indices = []
        
        for i, text in enumerate(texts):
            if text not in self.embedding_cache:
                new_texts.append(text)
                new_texts_indices.append(i)
        
        if not new_texts:
            return [self.embedding_cache[text] for text in texts]
        
        # Encode all new texts in a single batch operation
        new_embeddings = self.model.encode(new_texts, show_progress_bar=False, batch_size=self.batch_size)
        
        # Update cache and prepare results
        for i, idx in enumerate(new_texts_indices):
            self.embedding_cache[texts[idx]] = new_embeddings[i]
        
        # Return all embeddings in original order
        return [self.embedding_cache[text] for text in texts]
    
    def _get_entities(self, text):
        """Cache entities to avoid recomputation"""
        if text in self.entity_cache:
            return self.entity_cache[text]
        
        # Extract entities
        doc = self.ner_predictor.predict(text)
        important_keywords = {"war"}
        entities, important_keywords = ExtractTopic(doc, important_keywords)
        
        # Store in cache
        self.entity_cache[text] = entities
        return entities
    
    def process_query(self, query, passage=None):
        # Skip empty queries (used when setting a new passage)
        if not query.strip():
            if passage is not None:
                self.current_passage = passage
            return "", False
        
        # Update passage if provided
        if passage is not None:
            self.current_passage = passage
            # Preprocess the passage sentences
            self._preprocess_passage()
            
        # Apply attention to resolve the query using conversation history
        resolved_query = self._apply_attention(query)
        
        print("the query:: ",query)
        print("\nresolved query:: ",resolved_query)
        
        # queryType= predict_query_type(resolved_query,self.nlp,self.classifyQuery_model,self.classifyVectorizar)
        queryType= "QA"
        print("query type: ",queryType)

        # Check if current passage has the entities we need
        answer, confidence = self._get_answer_with_cosine_similarity(resolved_query)
        print("answer before checking confidence: ",answer)
        print("\nconfidence level: ",confidence)
        need_new_passage = confidence < 0.35
        if need_new_passage:
            print("I need new Passsage\n")
            # clear history
            self.conversation_history = []    
            # call NER Model to label query
            predictions = self.ner_predictor.predict(resolved_query)                           
            important_keywords = {"match", "war", "football", "born","places","news","achievements","today","now"}   # set important words
            entities = ExtractTopic(predictions, important_keywords)    # extract entities
            
            # generate prompt based on entities
            prompt = generate_passage_from_entity_tuple(entities,resolved_query)       
            print("prompt : ",prompt)

            # generate passage based on prompt
            passage_generated = generate_passage(prompt)
            print("new passage : ",passage_generated)
            self.set_passage(passage_generated) 
            # get closed answer from new passage
            # answer, confidence = self._get_answer_with_cosine_similarity(resolved_query)

            if queryType == "Summarization":
                answer = article_summarize(passage_generated)
            else: answer = self.answer_question(resolved_query,passage_generated)

            print("query type: ",queryType)
            # append to history
            self.conversation_history.append({
            "query": query,
            "resolved_query": resolved_query,
            "answer": answer,
            "entity": self.current_entity,
            "confidence": confidence
             })
            return answer, need_new_passage
        else:
            if queryType == "Summarization":
                answer = article_summarize(self.current_passage)
            else: answer = self.answer_question(resolved_query,self.current_passage)
            
            self.conversation_history.append({
                "query": query,
                "resolved_query": resolved_query,
                "answer": answer,
                "entity": self.current_entity,
                "confidence": confidence
            })
            if self.current_entity:
                if self.current_entity not in self.entity_info:
                    self.entity_info[self.current_entity] = []
                self.entity_info[self.current_entity].append({
                    "query": query,
                    "answer": answer,
                    "confidence": confidence
                })
            
            return answer, need_new_passage


    def resolve_query(self,context, query, pronoun):
        text = context +" "+ query

        # Find pronoun position
        pronoun_pattern = re.compile(r'\b' + re.escape(pronoun) + r'\b', re.IGNORECASE)
        matches = list(pronoun_pattern.finditer(query))  # Search in query only

        if not matches:
            return {"error": f"Pronoun '{pronoun}' not found in the query"}

        pronoun_position = matches[0].start()  # Get position in query

        extracted_labels = self.ner_predictor.predict(text)
        important_keywords = {"war"}
        candidates, important_keywords = ExtractTopic(extracted_labels, important_keywords)

        encoding = self.tokenizer(
            text, max_length=128, padding='max_length', truncation=True, return_tensors='pt'
        ).to(self.device)

        context_length = len(self.tokenizer.tokenize(context))  # Get token count for context
        pronoun_token_position = torch.tensor(
            [context_length + len(self.tokenizer.tokenize(query[:pronoun_position]))], dtype=torch.long
        ).to(self.device)

        # Process candidate encodings in batch
        candidate_encodings = self.tokenizer(
            candidates, max_length=20, padding='max_length', truncation=True, return_tensors='pt'
        ).to(self.device)

        num_candidates = torch.tensor([len(candidates)], dtype=torch.long).to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.pronoun_model(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask'],
                pronoun_position=pronoun_token_position,
                candidate_input_ids=candidate_encodings['input_ids'].unsqueeze(0),
                candidate_attention_masks=candidate_encodings['attention_mask'].unsqueeze(0),
                num_candidates=num_candidates
            )

        # Get prediction and confidence scores
        scores = outputs[0].cpu().numpy()
        probabilities = torch.softmax(outputs[0], dim=0).cpu().numpy()
        predicted_idx = int(torch.argmax(outputs, dim=1).item())
        resolved_candidate = candidates[predicted_idx]

        replaced_query = re.sub(r'\b' + re.escape(pronoun) + r'\b', resolved_candidate, query, count=1, flags=re.IGNORECASE)

        return replaced_query

    
    def _preprocess_passage(self):
        """Preprocess the passage to extract sentences and entities once"""
        # Clear cache for new passage
        self.processed_sentences = []
        self.sentence_cache = {}
        
        # Process passage into sentences
        doc = self.nlp(self.current_passage)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        # Store processed sentences
        self.processed_sentences = sentences
        
        # Batch compute sentence embeddings for efficiency
        embeddings = self._get_embeddings_batch(sentences)
        
        # Store sentence data in cache
        for i, (sent, embedding) in enumerate(zip(sentences, embeddings)):
            self.sentence_cache[i] = {
                'text': sent,
                'embedding': embedding,
                'entities': None  # Will be computed on demand
            }
    
    def _check_passage_relevance(self, resolved_query):
        """Check if the current passage contains information relevant to the query"""
        if not self.current_passage:
            return True
            
        # Extract entities from query
        query_entities = self._get_entities(resolved_query)
        
        # If no entities in query, check general relevance
        if not query_entities:
            # Use a simple keyword check for efficiency
            important_words = [word.lower() for word in resolved_query.split() 
                              if word.lower() not in self.question_words and len(word) > 3]
            
            # If any important word is in passage, consider it potentially relevant
            for word in important_words:
                if word in self.current_passage.lower():
                    return False
            
            # No important keywords found in passage
            return len(important_words) > 0
        
        # Check if query entities are in passage
        for entity in query_entities:
            if entity.lower() not in self.current_passage.lower():
                return True
                
        return False
    
    
    def _apply_attention(self, query):
        # If this is the first query, no resolution needed
        if not self.conversation_history:
            return query
        
        print("Historyy:::    ",self.conversation_history)
        # Quick check for pronouns
        pronoun_list = {"he", "him", "his", "she", "her", "it", "they", "them", "this", "that"}
    
        # Split query into words
        words = query.split()
        # Find pronoun in the query
        pronoun = next((word for word in words if word.lower() in pronoun_list), None)
        
        if not pronoun:
            return query
        
        if len(self.conversation_history) >=3:
            context = " ".join([entry['resolved_query'] for entry in self.conversation_history[-3:]])
        else: 
            context = self.conversation_history[-1]['resolved_query']

        print("context: ",context)
        print("query: ",query)
        print("pronoun: ",pronoun)
        resolved_query = self.resolve_query(context, query,pronoun)
      
        return resolved_query
    
    def _replace_pronouns(self, query, entities):
        if not entities:
            return query
            
        # Simple string replacement for speed
        words = query.split()
        replaced = False
        
        for i, word in enumerate(words):
            word_lower = word.lower()
            
            # Handle common pronouns
            if word_lower in ["he", "him", "she", "her", "it", "they", "them", "this", "that"] and not replaced:
                words[i] = entities[0].split()[0] if ' ' in entities[0] else entities[0]
                replaced = True
            elif word_lower in ["his", "her", "its", "their"] and not replaced:
                name = entities[0].split()[0] if ' ' in entities[0] else entities[0]
                words[i] = f"{name}'s"
                replaced = True
                
        resolved_query = " ".join(words)
        return resolved_query
    
    def assess_answer_quality(self, query, answer_sentence, similarity_score, question_type):
        
        query_keywords = set()
        query_doc = self.nlp(query)
        for token in query_doc:
            if (token.text.lower() not in self.question_words and 
                not token.is_stop and 
                token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ']):
                query_keywords.add(token.lemma_.lower())
        
        # If no keywords found, rely on similarity score
        if not query_keywords:
            return similarity_score
            
        # Quick keyword check for answer
        answer_doc = self.nlp(answer_sentence)
        answer_keywords = set()
        for token in answer_doc:
            if not token.is_stop and token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ']:
                answer_keywords.add(token.lemma_.lower())
        
        # Calculate keyword overlap
        keyword_overlap = len(query_keywords.intersection(answer_keywords))
        keyword_ratio = keyword_overlap / len(query_keywords) if query_keywords else 0
        
        # Check for named entities based on question type (simplified)
        has_relevant_entity = False
        if question_type == "who":
            has_relevant_entity = any(ent.label_ == "PERSON" for ent in answer_doc.ents)
        elif question_type == "where":
            has_relevant_entity = any(ent.label_ in ["GPE", "LOC", "FAC"] for ent in answer_doc.ents)
        elif question_type == "when":
            has_relevant_entity = any(ent.label_ == "DATE" or ent.label_ == "TIME" for ent in answer_doc.ents)
        
        # Weighted quality score
        quality_score = (similarity_score * 0.4) + (keyword_ratio * 0.4) + (0.2 if has_relevant_entity else 0)
        return quality_score
    
    @timing_decorator
    def _get_answer_with_cosine_similarity(self, query):
        
        if not self.processed_sentences:
            return "Insufficient information available.", 0.0
        
        # Categorize the question
        question_type = self._categorize_question(query)
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        print("QUESTION TYPE::::::  ", question_type)
  
        # Default handling: use all sentences
        sent_embeddings = np.array([sent_data['embedding'] for sent_data in self.sentence_cache.values()])
        
        # Calculate similarities in batch
        similarities = cosine_similarity([query_embedding], sent_embeddings)[0]
        
        best_idx = np.argmax(similarities)
        base_confidence = similarities[best_idx]
        answer = self.processed_sentences[best_idx]
        
        # Assess answer quality
        adjusted_confidence = self.assess_answer_quality(query, answer, base_confidence, question_type)
        
        # Reduce confidence for very short answers
        if len(answer.split()) < 5:
            adjusted_confidence *= 0.5
            
        return answer, adjusted_confidence

    def _categorize_question(self, query):
        query_lower = query.lower()
        
        if query_lower.startswith(("when", "what time", "what date")):
            return "when"
        elif query_lower.startswith(("where", "what place", "which city")):
            return "where"
        elif query_lower.startswith(("who", "what person")):
            return "who"
        elif query_lower.startswith("how"):
            return "how"
        elif query_lower.startswith("why"):
            return "why"
        else:
            return "general"
        
    def set_passage(self, new_passage):
        self.current_passage = new_passage 
        self._preprocess_passage()
        