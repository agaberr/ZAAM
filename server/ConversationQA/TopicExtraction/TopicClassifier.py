import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class TopicClassifier:
    def __init__(self, model_name='all-MiniLM-L6-v2', confidence_threshold=0.6):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.categories = []
        self.category_embeddings = {}
        self.category_centroids = {}
        self.confidence_threshold = confidence_threshold
        
    def add_examples(self, category_examples):
        self.categories = list(category_examples.keys())
        
        # Generate embeddings for all examples
        for category, examples in category_examples.items():
            print(f"Generating embeddings for category: {category} ({len(examples)} examples)")
            self.category_embeddings[category] = self.model.encode(examples)
            
            # Calculate category centroid (average embedding)
            self.category_centroids[category] = np.mean(self.category_embeddings[category], axis=0)
    
    def get_query_embedding(self, query):
        return self.model.encode([query])[0]
    
    def classify(self, query, method='max_similarity', threshold=0.5):
        query_embedding = self.model.encode([query])[0]
        
        if method == 'max_similarity':
            return self._classify_max_similarity(query_embedding, threshold)
        elif method == 'centroid':
            return self._classify_centroid(query_embedding, threshold)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'max_similarity' or 'centroid'.")
    
    def _classify_max_similarity(self, query_embedding, threshold):
        category_scores = {}
        
        for category in self.categories:
            # Calculate cosine similarity with each example in the category
            similarities = [
                cosine_similarity([query_embedding], [emb])[0][0] 
                for emb in self.category_embeddings[category]
            ]
            # Take the maximum similarity as the category score
            category_scores[category] = max(similarities)
        
        # Get best category and its score
        best_category = max(category_scores, key=category_scores.get)
        confidence = category_scores[best_category]
        
        # Return unknown if below threshold
        if confidence < threshold:
            return "unknown", confidence
        
        return best_category, confidence
    
    def _classify_centroid(self, query_embedding, threshold):
        category_scores = {}
        
        for category in self.categories:
            # Calculate cosine similarity with the category centroid
            similarity = cosine_similarity([query_embedding], [self.category_centroids[category]])[0][0]
            category_scores[category] = similarity
        
        # Get best category and its score
        best_category = max(category_scores, key=category_scores.get)
        confidence = category_scores[best_category]
        
        if confidence < threshold:
            return "unknown", confidence
        
        return best_category, confidence
    
    def evaluate(self, test_data):
        total = len(test_data)
        correct = 0
        category_metrics = {cat: {"correct": 0, "total": 0} for cat in self.categories}
        category_metrics["unknown"] = {"correct": 0, "total": 0}
        
        for query, true_category in test_data:
            pred_category, confidence = self.classify(query)
            
            # Update metrics
            if pred_category == true_category:
                correct += 1
                category_metrics[true_category]["correct"] += 1
            
            category_metrics[true_category]["total"] += 1
            
        # Calculate accuracy
        accuracy = correct / total if total > 0 else 0
        
        # Calculate per-category metrics
        for cat in category_metrics:
            if category_metrics[cat]["total"] > 0:
                category_metrics[cat]["accuracy"] = category_metrics[cat]["correct"] / category_metrics[cat]["total"]
            else:
                category_metrics[cat]["accuracy"] = 0
        
        return {
            "overall_accuracy": accuracy,
            "category_metrics": category_metrics
        }

    def interactive_classify(self, query, method='max_similarity'):
        query_embedding = self.model.encode([query])[0]
        
        if method == 'max_similarity':
            category, confidence = self._classify_max_similarity(query_embedding, 0)
        else:
            category, confidence = self._classify_centroid(query_embedding, 0)
            
        # Get all category scores for ranking
        category_scores = {}
        for cat in self.categories:
            if method == 'max_similarity':
                similarities = [
                    cosine_similarity([query_embedding], [emb])[0][0] 
                    for emb in self.category_embeddings[cat]
                ]
                category_scores[cat] = max(similarities)
            else:
                similarity = cosine_similarity([query_embedding], [self.category_centroids[cat]])[0][0]
                category_scores[cat] = similarity
        
        # Sort categories by confidence
        sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        
        # If confidence is below threshold, ask for confirmation
        if confidence < self.confidence_threshold:
            top_categories = sorted_categories  # All three categories
            
            # Generate confirmation message
            if category != "unknown":
                message = f"I think you're asking about {category} is that right? "
                message += "Please confirm if this is about football, cooking, or news."
                
                return {
                    "result": "needs_confirmation",
                    "suggested_category": category,
                    "confidence": confidence,
                    "message": message,
                    "top_categories": sorted_categories,
                    "query": query,
                    "query_embedding": query_embedding
                }
            else:
                message = "I'm not sure what Topic you are asking about. Could you tell me if it's about football, cooking, or news?"
                
                return {
                    "result": "needs_confirmation",
                    "suggested_category": "unknown",
                    "confidence": confidence,
                    "message": message,
                    "top_categories": sorted_categories,
                    "query": query,
                    "query_embedding": query_embedding
                }
        else:
            return {
                "result": "confident",
                "category": category,
                "confidence": confidence
            }
    
    def add_confirmed_example(self, query, correct_category, query_embedding=None):

        if correct_category not in self.categories or correct_category == "unknown":
            print(f"Warning: Cannot add example to '{correct_category}'. Valid categories are: {', '.join(self.categories)}")
            return
            
        # Get or compute embedding
        if query_embedding is None:
            query_embedding = self.model.encode([query])[0]
            
        # Add to category embeddings
        if len(self.category_embeddings.get(correct_category, [])) == 0:
            self.category_embeddings[correct_category] = np.array([query_embedding])
        else:
            self.category_embeddings[correct_category] = np.vstack([
                self.category_embeddings[correct_category], 
                query_embedding
            ])
            
        # Update centroid
        self.category_centroids[correct_category] = np.mean(
            self.category_embeddings[correct_category], 
            axis=0
        )
        
        print(f"Added example to category '{correct_category}': '{query}'")
