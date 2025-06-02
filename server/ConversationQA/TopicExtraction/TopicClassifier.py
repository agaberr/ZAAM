import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class TopicClassifier:
    def __init__(self, model_name='all-MiniLM-L6-v2', threshold=0.5):
       
        self.model = SentenceTransformer(model_name)
        self.categories = []
        self.categoryEmbeddings = {}
        self.categoryCentroids = {}
        self.confidenceThreshold = threshold
        
    def add_examples(self, category_examples):
        self.categories = list(category_examples.keys())
        
        for category, examples in category_examples.items():
            # print("henaa\n")
            self.categoryEmbeddings[category] =self.model.encode(examples)
            # print("aloooooooooooooooooooooooooooooooooo  ",self.categoryEmbeddings[category].shape)
            self.categoryCentroids[category] =np.mean(self.categoryEmbeddings[category], axis=0)

    
    def getEmbedding(self, query):

        #### bcalll model 3shan a7seb embedding


        em = self.model.encode([query])[0]
        return em
    
    def classify(self, query, threshold=0.5):
        emb = self.model.encode([query])[0]

        # bagebb similarity
        out = self.classifySimilarity(emb, threshold)
        return out
  
    def classifySimilarity(self, query_embedding, threshold):
        categoryScores = []
        
        for category in self.categories:
            ############## Half 3la kol categories el homa 3 ely 3ndyy #############


            for emb in self.categoryEmbeddings[category]:

                ##############  bageeb cosine similarity 3la kol embedding 3ndy ################

                Similarity = cosine_similarity([query_embedding], [emb])[0][0]
                
                ############### ba7outo 3ndy fi scoress 
                categoryScores.append((category, Similarity))

            ########### bag3bb awl 3 3ndy 
        topNKNN = sorted(categoryScores,key=lambda x: x[1], reverse=True)[:3]

        categories = [cat for cat, sim in topNKNN]
        voteCounts = {}

        ### bageeb votess 

        ################## 3shan a7seb confidence 
        for cat in categories:
            if cat in voteCounts:
                voteCounts[cat] += 1
            else:
                voteCounts[cat] = 1
        
        bestCategory = max(voteCounts, key=voteCounts.get)
        confidenceScores = [sim for cat, sim in categoryScores if cat == bestCategory]
                # ba7seb average confid
        confidence = sum(confidenceScores) / len(confidenceScores)
       


       ## lw confiden 2olyal awyy ###
        if confidence < threshold:
            return "unknown", confidence
        
        return bestCategory, confidence

    def TopicClassify(self, query):
        query_embedding = self.model.encode([query])[0]
        
       
        category, confidence = self.classifySimilarity(query_embedding, 0)
        
        # lw confidence smaller than threshold then i will apply fall back system and ask user
        if confidence < self.confidenceThreshold:
            # Generate confirmation message
            # print("alooooooooooooooooooooo confid",confidence)
            if category != "unknown":
                message = f"I think you're asking about {category} is that right? "
                message += "Please confirm if this is about football, cooking, or news."
                
                return {
                    "result": "needs_confirmation",
                    "suggested_category": category,
                    "confidence": confidence,
                    "message": message,
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
                    "query": query,
                    "query_embedding": query_embedding
                }
        else:
            return {
                "result": "confident",
                "category": category,
                "confidence": confidence
            }
    
    def addExample(self, query, correct_category, queryEmbedding=None):

        if correct_category not in self.categories or correct_category == "unknown":
            print(f"error")
            return
            
        if queryEmbedding is None:

            ### lw embedding msh 3ndy bagebo  ###########


            queryEmbedding = self.model.encode([query])[0]
            
        ############################## b3mal adding new example hena ################

        if len(self.categoryEmbeddings.get(correct_category, [])) == 0:
            self.categoryEmbeddings[correct_category] = np.array([queryEmbedding])
        else:
            self.categoryEmbeddings[correct_category] = np.vstack([
                self.categoryEmbeddings[correct_category], 
                queryEmbedding
            ])

        