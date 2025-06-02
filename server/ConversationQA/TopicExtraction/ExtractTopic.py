import datetime
from typing import List, Tuple, Dict, Any
import numpy as np


def ExtractTopic(predictions, important_keywords):
    topics = []
    important_words = []
    currentTopic = []
    current_tag = None
    
    for word, t in predictions:
       
       #### bahsouf kelma mwgoda 3ndy fi list wla eh

        if word.lower() in important_keywords:
            important_words.append(word)
        
        #
        ## handle tags ll kol kelma lw hya b tbd2 B- bashousf lw fi b3dha I- wla eh lw
        ## kda byb2o entity wahda 

        if t.startswith("B-"):
        
            if currentTopic:
        
        ### b3ml save ll current topic lw mwgoda
                topics.append(" ".join(currentTopic))
                currentTopic = []
            currentTopic.append(word)
            current_tag = t
            
        elif t.startswith("I-"):
         
         ## hena ba bhandle lw hya I- lw mafesh current ablha yba hya deh ablha O
            if currentTopic:
                currentTopic.append(word)
            else:
                currentTopic = [word]
                current_tag = t
        else:
        
            if currentTopic:
                ### hena ba lw tag kan o

                topics.append(" ".join(currentTopic))
                
                currentTopic = []
           
            current_tag = None

    if currentTopic:
        topics.append(" ".join(currentTopic))
    
    return topics, important_words