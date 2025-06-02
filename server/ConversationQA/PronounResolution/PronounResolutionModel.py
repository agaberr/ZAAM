import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
import os
from tqdm import tqdm
import random

class PronounResolutionModel(nn.Module):
    ############################################ INITIALIZATION ########################################################

    def __init__(self, bert_model_name='bert-base-uncased', dropout_rate=0.1):
        super(PronounResolutionModel, self).__init__()
        ## b3ml loading ll bert
        ### w bzbt size hidden layers
        self.bert =BertModel.from_pretrained(bert_model_name)
        self.hidden_size = self.bert.config.hidden_size
        
        ##### encoder bta3 entities 
        self.candidate_bert= BertModel.from_pretrained(bert_model_name)
        
        self.attention =nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size),nn.Tanh()
        ,nn.Dropout(dropout_rate),nn.Linear(self.hidden_size, 1))

            ### layer ll overfitting        
        self.dropout = nn.Dropout(dropout_rate)

    ################################################## PREDICT #######################################################

    def forward(self,input_ids,attention_mask,pronoun_position,candidate_input_ids,candidate_attention_masks,num_candidates):
        batchSize = input_ids.size(0)
        
        outputs =self.bert(input_ids=input_ids,attention_mask=attention_mask,return_dict=True)
        
        seqOutput = outputs.last_hidden_state  # [batchSize, seq_len, hidden_size]
        
        
        
        pronounRepresentations = []
        for i in range(batchSize):
            pos= pronoun_position[i]
            pronounRepresentations.append(seqOutput[i, pos])
        
        pronounRepresentations =torch.stack(pronounRepresentations)  # [batchSize, hidden_size]
        
        maxCandidates= candidate_input_ids.size(1)
        reps = []
        
        
#           ## mehtag reshaping 3shan processing        
        ids = candidate_input_ids.view(-1, candidate_input_ids.size(-1))
        masks = candidate_attention_masks.view(-1, candidate_attention_masks.size(-1))
        
        # Get candidate embeddings
        candidateOutputs=self.candidate_bert(input_ids=ids,attention_mask=masks,return_dict=True)
                                               
                                               
        ############ ba3ml reshape candidate
        candidateEmbeds =candidateOutputs.last_hidden_state[:, 0]  # [batchSize*maxCandidates, hidden_size]
        ###### Brg3 el reshape tany to [batchSize, maxCandidates, hidden_size]
        candidateBackEmbeds =candidateEmbeds.view(batchSize, maxCandidates, -1)
        


        ###### scores bta3et kol entity
        scores=[]
        for i in range(batchSize):
            
            
            n_cand=num_candidates[i].item()
            
            # Expand pronoun representation for each candidate
            Pronoun=pronounRepresentations[i].unsqueeze(0).expand(n_cand,-1)  # [n_cand, hidden_size]
            ###### Concatenate pronoun and candidate representations
            reps = torch.cat([Pronoun,candidateBackEmbeds[i, :n_cand]],dim=1)
            
            #### Score each candidate
            Cscores =self.attention(reps).squeeze(-1)  # [n_cand]
            

            paddedScores =torch.full((maxCandidates,),float('-inf'),device=Cscores.device)
            paddedScores[:n_cand]=Cscores
            
            scores.append(paddedScores)

        ################ # [batchSize, maxCandidates]   #####
        scores = torch.stack(scores)  
        
        return scores
