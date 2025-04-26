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

# Model Architecture
class PronounResolutionModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', dropout_rate=0.1):
        super(PronounResolutionModel, self).__init__()
        
        # BERT for contextualized embeddings
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.hidden_size = self.bert.config.hidden_size
        
        # Candidate encoder
        self.candidate_bert = BertModel.from_pretrained(bert_model_name)
        
        # Attention mechanism for candidate scoring
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size, 1)
        )
        
        # Additional layers
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, input_ids, attention_mask, pronoun_position, 
                candidate_input_ids, candidate_attention_masks, num_candidates):
        
        batch_size = input_ids.size(0)
        
        # Get contextualized embeddings from BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Extract pronoun representations
        pronoun_representations = []
        for i in range(batch_size):
            pos = pronoun_position[i]
            pronoun_representations.append(sequence_output[i, pos])
        
        pronoun_representations = torch.stack(pronoun_representations)  # [batch_size, hidden_size]
        
        # Process candidates
        max_candidates = candidate_input_ids.size(1)
        candidate_representations = []
        
        # Reshape for batch processing
        flat_candidate_ids = candidate_input_ids.view(-1, candidate_input_ids.size(-1))
        flat_candidate_masks = candidate_attention_masks.view(-1, candidate_attention_masks.size(-1))
        
        # Get candidate embeddings
        candidate_outputs = self.candidate_bert(
            input_ids=flat_candidate_ids,
            attention_mask=flat_candidate_masks,
            return_dict=True
        )
        
        # Use [CLS] token as candidate representation
        flat_candidate_embeds = candidate_outputs.last_hidden_state[:, 0]  # [batch_size*max_candidates, hidden_size]
        
        # Reshape back to [batch_size, max_candidates, hidden_size]
        candidate_embeds = flat_candidate_embeds.view(batch_size, max_candidates, -1)
        
        # Score each candidate
        scores = []
        for i in range(batch_size):
            n_cand = num_candidates[i].item()
            
            # Expand pronoun representation for each candidate
            expanded_pronoun = pronoun_representations[i].unsqueeze(0).expand(n_cand, -1)  # [n_cand, hidden_size]
            
            # Concatenate pronoun and candidate representations
            concat_reps = torch.cat([
                expanded_pronoun, 
                candidate_embeds[i, :n_cand]
            ], dim=1)  # [n_cand, hidden_size*2]
            
            # Score each candidate
            cand_scores = self.attention(concat_reps).squeeze(-1)  # [n_cand]
            
            # Pad scores for batch processing
            padded_scores = torch.full((max_candidates,), float('-inf'), device=cand_scores.device)
            padded_scores[:n_cand] = cand_scores
            scores.append(padded_scores)
        
        scores = torch.stack(scores)  # [batch_size, max_candidates]
        
        return scores
