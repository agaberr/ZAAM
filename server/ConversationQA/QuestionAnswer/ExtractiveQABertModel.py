import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizerFast, AdamW, get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm

class BertForQA(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super(BertForQA, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.qa_outputs = nn.Linear(self.bert.config.hidden_size, 2) 
        
    def forward(self, input_ids, attention_mask, token_type_ids=None, start_positions=None, end_positions=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        sequence_output = outputs.last_hidden_state
        logits = self.qa_outputs(sequence_output)
        
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        loss = None
        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2
        
        return {
            "loss": loss,
            "start_logits": start_logits,
            "end_logits": end_logits
        }