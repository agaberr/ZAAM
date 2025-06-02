import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizerFast, AdamW, get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm

class BertForQA(nn.Module):
############################################ INITIALIZATION ########################################################

    def __init__(self, model_name="bert-base-uncased"):

        # mafroud a load model bta3 bert 
        ### mn khelal function pretrained
        ####### wazabt size layers
        super(BertForQA, self).__init__()
        self.bert= BertModel.from_pretrained(model_name)
        self.qa_outputs= nn.Linear(self.bert.config.hidden_size, 2) 

############################################ FORWARD PASSSS ########################################################

    def forward(self,input_ids,attention_mask,token_type_ids=None,start_positions=None,end_positions=None):
        outputs= self.bert(
            input_ids=input_ids,attention_mask=attention_mask
            ,token_type_ids=token_type_ids
        )
        
        sequence_output= outputs.last_hidden_state
        logits=self.qa_outputs(sequence_output)
        
        startLogits, endLogits= logits.split(1, dim=-1)
        startLogits=startLogits.squeeze(-1)
        endLogits=endLogits.squeeze(-1)
        
        loss = None
        if start_positions is not None and end_positions is not None:
            LOSSfct= nn.CrossEntropyLoss()
            startLoss=LOSSfct(startLogits, start_positions)
            endLoss=LOSSfct(endLogits, end_positions)
            loss= (startLoss + endLoss) / 2
        
        return {
            "loss": loss,
            "start_logits": startLogits,
            "end_logits": endLogits
        }