import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertModel  # Changed to Fast tokenizer
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import os

# Define constants
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5
BERT_MODEL = 'bert-base-cased' 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_model_path(model_filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    models_dir = os.path.join(parent_dir, 'Models')
    return os.path.join(models_dir, model_filename)

tag2idx = {
    'O': 0,
    'B-PER': 1, 'I-PER': 2,
    'B-ORG': 3, 'I-ORG': 4,
    'B-LOC': 5, 'I-LOC': 6,
    'B-MISC': 7, 'I-MISC': 8
}
idx2tag = {v: k for k, v in tag2idx.items()}

class BERTSeq2SeqForNER(nn.Module):
    def __init__(self, bert_model_name, num_labels):
         #### mafroud a3ml calling bert constructor w a3ml loading
        # load model w
        super(BERTSeq2SeqForNER, self).__init__()
        self.bert =BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.num = num_labels
        
        
        # Decoder
        # Simple linear layer for token classification
        self.classifier=nn.Linear(self.bert.config.hidden_size, self.num)
        
        # Additional seq2seq components
        self.lstm=nn.LSTM(
            ### mafroud henna configuration input size
            input_size=self.bert.config.hidden_size,
    # config hidden size
            hidden_size=self.bert.config.hidden_size // 2,
        
            num_layers=2,
            batch_first=True,

            bidirectional=True
        )
    
    def forward(self, input_ids, attention_mask, labels=None):
        # Get BERT embeddings
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
       

        lstmOutput, _= self.lstm(out.last_hidden_state)
        
        # Apply dropout
        lstmOutput= self.dropout(lstmOutput)
        
        #### classifier 
        logits =self.classifier(lstmOutput)
        
        loss = None
        if labels is not None:
            LOSSfnc = nn.CrossEntropyLoss()
          
            loss= attention_mask.view(-1) == 1
            activeLogits =logits.view(-1, self.num)
            activeLabels= torch.where(
                activeLogits, labels.view(-1), torch.tensor(LOSSfnc.ignore_index).type_as(labels)
            )
            loss =LOSSfnc(activeLogits, activeLabels)
        
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

class NERPredictor:
############################################ INITIALIZATION ########################################################
    def __init__(self):
        self.device=torch.device("cuda" if torch.cuda.is_available()else "cpu")
        model_path = get_model_path('bert_seq2seq_ner.pt')

        #### 3ayeen n3mal evaluation 
        # load model w nenady evaluation
        self.model=BERTSeq2SeqForNER(BERT_MODEL, len(tag2idx))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        ## bwdeha ll device
        self.model.to(self.device)
        
        self.model.eval()

        # Initialize tokenizer 
        self.tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL)


################################################## PREDICT #######################################################

    def predict(self, text):
        # Tokenize input
        words =text.lower().split()
        inputs =self.tokenizer(
            words
            ,is_split_into_words=True,
            ### mafroud henna max length
            return_tensors= 'pt',
            padding= True
            ,truncation=True,max_length=MAX_LEN
        )

        #Get predictions
        with torch.no_grad():
            outputs= self.model(input_ids=inputs['input_ids'].to(self.device), attention_mask= inputs['attention_mask'].to(self.device)
)
            logits= outputs['logits']
            predictions=torch.argmax(logits, dim=2).cpu().numpy()[0]

        # Get word token mapping
        word_ids=inputs.word_ids(batch_index=0)
    #map predictions to words
        wordPredictions=[]
        prevIdx= None

        for idx, wordIdx in enumerate(word_ids):
        #############  bageeb tage with O 
            if not wordIdx or wordIdx==prevIdx:
                continue

            word=words[wordIdx]
            
            tag_idx = predictions[idx]
            
            ### get tag with O
            tag=idx2tag.get(tag_idx, "O")

            wordPredictions.append((word, tag))
            prevIdx= wordIdx

        return wordPredictions
