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
BERT_MODEL = 'bert-base-cased'  # Using cased variant as NER is case-sensitive
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base_path = os.path.dirname(__file__)


# CoNLL-2003 has these entity types
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
        super(BERTSeq2SeqForNER, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.num_labels = num_labels
        
        
        # Decoder
        # Simple linear layer for token classification
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
        # Additional seq2seq components
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=self.bert.config.hidden_size // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
    
    def forward(self, input_ids, attention_mask, labels=None):
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Process through LSTM
        lstm_output, _ = self.lstm(sequence_output)
        
        # Apply dropout
        lstm_output = self.dropout(lstm_output)
        
        # Apply classifier
        logits = self.classifier(lstm_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)
        
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

bert_seq2seq_ner_path = os.path.join(base_path, "../", "Models", "bert_seq2seq_ner.pt")

class NERPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the model
        self.model = BERTSeq2SeqForNER(BERT_MODEL, len(tag2idx))
        
        self.model.load_state_dict(torch.load(bert_seq2seq_ner_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Initialize tokenizer - Use Fast version
        self.tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL)

    def predict(self, text):
        # Tokenize input
        words = text.lower().split()
        inputs = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=MAX_LEN
        )

        # Move tensors to device
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        # Get predictions
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            predictions = torch.argmax(logits, dim=2).cpu().numpy()[0]

        # Get word-to-token mapping
        word_ids = inputs.word_ids(batch_index=0)

        # Map predictions to words
        word_predictions = []
        prev_word_idx = None

        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx == prev_word_idx:
                continue

            word = words[word_idx]
            tag_idx = predictions[token_idx]
            tag = idx2tag.get(tag_idx, "O")

            word_predictions.append((word, tag))
            prev_word_idx = word_idx

        return word_predictions
