import torch
import torch.nn as nn

class NERIntentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_intents, num_slots):
        super(NERIntentModel, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # BiLSTM layer for sequence labeling (NER)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # Linear layer for intent classification
        self.intent_fc = nn.Linear(hidden_dim * 2, num_intents)
        
        # Linear layer for slot tagging (NER)
        self.slot_fc = nn.Linear(hidden_dim * 2, num_slots)

    def forward(self, x):
        # Get token embeddings
        embedded = self.embedding(x)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(embedded)
        
        # Predict intent (classification for entire sentence)
        intent_pred = self.intent_fc(lstm_out[:, 0, :])  # Use the first token (CLS) for intent prediction
        
        # Predict slots (token-level classification)
        slot_pred = self.slot_fc(lstm_out)
        
        return intent_pred, slot_pred 