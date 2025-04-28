#!/usr/bin/env python3
"""
This script fixes the reminder model by updating the class reference
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add the current directory to the Python path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"Added {project_root} to Python path")

# Define the NERIntentModel class exactly the same as the original
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

def fix_reminder_model():
    print("Fixing reminder model...")
    
    # Define paths
    reminders_dir = os.path.join(project_root, 'reminders')
    original_model_path = os.path.join(reminders_dir, 'reminder_model.pth')
    backup_model_path = os.path.join(reminders_dir, 'reminder_model.pth.bak')
    
    # Check if model file exists
    if not os.path.exists(original_model_path):
        print(f"Error: Model file not found at {original_model_path}")
        return False
    
    try:
        # Create a backup
        print(f"Creating backup at {backup_model_path}")
        if not os.path.exists(backup_model_path):
            import shutil
            shutil.copy2(original_model_path, backup_model_path)
            print("Backup created")
        else:
            print("Backup already exists, skipping")
        
        # Load the model
        print("Loading model...")
        model_data = torch.load(original_model_path, map_location='cpu')
        print("Model loaded")
        
        # Check if the model is already fixed
        model_type = type(model_data).__name__
        print(f"Model type: {model_type}")
        
        # Extract model parameters
        if hasattr(model_data, 'embedding'):
            vocab_size = model_data.embedding.num_embeddings
            embedding_dim = model_data.embedding.embedding_dim
            
            # Get hidden dimension from LSTM
            hidden_dim = model_data.lstm.hidden_size
            
            # Get num intents from intent_fc
            num_intents = model_data.intent_fc.out_features
            
            # Get num slots from slot_fc
            num_slots = model_data.slot_fc.out_features
            
            print(f"Model parameters:")
            print(f"  vocab_size: {vocab_size}")
            print(f"  embedding_dim: {embedding_dim}")
            print(f"  hidden_dim: {hidden_dim}")
            print(f"  num_intents: {num_intents}")
            print(f"  num_slots: {num_slots}")
            
            # Create a new model
            new_model = NERIntentModel(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_intents=num_intents,
                num_slots=num_slots
            )
            
            # Copy the state dict
            new_model.load_state_dict(model_data.state_dict())
            
            # Save the fixed model
            print("Saving fixed model...")
            torch.save(new_model, original_model_path)
            print("Model saved successfully")
            
            return True
        else:
            print("Model does not have the expected structure, cannot fix automatically")
            return False
    
    except Exception as e:
        print(f"Error fixing model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = fix_reminder_model()
    sys.exit(0 if success else 1) 