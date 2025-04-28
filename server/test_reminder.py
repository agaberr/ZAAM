#!/usr/bin/env python3
"""
Test script for the reminder functionality
"""

import os
import sys
import traceback
from pathlib import Path

# Add the current directory to the Python path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"Added {project_root} to Python path")

def test_reminder_nlp():
    """Test the ReminderNLP functionality"""
    print("Testing ReminderNLP...")
    
    try:
        print("Importing ReminderNLP...")
        from models.reminder import ReminderNLP
        print("Import successful")
        
        # Check if model files exist
        reminders_dir = os.path.join(project_root, 'reminders')
        model_path = os.path.join(reminders_dir, 'reminder_model.pth')
        tokenizer_path = os.path.join(reminders_dir, 'reminder_tokenizer.pkl')
        intent_encoder_path = os.path.join(reminders_dir, 'reminder_intent_encoder.pkl')
        slot_encoder_path = os.path.join(reminders_dir, 'reminder_slot_encoder.pkl')
        
        print(f"Checking model files in {reminders_dir}:")
        print(f"  Model: {os.path.exists(model_path)}")
        print(f"  Tokenizer: {os.path.exists(tokenizer_path)}")
        print(f"  Intent encoder: {os.path.exists(intent_encoder_path)}")
        print(f"  Slot encoder: {os.path.exists(slot_encoder_path)}")
        
        # Initialize the ReminderNLP model
        print("Initializing ReminderNLP model...")
        nlp = ReminderNLP()
        print("ReminderNLP initialized successfully!")
        
        # Test some example texts
        test_texts = [
            "Remind me to buy groceries tomorrow",
            "Schedule a meeting with John at 3pm",
            "What's on my calendar for today?",
            "Add dentist appointment to my calendar for next week"
        ]
        
        for text in test_texts:
            print(f"\nProcessing: '{text}'")
            result = nlp.process_text(text)
            
            print(f"Intent: {result.get('predicted_intent', 'unknown')}")
            print(f"Action: {result.get('predicted_action', 'unknown')}")
            print(f"Time: {result.get('predicted_time', 'unknown')}")
            print(f"Days offset: {result.get('days_offset', 0)}")
    
    except Exception as e:
        print(f"Error testing ReminderNLP: {str(e)}")
        print("\nTraceback:")
        traceback.print_exc()
        return False
        
    print("\nAll tests completed successfully!")
    return True

if __name__ == "__main__":
    success = test_reminder_nlp()
    sys.exit(0 if success else 1) 