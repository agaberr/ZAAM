import os
import sys
from pathlib import Path

def initialize_qa_system():
    """Pre-initialize the QA system to load models at startup"""
    try:
        # Add ConversationQA to the path if needed
        conversation_qa_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        if conversation_qa_path not in sys.path:
            sys.path.append(conversation_qa_path)
        
        # Import and initialize the QA singleton
        from qa_singleton import get_qa_instance
        
        print("Pre-initializing QA system and loading models...")
        qa_system = get_qa_instance()
        print(f"QA system initialized on device: {qa_system.device}")
        print(f"ConversationQA models loaded successfully!")
        return True
    except Exception as e:
        print(f"Error initializing QA system: {str(e)}")
        return False 