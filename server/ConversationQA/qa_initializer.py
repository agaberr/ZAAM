import os
import sys
from pathlib import Path

def initialize_qa_system():
    try:
        conversation_qa_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        if conversation_qa_path not in sys.path:
            sys.path.append(conversation_qa_path)
        
        from qa_singleton import get_qa_instance
        return True
    except:
        return False 