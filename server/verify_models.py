import os
import sys
import torch

def main():
    """Verify that all required models can be loaded"""
    print("Starting model verification...")
    
    # Add ConversationQA to path
    conversation_qa_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ConversationQA")
    if conversation_qa_path not in sys.path:
        sys.path.append(conversation_qa_path)
    
    # Check if the Models directory exists
    models_dir = os.path.join(conversation_qa_path, "Models")
    if not os.path.exists(models_dir):
        print(f"ERROR: Models directory not found at {models_dir}")
        return False
    
    # Check if model files exist
    required_models = [
        "bert_seq2seq_ner.pt",
        "pronoun_resolution_model_full.pt",
        "extractiveQA.pt",
        "vectorizer.pkl",
        "classifier_model.pkl"
    ]
    
    missing_models = []
    for model_name in required_models:
        model_path = os.path.join(models_dir, model_name)
        if not os.path.exists(model_path):
            missing_models.append(model_name)
    
    if missing_models:
        print("\nWARNING: The following models are missing:")
        for model in missing_models:
            print(f"  - {model}")
        print("The ConversationQA functionality may not work correctly.\n")
    else:
        print("\nAll required model files are available.\n")
    
    # Try to load each model
    print("Attempting to load models...")
    
    # 1. Try loading extractiveQA.pt
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        extractive_qa_path = os.path.join(models_dir, "extractiveQA.pt")
        
        print(f"Loading extractiveQA.pt on {device}...")
        model_state = torch.load(extractive_qa_path, map_location=device)
        print("  => SUCCESS: extractiveQA.pt loaded successfully")
    except Exception as e:
        print(f"  => ERROR loading extractiveQA.pt: {str(e)}")
    
    # 2. Try loading bert_seq2seq_ner.pt
    try:
        bert_ner_path = os.path.join(models_dir, "bert_seq2seq_ner.pt")
        
        print(f"Loading bert_seq2seq_ner.pt on {device}...")
        model_state = torch.load(bert_ner_path, map_location=device)
        print("  => SUCCESS: bert_seq2seq_ner.pt loaded successfully")
    except Exception as e:
        print(f"  => ERROR loading bert_seq2seq_ner.pt: {str(e)}")
    
    # 3. Try loading pronoun_resolution_model_full.pt
    try:
        pronoun_path = os.path.join(models_dir, "pronoun_resolution_model_full.pt")
        
        print(f"Loading pronoun_resolution_model_full.pt on {device}...")
        model_state = torch.load(pronoun_path, map_location=device)
        print("  => SUCCESS: pronoun_resolution_model_full.pt loaded successfully")
    except Exception as e:
        print(f"  => ERROR loading pronoun_resolution_model_full.pt: {str(e)}")
    
    # 4. Try loading other models
    try:
        import joblib
        
        vectorizer_path = os.path.join(models_dir, "vectorizer.pkl")
        print(f"Loading vectorizer.pkl...")
        vectorizer = joblib.load(vectorizer_path)
        print("  => SUCCESS: vectorizer.pkl loaded successfully")
        
        classifier_path = os.path.join(models_dir, "classifier_model.pkl")
        print(f"Loading classifier_model.pkl...")
        classifier = joblib.load(classifier_path)
        print("  => SUCCESS: classifier_model.pkl loaded successfully")
    except Exception as e:
        print(f"  => ERROR loading joblib models: {str(e)}")
    
    print("\nVerification complete!")
    print("Run the server with 'python app.py' to start using the models.")

if __name__ == "__main__":
    main() 