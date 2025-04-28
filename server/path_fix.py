#!/usr/bin/env python3
"""
Path Fix Utility for ZAAM Server

This script fixes hardcoded Windows paths in the ZAAM server codebase,
replacing them with proper relative paths for cross-platform compatibility.
"""

import os
import sys
import re
from pathlib import Path

# Get the absolute path of the project root directory
project_root = Path(__file__).parent.absolute()

# Add to the Python path if not already there
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"Added {project_root} to Python path")

def fix_paths():
    """Fix hardcoded paths in all relevant files"""
    print("ZAAM Path Fix Utility")
    print("=====================")
    
    # Get the base directory (server)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Base directory: {base_dir}")
    
    # Create Models directory if it doesn't exist
    models_dir = os.path.join(base_dir, 'ConversationQA', 'Models')
    os.makedirs(models_dir, exist_ok=True)
    print(f"Models directory: {models_dir}")
    
    # Files to update
    files_to_update = {
        os.path.join(base_dir, 'ConversationQA', 'QA.py'): [
            (r'D:\\College\\Senior-2\\GP\\ZAAM project\\ZAAM\\server\\ConversationQA\\Models\\bert_seq2seq_ner.pt',
             'get_model_path("bert_seq2seq_ner.pt")'),
            (r'D:\\College\\Senior-2\\GP\\ZAAM project\\ZAAM\\server\\ConversationQA\\Models\\pronoun_resolution_model_full.pt',
             'get_model_path("pronoun_resolution_model_full.pt")'),
            (r'D:\\College\\Senior-2\\GP\\ZAAM project\\ZAAM\\server\\ConversationQA\\Models\\extractiveQA.pt',
             'get_model_path("extractiveQA.pt")'),
            (r'D:\\College\\Senior-2\\GP\\ZAAM project\\ZAAM\\server\\ConversationQA\\Models\\classifier_model.pkl',
             'get_model_path("classifier_model.pkl")'),
            (r'D:\\College\\Senior-2\\GP\\ZAAM project\\ZAAM\\server\\ConversationQA\\Models\\vectorizer.pkl',
             'get_model_path("vectorizer.pkl")')
        ],
        os.path.join(base_dir, 'ConversationQA', 'NameEntityModel', 'TopicExtractionModel.py'): [
            (r'D:\\College\\Senior-2\\GP\\ZAAM project\\ZAAM\\server\\ConversationQA\\Models\\bert_seq2seq_ner.pt',
             'get_model_path("bert_seq2seq_ner.pt")')
        ]
    }
    
    # Add the get_model_path function to each file that needs it
    get_model_path_function = """
# Define function to get model path
def get_model_path(model_filename):
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the Models directory path
    models_dir = os.path.join(current_dir, 'Models')
    # Return the full path to the model file
    return os.path.join(models_dir, model_filename)
"""

    get_model_path_function_ner = """
# Define function to get model path
def get_model_path(model_filename):
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory (ConversationQA)
    parent_dir = os.path.dirname(current_dir)
    # Get the Models directory path
    models_dir = os.path.join(parent_dir, 'Models')
    # Return the full path to the model file
    return os.path.join(models_dir, model_filename)
"""
    
    # Process each file
    for file_path, replacements in files_to_update.items():
        if os.path.exists(file_path):
            print(f"\nProcessing {os.path.basename(file_path)}...")
            
            try:
                # Read the file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check if we need to add the get_model_path function
                if 'get_model_path' not in content and 'import os' in content:
                    # Check if it's the NER file
                    if 'TopicExtractionModel' in file_path:
                        # Insert after the imports
                        import_match = re.search(r'import os.*?$', content, re.MULTILINE)
                        if import_match:
                            position = import_match.end()
                            content = content[:position] + get_model_path_function_ner + content[position:]
                            print("Added get_model_path function to NER file")
                    else:
                        # Insert after the imports
                        import_match = re.search(r'import os.*?$', content, re.MULTILINE)
                        if import_match:
                            position = import_match.end()
                            content = content[:position] + get_model_path_function + content[position:]
                            print("Added get_model_path function to file")
                
                # Perform all replacements
                updated_content = content
                for old_path, new_path in replacements:
                    # Count occurrences
                    count = updated_content.count(old_path)
                    if count > 0:
                        updated_content = updated_content.replace(old_path, new_path)
                        print(f"Replaced {count} occurrences of {old_path}")
                
                # Write back if changes were made
                if content != updated_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(updated_content)
                    print(f"Updated {file_path}")
                else:
                    print(f"No changes needed in {file_path}")
            except Exception as e:
                print(f"Error updating {file_path}: {e}")
        else:
            print(f"Warning: File {file_path} not found")
    
    print("\nPath fixing completed!")
    print("Remember to transfer your model files to the ConversationQA/Models directory!")

if __name__ == "__main__":
    fix_paths() 