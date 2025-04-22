#!/usr/bin/env python3
"""
Direct download script for ZAAM model files
This script downloads the model files directly from Google Drive URLs provided
"""

import os
import sys
import zipfile
import requests
import time
from tqdm import tqdm

def create_models_dir():
    """Create the models directory if it doesn't exist"""
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ConversationQA', 'Models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    return models_dir

def download_file(url, destination):
    """
    Download a file from a URL with progress bar
    """
    try:
        print(f"Downloading {os.path.basename(destination)}...")
        response = requests.get(url, stream=True)
        
        # Check if we're getting an HTML login page instead of the file
        if response.headers.get('content-type', '').startswith('text/html'):
            print(f"Warning: Received HTML content instead of a file. The URL might require authentication.")
            print(f"Please download this file manually from: {url}")
            return False
            
        total_size = int(response.headers.get('content-length', 0))
        
        # Show progress bar
        with open(destination, 'wb') as f, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
                
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

def extract_zip(zip_path, extract_dir):
    """Extract a ZIP file to the specified directory"""
    try:
        print(f"Extracting {os.path.basename(zip_path)}...")
        
        # First check if it's really a ZIP file by reading the first few bytes
        with open(zip_path, 'rb') as f:
            header = f.read(4)
            if header != b'PK\x03\x04':
                print("This doesn't appear to be a valid ZIP file.")
                print("The file might be a direct download from Google Drive, which isn't in ZIP format.")
                print("Please download the file manually.")
                return False
                
        with zipfile.ZipFile(zip_path) as zf:
            # Get list of files in the ZIP
            file_list = zf.namelist()
            
            # Create a subdirectory with the ZIP name if there are multiple files
            if len(file_list) > 1 and not any('/' in name for name in file_list):
                base_name = os.path.splitext(os.path.basename(zip_path))[0]
                extract_subdir = os.path.join(extract_dir, base_name)
                if not os.path.exists(extract_subdir):
                    os.makedirs(extract_subdir)
                zf.extractall(extract_subdir)
                print(f"Extracted {len(file_list)} files to {extract_subdir}")
            else:
                # Extract directly to the target directory
                zf.extractall(extract_dir)
                print(f"Extracted {len(file_list)} files to {extract_dir}")
        
        return True
    except Exception as e:
        print(f"Error extracting ZIP file: {e}")
        return False

def update_model_paths():
    """Update hardcoded model paths in the code files"""
    print("\nUpdating model paths in code files...")
    
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ConversationQA', 'Models')
    models_dir = models_dir.replace('\\', '\\\\')  # Escape backslashes for string literals
    
    # Update QA.py
    qa_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ConversationQA', 'QA.py')
    if os.path.exists(qa_path):
        try:
            with open(qa_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            content = content.replace(
                r'D:\College\Senior-2\GP\ZAAM project\ZAAM\server\ConversationQA\Models\bert_seq2seq_ner.pt',
                f'{models_dir}\\\\bert_seq2seq_ner.pt'
            )
            content = content.replace(
                r'D:\College\Senior-2\GP\ZAAM project\ZAAM\server\ConversationQA\Models\pronoun_resolution_model_full.pt',
                f'{models_dir}\\\\pronoun_resolution_model_full.pt'
            )
            content = content.replace(
                r'D:\College\Senior-2\GP\ZAAM project\ZAAM\server\ConversationQA\Models\extractiveQA.pt',
                f'{models_dir}\\\\extractiveQA.pt'
            )
            
            with open(qa_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"Updated model paths in {qa_path}")
        except Exception as e:
            print(f"Error updating QA.py: {e}")
    
    # Update TopicExtractionModel.py
    topic_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ConversationQA', 'NameEntityModel', 'TopicExtractionModel.py')
    if os.path.exists(topic_path):
        try:
            with open(topic_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            content = content.replace(
                r'D:\College\Senior-2\GP\ZAAM project\ZAAM\server\ConversationQA\Models\bert_seq2seq_ner.pt',
                f'{models_dir}\\\\bert_seq2seq_ner.pt'
            )
            
            with open(topic_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"Updated model paths in {topic_path}")
        except Exception as e:
            print(f"Error updating TopicExtractionModel.py: {e}")
    
    print("Model path updates complete!")

def main():
    print("=== ZAAM Model Direct Download ===")
    
    # Create temp directory for downloads
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_downloads')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    # Create models directory
    models_dir = create_models_dir()
    print(f"Models will be saved to: {models_dir}")
    
    # Define the files to download
    files = {
        "bert_seq2seq_ner.zip": "https://drive.google.com/uc?id=1lOSkIPGU4TX7L727OmkBL3f0Fh_W3_Fv&export=download",
        "pronoun_resolution_model_full.zip": "https://drive.google.com/uc?id=1DhlkILm1kzD8gPbEU0NlTUGeDA_uGisk&export=download",
        "extractiveQA.zip": "https://drive.google.com/uc?id=1ph3yhuAz7fmTv8lat8fbdwzlHYSa4y4U&export=download",
        # Add vectorizer.zip if needed
        # "vectorizer.zip": "https://drive.google.com/uc?id=1QqxFX0VhLYKdgWWJBYnE9YVnEP6bZKcY&export=download",
    }
    
    # Download and process each file
    for filename, url in files.items():
        destination = os.path.join(temp_dir, filename)
        
        # Download file
        if download_file(url, destination):
            print(f"Downloaded {filename} successfully to {destination}")
            
            # Extract the zip file
            model_name = os.path.splitext(filename)[0]  # Remove .zip extension
            
            if extract_zip(destination, models_dir):
                print(f"Extracted {filename} to {models_dir}")
            else:
                print(f"Failed to extract {filename}")
                # Try to rename the file if extraction failed (might be direct file download)
                try:
                    import shutil
                    final_dest = os.path.join(models_dir, model_name)
                    shutil.copy2(destination, final_dest)
                    print(f"Copied {filename} as {model_name} to {models_dir}")
                except Exception as e:
                    print(f"Error copying file: {e}")
        else:
            print(f"Failed to download {filename}")
            print(f"\nManual download instructions:")
            print(f"1. Visit: {url.replace('uc?id=', 'file/d/')}")
            print(f"2. Click 'Download' button")
            print(f"3. Save the file as {filename}")
            print(f"4. Extract the contents to: {models_dir}\n")
    
    # Print instructions for manual download if needed
    print("\nIf automatic download failed, please follow these steps:")
    print("1. Visit the Google Drive links:")
    print("   - bert_seq2seq_ner.pt:")
    print("     https://drive.google.com/file/d/1lOSkIPGU4TX7L727OmkBL3f0Fh_W3_Fv/view")
    print("   - pronoun_resolution_model_full.pt:")
    print("     https://drive.google.com/file/d/1DhlkILm1kzD8gPbEU0NlTUGeDA_uGisk/view")
    print("   - extractiveQA.pt:")
    print("     https://drive.google.com/file/d/1ph3yhuAz7fmTv8lat8fbdwzlHYSa4y4U/view")
    print(f"2. Download the files to your computer")
    print(f"3. Extract them if they're zip files or rename them to .pt files as needed")
    print(f"4. Move all .pt files to: {models_dir}")
    
    # Update model paths in code files
    update_model_paths()
    
    # Verify that all required models exist
    required_models = [
        "bert_seq2seq_ner.pt",
        "pronoun_resolution_model_full.pt", 
        "extractiveQA.pt"
    ]
    
    missing_models = []
    for model in required_models:
        model_path = os.path.join(models_dir, model)
        if not os.path.exists(model_path):
            missing_models.append(model)
    
    if missing_models:
        print("\nWARNING: The following models are still missing:")
        for model in missing_models:
            print(f"  - {model}")
        print("Please download them manually using the instructions above.")
    else:
        print("\nAll required models have been successfully downloaded!")
    
    print("\nSetup complete!")
    print("Next step: Run 'python app.py' to start the server")
    
    # Clean up temp directory
    try:
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        os.rmdir(temp_dir)
        print("Cleaned up temporary files")
    except Exception as e:
        print(f"Error cleaning up temporary files: {e}")

if __name__ == "__main__":
    # Install required packages if not already installed
    try:
        import importlib
        
        required_packages = ["requests", "tqdm"]
        for package in required_packages:
            try:
                importlib.import_module(package)
            except ImportError:
                print(f"Installing required package: {package}")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except Exception as e:
        print(f"Warning: Failed to install required packages: {str(e)}")
    
    main() 