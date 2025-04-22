import os
import sys
import tempfile
import shutil
import subprocess
import zipfile
import urllib.request
from pathlib import Path
import webbrowser

def check_dependencies():
    """Check if required dependencies are installed"""
    missing_packages = []
    
    try:
        import gdown
    except ImportError:
        missing_packages.append("gdown")
    
    if missing_packages:
        print(f"Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            return True
        except Exception as e:
            print(f"Error installing dependencies: {e}")
            return False
    
    return True

def create_models_dir():
    """Create the models directory if it doesn't exist"""
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ConversationQA', 'Models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    return models_dir

def extract_file(file_path, extract_dir):
    """Extract ZIP file using Python libraries"""
    try:
        file_name = os.path.basename(file_path)
        print(f"Extracting {file_name}...")
        
        if file_path.lower().endswith('.zip'):
            # Use zipfile for ZIP extraction
            with zipfile.ZipFile(file_path) as zf:
                zf.extractall(extract_dir)
            print(f"Successfully extracted {file_name}")
            return True
        else:
            print(f"Unsupported file format: {file_path}")
            return False
        
    except Exception as e:
        print(f"Error extracting file: {e}")
        return False

def download_from_google_drive():
    """Download model files from Google Drive"""
    import gdown
    
    models_dir = create_models_dir()
    print(f"Models will be saved to: {models_dir}")
    
    # Define file URLs
    file_urls = {
        "Models.zip": "https://drive.google.com/uc?id=1lrKO1AFffUs9bi4OsqSxscRkMV4Q_kCo&export=download",
        "extractiveQA.zip": "https://drive.google.com/uc?id=1ph3yhuAz7fmTv8lat8fbdwzlHYSa4y4U&export=download",
        "vectorizer.zip": "https://drive.google.com/uc?id=1QqxFX0VhLYKdgWWJBYnE9YVnEP6bZKcY&export=download"
    }
    
    success_count = 0
    
    # Temporary directory for downloads
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Download each file
        for file_name, file_url in file_urls.items():
            output_path = os.path.join(temp_dir, file_name)
            print(f"Downloading {file_name}...")
            
            try:
                downloaded = gdown.download(file_url, output_path, quiet=False)
                
                if downloaded:
                    # Check if file exists and has content
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        print(f"Successfully downloaded {file_name}")
                        
                        # Extract the file
                        if extract_file(output_path, models_dir):
                            print(f"Successfully extracted {file_name}")
                            success_count += 1
                        else:
                            print(f"Failed to extract {file_name}")
                    else:
                        print(f"Failed to download {file_name} (file empty or does not exist)")
                else:
                    print(f"Failed to download {file_name}")
            except Exception as e:
                print(f"Error downloading {file_name}: {e}")
    
    if success_count == len(file_urls):
        print("All files downloaded and extracted successfully!")
        return True
    else:
        print(f"Downloaded and extracted {success_count} out of {len(file_urls)} files")
        return False

def manual_download_instructions():
    """Provide manual download instructions"""
    drive_url = "https://drive.google.com/drive/folders/1wFJAiELAhvXB6rIiokTJSWhUnq3aSX2K"
    models_dir = create_models_dir()
    
    print("\n====== MANUAL DOWNLOAD INSTRUCTIONS ======")
    print(f"1. Open this Google Drive URL in your browser: {drive_url}")
    print("2. Download the following files:")
    print("   - Models.zip (contains bert_seq2seq_ner.pt and pronoun_resolution_model_full.pt)")
    print("   - extractiveQA.zip (contains extractiveQA.pt)")
    print("   - vectorizer.zip (contains vectorizer.pkl and classifier_model.pkl)")
    print(f"3. Extract the files to: {models_dir}")
    print("4. Run this script again after downloading and extracting the files to update the paths")
    print("=======================================\n")
    
    # Open the browser with the Google Drive URL
    print("Opening Google Drive folder in your browser...")
    webbrowser.open(drive_url)
    
    # Ask if user wants to continue with the script after downloading
    downloads_complete = input("Have you downloaded and extracted the files? (yes/no): ").strip().lower()
    
    if downloads_complete == 'yes':
        # Check if the files are in the models directory
        expected_files = [
            os.path.join(models_dir, 'bert_seq2seq_ner.pt'),
            os.path.join(models_dir, 'pronoun_resolution_model_full.pt'),
            os.path.join(models_dir, 'extractiveQA.pt'),
            os.path.join(models_dir, 'vectorizer.pkl'),
            os.path.join(models_dir, 'classifier_model.pkl')
        ]
        
        missing_files = [f for f in expected_files if not os.path.exists(f)]
        
        if missing_files:
            print("Some files are still missing:")
            for file in missing_files:
                print(f"  - {os.path.basename(file)}")
            
            # Ask about downloaded ZIP files
            zip_files = [
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "Models.zip"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "extractiveQA.zip"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectorizer.zip")
            ]
            
            found_zips = [f for f in zip_files if os.path.exists(f)]
            
            if found_zips:
                print("\nFound downloaded ZIP files that need extraction:")
                for zip_file in found_zips:
                    print(f"  - {os.path.basename(zip_file)}")
                
                extract_prompt = input("Do you want to extract these files? (yes/no): ").strip().lower()
                if extract_prompt == 'yes':
                    for zip_file in found_zips:
                        if extract_file(zip_file, models_dir):
                            print(f"Successfully extracted {os.path.basename(zip_file)}")
                        else:
                            print(f"Failed to extract {os.path.basename(zip_file)}")
            
            return False
        
        return True
    else:
        print("Please download and extract the files before continuing.")
        return False

def update_model_paths():
    """Update hardcoded model paths in code files to use the correct paths"""
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ConversationQA', 'Models')
    models_dir = models_dir.replace('\\', '\\\\')  # Escape backslashes for string literals
    
    qa_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ConversationQA', 'QA.py')
    topic_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ConversationQA', 'NameEntityModel', 'TopicExtractionModel.py')
    
    updated_files = []
    
    # Update QA.py if it exists
    if os.path.exists(qa_path):
        try:
            with open(qa_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Update model paths
            original_content = content
            
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
            
            # Only write if content changed
            if content != original_content:
                with open(qa_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                updated_files.append(qa_path)
                print(f"Updated model paths in {qa_path}")
            else:
                print(f"No changes needed in {qa_path}")
        except Exception as e:
            print(f"Error updating {qa_path}: {str(e)}")
    
    # Update TopicExtractionModel.py if it exists
    if os.path.exists(topic_path):
        try:
            with open(topic_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Update model paths
            original_content = content
            
            content = content.replace(
                r'D:\College\Senior-2\GP\ZAAM project\ZAAM\server\ConversationQA\Models\bert_seq2seq_ner.pt',
                f'{models_dir}\\\\bert_seq2seq_ner.pt'
            )
            
            # Only write if content changed
            if content != original_content:
                with open(topic_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                updated_files.append(topic_path)
                print(f"Updated model paths in {topic_path}")
            else:
                print(f"No changes needed in {topic_path}")
        except Exception as e:
            print(f"Error updating {topic_path}: {str(e)}")
    
    return len(updated_files) > 0

def verify_models():
    """Check if all models are available"""
    models_dir = create_models_dir()
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
        print("\nThe following models are missing:")
        for model in missing_models:
            print(f"  - {model}")
        return False
    else:
        print("\nAll required models are available.")
        return True

def main():
    print("=== ZAAM AI Model Setup ===")
    
    # Check dependencies
    if not check_dependencies():
        print("Failed to install required dependencies.")
        sys.exit(1)
    
    # Verify if models are available
    if verify_models():
        print("All required models are already available.")
        # Just update paths
        if update_model_paths():
            print("Updated model paths in code files.")
        else:
            print("No path updates were needed.")
        print("\nSetup complete! You can now run the server.")
        return True
    
    print("Some models are missing. Let's download them.")
    
    # Try downloading from Google Drive
    print("\nMethod 1: Automatic download from Google Drive")
    if download_from_google_drive():
        print("Automatic download successful!")
        update_model_paths()
        print("\nSetup complete! You can now run the server.")
        return True
    
    # If automatic download fails, provide manual instructions
    print("\nMethod 2: Manual download")
    if manual_download_instructions():
        print("Manual download successful!")
        update_model_paths()
        print("\nSetup complete! You can now run the server.")
        return True
    
    print("\nCould not complete model setup. Please try again or contact support.")
    return False

if __name__ == "__main__":
    main() 