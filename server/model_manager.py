import os
import sys
import logging
import importlib
import subprocess
from pathlib import Path

# Configure logging for model manager
logger = logging.getLogger(__name__)

def ensure_required_packages():
    """Ensure required packages are installed"""
    required_packages = ["gdown", "requests"]
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            print(f"Installing required package: {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def verify_models():
    """Check if all required models are available"""
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ConversationQA', 'Models')
    weather_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Weather')
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)
    if not os.path.exists(weather_dir):
        os.makedirs(weather_dir, exist_ok=True)
        
    required_models = [
        "bert_seq2seq_ner.pt",
        "pronoun_resolution_model_full.pt",
        "extractiveQA.pt",
        "vectorizer.pkl",
        "classifier_model.pkl",
    ]
    
    # Add weather model check
    weather_model = "weather_model.pt"
    weather_model_path = os.path.join(weather_dir, weather_model)
    
    missing_models = []
    for model_name in required_models:
        model_path = os.path.join(models_dir, model_name)
        if not os.path.exists(model_path):
            missing_models.append(model_name)
    
    # Check weather model
    if not os.path.exists(weather_model_path):
        missing_models.append(weather_model)
    
    if missing_models:
        print("\nWARNING: The following models are missing:")
        for model in missing_models:
            print(f"  - {model}")
        print("The ConversationQA and Weather functionality may not work correctly.\n")
        return False, missing_models
    else:
        print("\nAll required AI models are available.\n")
        return True, []

def downloadModels(missing_models):
    """Download missing model files"""
    import tempfile
    import gdown
    import zipfile
    import requests
    
    print("\nDownloading missing models...")
    
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ConversationQA', 'Models')
    weather_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Weather')
    
    # Direct download URLs (not through Google Drive API)
    direct_urls = {
        "bert_seq2seq_ner.zip": "https://drive.google.com/uc?id=1lOSkIPGU4TX7L727OmkBL3f0Fh_W3_Fv&export=download",
        "pronoun_resolution_model_full.zip": "https://drive.google.com/uc?id=1DhlkILm1kzD8gPbEU0NlTUGeDA_uGisk&export=download",
        "extractiveQA.zip": "https://drive.google.com/uc?id=1ph3yhuAz7fmTv8lat8fbdwzlHYSa4y4U&export=download",
        "vectorizer.zip": "https://drive.google.com/uc?id=1QqxFX0VhLYKdgWWJBYnE9YVnEP6bZKcY&export=download",
        "weather_model.zip": "https://drive.google.com/uc?id=1KBTMhfV9MaLuM2V7ID2ZEEi7ov5szG6M&export=download"
    }
    
    # Map models to their containing archives
    model_to_archive = {
        "bert_seq2seq_ner.pt": "bert_seq2seq_ner.zip",
        "pronoun_resolution_model_full.pt": "pronoun_resolution_model_full.zip",
        "extractiveQA.pt": "extractiveQA.zip",
        "vectorizer.pkl": "vectorizer.zip",
        "classifier_model.pkl": "vectorizer.zip",
        "weather_model.pt": "weather_model.zip"
    }
    
    # Map models to their target directories
    model_to_dir = {
        "weather_model.pt": weather_dir
    }
    
    # Determine which archives to download
    archives_to_download = set()
    for model in missing_models:
        if model in model_to_archive:
            archives_to_download.add(model_to_archive[model])
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for archive in archives_to_download:
            archive_path = os.path.join(temp_dir, archive)
            url = direct_urls.get(archive)
            
            if not url:
                print(f"No URL found for {archive}")
                continue
            
            print(f"Downloading {archive}...")
            
            # First try gdown
            try:
                gdown.download(url, archive_path, quiet=False)
                
                # Check if downloaded successfully
                if os.path.exists(archive_path) and os.path.getsize(archive_path) > 0:
                    print(f"Successfully downloaded {archive}")
                    
                    # Extract the file
                    try:
                        print(f"Extracting {archive}...")
                        
                        # Determine target directory for extraction
                        target_dir = models_dir
                        for model in missing_models:
                            if model_to_archive.get(model) == archive:
                                target_dir = model_to_dir.get(model, models_dir)
                                break
                        
                        # Handle ZIP extraction
                        try:
                            with zipfile.ZipFile(archive_path) as zf:
                                zf.extractall(target_dir)
                            print(f"Extracted {archive} using zipfile")
                        except Exception as e:
                            print(f"zipfile extraction failed: {e}")
                            
                            # Try with subprocess
                            try:
                                subprocess.run(['unzip', archive_path, '-d', target_dir], check=True)
                                print(f"Extracted {archive} using unzip command")
                            except Exception as e2:
                                print(f"unzip command failed: {e2}")
                                
                                # Try with 7z as last resort
                                try:
                                    subprocess.run(['7z', 'x', archive_path, f'-o{target_dir}'], check=True)
                                    print(f"Extracted {archive} using 7z")
                                except Exception as e3:
                                    print(f"7z extraction failed: {e3}")
                                    print(f"Failed to extract {archive} with any method")
                    except Exception as e:
                        print(f"Error during extraction: {e}")
                else:
                    print(f"Failed to download {archive} with gdown")
            except Exception as e:
                print(f"Error downloading with gdown: {e}")
                print("Trying direct download method...")
                
                # Try direct download with requests as fallback
                try:
                    response = requests.get(url, stream=True)
                    if response.status_code == 200:
                        with open(archive_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        print(f"Downloaded {archive} with requests")
                        
                        # Extract the file
                        try:
                            print(f"Extracting {archive}...")
                            
                            # Determine target directory for extraction
                            target_dir = models_dir
                            for model in missing_models:
                                if model_to_archive.get(model) == archive:
                                    target_dir = model_to_dir.get(model, models_dir)
                                    break
                            
                            # Handle ZIP extraction
                            try:
                                with zipfile.ZipFile(archive_path) as zf:
                                    zf.extractall(target_dir)
                                print(f"Extracted {archive} using zipfile")
                            except Exception as e:
                                print(f"zipfile extraction failed: {e}")
                                
                                # Try with subprocess
                                try:
                                    subprocess.run(['unzip', archive_path, '-d', target_dir], check=True)
                                    print(f"Extracted {archive} using unzip command")
                                except Exception as e2:
                                    print(f"unzip command failed: {e2}")
                                    
                                    # Try with 7z as last resort
                                    try:
                                        subprocess.run(['7z', 'x', archive_path, f'-o{target_dir}'], check=True)
                                        print(f"Extracted {archive} using 7z")
                                    except Exception as e3:
                                        print(f"7z extraction failed: {e3}")
                                        print(f"Failed to extract {archive} with any method")
                        except Exception as e:
                            print(f"Error during extraction: {e}")
                    else:
                        print(f"Failed to download {archive} with requests: {response.status_code}")
                except Exception as e:
                    print(f"Error with direct download: {e}")
    
    # Verify models again
    models_available, still_missing = verify_models()
    if still_missing:
        print("\nSome models are still missing after download attempts:")
        for model in still_missing:
            print(f"  - {model}")
        return False
    else:
        print("\nAll models successfully downloaded and verified!")
        return True

def setup_models():
    """Main function to setup and verify models"""
    print("\n===== ZAAM Model Setup =====")
    print("Ensuring required packages are installed...")
    ensure_required_packages()
    
    print("Checking for AI models...")
    models_available, missing_models = verify_models()

    if not models_available:
        print("Missing models detected. Attempting to download...")
        if downloadModels(missing_models):
            print("Model setup completed successfully!")
            return True
        else:
            print("Model setup failed!")
            return False
    else:
        print("All models are available!")
        return True 