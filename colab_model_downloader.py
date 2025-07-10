#!/usr/bin/env python3
"""
Model weights download script for Colab environment
Handles downloading and managing large model files efficiently
"""

import os
import sys
import requests
import hashlib
import json
import shutil
from pathlib import Path
from urllib.parse import urlparse
from tqdm import tqdm
import zipfile
import tarfile
import gzip
import time
from IPython.display import clear_output, display, HTML

class ModelDownloader:
    def __init__(self, models_dir="/content/models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Model configurations
        self.models = {
            "alphafold": {
                "dir": self.models_dir / "alphafold",
                "files": {
                    "params_model_1.npz": {
                        "url": "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar",
                        "extract": True,
                        "size_gb": 3.5,
                        "required": True
                    }
                }
            },
            "proteinmpnn": {
                "dir": self.models_dir / "proteinmpnn", 
                "files": {
                    "proteinmpnn_v_48_020.pt": {
                        "url": "https://files.ipd.uw.edu/pub/training_sets/proteinmpnn_v_48_020.pt",
                        "size_gb": 0.5,
                        "required": True
                    },
                    "ligandmpnn_v_32_010_25.pt": {
                        "url": "https://files.ipd.uw.edu/pub/training_sets/ligandmpnn_v_32_010_25.pt", 
                        "size_gb": 0.5,
                        "required": True
                    }
                }
            },
            "rf_diffusion": {
                "dir": self.models_dir / "rf_diffusion",
                "files": {
                    "rf_diffusion_all_atom.pt": {
                        "url": "https://files.ipd.uw.edu/pub/RF-All-Atom/weights/rf_diffusion_all_atom.pt",
                        "size_gb": 2.0,
                        "required": True
                    }
                }
            }
        }
    
    def check_disk_space(self, required_gb):
        """Check if there's enough disk space"""
        try:
            statvfs = os.statvfs(self.models_dir)
            free_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
            return free_gb >= required_gb
        except:
            # If we can't check, assume it's OK (Colab usually has enough space)
            return True
    
    def download_file(self, url, filepath, chunk_size=8192, max_retries=3):
        """Download a file with progress bar and retries"""
        for attempt in range(max_retries):
            try:
                # Create a session with longer timeout
                session = requests.Session()
                session.mount('https://', requests.adapters.HTTPAdapter(max_retries=3))
                
                response = session.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                # Create parent directory if it doesn't exist
                filepath.parent.mkdir(parents=True, exist_ok=True)
                
                with open(filepath, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {filepath.name}") as pbar:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                
                # Verify download completed successfully
                if filepath.stat().st_size > 0:
                    return True
                else:
                    print(f"⚠️ Downloaded file {filepath} is empty, retrying...")
                    if attempt < max_retries - 1:
                        time.sleep(2)  # Wait before retry
                    continue
                    
            except (requests.exceptions.RequestException, IOError) as e:
                print(f"⚠️ Attempt {attempt+1} failed: {e}")
                if filepath.exists():
                    filepath.unlink()
                
                if attempt < max_retries - 1:
                    print(f"Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print(f"❌ Failed to download after {max_retries} attempts")
                    return False
        
        return False
    
    def extract_archive(self, filepath, extract_dir):
        """Extract archive files with error handling"""
        try:
            print(f"📂 Extracting {filepath.name}...")
            
            if filepath.suffix == '.tar' or '.tar.' in filepath.name:
                with tarfile.open(filepath, 'r:*') as tar:
                    # Check for suspicious paths
                    for member in tar.getmembers():
                        if member.name.startswith('/') or '..' in member.name:
                            print(f"⚠️ Skipping suspicious path: {member.name}")
                            continue
                    # Extract files
                    tar.extractall(path=extract_dir)
            elif filepath.suffix == '.zip':
                with zipfile.ZipFile(filepath, 'r') as zip_file:
                    # Check for suspicious paths
                    for member in zip_file.namelist():
                        if member.startswith('/') or '..' in member:
                            print(f"⚠️ Skipping suspicious path: {member}")
                            continue
                    # Extract files
                    zip_file.extractall(path=extract_dir)
            elif filepath.suffix == '.gz':
                output_file = extract_dir / filepath.stem
                with gzip.open(filepath, 'rb') as gz_file:
                    with open(output_file, 'wb') as out_file:
                        shutil.copyfileobj(gz_file, out_file)
            else:
                print(f"⚠️ Unknown archive format: {filepath}")
                return False
            
            print(f"✓ Extracted {filepath.name}")
            return True
            
        except Exception as e:
            print(f"❌ Error extracting {filepath}: {e}")
            return False
    
    def verify_file(self, filepath, expected_size_gb=None):
        """Verify downloaded file"""
        if not filepath.exists():
            return False
        
        # Check file size
        size_gb = filepath.stat().st_size / (1024**3)
        
        if expected_size_gb and abs(size_gb - expected_size_gb) > 0.2:  # Allow 200MB tolerance
            print(f"⚠️ File size mismatch: {size_gb:.2f}GB vs expected {expected_size_gb:.2f}GB")
            return False
        
        return True
    
    def download_model_category(self, category):
        """Download all models for a specific category"""
        if category not in self.models:
            print(f"❌ Unknown model category: {category}")
            return False
        
        model_config = self.models[category]
        model_dir = model_config["dir"]
        model_dir.mkdir(exist_ok=True)
        
        print(f"📦 Downloading {category} models...")
        
        # Calculate total space needed
        total_size = sum(info.get("size_gb", 0) for info in model_config["files"].values())
        
        if not self.check_disk_space(total_size + 1):  # +1GB buffer
            print(f"❌ Not enough disk space. Need {total_size:.1f}GB")
            return False
        
        success = True
        required_success = True
        
        for filename, info in model_config["files"].items():
            filepath = model_dir / filename
            is_required = info.get("required", False)
            
            # Skip if already exists and is valid
            if self.verify_file(filepath, info.get("size_gb")):
                print(f"✓ {filename} already exists")
                continue
            
            print(f"📥 Downloading {filename}...")
            
            # Download file
            temp_filepath = filepath.with_suffix(filepath.suffix + ".tmp")
            if self.download_file(info["url"], temp_filepath):
                # Verify download
                if self.verify_file(temp_filepath, info.get("size_gb")):
                    # Extract if needed
                    if info.get("extract", False):
                        if self.extract_archive(temp_filepath, model_dir):
                            temp_filepath.unlink()  # Remove archive after extraction
                        else:
                            print(f"❌ Failed to extract {filename}")
                            success = False
                            if is_required:
                                required_success = False
                    else:
                        # Move to final location
                        temp_filepath.rename(filepath)
                        print(f"✓ {filename} downloaded")
                else:
                    print(f"❌ {filename} verification failed")
                    success = False
                    if is_required:
                        required_success = False
            else:
                print(f"❌ Failed to download {filename}")
                success = False
                if is_required:
                    required_success = False
        
        if not required_success:
            print(f"❌ Failed to download required models for {category}")
        
        return required_success
    
    def download_all_models(self):
        """Download all required models"""
        print("🚀 Starting model download process...")
        print("=" * 50)
        
        # Calculate total space needed
        total_size = 0
        for category, config in self.models.items():
            for info in config["files"].values():
                total_size += info.get("size_gb", 0)
        
        print(f"📊 Total download size: ~{total_size:.1f}GB")
        print("Note: This may take 10-30 minutes depending on your connection speed")
        
        if not self.check_disk_space(total_size + 2):  # +2GB buffer
            print(f"❌ Not enough disk space. Need {total_size + 2:.1f}GB")
            return False
        
        # Download each category
        all_required_success = True
        for category in self.models.keys():
            if not self.download_model_category(category):
                all_required_success = False
        
        if all_required_success:
            print("\n✅ All required models downloaded successfully!")
        else:
            print("\n⚠️ Some required models failed to download")
            print("Please check the logs and try again")
        
        return all_required_success
    
    def create_symlinks(self, target_dir):
        """Create symlinks to models in the project directory"""
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Create symlinks for easy access
        links = {
            "alphafold_models": self.models_dir / "alphafold",
            "proteinmpnn_models": self.models_dir / "proteinmpnn", 
            "rf_diffusion_models": self.models_dir / "rf_diffusion"
        }
        
        for link_name, source in links.items():
            link_path = target_dir / link_name
            
            if link_path.exists():
                if link_path.is_symlink():
                    link_path.unlink()
                else:
                    shutil.rmtree(link_path)
            
            try:
                os.symlink(source, link_path, target_is_directory=True)
                print(f"✓ Created symlink: {link_name}")
            except Exception as e:
                # Symlinks might not work in Colab, so copy instead
                print(f"⚠️ Could not create symlink, copying files instead: {e}")
                try:
                    shutil.copytree(source, link_path)
                    print(f"✓ Copied files to: {link_name}")
                except Exception as copy_error:
                    print(f"❌ Failed to copy files: {copy_error}")
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        for category_dir in self.models_dir.iterdir():
            if category_dir.is_dir():
                for file in category_dir.glob("*.tmp"):
                    file.unlink()
                    print(f"🧹 Cleaned up {file.name}")
    
    def get_model_info(self):
        """Get information about downloaded models"""
        info = {}
        
        for category, config in self.models.items():
            info[category] = {
                "dir": str(config["dir"]),
                "files": {},
                "total_size_gb": 0
            }
            
            for filename, file_info in config["files"].items():
                filepath = config["dir"] / filename
                
                if filepath.exists():
                    size_gb = filepath.stat().st_size / (1024**3)
                    info[category]["files"][filename] = {
                        "exists": True,
                        "size_gb": size_gb,
                        "path": str(filepath)
                    }
                    info[category]["total_size_gb"] += size_gb
                else:
                    info[category]["files"][filename] = {
                        "exists": False,
                        "size_gb": 0,
                        "path": str(filepath)
                    }
        
        return info
    
    def print_status(self):
        """Print download status"""
        print("\n📋 Model Download Status:")
        print("=" * 40)
        
        info = self.get_model_info()
        
        for category, data in info.items():
            print(f"\n{category.upper()}:")
            for filename, file_info in data["files"].items():
                status = "✓" if file_info["exists"] else "❌"
                size = f"{file_info['size_gb']:.2f}GB" if file_info["exists"] else "Not downloaded"
                print(f"  {status} {filename}: {size}")
            
            print(f"  Total: {data['total_size_gb']:.2f}GB")
        
        # Display a visual summary
        html = """
        <style>
        .model-status {
            font-family: monospace;
            margin: 10px 0;
            padding: 10px;
            border-radius: 5px;
            background-color: #f5f5f5;
        }
        .model-status .category {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .model-status .success {
            color: green;
        }
        .model-status .failure {
            color: red;
        }
        </style>
        <div class="model-status">
        """
        
        for category, data in info.items():
            all_exist = all(file_info["exists"] for file_info in data["files"].values())
            status_class = "success" if all_exist else "failure"
            html += f'<div class="category {status_class}">{category.upper()}: {"✓ Complete" if all_exist else "❌ Incomplete"}</div>'
        
        html += "</div>"
        
        display(HTML(html))

def main():
    """Main download function"""
    downloader = ModelDownloader()
    
    print("🔍 Checking current model status...")
    downloader.print_status()
    
    # Ask user what to download
    print("\nWhat would you like to download?")
    print("1. All models (recommended)")
    print("2. AlphaFold models only")
    print("3. ProteinMPNN models only") 
    print("4. RF Diffusion models only")
    print("5. Check status only")
    
    try:
        choice = input("Enter choice (1-5) [1]: ").strip() or "1"
        
        if choice == "1":
            success = downloader.download_all_models()
        elif choice == "2":
            success = downloader.download_model_category("alphafold")
        elif choice == "3":
            success = downloader.download_model_category("proteinmpnn")
        elif choice == "4":
            success = downloader.download_model_category("rf_diffusion")
        elif choice == "5":
            success = True
        else:
            print("Invalid choice, defaulting to all models")
            success = downloader.download_all_models()
    except KeyboardInterrupt:
        print("\n⚠️ Download interrupted by user")
        success = False
    
    # Cleanup and show final status
    downloader.cleanup_temp_files()
    
    # Clear output for cleaner display
    time.sleep(1)
    clear_output(wait=True)
    
    print("📋 Final Download Status:")
    downloader.print_status()
    
    # Create symlinks in the project directory
    project_dir = "/content/heme_binder_diffusion"
    if os.path.exists(project_dir):
        downloader.create_symlinks(project_dir)
    
    return success

if __name__ == "__main__":
    main()