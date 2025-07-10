#!/usr/bin/env python3
"""
Google Colab setup script for heme_binder_diffusion project
This script handles the installation and configuration of dependencies for Colab Pro
"""

import os
import subprocess
import sys
from pathlib import Path
import requests
import zipfile
import shutil
import time
from IPython.display import clear_output, display, HTML
from tqdm import tqdm

class ColabSetup:
    def __init__(self):
        self.base_dir = "/content/heme_binder_diffusion"
        self.models_dir = "/content/models"
        self.drive_mount = "/content/drive"
        
    def check_gpu(self):
        """Check if GPU is available"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print(f"✓ GPU detected: {gpu_name}")
                
                # Display GPU info in a nicer format
                html = f"""
                <div style="background-color:#dff0d8; padding:10px; border-radius:5px; margin:10px 0;">
                <h3 style="color:#3c763d;">✅ GPU Available</h3>
                <p><b>Model:</b> {gpu_name}</p>
                <p>GPU acceleration is available for this session.</p>
                </div>
                """
                display(HTML(html))
                return True
            else:
                html = """
                <div style="background-color:#f2dede; padding:10px; border-radius:5px; margin:10px 0;">
                <h3 style="color:#a94442;">⚠️ No GPU Detected</h3>
                <p>This project requires GPU acceleration.</p>
                <p>Please check your runtime type by going to:</p>
                <p><code>Runtime > Change runtime type > Hardware accelerator > GPU</code></p>
                </div>
                """
                display(HTML(html))
                print("⚠ No GPU detected. This project requires GPU!")
                return False
        except ImportError:
            print("⚠ PyTorch not installed yet")
            return False
    
    def mount_drive(self):
        """Mount Google Drive for persistent storage"""
        try:
            from google.colab import drive
            drive.mount(self.drive_mount)
            print("✓ Google Drive mounted successfully")
            
            # Create project directory in Drive
            drive_project_dir = f"{self.drive_mount}/MyDrive/heme_binder_diffusion"
            os.makedirs(drive_project_dir, exist_ok=True)
            print(f"✓ Project directory created: {drive_project_dir}")
            
            return drive_project_dir
        except ImportError:
            print("⚠ Not running in Colab environment")
            return None
        except Exception as e:
            print(f"❌ Error mounting drive: {e}")
            return None
    
    def install_system_dependencies(self):
        """Install system-level dependencies"""
        print("Installing system dependencies...")
        
        commands = [
            "apt-get update -q",
            "apt-get install -y -q wget curl git",
            "apt-get install -y -q build-essential",
            "apt-get install -y -q libssl-dev libffi-dev",
            "apt-get install -y -q cmake",  # Required for some Python packages
        ]
        
        for cmd in commands:
            try:
                print(f"Running: {cmd}")
                process = subprocess.run(cmd.split(), check=True, capture_output=True)
                print(f"✓ Command completed")
            except subprocess.CalledProcessError as e:
                print(f"⚠ Warning: {cmd} failed: {e}")
        
        print("✓ System dependencies installed")
    
    def clone_repository(self):
        """Clone the main repository and submodules"""
        print("Cloning repository...")
        
        if os.path.exists(self.base_dir):
            print(f"Repository directory already exists, removing: {self.base_dir}")
            shutil.rmtree(self.base_dir)
        
        # Clone main repository
        try:
            subprocess.run([
                "git", "clone", 
                "https://github.com/ikalvet/heme_binder_diffusion.git",
                self.base_dir
            ], check=True)
            
            # Change to project directory
            os.chdir(self.base_dir)
            
            # Initialize and update submodules
            subprocess.run(["git", "submodule", "init"], check=True)
            subprocess.run(["git", "submodule", "update"], check=True)
            
            print("✓ Repository cloned successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error cloning repository: {e}")
            return False
    
    def install_python_dependencies(self):
        """Install Python dependencies"""
        print("Installing Python dependencies...")
        
        # Install basic dependencies
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"
        ], check=True)
        
        # Install core packages
        packages = [
            "torch>=1.12.0",
            "torchvision",
            "numpy",
            "pandas",
            "matplotlib",
            "seaborn",
            "biopython",
            "prody",
            "rdkit-pypi",
            "openmm",
            "pdbfixer",
            "mdtraj",
            "biotite",
            "py3dmol",
        ]
        
        for package in packages:
            try:
                print(f"Installing {package}...")
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "-q", package
                ], check=True)
                print(f"✓ {package}")
            except subprocess.CalledProcessError as e:
                print(f"⚠ Failed to install {package}: {e}")
        
        print("✓ Python dependencies installed")
    
    def install_jax(self):
        """Install JAX with CUDA support"""
        print("Installing JAX with CUDA support...")
        
        try:
            # Install JAX with CUDA support
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-q",
                "jax[cuda12_pip]", "-f", 
                "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
            ], check=True)
            
            # Install additional JAX dependencies
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-q",
                "jaxlib", "flax", "optax"
            ], check=True)
            
            print("✓ JAX with CUDA support installed")
            
            # Verify JAX installation
            try:
                import jax
                jax_devices = jax.devices()
                if jax_devices:
                    print(f"✓ JAX devices available: {jax_devices}")
                else:
                    print("⚠ No JAX devices found")
            except Exception as e:
                print(f"⚠ JAX import test failed: {e}")
                
        except subprocess.CalledProcessError as e:
            print(f"⚠ JAX installation failed: {e}")
            # Fallback to CPU version
            print("Trying fallback to CPU version...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-q",
                "jax", "jaxlib"
            ], check=True)
            print("✓ JAX (CPU version) installed")
    
    def download_file(self, url, filepath, chunk_size=8192):
        """Download a file with progress bar"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            # Create parent directory if it doesn't exist
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, "wb") as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {filepath.name}") as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            return True
            
        except Exception as e:
            print(f"❌ Error downloading {url}: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return False
    
    def download_alphafold_models(self):
        """Download AlphaFold2 model weights"""
        print("Downloading AlphaFold2 models...")
        
        os.makedirs(self.models_dir, exist_ok=True)
        af2_models_dir = f"{self.models_dir}/alphafold"
        os.makedirs(af2_models_dir, exist_ok=True)
        
        # Download essential AlphaFold2 models (reduced set for Colab)
        models_to_download = [
            "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar",
        ]
        
        for model_url in models_to_download:
            filename = model_url.split("/")[-1]
            filepath = Path(f"{af2_models_dir}/{filename}")
            
            if not filepath.exists():
                print(f"Downloading {filename}...")
                if self.download_file(model_url, filepath):
                    # Extract if it's a tar file
                    if filename.endswith(".tar"):
                        print(f"Extracting {filename}...")
                        try:
                            subprocess.run(["tar", "-xf", filepath, "-C", af2_models_dir], check=True)
                            os.remove(filepath)  # Remove tar file after extraction
                            print(f"✓ {filename} extracted")
                        except subprocess.CalledProcessError as e:
                            print(f"❌ Failed to extract {filename}: {e}")
                else:
                    print(f"❌ Failed to download {filename}")
            else:
                print(f"✓ {filename} already exists")
        
        print("✓ AlphaFold2 models downloaded")
    
    def download_proteinmpnn_models(self):
        """Download ProteinMPNN model weights"""
        print("Downloading ProteinMPNN models...")
        
        mpnn_models_dir = Path(f"{self.models_dir}/proteinmpnn")
        mpnn_models_dir.mkdir(exist_ok=True)
        
        # Download ProteinMPNN models
        models = {
            "proteinmpnn_v_48_020.pt": "https://files.ipd.uw.edu/pub/training_sets/proteinmpnn_v_48_020.pt",
            "ligandmpnn_v_32_010_25.pt": "https://files.ipd.uw.edu/pub/training_sets/ligandmpnn_v_32_010_25.pt",
        }
        
        for model_name, model_url in models.items():
            filepath = mpnn_models_dir / model_name
            
            if not filepath.exists():
                print(f"Downloading {model_name}...")
                if self.download_file(model_url, filepath):
                    print(f"✓ {model_name} downloaded")
                else:
                    print(f"❌ Failed to download {model_name}")
            else:
                print(f"✓ {model_name} already exists")
        
        print("✓ ProteinMPNN models downloaded")
    
    def setup_rf_diffusion(self):
        """Setup RFdiffusion (simplified version for Colab)"""
        print("Setting up RFdiffusion...")
        
        rf_dir = f"{self.base_dir}/rf_diffusion"
        
        try:
            # Clone RFdiffusion repository
            if not os.path.exists(rf_dir):
                subprocess.run([
                    "git", "clone", 
                    "https://github.com/baker-laboratory/rf_diffusion_all_atom.git",
                    rf_dir
                ], check=True)
            
            # Install RFdiffusion dependencies
            os.chdir(rf_dir)
            if os.path.exists("requirements.txt"):
                print("Installing RFdiffusion requirements...")
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"
                ], check=True)
            
            # Download RFdiffusion model weights
            rf_weights_dir = Path(f"{self.models_dir}/rf_diffusion")
            rf_weights_dir.mkdir(exist_ok=True)
            
            model_url = "https://files.ipd.uw.edu/pub/RF-All-Atom/weights/rf_diffusion_all_atom.pt"
            model_path = rf_weights_dir / "rf_diffusion_all_atom.pt"
            
            if not model_path.exists():
                print("Downloading RFdiffusion model weights...")
                if self.download_file(model_url, model_path):
                    print("✓ RFdiffusion model weights downloaded")
                else:
                    print("❌ Failed to download RFdiffusion model weights")
            else:
                print("✓ RFdiffusion model weights already exist")
            
            print("✓ RFdiffusion setup completed")
        except Exception as e:
            print(f"⚠ RFdiffusion setup failed: {e}")
        
        # Return to base directory
        os.chdir(self.base_dir)
    
    def create_config_files(self):
        """Create configuration files for Colab environment"""
        print("Creating configuration files...")
        
        # Create paths configuration
        config_content = f"""
# Colab-specific paths configuration
BASE_DIR = "{self.base_dir}"
MODELS_DIR = "{self.models_dir}"
DRIVE_DIR = "{self.drive_mount}/MyDrive/heme_binder_diffusion"

# Python executables (all use the same environment in Colab)
PYTHON_PATHS = {{
    "diffusion": "/usr/bin/python3",
    "af2": "/usr/bin/python3", 
    "proteinMPNN": "/usr/bin/python3",
    "general": "/usr/bin/python3"
}}

# Script paths
SCRIPTS = {{
    "alphafold": "{self.base_dir}/scripts/af2/af2.py",
    "proteinmpnn": "{self.base_dir}/lib/LigandMPNN/run.py",
    "rf_diffusion": "{self.base_dir}/rf_diffusion/run_inference.py"
}}

# Model paths
MODEL_PATHS = {{
    "alphafold": "{self.models_dir}/alphafold",
    "proteinmpnn": "{self.models_dir}/proteinmpnn",
    "rf_diffusion": "{self.models_dir}/rf_diffusion"
}}
"""
        
        config_file = f"{self.base_dir}/colab_config.py"
        with open(config_file, "w") as f:
            f.write(config_content)
        
        print("✓ Configuration files created")
    
    def setup_example_data(self):
        """Extract example data for testing"""
        print("Setting up example data...")
        
        try:
            # Extract example input models if they exist
            input_dir = f"{self.base_dir}/input"
            
            zip_files = [
                "P450_HBA_input_models.zip",
                "UPO_HBA_input_models.zip"
            ]
            
            for zip_file in zip_files:
                zip_path = f"{input_dir}/{zip_file}"
                if os.path.exists(zip_path):
                    print(f"Extracting {zip_file}...")
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(input_dir)
                    print(f"✓ {zip_file} extracted")
            
            print("✓ Example data setup completed")
        except Exception as e:
            print(f"⚠ Example data setup failed: {e}")
    
    def run_setup(self):
        """Run complete setup process"""
        start_time = time.time()
        print("🚀 Starting Colab setup for heme_binder_diffusion...")
        print("=" * 50)
        
        # Check GPU
        has_gpu = self.check_gpu()
        if not has_gpu:
            print("⚠ Warning: No GPU detected. Setup will continue but performance will be limited.")
        
        # Mount Drive
        drive_dir = self.mount_drive()
        
        # Install system dependencies
        self.install_system_dependencies()
        
        # Clone repository
        if not self.clone_repository():
            print("❌ Failed to clone repository. Setup cannot continue.")
            return False
        
        # Install Python dependencies
        self.install_python_dependencies()
        
        # Install JAX
        self.install_jax()
        
        # Download models (this will take time)
        print("📥 Downloading model weights (this may take 10-20 minutes)...")
        self.download_alphafold_models()
        self.download_proteinmpnn_models()
        
        # Setup RFdiffusion
        self.setup_rf_diffusion()
        
        # Create config files
        self.create_config_files()
        
        # Setup example data
        self.setup_example_data()
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(elapsed_time, 60)
        
        print("\n" + "=" * 50)
        print(f"✅ Setup completed in {int(minutes)} minutes and {int(seconds)} seconds!")
        print(f"📁 Project directory: {self.base_dir}")
        print(f"📁 Models directory: {self.models_dir}")
        if drive_dir:
            print(f"📁 Drive backup: {drive_dir}")
        
        print("\n📋 Next steps:")
        print("1. Run the simplified pipeline notebook")
        print("2. Start with small test datasets")
        print("3. Monitor GPU usage and session time")
        
        # Display a nice completion message
        html = f"""
        <div style="background-color:#dff0d8; padding:15px; border-radius:5px; margin:10px 0;">
        <h2 style="color:#3c763d;">✅ Setup Complete!</h2>
        <p><b>Time:</b> {int(minutes)} minutes and {int(seconds)} seconds</p>
        <p><b>Project directory:</b> {self.base_dir}</p>
        <p><b>Models directory:</b> {self.models_dir}</p>
        <p>You're now ready to run the heme binder diffusion pipeline!</p>
        </div>
        """
        display(HTML(html))
        
        return True

def main():
    """Main setup function"""
    setup = ColabSetup()
    return setup.run_setup()

if __name__ == "__main__":
    main()