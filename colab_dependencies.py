#!/usr/bin/env python3
"""
Dependency installation script for Colab environment
Handles the installation of all required packages with proper version management
"""

import subprocess
import sys
import os
import importlib.util
from pathlib import Path
import time
from IPython.display import clear_output

class DependencyInstaller:
    def __init__(self):
        self.requirements = {
            "core": [
                "torch>=1.12.0",
                "torchvision",
                "numpy>=1.21.0",
                "pandas>=1.3.0",
                "matplotlib>=3.5.0",
                "seaborn",
                "scipy>=1.7.0",
                "scikit-learn",
            ],
            "bio": [
                "biopython>=1.79",
                "prody>=2.0",
                "biotite>=0.34",
                "mdtraj>=1.9",
                "openmm>=7.6",
                "pdbfixer",
                "rdkit-pypi",
                "py3dmol>=1.8",
            ],
            "ml": [
                "jax>=0.4.0",
                "jaxlib>=0.4.0",
                "flax>=0.6.0",
                "optax>=0.1.4",
                "dm-haiku>=0.0.8",
                "chex>=0.1.5",
            ],
            "utils": [
                "tqdm",
                "wandb",
                "tensorboard",
                "ipywidgets",
                "plotly",
                "requests",
                "urllib3",
            ]
        }
        
        self.conda_packages = [
            "openmm",
            "pdbfixer",
        ]
        
        self.special_installs = {
            "pyrosetta": self._install_pyrosetta,
            "colabfold": self._install_colabfold,
            "alphafold": self._install_alphafold_deps,
        }
    
    def check_package(self, package_name):
        """Check if a package is already installed"""
        try:
            spec = importlib.util.find_spec(package_name)
            return spec is not None
        except (ImportError, ModuleNotFoundError):
            return False
    
    def install_package(self, package, upgrade=False, max_retries=3):
        """Install a single package with retries"""
        cmd = [sys.executable, "-m", "pip", "install", "-q"]
        if upgrade:
            cmd.append("--upgrade")
        cmd.append(package)
        
        for attempt in range(max_retries):
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                return True
            except subprocess.CalledProcessError as e:
                if attempt < max_retries - 1:
                    print(f"⚠️ Attempt {attempt+1} failed for {package}, retrying...")
                    time.sleep(2)  # Wait before retry
                else:
                    print(f"❌ Failed to install {package} after {max_retries} attempts: {e.stderr.decode()}")
                    return False
    
    def install_requirements(self, category="all"):
        """Install requirements by category"""
        if category == "all":
            categories = self.requirements.keys()
        else:
            categories = [category] if category in self.requirements else []
        
        for cat in categories:
            print(f"\n📦 Installing {cat} packages...")
            packages = self.requirements[cat]
            
            for package in packages:
                package_name = package.split(">=")[0].split("==")[0]
                
                if self.check_package(package_name):
                    print(f"✓ {package_name} already installed")
                else:
                    print(f"Installing {package}...")
                    if self.install_package(package):
                        print(f"✓ {package_name} installed")
                    else:
                        print(f"❌ Failed to install {package_name}")
    
    def _install_pyrosetta(self):
        """Install PyRosetta (requires registration)"""
        print("Installing PyRosetta...")
        
        # Try to install PyRosetta4 conda package
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-q",
                "pyrosetta-installer"
            ], check=True)
            
            print("✓ PyRosetta installer installed")
            print("⚠ You need to run pyrosetta-installer with your credentials")
            print("  Visit: https://www.pyrosetta.org/downloads")
            
        except subprocess.CalledProcessError:
            print("❌ PyRosetta installation failed")
            print("ℹ Manual installation required - see PyRosetta documentation")
            print("Note: PyRosetta is optional for this pipeline in Colab")
    
    def _install_colabfold(self):
        """Install ColabFold"""
        print("Installing ColabFold...")
        
        try:
            # Install ColabFold
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-q",
                "colabfold[alphafold]"
            ], check=True)
            
            print("✓ ColabFold installed")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ ColabFold installation failed: {e}")
            print("Note: ColabFold is optional for this pipeline")
    
    def _install_alphafold_deps(self):
        """Install AlphaFold dependencies"""
        print("Installing AlphaFold dependencies...")
        
        alphafold_deps = [
            "absl-py>=1.0.0",
            "chex>=0.1.5",
            "dm-haiku>=0.0.8",
            "dm-tree>=0.1.6",
            "immutabledict>=2.0.0",
            "jax>=0.4.0",
            "jaxlib>=0.4.0",
            "ml-collections>=0.1.0",
            "numpy>=1.21.0",
            "scipy>=1.7.0",
            "tensorflow>=2.9.0",
        ]
        
        for dep in alphafold_deps:
            if self.install_package(dep):
                print(f"✓ {dep.split('>=')[0]} installed")
    
    def install_from_git(self, repo_url, package_name=None):
        """Install package from git repository"""
        if package_name:
            if self.check_package(package_name):
                print(f"✓ {package_name} already installed")
                return True
        
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-q",
                f"git+{repo_url}"
            ], check=True)
            
            print(f"✓ Installed from {repo_url}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install from {repo_url}: {e}")
            return False
    
    def setup_conda_packages(self):
        """Setup packages that work better with conda"""
        print("Setting up conda packages...")
        
        # Check if conda is available
        try:
            subprocess.run(["conda", "--version"], check=True, capture_output=True)
            conda_available = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            conda_available = False
        
        if conda_available:
            for package in self.conda_packages:
                try:
                    subprocess.run([
                        "conda", "install", "-y", "-q", package
                    ], check=True)
                    print(f"✓ {package} installed via conda")
                except subprocess.CalledProcessError:
                    print(f"⚠ {package} conda install failed, trying pip")
                    self.install_package(package)
        else:
            print("Conda not available in Colab, using pip for all packages")
            for package in self.conda_packages:
                self.install_package(package)
    
    def install_special_packages(self):
        """Install packages that require special handling"""
        print("\n🔧 Installing special packages...")
        
        for package_name, install_func in self.special_installs.items():
            try:
                install_func()
            except Exception as e:
                print(f"❌ Failed to install {package_name}: {e}")
    
    def install_colab_specific_packages(self):
        """Install Colab-specific packages"""
        print("\n📱 Installing Colab-specific packages...")
        
        colab_packages = [
            "google-colab",
            "ipywidgets",
            "py3Dmol",
        ]
        
        for package in colab_packages:
            if self.check_package(package):
                print(f"✓ {package} already installed")
            else:
                if self.install_package(package):
                    print(f"✓ {package} installed")
    
    def verify_installation(self):
        """Verify that key packages are installed correctly"""
        print("\n🔍 Verifying installation...")
        
        key_packages = [
            "torch", "numpy", "pandas", "matplotlib",
            "biopython", "prody", "jax", "jaxlib"
        ]
        
        results = {}
        for package in key_packages:
            try:
                module = importlib.import_module(package)
                version = getattr(module, "__version__", "unknown")
                results[package] = f"✓ {version}"
            except ImportError:
                results[package] = "❌ Not installed"
        
        print("\n📋 Installation Summary:")
        for package, status in results.items():
            print(f"  {package}: {status}")
        
        return results
    
    def run_installation(self):
        """Run complete installation process"""
        print("🚀 Starting dependency installation...")
        print("=" * 50)
        
        try:
            # Upgrade pip first
            print("Upgrading pip...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"
            ], check=True)
            
            # Install core requirements
            self.install_requirements("core")
            
            # Install bio packages
            self.install_requirements("bio")
            
            # Install ML packages
            self.install_requirements("ml")
            
            # Install utility packages
            self.install_requirements("utils")
            
            # Install Colab-specific packages
            self.install_colab_specific_packages()
            
            # Setup conda packages (if available)
            self.setup_conda_packages()
            
            # Install special packages
            self.install_special_packages()
            
            # Verify installation
            results = self.verify_installation()
            
            print("\n" + "=" * 50)
            print("✅ Dependency installation completed!")
            
            # Check for failed installations
            failed = [pkg for pkg, status in results.items() if "❌" in status]
            if failed:
                print(f"⚠ Some packages failed to install: {', '.join(failed)}")
                print("You may need to install these manually")
                
                # Try to provide workarounds for common failures
                if "jax" in failed or "jaxlib" in failed:
                    print("\nTry installing JAX with this specific command:")
                    print("!pip install --upgrade jax jaxlib==0.4.23 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
                
                if "prody" in failed:
                    print("\nTry installing Prody with:")
                    print("!pip install -U ProDy")
                
                if "openmm" in failed or "pdbfixer" in failed:
                    print("\nTry installing OpenMM with:")
                    print("!pip install openmm pdbfixer")
            
            return len(failed) == 0
            
        except Exception as e:
            print(f"❌ Installation process failed: {e}")
            return False

def main():
    """Main installation function"""
    installer = DependencyInstaller()
    success = installer.run_installation()
    
    # Clear output to keep the notebook clean
    time.sleep(2)
    clear_output(wait=True)
    
    if success:
        print("✅ All dependencies installed successfully!")
    else:
        print("⚠️ Some dependencies could not be installed. Check the logs for details.")
    
    return success

if __name__ == "__main__":
    main()