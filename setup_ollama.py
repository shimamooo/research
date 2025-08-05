#!/usr/bin/env python3
"""
Setup script for Ollama LLM Benchmarking
This script helps install Ollama and pull required models.
"""

import subprocess
import sys
import platform
import os
import requests
import json
from typing import List, Dict, Any

class OllamaSetup:
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.default_models = [
            "llama2:7b",
            "mistral:7b", 
            "codellama:7b",
            "neural-chat:7b",
            "qwen:7b"
        ]
    
    def check_ollama_installed(self) -> bool:
        """Check if Ollama is installed"""
        try:
            result = subprocess.run(["ollama", "--version"], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def check_ollama_running(self) -> bool:
        """Check if Ollama service is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_installed_models(self) -> List[str]:
        """Get list of installed models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
            return []
        except:
            return []
    
    def install_ollama(self):
        """Install Ollama based on platform"""
        system = platform.system().lower()
        
        print("🔧 Installing Ollama...")
        
        if system == "windows":
            print("📥 Downloading Ollama for Windows...")
            print("Please visit: https://ollama.ai/download")
            print("Or run: winget install Ollama.Ollama")
            return False
        
        elif system == "darwin":  # macOS
            try:
                subprocess.run([
                    "curl", "-fsSL", "https://ollama.ai/install.sh"
                ], check=True)
                print("✅ Ollama installed successfully on macOS")
                return True
            except subprocess.CalledProcessError:
                print("❌ Failed to install Ollama on macOS")
                return False
        
        elif system == "linux":
            try:
                subprocess.run([
                    "curl", "-fsSL", "https://ollama.ai/install.sh", "|", "sh"
                ], shell=True, check=True)
                print("✅ Ollama installed successfully on Linux")
                return True
            except subprocess.CalledProcessError:
                print("❌ Failed to install Ollama on Linux")
                return False
        
        else:
            print(f"❌ Unsupported platform: {system}")
            return False
    
    def start_ollama(self):
        """Start Ollama service"""
        print("🚀 Starting Ollama service...")
        try:
            # Start Ollama in background
            subprocess.Popen(["ollama", "serve"], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            
            # Wait a moment for service to start
            import time
            time.sleep(3)
            
            if self.check_ollama_running():
                print("✅ Ollama service started successfully")
                return True
            else:
                print("❌ Failed to start Ollama service")
                return False
        except Exception as e:
            print(f"❌ Error starting Ollama: {e}")
            return False
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a specific model"""
        print(f"📥 Pulling model: {model_name}")
        try:
            result = subprocess.run(
                ["ollama", "pull", model_name],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                print(f"✅ Successfully pulled {model_name}")
                return True
            else:
                print(f"❌ Failed to pull {model_name}: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout while pulling {model_name}")
            return False
        except Exception as e:
            print(f"❌ Error pulling {model_name}: {e}")
            return False
    
    def setup_models(self, models: List[str] = None):
        """Setup all required models"""
        if models is None:
            models = self.default_models
        
        installed_models = self.get_installed_models()
        models_to_pull = [model for model in models if model not in installed_models]
        
        if not models_to_pull:
            print("✅ All required models are already installed!")
            return True
        
        print(f"📦 Installing {len(models_to_pull)} models...")
        
        successful_pulls = 0
        for model in models_to_pull:
            if self.pull_model(model):
                successful_pulls += 1
        
        print(f"✅ Successfully installed {successful_pulls}/{len(models_to_pull)} models")
        return successful_pulls == len(models_to_pull)
    
    def run_setup(self):
        """Run complete setup process"""
        print("🚀 Ollama LLM Benchmark Setup")
        print("=" * 40)
        
        # Check if Ollama is installed
        if not self.check_ollama_installed():
            print("❌ Ollama is not installed")
            if not self.install_ollama():
                print("Please install Ollama manually from https://ollama.ai")
                return False
        else:
            print("✅ Ollama is already installed")
        
        # Check if Ollama is running
        if not self.check_ollama_running():
            print("❌ Ollama service is not running")
            if not self.start_ollama():
                print("Please start Ollama manually with: ollama serve")
                return False
        else:
            print("✅ Ollama service is running")
        
        # Setup models
        print("\n📦 Setting up models...")
        if self.setup_models():
            print("\n🎉 Setup completed successfully!")
            print("\nYou can now run the benchmark with:")
            print("python main.py")
            return True
        else:
            print("\n⚠️  Setup completed with some issues")
            print("You can still run the benchmark, but some models may not be available")
            return False

def main():
    """Main setup function"""
    setup = OllamaSetup()
    
    try:
        success = setup.run_setup()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 