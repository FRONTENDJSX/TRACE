#!/usr/bin/env python3
"""
TRACE Setup Script
Installs and configures the TRACE facial recognition system.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"   Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python {sys.version.split()[0]} is compatible")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    
    # Check if pip is available
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("❌ pip is not available. Please install pip first.")
        return False
    
    # Install requirements
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing core dependencies"):
        return False
    
    # Install development dependencies if requested
    if "--dev" in sys.argv:
        if not run_command(f"{sys.executable} -m pip install -r requirements-dev.txt", "Installing development dependencies"):
            return False
    
    return True

def setup_environment():
    """Set up environment configuration."""
    print("⚙️ Setting up environment...")
    
    # Check if .env file exists
    env_file = Path(".env")
    env_example = Path("env.example")
    
    if not env_file.exists():
        if env_example.exists():
            shutil.copy(env_example, env_file)
            print("✅ Created .env file from env.example")
            print("   Please edit .env and add your GEMINI_API_KEY")
        else:
            print("⚠️ No env.example file found")
    else:
        print("✅ .env file already exists")
    
    return True

def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    
    directories = ["images", "logs", "temp"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def test_installation():
    """Test the installation."""
    print("🧪 Testing installation...")
    
    try:
        # Test imports
        import cv2
        import numpy as np
        from PIL import Image
        import flask
        import google.generativeai as genai
        from dotenv import load_dotenv
        
        print("✅ All required modules imported successfully")
        
        # Test OpenCV
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Camera access test passed")
            cap.release()
        else:
            print("⚠️ Camera not accessible (this is normal if no camera is connected)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 TRACE Setup Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Setup environment
    if not setup_environment():
        print("❌ Failed to setup environment")
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        print("❌ Failed to create directories")
        sys.exit(1)
    
    # Test installation
    if not test_installation():
        print("❌ Installation test failed")
        sys.exit(1)
    
    print("\n🎉 TRACE setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Edit .env file and add your GEMINI_API_KEY")
    print("2. Run: python example.py")
    print("3. Or run: python trace_api.py (for API server)")
    print("4. Open trace_web_client.html in your browser")
    
    if "--dev" in sys.argv:
        print("\n🔧 Development setup:")
        print("- Install pre-commit hooks: pre-commit install")
        print("- Run tests: python -m pytest")
        print("- Format code: black TRACE/")
        print("- Lint code: flake8 TRACE/")

if __name__ == "__main__":
    main()
