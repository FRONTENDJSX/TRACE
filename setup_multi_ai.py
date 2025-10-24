#!/usr/bin/env python3
"""
TRACE Multi-AI Setup Script
Helps users configure TRACE with multiple AI providers
"""

import os
import sys
from pathlib import Path

def main():
    """Main setup function"""
    print("=== TRACE Multi-AI Setup ===")
    print("This script will help you configure TRACE with multiple AI providers")
    print()
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print("✅ Python version check passed")
    
    # Check if .env exists
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file already exists")
        overwrite = input("Do you want to overwrite it? (y/N): ").strip().lower()
        if overwrite != 'y':
            print("Skipping .env creation")
            return
    else:
        print("📝 Creating .env file...")
    
    # Create .env file
    env_content = """# TRACE Multi-AI Configuration
# Copy this to .env and fill in your API keys

# System settings
TRACE_DATA_FILE=TRACE.json
TRACE_IMAGES_DIR=images/
ANATOMICAL_THRESHOLD=0.85
ANATOMICAL_WEIGHT=0.95
ACCESSORY_WEIGHT=0.02
EXPRESSION_WEIGHT=0.0

# AI Provider (choose one or more)
# Options: gemini, openai, claude, ollama
AI_PROVIDER=gemini

# Google Gemini API
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash

# OpenAI API
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4-vision-preview

# Anthropic Claude API
ANTHROPIC_API_KEY=your_anthropic_api_key_here
CLAUDE_MODEL=claude-3-5-sonnet-20241022

# Local Ollama
OLLAMA_ENABLED=false
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llava:7b

# API settings
API_HOST=0.0.0.0
API_PORT=5000
API_DEBUG=true

# Logging
LOG_LEVEL=INFO
LOG_FILE=trace.log
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("✅ .env file created")
    
    # Check dependencies
    print("\n🔍 Checking dependencies...")
    
    required_packages = [
        "flask",
        "flask-cors",
        "opencv-python",
        "pillow",
        "numpy",
        "google-generativeai",
        "openai",
        "anthropic",
        "requests",
        "python-dotenv"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them with: pip install -r requirements.txt")
        return
    
    print("✅ All dependencies are available")
    
    # Test AI providers
    print("\n🧪 Testing AI providers...")
    
    from config import get_config
    config = get_config()
    
    available_providers = config.get_available_providers()
    print(f"Available providers: {', '.join(available_providers)}")
    
    if not available_providers:
        print("❌ No AI providers are configured")
        print("Please edit .env file and add at least one API key")
        return
    
    # Test each provider
    for provider in available_providers:
        print(f"\n🔄 Testing {provider} provider...")
        
        try:
            from ai_providers import AIProviderFactory
            provider_config = config.get_ai_provider_config(provider)
            provider_instance = AIProviderFactory.create_provider(provider, **provider_config)
            
            if provider_instance.is_available():
                print(f"✅ {provider} provider is working")
            else:
                print(f"❌ {provider} provider is not available")
        except Exception as e:
            print(f"❌ {provider} provider error: {e}")
    
    print("\n🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Edit .env file and add your API keys")
    print("2. Run: python TRACE_multi_ai.py")
    print("3. Or run the API server: python trace_api_multi_ai.py")
    print("\nFor more information, see README_MULTI_AI.md")

if __name__ == "__main__":
    main()
