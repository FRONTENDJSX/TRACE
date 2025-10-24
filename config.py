"""
Configuration system for TRACE with multiple AI providers
"""

import os
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

class TRACEConfig:
    """Configuration manager for TRACE system"""
    
    def __init__(self, config_file: str = None):
        """Initialize configuration"""
        self.config_file = config_file
        self.config = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from environment and config file"""
        # Load environment variables
        load_dotenv()
        
        # Default configuration
        self.config = {
            # System settings
            "data_file": os.getenv("TRACE_DATA_FILE", "TRACE.json"),
            "images_dir": os.getenv("TRACE_IMAGES_DIR", "images/"),
            
            # Recognition thresholds
            "anatomical_threshold": float(os.getenv("ANATOMICAL_THRESHOLD", "0.85")),
            "anatomical_weight": float(os.getenv("ANATOMICAL_WEIGHT", "0.95")),
            "accessory_weight": float(os.getenv("ACCESSORY_WEIGHT", "0.02")),
            "expression_weight": float(os.getenv("EXPRESSION_WEIGHT", "0.0")),
            
            # AI Provider settings
            "ai_provider": os.getenv("AI_PROVIDER", "gemini"),
            "ai_providers": {
                "gemini": {
                    "api_key": os.getenv("GEMINI_API_KEY"),
                    "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                    "enabled": bool(os.getenv("GEMINI_API_KEY"))
                },
                "openai": {
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "model": os.getenv("OPENAI_MODEL", "gpt-4-vision-preview"),
                    "enabled": bool(os.getenv("OPENAI_API_KEY"))
                },
                "claude": {
                    "api_key": os.getenv("ANTHROPIC_API_KEY"),
                    "model": os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022"),
                    "enabled": bool(os.getenv("ANTHROPIC_API_KEY"))
                },
                "ollama": {
                    "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                    "model": os.getenv("OLLAMA_MODEL", "llava:7b"),
                    "enabled": os.getenv("OLLAMA_ENABLED", "false").lower() == "true"
                }
            },
            
            # API settings
            "api_host": os.getenv("API_HOST", "0.0.0.0"),
            "api_port": int(os.getenv("API_PORT", "5000")),
            "api_debug": os.getenv("API_DEBUG", "true").lower() == "true",
            
            # Logging settings
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "log_file": os.getenv("LOG_FILE", "trace.log")
        }
        
        # Load from config file if specified
        if self.config_file and os.path.exists(self.config_file):
            self._load_config_file()
    
    def _load_config_file(self):
        """Load configuration from file"""
        try:
            import json
            with open(self.config_file, 'r') as f:
                file_config = json.load(f)
                self.config.update(file_config)
        except Exception as e:
            print(f"Warning: Could not load config file {self.config_file}: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        self.config[key] = value
    
    def get_ai_provider_config(self, provider: str) -> Dict[str, Any]:
        """Get configuration for specific AI provider"""
        return self.config.get("ai_providers", {}).get(provider, {})
    
    def get_available_providers(self) -> List[str]:
        """Get list of available AI providers"""
        available = []
        for provider, config in self.config.get("ai_providers", {}).items():
            if config.get("enabled", False):
                available.append(provider)
        return available
    
    def get_primary_provider(self) -> str:
        """Get the primary AI provider"""
        return self.config.get("ai_provider", "gemini")
    
    def set_primary_provider(self, provider: str):
        """Set the primary AI provider"""
        if provider in self.get_available_providers():
            self.config["ai_provider"] = provider
        else:
            raise ValueError(f"Provider {provider} is not available or not enabled")
    
    def enable_provider(self, provider: str, **kwargs):
        """Enable and configure an AI provider"""
        if provider not in self.config.get("ai_providers", {}):
            self.config["ai_providers"][provider] = {}
        
        provider_config = self.config["ai_providers"][provider]
        provider_config.update(kwargs)
        provider_config["enabled"] = True
    
    def disable_provider(self, provider: str):
        """Disable an AI provider"""
        if provider in self.config.get("ai_providers", {}):
            self.config["ai_providers"][provider]["enabled"] = False
    
    def save_config(self, config_file: str = None):
        """Save configuration to file"""
        if config_file:
            self.config_file = config_file
        
        if not self.config_file:
            return
        
        try:
            import json
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Error saving config file {self.config_file}: {e}")
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Check if at least one provider is enabled
        available_providers = self.get_available_providers()
        if not available_providers:
            issues.append("No AI providers are enabled. Please configure at least one provider.")
        
        # Check primary provider
        primary_provider = self.get_primary_provider()
        if primary_provider not in available_providers:
            issues.append(f"Primary provider '{primary_provider}' is not available or not enabled.")
        
        # Check provider configurations
        for provider, config in self.config.get("ai_providers", {}).items():
            if config.get("enabled", False):
                if provider == "gemini" and not config.get("api_key"):
                    issues.append("Gemini provider is enabled but no API key is configured.")
                elif provider == "openai" and not config.get("api_key"):
                    issues.append("OpenAI provider is enabled but no API key is configured.")
                elif provider == "claude" and not config.get("api_key"):
                    issues.append("Claude provider is enabled but no API key is configured.")
                elif provider == "ollama" and not config.get("base_url"):
                    issues.append("Ollama provider is enabled but no base URL is configured.")
        
        return issues
    
    def get_env_example(self) -> str:
        """Get example environment variables"""
        return """
# TRACE Configuration
# Copy this to .env and fill in your API keys

# System settings
TRACE_DATA_FILE=TRACE.json
TRACE_IMAGES_DIR=images/
ANATOMICAL_THRESHOLD=0.85
ANATOMICAL_WEIGHT=0.95
ACCESSORY_WEIGHT=0.02
EXPRESSION_WEIGHT=0.0

# AI Provider (choose one or more)
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

# Global configuration instance
config = TRACEConfig()

def get_config() -> TRACEConfig:
    """Get global configuration instance"""
    return config

def reload_config():
    """Reload global configuration"""
    global config
    config = TRACEConfig()
    return config
