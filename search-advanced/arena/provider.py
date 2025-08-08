from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from enum import Enum
import os
import json
from pathlib import Path

class ModelProvider(str, Enum):
    OLLAMA = "ollama"  # Listed first to be the default
    HUGGINGFACE = "huggingface"

    @classmethod
    def from_str(cls, value: str) -> 'ModelProvider':
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"Invalid provider: {value}. Must be one of {[p.value for p in cls]}")

@dataclass
class ProviderConfig:
    """Configuration for a specific model provider"""
    name: str
    default_model: str
    collection_name: str
    embedding_model: str
    max_tokens: int = 512
    additional_params: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProviderConfig':
        """Create config from dictionary"""
        return cls(**data)

    @classmethod
    def from_env(cls, provider: str) -> 'ProviderConfig':
        """Create provider config from environment variables"""
        # Common Milvus connection parameters - stored separately
        milvus_params = {
            "host": os.getenv("MILVUS_HOST", "localhost"),
            "port": os.getenv("MILVUS_PORT", "19530")
        }
        
        if provider == ModelProvider.HUGGINGFACE:
            # Model-specific parameters only (no Milvus params)
            model_params = {
                "task": "conversational",
                "do_sample": False,
                "repetition_penalty": 1.03
            }
            
            # Store Milvus params separately in the additional_params
            additional_params = {
                "model_params": model_params,
                "milvus_params": milvus_params
            }
            
            return cls(
                name=ModelProvider.HUGGINGFACE,
                # Use a proper text generation model, not an embedding model
                default_model=os.getenv("HF_LLM_MODEL", ''),
                # Use provider-specific collection name
                collection_name="milvus_hf_collection",
                # Use the embedding model from env or default
                embedding_model=os.getenv("HF_EMBEDDING_MODEL", ''),
                max_tokens=int(os.getenv("MAX_TOKENS_HF", "512")),
                additional_params=additional_params
            )
        elif provider == ModelProvider.OLLAMA:
            # Model-specific parameters only
            model_params = {
                "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            }
            
            # Store Milvus params separately
            additional_params = {
                "model_params": model_params,
                "milvus_params": milvus_params
            }
            
            return cls(
                name=ModelProvider.OLLAMA,
                default_model=os.getenv("OLLAMA_LLM_MODEL", ''),
                # Use provider-specific collection name
                collection_name="milvus_ollama_collection",
                # Always use nomic-embed-text which has consistent 768 dimensions
                embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", ''),
                max_tokens=int(os.getenv("MAX_TOKENS_OLLAMA", "512")),
                additional_params=additional_params
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

class ModelRegistry:
    """Registry for model providers and their configurations"""
    CONFIG_FILE = Path.home() / ".advanced_search_config.json"
    
    def __init__(self):
        self.providers: Dict[str, ProviderConfig] = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file or initialize defaults"""
        if self.CONFIG_FILE.exists():
            try:
                with open(self.CONFIG_FILE) as f:
                    data = json.load(f)
                    self.providers = {
                        name: ProviderConfig.from_dict(config)
                        for name, config in data.items()
                    }
            except Exception as e:
                print(f"Error loading config file: {e}")
                self._load_default_providers()
        else:
            self._load_default_providers()
    
    def _load_default_providers(self):
        """Load default provider configurations"""
        for provider in ModelProvider:
            # Use the provider value (string) as the key
            self.providers[provider.value] = ProviderConfig.from_env(provider.value)
        self._save_config()
    
    def _save_config(self):
        """Save current configuration to file"""
        try:
            config_data = {
                name: config.to_dict()
                for name, config in self.providers.items()
            }
            self.CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(self.CONFIG_FILE, 'w') as f:
                json.dump(config_data, f, indent=2)
        except Exception as e:
            print(f"Error saving config file: {e}")
    
    def get_provider(self, name: str) -> ProviderConfig:
        """Get provider configuration by name"""
        # Convert name to lowercase for case-insensitive comparison
        name_lower = name.lower()
        
        # Try to find the provider by exact match
        if name_lower in self.providers:
            return self.providers[name_lower]
        
        # Try to find the provider by enum value
        for provider_name, provider_config in self.providers.items():
            if provider_name.lower() == name_lower or provider_config.name.lower() == name_lower:
                return provider_config
                
        raise ValueError(f"Provider {name} not found in registry")
    
    def get_provider_model(self, provider: str) -> str:
        """Get the default model for a provider"""
        if provider not in self.providers:
            raise ValueError(f"Provider {provider} not found in registry")
        return self.providers[provider].default_model        

    def add_provider(self, config: ProviderConfig):
        """Add or update a provider configuration"""
        self.providers[config.name] = config
        self._save_config()
    
    def set_default_model(self, provider: str, model: str):
        """Set the default model for a provider"""
        if provider not in self.providers:
            raise ValueError(f"Provider {provider} not found in registry")
        config = self.providers[provider]
        config.default_model = model
        self._save_config()
    
    def reset_to_defaults(self):
        """Reset all configurations to environment-based defaults"""
        self._load_default_providers()
        
    def debug_info(self) -> str:
        """Get debug information about the registry"""
        info = ["Model Registry Debug Info:"]
        info.append(f"Config file: {self.CONFIG_FILE}")
        info.append(f"Config file exists: {self.CONFIG_FILE.exists()}")
        info.append(f"Number of providers: {len(self.providers)}")
        
        for key, config in self.providers.items():
            info.append(f"Provider key: '{key}', name: '{config.name}', model: '{config.default_model}'")
            
        return "\n".join(info)

# Global model registry instance
model_registry = ModelRegistry()