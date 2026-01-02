"""
Configuration module for LLM Council
Manages model settings and environment variables
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv


@dataclass
class ModelConfig:
    """Configuration for a single model deployment."""
    deployment_name: str
    temperature: float
    role: str
    display_name: str


@dataclass
class CouncilConfig:
    """Configuration for the entire LLM Council system."""
    # Azure AI Foundry settings
    project_endpoint: str
    api_key: Optional[str]
    
    # Model configurations
    member_a: ModelConfig
    member_b: ModelConfig
    chair: ModelConfig
    
    # System settings
    timeout: int
    
    @classmethod
    def from_env(cls) -> 'CouncilConfig':
        """
        Load configuration from environment variables.
        
        Returns:
            CouncilConfig instance with loaded settings
            
        Raises:
            ValueError: If required environment variables are missing
        """
        load_dotenv()
        
        # Load Azure settings
        project_endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
        if not project_endpoint:
            raise ValueError(
                "AZURE_AI_PROJECT_ENDPOINT not found. "
                "Please copy .env.template to .env and configure your Azure settings."
            )
        
        api_key = os.getenv("AZURE_AI_API_KEY")
        # api_key can be None - will use Azure CLI authentication
        
        # Load model deployment names
        model_gpt41 = os.getenv("MODEL_DEPLOYMENT_GPT41", "gpt-4.1")
        model_deepseek = os.getenv("MODEL_DEPLOYMENT_DEEPSEEK", "DeepSeek-V3.1")
        model_grok = os.getenv("MODEL_DEPLOYMENT_GROK", "grok-3")
        
        # Load temperature settings
        temp_council = float(os.getenv("TEMPERATURE_COUNCIL_MEMBERS", "0.7"))
        temp_chair = float(os.getenv("TEMPERATURE_CHAIR", "0.3"))
        
        # Load timeout
        timeout = int(os.getenv("AGENT_TIMEOUT", "60"))
        
        # Create model configurations
        member_a = ModelConfig(
            deployment_name=model_gpt41,
            temperature=temp_council,
            role="analytical",
            display_name="Council Member A (Analytical - GPT-4.1)"
        )
        
        member_b = ModelConfig(
            deployment_name=model_deepseek,
            temperature=temp_council,
            role="critical",
            display_name="Council Member B (Critical - DeepSeek-V3.1)"
        )
        
        chair = ModelConfig(
            deployment_name=model_grok,
            temperature=temp_chair,
            role="chair",
            display_name="Chair (Grok-3)"
        )
        
        return cls(
            project_endpoint=project_endpoint,
            api_key=api_key if api_key else None,
            member_a=member_a,
            member_b=member_b,
            chair=chair,
            timeout=timeout
        )
    
    def validate(self):
        """
        Validate that all required configuration is present.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if not self.project_endpoint:
            raise ValueError("Azure AI Project Endpoint is required")
        
        if not self.member_a.deployment_name:
            raise ValueError("MODEL_DEPLOYMENT_GPT41 is required")
        
        if not self.member_b.deployment_name:
            raise ValueError("MODEL_DEPLOYMENT_DEEPSEEK is required")
        
        if not self.chair.deployment_name:
            raise ValueError("MODEL_DEPLOYMENT_GROK is required")
        
        if self.timeout <= 0:
            raise ValueError("AGENT_TIMEOUT must be positive")


# Singleton instance
_config_instance: Optional[CouncilConfig] = None


def get_config() -> CouncilConfig:
    """
    Get the global configuration instance.
    
    Returns:
        CouncilConfig instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = CouncilConfig.from_env()
        _config_instance.validate()
    return _config_instance


def reload_config() -> CouncilConfig:
    """
    Force reload configuration from environment.
    
    Returns:
        New CouncilConfig instance
    """
    global _config_instance
    _config_instance = CouncilConfig.from_env()
    _config_instance.validate()
    return _config_instance
