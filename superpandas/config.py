"""
Configuration settings for SuperPandas
"""
from typing import Literal, Optional, Type, Union, Dict, Any
import json
import os
from smolagents import Model
from .templates import system_template, user_template  # Import the system_template

class SuperPandasConfig:
    """
    Global configuration settings for SuperPandas.
    
    Attributes:
        provider (Type[Model]): The default LLM provider class to use
        model (Union[str, Model]): The default model name or instance to use
        llm_kwargs (dict): Additional keyword arguments for LLM initialization, including:
            - existing_values: How to handle existing metadata ('warn', 'skip', 'overwrite')
        system_template (str): The default system template for LLM queries
        user_template (str): The default user template for LLM queries
    """
    
    DEFAULT_CONFIG_PATH = os.path.expanduser("~/.cache/superpandas/config.json")
    
    def __init__(self):
        # Initialize default settings
        self._provider = "HfApiModel"
        self._model = "meta-llama/Llama-3.2-3B-Instruct"
        self._llm_kwargs = {'existing_values': 'warn'}
        self._system_template = system_template  # Set default system template
        self._user_template = user_template  # Set default user template

    @property
    def provider(self) -> Literal['LiteLLMModel', 'OpenAIServerModel', 'HfApiModel', 'TransformersModel', 'VLLMModel', 'MLXModel', 'AzureOpenAIServerModel']:
        """Get the default LLM provider class"""
        return self._provider
    
    @provider.setter
    def provider(self, value: Literal['LiteLLMModel', 'OpenAIServerModel', 'HfApiModel', 'TransformersModel', 'VLLMModel', 'MLXModel', 'AzureOpenAIServerModel']):
        """Set the default LLM provider class"""
        self._provider = value
    
    @property
    def model(self) -> str:
        """Get the default LLM model"""
        return self._model
    
    @model.setter
    def model(self, value: str):
        """Set the default LLM model"""
        self._model = value
    
    @property
    def llm_kwargs(self) -> dict:
        """Get additional LLM initialization kwargs"""
        return self._llm_kwargs
    
    @llm_kwargs.setter
    def llm_kwargs(self, kwargs: dict):
        """Set additional LLM initialization kwargs"""
        if 'existing_values' in kwargs and kwargs['existing_values'] not in ('warn', 'skip', 'overwrite'):
            raise ValueError("existing_values must be one of: 'warn', 'skip', 'overwrite'")
        self._llm_kwargs = kwargs
    
    @property
    def system_template(self) -> str:
        """Get the default system template"""
        return self._system_template
    
    @system_template.setter
    def system_template(self, value: str):
        """Set the default system template"""
        self._system_template = value

    @property
    def user_template(self) -> str:
        """Get the default user template"""
        return self._user_template
    
    @user_template.setter
    def user_template(self, value: str):
        """Set the default user template"""
        self._user_template = value

    def __str__(self):
        return (f"SuperPandasConfig("
                f"provider={self.provider}, "
                f"model={self.model}, "
                f"llm_kwargs={self.llm_kwargs}, "
                f"system_template={self.system_template})")

    def __repr__(self):
        return self.__str__()

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
        return {
            'provider': self._provider,
            'model': self._model,
            'llm_kwargs': self._llm_kwargs,
            'system_template': self._system_template,
            'user_template': self._user_template
        }
    
    def from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Load configuration from a dictionary."""
        if 'provider' in config_dict:
            self._provider = config_dict['provider']
        if 'model' in config_dict:
            self._model = config_dict['model']
        if 'llm_kwargs' in config_dict:
            self._llm_kwargs = config_dict['llm_kwargs']
        if 'system_template' in config_dict:
            self._system_template = config_dict['system_template']
        if 'user_template' in config_dict:
            self._user_template = config_dict['user_template']
    
    def save(self, filepath: Optional[str] = None) -> None:
        """
        Save the current configuration to a JSON file.
        
        Args:
            filepath: Path to save the configuration file. If None, uses the default path.
        """
        if filepath is None:
            filepath = self.DEFAULT_CONFIG_PATH
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
    
    def load(self, filepath: Optional[str] = None) -> None:
        """
        Load configuration from a JSON file.
        
        Args:
            filepath: Path to load the configuration file from. If None, uses the default path.
        """
        if filepath is None:
            filepath = self.DEFAULT_CONFIG_PATH
            
        if not os.path.exists(filepath):
            return
            
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
            self.from_dict(config_dict)