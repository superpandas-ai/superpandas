"""
Configuration settings for SuperPandas
"""
from typing import Optional, Type, Union
from smolagents import Model
from .templates import system_template  # Import the system_template

class SuperPandasConfig:
    """
    Global configuration settings for SuperPandas.
    
    Attributes:
        llm_provider (Type[Model]): The default LLM provider class to use
        llm_model (Union[str, Model]): The default model name or instance to use
        llm_kwargs (dict): Additional keyword arguments for LLM initialization, including:
            - existing_values: How to handle existing metadata ('warn', 'skip', 'overwrite')
        system_template (str): The default system template for LLM queries
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SuperPandasConfig, cls).__new__(cls)
            # Initialize default settings
            cls._instance._llm_provider = "HfApiModel"
            cls._instance._llm_model = "meta-llama/Llama-3.2-3B-Instruct"
            cls._instance._llm_kwargs = {'existing_values': 'warn'}
            cls._instance._system_template = system_template  # Set default system template
        return cls._instance
    
    @property
    def llm_provider(self) -> Optional[Type[Model]]:
        """Get the default LLM provider class"""
        return self._llm_provider
    
    @llm_provider.setter
    def llm_provider(self, value: Optional[Type[Model]]):
        """Set the default LLM provider class"""
        self._llm_provider = value
    
    @property
    def llm_model(self) -> Optional[Union[str, Model]]:
        """Get the default LLM model"""
        return self._llm_model
    
    @llm_model.setter
    def llm_model(self, value: Optional[Union[str, Model]]):
        """Set the default LLM model"""
        self._llm_model = value
    
    @property
    def llm_kwargs(self) -> dict:
        """Get additional LLM initialization kwargs"""
        return self._llm_kwargs
    
    @llm_kwargs.setter
    def llm_kwargs(self, value: dict):
        """Set additional LLM initialization kwargs"""
        if 'existing_values' in value and value['existing_values'] not in ('warn', 'skip', 'overwrite'):
            raise ValueError("existing_values must be one of: 'warn', 'skip', 'overwrite'")
        self._llm_kwargs = value
    
    @property
    def system_template(self) -> str:
        """Get the default system template"""
        return self._system_template
    
    @system_template.setter
    def system_template(self, value: str):
        """Set the default system template"""
        self._system_template = value

    def configure_llm(self, 
                     provider_class: Optional[Type[Model]] = None,
                     model: Optional[Union[str, Model]] = None,
                     system_template: Optional[str] = None,
                     **kwargs):
        """
        Configure the default LLM settings.
        
        Args:
            provider_class: The LLM provider class to use
            model: The model name or instance to use
            system_template: The system template for LLM queries
            **kwargs: Additional keyword arguments for LLM initialization, including:
                - existing_values: How to handle existing metadata ('warn', 'skip', 'overwrite')
        """
        if provider_class is not None:
            self.llm_provider = provider_class
        if model is not None:
            self.llm_model = model
        if system_template is not None:
            self.system_template = system_template
        if kwargs:
            if 'existing_values' in kwargs and kwargs['existing_values'] not in ('warn', 'skip', 'overwrite'):
                raise ValueError("existing_values must be one of: 'warn', 'skip', 'overwrite'")
            self.llm_kwargs.update(kwargs)

    def __str__(self):
        return (f"SuperPandasConfig("
                f"llm_provider={self.llm_provider}, "
                f"llm_model={self.llm_model}, "
                f"llm_kwargs={self.llm_kwargs}, "
                f"system_template={self.system_template})")

    def __repr__(self):
        return self.__str__()

# Global config instance
config = SuperPandasConfig() 