"""
Configuration settings for SuperPandas
"""
from typing import Optional, Type, Union, Literal
from smolagents import Model

class SuperPandasConfig:
    """
    Global configuration settings for SuperPandas.
    
    Attributes:
        llm_provider_class (Type[Model]): The default LLM provider class to use
        llm_model (Union[str, Model]): The default model name or instance to use
        llm_kwargs (dict): Additional keyword arguments for LLM initialization, including:
            - existing_values: How to handle existing metadata ('warn', 'skip', 'overwrite')
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SuperPandasConfig, cls).__new__(cls)
            # Initialize default settings
            cls._instance._llm_provider_class = None
            cls._instance._llm_model = None
            cls._instance._llm_kwargs = {'existing_values': 'warn'}
        return cls._instance
    
    @property
    def llm_provider_class(self) -> Optional[Type[Model]]:
        """Get the default LLM provider class"""
        return self._llm_provider_class
    
    @llm_provider_class.setter
    def llm_provider_class(self, value: Optional[Type[Model]]):
        """Set the default LLM provider class"""
        self._llm_provider_class = value
    
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
    
    def configure_llm(self, 
                     provider_class: Optional[Type[Model]] = None,
                     model: Optional[Union[str, Model]] = None,
                     **kwargs):
        """
        Configure the default LLM settings.
        
        Args:
            provider_class: The LLM provider class to use
            model: The model name or instance to use
            **kwargs: Additional keyword arguments for LLM initialization, including:
                - existing_values: How to handle existing metadata ('warn', 'skip', 'overwrite')
        """
        if provider_class is not None:
            self.llm_provider_class = provider_class
        if model is not None:
            self.llm_model = model
        if kwargs:
            if 'existing_values' in kwargs and kwargs['existing_values'] not in ('warn', 'skip', 'overwrite'):
                raise ValueError("existing_values must be one of: 'warn', 'skip', 'overwrite'")
            self.llm_kwargs.update(kwargs)

# Global config instance
config = SuperPandasConfig() 