"""
Configuration settings for SuperPandas
"""
from typing import Literal, Optional, Dict, Any
import json
import os

from .templates import system_template, user_template, schema_template

from .providers import available_providers

class SuperPandasConfig:
    """
    Manages global configuration settings for the SuperPandas library.

    This class handles settings such as the default LLM provider and model,
    keyword arguments for LLM initialization, and templates for system and user prompts.
    Configurations can be saved to and loaded from a JSON file.

    :cvar DEFAULT_CONFIG_PATH: The default path for storing the configuration file
                               (`~/.cache/superpandas/config.json`).
    :cvar default_config: Class-level default configuration instance that can be modified by users.
    :ivar _provider: Name of the default LLM provider.
    :ivar _model: Name or identifier of the default LLM model.
    :ivar llm_kwargs: Dictionary of keyword arguments for LLM invocation.
    :ivar _existing_values: Strategy for handling existing metadata values ('warn', 'skip', 'overwrite').
    :ivar system_template: Default system prompt template.
    :ivar _user_template: Default user prompt template.
    :ivar _schema_template: Default schema prompt template.
    """

    DEFAULT_CONFIG_PATH = os.path.expanduser(
        "~/.cache/superpandas/config.json")
    
    # Class-level default config that can be modified by users
    default_config = None

    def __init__(self, load_saved: bool = False):
        """Initializes SuperPandasConfig with default values.
        
        Parameters:
        -----------
        load_saved : bool, default False
            Whether to load saved configuration from file
        """
        self._provider: str = "hf"
        self.model: str = "meta-llama/Llama-3.2-3B-Instruct"
        self.llm_kwargs: Dict[str, Any] = {}
        self._existing_values: Literal['warn', 'skip', 'overwrite'] = 'warn'
        self.system_template: str = system_template  # Default from templates.py
        self._user_template: str = user_template  # Default from templates.py
        self._schema_template: str = schema_template  # Default from templates.py
        if load_saved:
            self.load()  # Only load saved configuration if requested

    @classmethod
    def set_default_config(cls, config: 'SuperPandasConfig'):
        """
        Set the default configuration instance for the library.
        
        Parameters:
        -----------
        config : SuperPandasConfig
            The configuration instance to use as default
        """
        cls.default_config = config
        # Save the config to the default path
        config.save()

    @classmethod
    def get_default_config(cls) -> 'SuperPandasConfig':
        """
        Get the default configuration instance, creating one if it doesn't exist.
        
        Returns:
        --------
        SuperPandasConfig
            The default configuration instance
        """
        if cls.default_config is None:
            # Try to load from the config file
            if os.path.exists(cls.DEFAULT_CONFIG_PATH):
                config = cls()
                config.load()
                cls.default_config = config
            else:
                # If no config file exists, create one with package defaults
                config = cls()
                config.save()  # Save the default config
                cls.default_config = config
        return cls.default_config

    @property
    def provider(self) -> Literal['lite', 
                                  'openai', 
                                  'hf', 
                                  'tf', 
                                  'vllm', 
                                  'mlx', 
                                  'openai_az',
                                  'bedrock',
                                  ]:
        """The default LLM provider name (e.g., 'openai', 'hf', 'vllm', 'mlx', 'bedrock' etc)."""

        return self._provider

    @provider.setter
    def provider(self, value: Literal['lite', 
                                      'openai', 
                                      'hf', 
                                      'tf', 
                                      'vllm', 
                                      'mlx', 
                                      'openai_az',
                                      'bedrock',
                                      ]):
        """Sets the default LLM provider name."""
        if value not in available_providers:
            raise ValueError(f"Provider {value} is not available. Please install smolagents with the provider.")
        self._provider = value

    @property
    def existing_values(self) -> Literal['warn', 'skip', 'overwrite']:
        """
        Strategy for handling existing metadata values.
        
        Returns:
        --------
        Literal['warn', 'skip', 'overwrite']
            The strategy to use when handling existing metadata values
        """
        return self._existing_values

    @existing_values.setter
    def existing_values(self, value: Literal['warn', 'skip', 'overwrite']):
        """
        Sets the strategy for handling existing metadata values.
        
        Parameters:
        -----------
        value : Literal['warn', 'skip', 'overwrite']
            The strategy to use when handling existing metadata values
            
        Raises:
        -------
        ValueError
            If value is not one of 'warn', 'skip', or 'overwrite'
        """
        if value not in ('warn', 'skip', 'overwrite'):
            raise ValueError("existing_values must be one of: 'warn', 'skip', 'overwrite'")
        self._existing_values = value

    @property
    def schema_template(self) -> str:
        """Get the system template"""
        return self._schema_template
    
    @schema_template.setter
    def schema_template(self, value: str):
        """Set the system template
        
        Parameters:
        -----------
        value : str
            The system template string. Must contain the following placeholders:
            - {name}: The dataframe name
            - {description}: The dataframe description
            - {column_info}: The formatted column information
            - {shape}: The dataframe shape
            
        Raises:
        -------
        ValueError
            If the template is missing any required placeholders
        """
        required_placeholders = ['{name}', '{description}', '{column_info}', '{shape}'] # TODO: check if we can make any optional.
        missing_placeholders = [p for p in required_placeholders if p not in value]
        
        if missing_placeholders:
            raise ValueError(
                f"System template is missing required placeholders: {', '.join(missing_placeholders)}. "
                "All templates must include: {name}, {description}, {column_info}, and {shape}"
            )
            
        self._schema_template = value

    @property
    def user_template(self) -> str:
        """Get the user template"""
        return self._user_template
    
    @user_template.setter
    def user_template(self, value: str):
        """Set the user template
        
        Parameters:
        -----------
        value : str
            The user template string. Must contain the following placeholders:
            - {question}: The user's question
            - {schema}: The dataframe schema information
            
        Raises:
        -------
        ValueError
            If the template is missing any required placeholders
        """
        required_placeholders = ['{question}', '{schema}']
        missing_placeholders = [p for p in required_placeholders if p not in value]
        
        if missing_placeholders:
            raise ValueError(
                f"User template is missing required placeholders: {', '.join(missing_placeholders)}. "
                "All templates must include: {question} and {schema}"
            )
            
        self._user_template = value

    def __str__(self) -> str:
        return (f"SuperPandasConfig("
                f"provider={self.provider}, "
                f"model={self.model}, "
                f"llm_kwargs={self.llm_kwargs}, "
                f"existing_values={self.existing_values}, "
                f"system_template={self.system_template}, "
                f"schema_template={self.schema_template}, "
                f"user_template={self.user_template})")

    def __repr__(self):
        return self.__str__()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the current configuration to a dictionary.

        :return: A dictionary representation of the configuration.
        :rtype: Dict[str, Any]
        """
        return {
            'provider': self.provider,
            'model': self.model,
            'llm_kwargs': self.llm_kwargs,
            'existing_values': self.existing_values,
            'system_template': self.system_template,
            'schema_template': self.schema_template,
            'user_template': self.user_template
        }

    def from_dict(self, config_dict: Dict[str, Any]):
        """
        Load configuration settings from a dictionary.

        Updates the instance's attributes with values from the dictionary.
        Keys not present in the dictionary will retain their current values.

        :param config_dict: A dictionary containing configuration settings.
        :type config_dict: Dict[str, Any]
        """
        self.provider = config_dict.get('provider', self.provider)
        self.model = config_dict.get('model', self.model)
        self.llm_kwargs = config_dict.get('llm_kwargs', self.llm_kwargs)
        self.existing_values = config_dict.get('existing_values', self.existing_values)
        self.system_template = config_dict.get( # TODO: Should we default to system_template?
            'system_template', self.system_template)
        self.schema_template = config_dict.get( # TODO: Should we default to schema_template?
            'schema_template', self.schema_template)
        self.user_template = config_dict.get( # TODO: Should we default to user_template?
            'user_template', self.user_template)

    def save(self, filepath: Optional[str] = None):
        """
        Save the current configuration to a JSON file.

        The directory for the file will be created if it does not exist.

        :param filepath: Path to save the configuration file.
                         If None, uses `DEFAULT_CONFIG_PATH`.
        :type filepath: Optional[str]
        """
        if filepath is None:
            filepath = self.DEFAULT_CONFIG_PATH

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=4)

    def load(self, filepath: Optional[str] = None):
        """
        Load configuration from a JSON file.

        If the specified file does not exist, the current configuration remains unchanged.

        :param filepath: Path to load the configuration file from.
                         If None, uses `DEFAULT_CONFIG_PATH`.
        :type filepath: Optional[str]
        """
        if filepath is None:
            filepath = self.DEFAULT_CONFIG_PATH

        if not os.path.exists(filepath):
            return  # Keep defaults if file doesn't exist

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
                self.from_dict(config_dict)
        except (json.JSONDecodeError, IOError) as e:
            # Log error or handle appropriately, e.g., fall back to defaults
            print(
                f"Warning: Could not load configuration from {filepath}: {e}. Using defaults.")
            # Re-initialize defaults if loading fails badly
            self._provider = "HfApiModel"
            self._model = "meta-llama/Llama-3.2-3B-Instruct"
            self.llm_kwargs = {}
            self._existing_values = 'warn'
            self.system_template = system_template
            self._schema_template = schema_template
            self._user_template = user_template
