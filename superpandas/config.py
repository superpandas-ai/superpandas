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
    Manages global configuration settings for the SuperPandas library.

    This class handles settings such as the default LLM provider and model,
    keyword arguments for LLM initialization, and templates for system and user prompts.
    Configurations can be saved to and loaded from a JSON file.

    :cvar DEFAULT_CONFIG_PATH: The default path for storing the configuration file
                               (`~/.cache/superpandas/config.json`).
    :ivar _provider: Name of the default LLM provider.
    :ivar _model: Name or identifier of the default LLM model.
    :ivar _llm_kwargs: Dictionary of keyword arguments for LLM initialization.
                       Includes `existing_values` strategy ('warn', 'skip', 'overwrite').
    :ivar _system_template: Default system prompt template.
    :ivar _user_template: Default user prompt template.
    """

    DEFAULT_CONFIG_PATH = os.path.expanduser(
        "~/.cache/superpandas/config.json")

    def __init__(self):
        """Initializes SuperPandasConfig with default values and attempts to load from file."""
        self._provider: str = "HfApiModel"
        self._model: str = "meta-llama/Llama-3.2-3B-Instruct"
        self._llm_kwargs: Dict[str, Any] = {'existing_values': 'warn'}
        self._system_template: str = system_template  # Default from templates.py
        self._user_template: str = user_template    # Default from templates.py
        self.load()  # Attempt to load saved configuration

    @property
    def provider(self) -> Literal['LiteLLMModel', 'OpenAIServerModel', 'HfApiModel', 'TransformersModel', 'VLLMModel', 'MLXModel', 'AzureOpenAIServerModel']:
        """The default LLM provider name (e.g., 'OpenAIServerModel', 'HfApiModel')."""
        return self._provider

    @provider.setter
    def provider(self, value: Literal['LiteLLMModel', 'OpenAIServerModel', 'HfApiModel', 'TransformersModel', 'VLLMModel', 'MLXModel', 'AzureOpenAIServerModel']):
        """Sets the default LLM provider name."""
        self._provider = value

    @property
    def model(self) -> str:
        """The default LLM model name or identifier (e.g., 'gpt-3.5-turbo')."""
        return self._model

    @model.setter
    def model(self, value: str):
        """Sets the default LLM model name."""
        self._model = value

    @property
    def llm_kwargs(self) -> Dict[str, Any]:
        """
        Additional keyword arguments for LLM initialization.

        This can include provider-specific parameters or the `existing_values`
        strategy ('warn', 'skip', 'overwrite') for metadata generation.
        """
        return self._llm_kwargs

    @llm_kwargs.setter
    def llm_kwargs(self, kwargs: Dict[str, Any]):
        """
        Sets additional LLM initialization keyword arguments.

        :raises ValueError: If `existing_values` is provided and not one of
                            'warn', 'skip', or 'overwrite'.
        """
        if 'existing_values' in kwargs and kwargs['existing_values'] not in ('warn', 'skip', 'overwrite'):
            raise ValueError(
                "existing_values must be one of: 'warn', 'skip', 'overwrite'")
        self._llm_kwargs = kwargs

    @property
    def system_template(self) -> str:
        """The default system prompt template for LLM queries."""
        return self._system_template

    @system_template.setter
    def system_template(self, value: str):
        """Sets the default system prompt template."""
        self._system_template = value

    @property
    def user_template(self) -> str:
        """The default user prompt template for LLM queries."""
        return self._user_template

    @user_template.setter
    def user_template(self, value: str):
        """Sets the default user prompt template."""
        self._user_template = value

    def __str__(self) -> str:
        return (f"SuperPandasConfig("
                f"provider={self.provider}, "
                f"model={self.model}, "
                f"llm_kwargs={self.llm_kwargs}, "
                f"system_template={self.system_template})")

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
            'system_template': self.system_template,
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
        self.system_template = config_dict.get(
            'system_template', self.system_template)
        self.user_template = config_dict.get(
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
            self._llm_kwargs = {'existing_values': 'warn'}
            self._system_template = system_template
            self._user_template = user_template
