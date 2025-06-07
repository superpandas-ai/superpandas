from typing import Dict, Literal, Optional, Union, List, Type, TYPE_CHECKING
import pandas as pd
from smolagents import Model
from pydantic import BaseModel
from .config import SuperPandasConfig

# Import individual providers
providers = {}

try:
    from smolagents import LiteLLMModel
    providers['LiteLLMModel'] = LiteLLMModel
except ImportError:
    pass

try:
    from smolagents import OpenAIServerModel
    providers['OpenAIServerModel'] = OpenAIServerModel
except ImportError:
    pass

try:
    from smolagents import HfApiModel
    providers['HfApiModel'] = HfApiModel
except ImportError:
    pass

try:
    from smolagents import TransformersModel
    providers['TransformersModel'] = TransformersModel
except ImportError:
    pass

try:
    from smolagents import VLLMModel
    providers['VLLMModel'] = VLLMModel
except ImportError:
    pass

try:
    from smolagents import MLXModel
    providers['MLXModel'] = MLXModel
except ImportError:
    pass

try:
    from smolagents import AzureOpenAIServerModel
    providers['AzureOpenAIServerModel'] = AzureOpenAIServerModel
except ImportError:
    pass


class LLMMessage(BaseModel):
    """
    Represents a single message in a conversation with an LLM.

    :param role: The role of the message sender ('system', 'user', or 'assistant').
    :type role: Literal['system', 'user', 'assistant']
    :param content: The content of the message. Can be a simple string or a list of content blocks
                    (e.g., for multimodal inputs, though SuperPandas primarily uses text).
    :type content: Union[str, List[Dict[str, str]]]
    """
    role: Literal['system', 'user', 'assistant']
    content: Union[str, List[Dict[str, str]]]


class LLMResponse(BaseModel):
    """
    Represents a response from an LLM.

    :param content: The content of the response. Can be a simple string or a list of content blocks.
    :type content: Union[str, List[Dict[str, str]]]
    """
    content: Union[str, List[Dict[str, str]]]


class LLMClient:
    """
    Client for interacting with Large Language Models (LLMs).

    This class provides a unified interface to various LLM providers available through
    the `smolagents` library. It handles model initialization, query construction,
    and response parsing. It can also be used to generate descriptive metadata for DataFrames.

    :ivar model: The underlying LLM model instance from `smolagents`.
    """

    # Default to using HfApiModel with Llama 3.2
    try:
        DEFAULT_LLM = providers.get('HfApiModel', lambda x: None)(
            "meta-llama/Llama-3.2-3B-Instruct")  # Qwen/Qwen2.5-Coder-32B-Instruct
    except Exception as e:
        print(f"Error initializing default LLM: {e}")
        DEFAULT_LLM = None

    @staticmethod
    def available_providers() -> Dict[str, Type[Model]]:
        """
        Get a dictionary of all available LLM providers from `smolagents`.

        :return: A dictionary mapping provider names (e.g., 'OpenAIServerModel')
                 to their corresponding model classes.
        :rtype: Dict[str, Type[Model]]

        Example::

            providers = LLMClient.available_providers()
            if 'OpenAIServerModel' in providers:
                client = LLMClient(model="gpt-3.5-turbo", provider='OpenAIServerModel')
            else:
                print("OpenAI provider not available.")
        """
        return providers

    def __init__(self,
                 model: Optional[Union[str, Model]] = None,
                 provider: Optional[str] = None,
                 config: Optional[SuperPandasConfig] = None,
                 **model_kwargs):
        """
        Initialize the LLMClient.

        The client can be configured by specifying a model name and provider,
        or by passing a `SuperPandasConfig` object. If no parameters are provided,
        it attempts to use defaults from a loaded `SuperPandasConfig` or falls back
        to a predefined default LLM.

        :param model: The name of the model to use (e.g., "gpt-3.5-turbo") or an instance
                      of a `smolagents.Model`. If None, uses `config.model`.
        :type model: Optional[Union[str, Model]]
        :param provider: The name of the LLM provider (e.g., "OpenAIServerModel").
                         If None, uses `config.provider`.
        :type provider: Optional[str]
        :param config: A `SuperPandasConfig` instance. If None, a default config is loaded/created.
        :type config: Optional[SuperPandasConfig]
        :param model_kwargs: Additional keyword arguments to pass to the underlying
                             `smolagents` model provider during initialization.
        """
        if config is None:
            config = SuperPandasConfig()  # Load default or saved config

        if model is None:
            model = config.model
        if provider is None:
            provider = config.provider

        if isinstance(model, (Model, DummyLLMClient)):
            self.model = model
        elif isinstance(model, str):
            model_kwargs = {**config.llm_kwargs, **model_kwargs}

            # Remove existing_values from model_kwargs which are not required for the LLM client
            model_kwargs.pop('existing_values', None)

            provider_class = providers.get(provider)

            if not provider_class:
                print(
                    f"LLM provider {provider} not available. Please install smolagents with the provider. Using default provider: {self.DEFAULT_LLM.model_id}")
                self.model = self.DEFAULT_LLM
                return

            try:
                self.model = provider_class(model_id=model, **model_kwargs)
            except Exception as e:
                print(
                    f"Error {e} initializing model: {model} with provider {provider}. Using default provider: {self.DEFAULT_LLM.model_id}")
                self.model = self.DEFAULT_LLM

    def query(self,
              prompt: Union[str, LLMMessage, List[LLMMessage]],
              **kwargs) -> LLMResponse:
        """
        Send a query to the configured LLM and return its response.

        The prompt can be a simple string, a single `LLMMessage`, or a list of
        `LLMMessage` objects for multi-turn conversations.

        :param prompt: The prompt or messages to send to the LLM.
        :type prompt: Union[str, LLMMessage, List[LLMMessage]]
        :param kwargs: Additional keyword arguments to pass to the underlying
                       `smolagents` model's call method.
        :return: An `LLMResponse` object containing the model's output.
        :rtype: LLMResponse
        :raises RuntimeError: If no LLM provider is available or configured.
        """
        if not self.model:
            raise RuntimeError(
                "No LLM provider available or configured for the LLMClient.")

        if isinstance(prompt, str):
            messages = [LLMMessage(role='user', content=[
                                   {"type": "text", "text": prompt}])]
        elif isinstance(prompt, LLMMessage):
            messages = [prompt]
        elif isinstance(prompt, list):
            messages = prompt

        payload = [message.model_dump() for message in messages]

        response = self.model(payload, **kwargs)
        return LLMResponse(content=response.content)

    def generate_df_description(self, df: pd.DataFrame) -> str:
        """
        Generate a textual description for a DataFrame using the LLM.

        The DataFrame's schema (obtained via `df.super.get_schema()`) is used as context.

        :param df: The pandas DataFrame to describe.
        :type df: pd.DataFrame
        :return: A string containing the LLM-generated description.
        :rtype: str
        """
        prompt = f"""
        Given the following DataFrame information, provide a concise description of its contents and purpose:
        
        {df.super.get_schema()}
        
        Please provide a clear, concise description of what this DataFrame represents.
        Response should be a single paragraph, no more than 2-3 sentences.
        """

        return self.query(prompt).content

    def generate_column_descriptions(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Generate descriptions for each column in a DataFrame using the LLM.

        Column types and sample data from the DataFrame are used as context.
        The method attempts to parse the LLM's response as a Python dictionary.

        :param df: The pandas DataFrame whose columns are to be described.
        :type df: pd.DataFrame
        :return: A dictionary mapping column names to their LLM-generated descriptions.
                 If parsing fails, a fallback description is provided.
        :rtype: Dict[str, str]
        """
        prompt = f"""
        Given the following DataFrame information, provide concise descriptions for each column:
        
        Column Types:
        {df.super.column_types}
        
        Sample Data:
        {df.head(3).to_string()}
        
        Please provide short, clear descriptions for each column.
        Format your response as a Python dictionary with column names as keys and descriptions as values.
        Example format:
        {{"column_name": "Description of what this column represents"}}
        Response should be a Python dictionary, nothing else.
        """

        response = self.query(prompt).content
        try:
            # Safely evaluate the response string as a Python dictionary
            descriptions = eval(response)
            if not isinstance(descriptions, dict):
                raise ValueError("Response is not a dictionary")
            return descriptions
        except:
            # Fallback if response isn't properly formatted
            return {col: "No description available" for col in df.columns}

    def generate_df_name(self, df: pd.DataFrame) -> str:
        """
        Generate a concise, descriptive name for a DataFrame using the LLM.

        The DataFrame's schema is used as context. The name is expected
        to be in snake_case format.

        :param df: The pandas DataFrame to name.
        :type df: pd.DataFrame
        :return: A string containing the LLM-generated name in snake_case.
        :rtype: str
        """
        prompt = f"""
        Given the following DataFrame information, generate a concise, descriptive name for it:
        
        {df.super.get_schema()}
        
        Please provide a short, clear name (up to 3 words) that captures the essence of this dataset.
        The name should be in snake_case format (lowercase with underscores).
        Response should only contain the name, nothing else.
        """

        return self.query(prompt).content.strip()


class DummyLLMClient(LLMClient):
    """
    A dummy LLM client for testing and development purposes.

    This client does not make any actual API calls. Instead, it returns predefined
    dummy responses. It inherits from `LLMClient` but overrides its core methods.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the DummyLLMClient.

        Ignores all arguments and skips parent initialization to avoid actual model loading.
        """
        # Skip parent __init__ by calling object's __init__ or just pass
        # A bit of a trick to avoid LLMClient's __init__ logic effectively
        super(LLMClient, self).__init__()
        self.model = self  # Makes sure self.model is not None for internal checks

    def query(self, prompt: Union[str, LLMMessage, List[LLMMessage]], **kwargs) -> LLMResponse:
        """
        Return a dummy response acknowledging the prompt.

        :param prompt: The input prompt.
        :type prompt: Union[str, LLMMessage, List[LLMMessage]]
        :param kwargs: Additional keyword arguments (ignored).
        :return: A dummy `LLMResponse`.
        :rtype: LLMResponse
        """
        return LLMResponse(content=f"This is a dummy response for prompt: {prompt}")

    def generate_df_name(self, df: pd.DataFrame) -> str:
        """
        Return a dummy DataFrame name.

        :param df: The DataFrame (ignored).
        :type df: pd.DataFrame
        :return: The string "dummy_dataframe_name".
        :rtype: str
        """
        return "dummy_dataframe_name"

    def generate_df_description(self, df: pd.DataFrame) -> str:
        """
        Return a dummy DataFrame description.

        :param df: The DataFrame (ignored).
        :type df: pd.DataFrame
        :return: A fixed dummy description string.
        :rtype: str
        """
        return "This is a dummy response for DataFrame description."

    def generate_column_descriptions(self, df: pd.DataFrame) -> dict:
        """
        Return dummy descriptions for each column in the DataFrame.

        :param df: The DataFrame.
        :type df: pd.DataFrame
        :return: A dictionary mapping column names to a fixed dummy description.
        :rtype: dict
        """
        return {col: "This is a dummy response for column description." for col in df.columns}

    def __call__(self, payload, **kwargs):
        """
        Mock the model's `__call__` method to return a dummy response.
        Needed because `self.model` is set to `self`.

        :param payload: The payload (ignored).
        :param kwargs: Additional keyword arguments (ignored).
        :return: A dummy `LLMResponse`.
        :rtype: LLMResponse
        """
        return LLMResponse(content="This is a dummy response")
