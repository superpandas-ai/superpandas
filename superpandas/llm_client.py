from textwrap import dedent
from typing import Dict, Literal, Optional, Union, List
import pandas as pd
from smolagents import Model
from pydantic import BaseModel
from .config import SuperPandasConfig
from .providers import available_providers


class LLMMessage(BaseModel):
    """
    Represents a single message in a conversation with an LLM.

    :param role: The role of the message sender ('system', 'user', or 'assistant').
    :type role: Literal['system', 'user', 'assistant']
    :param content: The content of the message. Can be a simple string or a list of content blocks
                    (e.g., for multimodal inputs, though SuperPandas primarily uses text).
    :type content: Union[str, List[Dict[str, str]]]. Can be a simple string or a list of content blocks (e.g., for multimodal inputs, though SuperPandas primarily uses text).
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
    :ivar provider: The name of the LLM provider.
    :ivar config: The `SuperPandasConfig` instance used for configuration.
    """

    @staticmethod
    def available_providers() -> List[str]:
        """
        Get a list of all available LLM providers from `smolagents` which are installed on your system.

        :return: A list of available provider names (e.g., 'openai')
        :rtype: List[str]

        Example::

            available_providers = LLMClient.available_providers()
            if 'openai' in available_providers:
                client = LLMClient(model="gpt-3.5-turbo", provider='openai')
            else:
                print("OpenAI provider not available.")
        """
        return list(available_providers.keys())

    def __init__(self,
                 model: Optional[Union[str, Model]] = None,
                 provider: Optional[str] = None,
                 config: Optional[SuperPandasConfig] = SuperPandasConfig.get_default_config(),
                 **model_kwargs):
        """
        Initialize the LLMClient.

        The client can be configured by specifying a model name and provider,
        or by passing a `SuperPandasConfig` object. If no parameters are provided,
        it attempts to use defaults from a loaded `SuperPandasConfig`.

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

        if model is None:
            model = config.model
        if provider is None:
            provider = config.provider

        if isinstance(model, Model):
            self.model = model
        elif isinstance(model, str):
            model_kwargs = {**config.llm_kwargs, **model_kwargs}
            provider_class = available_providers.get(provider)

            if not provider_class:
                raise RuntimeError(f"LLM provider {provider} not available. Available providers: {available_providers.keys()}. Please install smolagents with the provider or use one of the available providers.")

            try:
                self.model = provider_class(model_id=model, **model_kwargs)
            except Exception as e:
                raise RuntimeError(f"Error initializing model: {model} with provider {provider}: {e}")

    def query(self,
              messages: Union[str, LLMMessage, List[LLMMessage]],
              **kwargs) -> LLMResponse:
        """
        Send a query to the configured LLM and return its response.

        The prompt can be a simple string, a single `LLMMessage`, or a list of
        `LLMMessage` objects for multi-turn conversations.

        :param messages: The messages to send to the LLM.
        :type messages: Union[str, LLMMessage, List[LLMMessage]]
        :param kwargs: Additional keyword arguments to pass to the underlying
                       `smolagents` model's call method.
        :return: An `LLMResponse` object containing the model's output.
        :rtype: LLMResponse
        :raises RuntimeError: If no LLM provider is available or configured.
        """
        if not self.model:
            raise RuntimeError(
                "No LLM provider available or configured for the LLMClient.")

        if isinstance(messages, str):
            messages = [LLMMessage(role='user', content=[
                                   {"type": "text", "text": messages}])]
        elif isinstance(messages, LLMMessage):
            messages = [messages]
        elif isinstance(messages, list) and all(isinstance(m, LLMMessage) for m in messages):
            messages = messages
        else:
            raise ValueError(f"Invalid messages type: {type(messages)}. Messages must be a string, LLMMessage, or list of LLMMessage objects.")

        # payload = [message.model_dump() for message in messages]

        response = self.model(messages, **kwargs)
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
        prompt = dedent(f"""
        Given the following PandasDataFrame schema, provide a concise description of the contents and purpose of the DataFrame:
        
        {df.super.get_schema()}
        
        Please provide a clear, concise description of what this DataFrame represents.
        Response should be a single paragraph, no more than 2-3 sentences.
        """)

        return self.query(prompt).content

    def generate_column_descriptions(self, df: pd.DataFrame, num_tries: int = 3) -> Dict[str, str]:
        """
        Generate descriptions for each column in a DataFrame using the LLM.

        Column types and sample data from the DataFrame are used as context.
        The method attempts to parse the LLM's response as a Python dictionary.
        Will retry up to num_tries times if the response is not a valid dictionary.

        :param df: The pandas DataFrame whose columns are to be described.
        :type df: pd.DataFrame
        :param num_tries: Maximum number of attempts to get a valid dictionary response.
        :type num_tries: int
        :return: A dictionary mapping column names to their LLM-generated descriptions.
        :rtype: Dict[str, str]
        :raises ValueError: If all attempts fail to produce a valid dictionary response.
        """
        prompt = dedent(f"""
        Given the following Pandas DataFrame schema, provide concise descriptions for each column:
        
        Column Types:
        {df.super.column_types}
        
        Sample Data:
        {df.head(3).to_string()}
        
        Please provide short, clear descriptions for each column.
        Format your response as a Python dictionary with column names as keys and descriptions as values.
        Example format:
        {{"column_name": "Description of what this column represents"}}
        Response should be a Python dictionary, nothing else.
        """)

        for attempt in range(num_tries):
            response = self.query(prompt).content
            try:
                # Safely evaluate the response string as a Python dictionary
                descriptions = eval(response)
                if not isinstance(descriptions, dict):
                    if attempt < num_tries - 1:
                        prompt += "\n \nLast attempt did not return a dictionary. Please try again."
                        continue
                    raise ValueError("Response is not a dictionary")
                return descriptions
            except:
                if attempt < num_tries - 1:
                    continue
                raise ValueError(f"Failed to parse LLM response as a dictionary after {num_tries} attempts")

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
        prompt = dedent(f"""
        Given the following Pandas DataFrame information, generate a concise, descriptive name for it:
        
        {df.super.get_schema()}
        
        Please provide a short, clear name (up to 3 words) that captures the essence of this dataset.
        The name should be in snake_case format (lowercase with underscores).
        Response should only contain the name, nothing else.
        """)

        return self.query(prompt).content.strip()


# class DummyLLMClient(LLMClient):
#     """
#     A dummy LLM client for testing and development purposes.

#     This client does not make any actual API calls. Instead, it returns predefined
#     dummy responses. It inherits from `LLMClient` but overrides its core methods.
#     """

#     def __init__(self, *args, **kwargs):
#         """
#         Initialize the DummyLLMClient.

#         Ignores all arguments and skips parent initialization to avoid actual model loading.
#         """
#         # Skip parent __init__ by calling object's __init__ or just pass
#         # A bit of a trick to avoid LLMClient's __init__ logic effectively
#         super(LLMClient, self).__init__()
#         self.model = self  # Makes sure self.model is not None for internal checks

#     def query(self, prompt: Union[str, LLMMessage, List[LLMMessage]], **kwargs) -> LLMResponse:
#         """
#         Return a dummy response acknowledging the prompt.

#         :param prompt: The input prompt.
#         :type prompt: Union[str, LLMMessage, List[LLMMessage]]
#         :param kwargs: Additional keyword arguments (ignored).
#         :return: A dummy `LLMResponse`.
#         :rtype: LLMResponse
#         """
#         return LLMResponse(content=f"This is a dummy response for prompt: {prompt}")

#     def generate_df_name(self, df: pd.DataFrame) -> str:
#         """
#         Return a dummy DataFrame name.

#         :param df: The DataFrame (ignored).
#         :type df: pd.DataFrame
#         :return: The string "dummy_dataframe_name".
#         :rtype: str
#         """
#         return "dummy_dataframe_name"

#     def generate_df_description(self, df: pd.DataFrame) -> str:
#         """
#         Return a dummy DataFrame description.

#         :param df: The DataFrame (ignored).
#         :type df: pd.DataFrame
#         :return: A fixed dummy description string.
#         :rtype: str
#         """
#         return "This is a dummy response for DataFrame description."

#     def generate_column_descriptions(self, df: pd.DataFrame) -> dict:
#         """
#         Return dummy descriptions for each column in the DataFrame.

#         :param df: The DataFrame.
#         :type df: pd.DataFrame
#         :return: A dictionary mapping column names to a fixed dummy description.
#         :rtype: dict
#         """
#         return {col: "This is a dummy response for column description." for col in df.columns}

#     def __call__(self, payload, **kwargs):
#         """
#         Mock the model's `__call__` method to return a dummy response.
#         Needed because `self.model` is set to `self`.

#         :param payload: The payload (ignored).
#         :param kwargs: Additional keyword arguments (ignored).
#         :return: A dummy `LLMResponse`.
#         :rtype: LLMResponse
#         """
#         return LLMResponse(content="This is a dummy response")
