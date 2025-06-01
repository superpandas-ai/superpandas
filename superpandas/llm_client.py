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
    role: Literal['system', 'user', 'assistant']
    content: Union[str, List[Dict[str, str]]]

class LLMResponse(BaseModel):
    content: Union[str, List[Dict[str, str]]]

class LLMClient:
    """
    Base class for LLM clients that can be used with pandas DataFrames via the .super accessor.
    
    This is a simple interface that can be extended to work with different LLM providers.
    """
    
    # Default to using HfApiModel with Llama 3.2
    try:
        DEFAULT_LLM = providers.get('HfApiModel', lambda x: None)("meta-llama/Llama-3.2-3B-Instruct") # Qwen/Qwen2.5-Coder-32B-Instruct
    except Exception as e:
        print(f"Error initializing default LLM: {e}")
        DEFAULT_LLM = None
    
    @staticmethod
    def available_providers() -> Dict[str, Type[Model]]:
        """
        Get a dictionary of all available LLM providers.
        
        Returns:
            Dict[str, Type[Model]]: Dictionary mapping provider names to their corresponding classes
        
        Example:
            >>> providers = LLMClient.available_providers()
            >>> print(providers.keys())  # Shows all available provider names
            >>> my_client = LLMClient(provider_class=providers['OpenAIServerModel'])
        """
        return providers

    def __init__(self, 
                 model: Optional[Union[str, Model]] = None,
                 provider: Optional[str] = None,
                 config: Optional[SuperPandasConfig] = SuperPandasConfig(),
                 **model_kwargs):
        """
        Initialize with specified LLM provider.
        
        Args:
            model: Model name or instance of Model class
            provider: Class to use for model provider (LiteLLMModel, OpenAIServerModel, HfApiModel, etc.)
            config: SuperPandasConfig instance. If None, uses the default SuperPandasConfig.
            **model_kwargs: Additional arguments to pass to the model provider
        """

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
                print(f"LLM provider {provider} not available. Please install smolagents with the provider. Using default provider: {self.DEFAULT_LLM.model_id}")
                self.model = self.DEFAULT_LLM
                return
            
            try:
                self.model = provider_class(model_id=model, **model_kwargs)
            except Exception as e:
                print(f"Error {e} initializing model: {model} with provider {provider}. Using default provider: {self.DEFAULT_LLM.model_id}")
                self.model = self.DEFAULT_LLM

    def query(self, 
              prompt: Union[str, LLMMessage, List[LLMMessage]],
              **kwargs) -> LLMResponse:
        """
        Send a query to the LLM and return the response.
        
        Parameters:
        -----------
        prompt : str | LLMMessage | list[LLMMessage]
            The prompt to send to the model. If a string, it will be converted to a LLMMessage with role 'user'. If a LLMMessage, it will be sent as is. If a list of LLMMessage, it will be sent as a list of messages.
        **kwargs : dict
            Additional arguments to pass to the model
        
        Returns:
        --------
        LLMResponse
            The model's response
        
        Raises:
        -------
        RuntimeError
            If no LLM provider is available
        """
        if not self.model:
            raise RuntimeError("No LLM provider available")

        if isinstance(prompt, str):
            messages = [LLMMessage(role='user', content=[{"type": "text", "text": prompt}])]
        elif isinstance(prompt, LLMMessage):
            messages = [prompt]
        elif isinstance(prompt, list):
            messages = prompt

        payload = [message.model_dump() for message in messages]
        
        response = self.model(payload, **kwargs)
        return LLMResponse(content=response.content)
    
    def generate_df_description(self, df: pd.DataFrame) -> str:
        """
        Generate a description for the entire DataFrame based on its contents.
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
        Generate descriptions for each column in the DataFrame.
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
        Generate a concise name for the DataFrame based on its contents.
        
        Args:
            df: DataFrame to name
            
        Returns:
            str: Generated name for the DataFrame
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
    """A dummy LLM client for testing purposes"""
    
    def __init__(self, *args, **kwargs):
        """Initialize the dummy client"""
        # Skip parent initialization
        self.model = self  # Set self as the model to prevent None
    
    def query(self, prompt: Union[str, LLMMessage, List[LLMMessage]], **kwargs) -> LLMResponse:
        """Return a simple acknowledgment of the prompt"""
        return LLMResponse(content=f"This is a dummy response for prompt: {prompt}")
    
    def generate_df_name(self, df: pd.DataFrame) -> str:
        """Generate a dummy DataFrame name"""
        return "dummy_dataframe_name"
    
    def generate_df_description(self, df: pd.DataFrame) -> str:
        """Generate a dummy DataFrame description"""
        return "This is a dummy response for DataFrame description."
    
    def generate_column_descriptions(self, df: pd.DataFrame) -> dict:
        """Generate dummy column descriptions"""
        return {col: "This is a dummy response for column description." for col in df.columns}
    
    def __call__(self, payload, **kwargs):
        """Mock the model's __call__ method"""
        return LLMResponse(content="This is a dummy response") 