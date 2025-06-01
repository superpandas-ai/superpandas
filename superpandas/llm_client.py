import pdb
from typing import Dict, Optional, Union, List, Type
import pandas as pd
from smolagents import Model
from .config import config

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

class LLMClient:
    """
    Base class for LLM clients that can be used with pandas DataFrames via the .super accessor.
    
    This is a simple interface that can be extended to work with different LLM providers.
    """
    
    # Default to using HfApiModel with Llama 3.2
    DEFAULT_LLM = providers.get('HfApiModel', lambda x: None)("meta-llama/Llama-3.2-3B-Instruct") # Qwen/Qwen2.5-Coder-32B-Instruct
    
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
                 provider_class: Optional[Type[Model] | str] = None,
                 **model_kwargs):
        """
        Initialize with specified LLM provider.
        
        Args:
            model: Model name or instance of Model class
            provider_class: Class to use for model provider (LiteLLMModel, OpenAIServerModel, HfApiModel, etc.)
            **model_kwargs: Additional arguments to pass to the model provider
        """
        if isinstance(model, (Model, DummyLLMClient)):
            self.model = model
        elif isinstance(model, str) or model is None:
            # Use provided values or fall back to config values
            model = model or config.llm_model
            provider_class = provider_class or config.llm_provider
            model_kwargs = {**config.llm_kwargs, **model_kwargs}

            # Remove existing_values from model_kwargs which are not required for the LLM client
            model_kwargs.pop('existing_values', None)
            
            if model is None and provider_class is None:
                provider_class = providers.get('HfApiModel')

            if isinstance(provider_class, str):
                provider_class = providers.get(provider_class)
            
            if not provider_class:
                print("No LLM provider available. Please install smolagents with the desired provider.")
                self.model = None
                return
            
            try:
                self.model = provider_class(model_id=model, **model_kwargs)
            except Exception as e:
                print(f"Error {e} initializing model: {model} with provider {provider_class}")
                if self.DEFAULT_LLM:
                    print(f"Using default provider: {self.DEFAULT_LLM.model_id}")
                    self.model = self.DEFAULT_LLM
                else:
                    print("No default provider available. Please install smolagents with the desired provider.")
                    self.model = None

    
    def query(self, user_message: str, 
              system_message: Optional[str] = None, 
              **kwargs) -> str:
        """
        Send a query to the LLM and return the response.
        
        Parameters:
        -----------
        user_message : str
            The user message to send to the model
        system_message : str, optional
            The system message to use for the model, by default None
        **kwargs : dict
            Additional arguments to pass to the model
        
        Returns:
        --------
        str
            The model's response
        
        Raises:
        -------
        RuntimeError
            If no LLM provider is available
        """
        if not self.model:
            raise RuntimeError("No LLM provider available")
            
        if system_message:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        else:
            messages = [
                {"role": "user", "content": [{"type": "text", "text": user_message}]}
            ]
        
        response = self.model(messages, **kwargs)
        return response.content
    
    def generate_df_description(self, df: pd.DataFrame) -> str:
        """
        Generate a description for the entire DataFrame based on its contents.
        """
        prompt = f"""
        Given the following DataFrame information, provide a concise description of its contents and purpose:
        
        {df.super.schema()}
        
        Please provide a clear, concise description of what this DataFrame represents.
        Response should be a single paragraph, no more than 2-3 sentences.
        """
        
        return self.query(prompt)
    
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
        """
        
        response = self.query(prompt)
        try:
            # Safely evaluate the response string as a Python dictionary
            descriptions = eval(response)
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
        
        {df.super.schema()}
        
        Please provide a short, clear name (1-5 words) that captures the essence of this dataset.
        The name should be in snake_case format (lowercase with underscores).
        Response should only contain the name, nothing else.
        """
        
        return self.query(prompt).strip()

class DummyLLMClient(LLMClient):
    """A dummy LLM client for testing purposes"""
    
    def __init__(self, *args, **kwargs):
        """Initialize the dummy client"""
        # Skip parent initialization
        self.model = self  # Set self as the model to prevent None
    
    def query(self, user_message: str, system_message: Optional[str] = None, **kwargs) -> str:
        """Return a simple acknowledgment of the prompt"""
        return f"Received user message of length {len(user_message)}. This is a dummy response."
    
    def generate_df_name(self, df: pd.DataFrame) -> str:
        """Generate a dummy DataFrame name"""
        return "dummy_dataframe_name"
    
    def generate_df_description(self, df: pd.DataFrame) -> str:
        """Generate a dummy DataFrame description"""
        return "This is a dummy response for DataFrame description."
    
    def generate_column_descriptions(self, df: pd.DataFrame) -> dict:
        """Generate dummy column descriptions"""
        return {col: "This is a dummy response for column description." for col in df.columns}
    
    def __call__(self, messages, **kwargs):
        """Mock the model's __call__ method"""
        class DummyResponse:
            content = "This is a dummy response"
        return DummyResponse() 