from typing import Dict, Optional, Union, List, Type
import pandas as pd
from smolagents import Model
from .superdataframe import SuperDataFrame
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
    from huggingface_hub.errors import HfHubHTTPError
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
    Base class for LLM clients that can be used with SuperDataFrame.
    
    This is a simple interface that can be extended to work with different LLM providers.
    """
    
    # Default to using HfApiModel with Llama 3.2
    DEFAULT_LLM = providers.get('HfApiModel', lambda x: None)("meta-llama/Llama-3.2-3B-Instruct")
    
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
                 provider_class: Optional[Type[Model]] = None,
                 **model_kwargs):
        """
        Initialize with specified LLM provider.
        
        Args:
            model: Model name or instance of Model class
            provider_class: Class to use for model provider (LiteLLMModel, OpenAIServerModel, HfApiModel, etc.)
            **model_kwargs: Additional arguments to pass to the model provider
        """
        if isinstance(model, Model):
            self.model = model
        elif isinstance(model, str):
            provider_class = provider_class or providers.get('HfApiModel')
            if not provider_class:
                print("No LLM provider available. Please install smolagents with the desired provider.")
                self.model = None
                return
                
            try:
                self.model = provider_class(model, **model_kwargs)
                # Validate HfApiModel endpoint exists
                # if provider_class == HfApiModel:
                #     try:
                #         self.model.client.get_endpoint_info()
                #     except HfHubHTTPError as e:
                #         raise ValueError(f"Model {model} not found on Hugging Face Hub: {str(e)}")
            except Exception as e:
                print(f"Error {e} initializing model: {model} with provider {provider_class}")
                if self.DEFAULT_LLM:
                    print(f"Using default provider: {self.DEFAULT_LLM}")
                    self.model = self.DEFAULT_LLM
                else:
                    print("No default provider available. Please install smolagents with the desired provider.")
                    self.model = None
        else:
            if self.DEFAULT_LLM:
                print(f"Using default provider: {str(self.DEFAULT_LLM)}")
                self.model = self.DEFAULT_LLM
            else:
                print("No default provider available. Please install smolagents with the desired provider.")
                self.model = None
    
    def query(self, prompt: str, **kwargs) -> str:
        """
        Send a query to the LLM and return the response.
        
        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional arguments to pass to the model
        
        Returns:
            str: The model's response
        
        Raises:
            RuntimeError: If no LLM provider is available
        """
        if not self.model:
            raise RuntimeError("No LLM provider available")
            
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        response = self.model(messages, **kwargs)
        return response.content
    
    def analyze_dataframe(self, sdf: SuperDataFrame, query: str, **kwargs) -> str:
        """
        Analyze a dataframe using the LLM.
        
        Args:
            sdf: SuperDataFrame to analyze
            query: The analysis query/question about the dataframe
            **kwargs: Additional arguments to pass to the model
            
        Returns:
            str: The model's analysis response
            
        Example:
            >>> df = SuperDataFrame(...)
            >>> client = LLMClient(...)
            >>> result = client.analyze_dataframe(df, "What are the key trends in this data?")
        """
        prompt = f"""
Please analyze the following dataframe:

{sdf.schema()}

Analysis request: {query}

Provide a clear and concise response based on the data shown above.
"""
        return self.query(prompt, **kwargs)
    
    def generate_df_description(self, sdf: SuperDataFrame) -> str:
        """
        Generate a description for the entire DataFrame based on its contents.
        """
        prompt = f"""
        Given the following DataFrame information, provide a concise description of its contents and purpose:
        
        {sdf.schema()}
        
        Please provide a clear, concise description of what this DataFrame represents.
        Response should be a single paragraph, no more than 2-3 sentences.
        """
        
        return self.query(prompt)
    
    def generate_column_descriptions(self, sdf: SuperDataFrame) -> Dict[str, str]:
        """
        Generate descriptions for each column in the DataFrame.
        """
        prompt = f"""
        Given the following DataFrame information, provide concise descriptions for each column:
        
        Column Types:
        {sdf.column_types}
        
        Sample Data:
        {sdf.head(3).to_string()}
        
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
            return {col: "No description available" for col in sdf.columns}

def auto_describe_dataframe(
    df: pd.DataFrame,
    model: Optional[Union[str, Model]] = None,
    provider_class: Optional[Type[Model]] = None,
    generate_df_description: bool = True,
    generate_column_descriptions: bool = True,
    _is_auto_describing: bool = False,
    **model_kwargs
) -> SuperDataFrame:
    """
    Automatically generate descriptions for a DataFrame using LLMs.
    
    Args:
        df: Input DataFrame
        model: Model name or instance of Model class
        provider_class: Class to use for model provider (LiteLLMModel, OpenAIServerModel, HfApiModel, etc.)
        generate_df_description: Whether to generate overall DataFrame description
        generate_column_descriptions: Whether to generate column descriptions
        _is_auto_describing: Internal flag to indicate auto-describing process
        **model_kwargs: Additional arguments to pass to the model provider
    
    Returns:
        DataFrame with generated descriptions
    """
    llm_client = LLMClient(model=model, provider_class=provider_class, **model_kwargs)
    
    df_description = ""
    column_descriptions = {}
    
    # Convert to SuperDataFrame first if needed
    if not isinstance(df, SuperDataFrame):
        sdf = SuperDataFrame.from_pandas(df, _is_auto_describing=True)
    else:
        sdf = df
    
    if generate_df_description:
        df_description = llm_client.generate_df_description(sdf)
        sdf.description = df_description
    
    if generate_column_descriptions:
        column_descriptions = llm_client.generate_column_descriptions(sdf)
        sdf.set_column_descriptions(column_descriptions)
    
    return sdf


class DummyLLMClient(LLMClient):
    """A dummy LLM client for testing purposes"""
    
    def query(self, prompt: str, **kwargs) -> str:
        """Return a simple acknowledgment of the prompt"""
        return f"Received prompt of length {len(prompt)}. This is a dummy response." 