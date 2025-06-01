import pytest
from superpandas import LLMClient, SuperPandasConfig
from superpandas.llm_client import LLMMessage, LLMResponse, DummyLLMClient
import pandas as pd
from unittest.mock import Mock, patch
from pydantic import ValidationError

class TestLLMClient:
    """Test LLM client functionality"""
    
    def test_llm_message_model(self):
        """Test LLMMessage Pydantic model"""
        # Test with text content
        message = LLMMessage(role='user', content="test message")
        assert message.role == 'user'
        assert message.content == "test message"
        
        # Test with list content
        content = [{"type": "text", "text": "test message"}]
        message = LLMMessage(role='user', content=content)
        assert message.role == 'user'
        assert message.content == content
        
        # Test validation
        with pytest.raises(ValidationError):
            LLMMessage(role='invalid_role', content="test message")
        
        with pytest.raises(ValidationError):
            LLMMessage(role='user', content=123)  # Invalid content type

    def test_llm_response_model(self):
        """Test LLMResponse Pydantic model"""
        # Test with text content
        response = LLMResponse(content="test response")
        assert response.content == "test response"
        
        # Test with list content
        content = [{"type": "text", "text": "test response"}]
        response = LLMResponse(content=content)
        assert response.content == content
        
        # Test validation
        with pytest.raises(ValidationError):
            LLMResponse(content=123)  # Invalid content type

    def test_dummy_llm_client(self):
        """Test the DummyLLMClient"""
        client = DummyLLMClient()
        
        # Test query with string
        response = client.query("test message")
        assert isinstance(response, LLMResponse)
        assert "This is a dummy response" in response.content
        
        # Test query with LLMMessage
        message = LLMMessage(role='user', content="test message")
        response = client.query(message)
        assert isinstance(response, LLMResponse)
        
        # Test query with list of messages
        messages = [LLMMessage(role='user', content="test message")]
        response = client.query(messages)
        assert isinstance(response, LLMResponse)

    def test_llm_client_initialization(self):
        """Test LLMClient initialization"""
        # Test with default values
        client = LLMClient()
        assert client.model is not None
        
        # Test with custom config
        config = SuperPandasConfig()
        client = LLMClient(config=config)
        assert client.model is not None
        
        # Test with model string and provider
        client = LLMClient(model="test-model", provider="HfApiModel")
        assert client.model is not None
        
        # Test with existing model instance
        dummy_model = DummyLLMClient()
        client = LLMClient(model=dummy_model)
        assert client.model == dummy_model

    def test_llm_client_query(self):
        """Test LLMClient query method"""
        client = DummyLLMClient()
        
        # Test with string
        response = client.query("test message")
        assert isinstance(response, LLMResponse)
        assert response.content == "This is a dummy response for prompt: test message"
        
        # Test with LLMMessage
        message = LLMMessage(role='user', content="test message")
        response = client.query(message)
        assert isinstance(response, LLMResponse)
        
        # Test with list of messages
        messages = [LLMMessage(role='user', content="test message")]
        response = client.query(messages)
        assert isinstance(response, LLMResponse)
        
        # TODO: Test error handling
        # client.model = None
        # with pytest.raises(RuntimeError, match="No LLM provider available"):
        #     client.query("test message")

    def test_llm_client_description_methods(self, sample_df):
        """Test LLMClient description generation methods"""
        client = DummyLLMClient()
        
        # Test generate_df_description
        description = client.generate_df_description(sample_df)
        assert isinstance(description, str)
        assert "This is a dummy response" in description
        
        # Test generate_column_descriptions
        col_descriptions = client.generate_column_descriptions(sample_df)
        assert isinstance(col_descriptions, dict)
        assert all(col in col_descriptions for col in sample_df.columns)
        assert all(isinstance(desc, str) for desc in col_descriptions.values())
        
        # Test generate_df_name
        name = client.generate_df_name(sample_df)
        assert isinstance(name, str)
        assert name == "dummy_dataframe_name"

    def test_available_providers(self):
        """Test getting available LLM providers"""
        providers = LLMClient.available_providers()
        assert isinstance(providers, dict)
        # Check that providers dictionary is not empty
        assert len(providers) > 0

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing"""
        return pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c'],
            'C': [1.1, 2.2, 3.3]
        }) 