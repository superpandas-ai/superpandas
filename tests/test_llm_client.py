import pytest
from superpandas import LLMClient, SuperPandasConfig, create_super_dataframe
from superpandas.llm_client import LLMMessage, LLMResponse
import pandas as pd
from unittest.mock import Mock, patch
from pydantic import ValidationError
from smolagents import Model

class MockModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_value = LLMResponse(content="test response")

    def __call__(self, *args, **kwargs):
        # Check if this is a column description request by looking at the prompt
        if args and isinstance(args[0], list) and len(args[0]) > 0:
            messages = args[0]
            if isinstance(messages, list) and len(messages) > 0:
                message_content = messages[0].content
                if isinstance(message_content, list) and len(message_content) > 0:
                    text = message_content[0].get('text', '')
                    # Only return dictionary for column description requests
                    if 'column' in text.lower() and 'description' in text.lower() and 'format your response as a python dictionary' in text.lower():
                        return LLMResponse(content='{"A": "desc1", "B": "desc2", "C": "desc3"}')
        # Return the default response for other operations (name, description)
        return self.return_value

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

    def test_llm_client_initialization(self):
        """Test LLMClient initialization"""
        # Test with default values
        client = LLMClient()
        assert client.model is not None
        
        # Test with custom config
        config = SuperPandasConfig()
        client = LLMClient(config=config)
        assert client.model is not None
        
        # Test with existing model instance
        mock_model = MockModel()
        client = LLMClient(model=mock_model)
        assert client.model == mock_model
        
        # Test with model string and provider - use mock to avoid dependency issues
        with patch('superpandas.llm_client.available_providers', {'test_provider': MockModel}):
            client = LLMClient(model="test-model", provider="test_provider")
            assert client.model is not None

    def test_llm_client_query(self):
        """Test LLMClient query method"""
        # Create a mock model that returns a predictable response
        mock_model = MockModel()
        client = LLMClient(model=mock_model)
        
        # Test with string
        response = client.query("test message")
        assert isinstance(response, LLMResponse)
        assert response.content == "test response"
        
        # Test with LLMMessage
        message = LLMMessage(role='user', content="test message")
        response = client.query(message)
        assert isinstance(response, LLMResponse)
        
        # Test with list of messages
        messages = [LLMMessage(role='user', content="test message")]
        response = client.query(messages)
        assert isinstance(response, LLMResponse)
        
        # Test error handling
        client.model = None
        with pytest.raises(RuntimeError, match="No LLM provider available"):
            client.query("test message")

    def test_llm_client_description_methods(self, sample_df):
        """Test LLMClient description generation methods"""
        # Create a mock model that returns predictable responses
        mock_model = MockModel()
        client = LLMClient(model=mock_model)
        
        # Test generate_df_description
        description = client.generate_df_description(sample_df)
        assert isinstance(description, str)
        assert description == "test response"
        
        # Test generate_column_descriptions
        mock_model.return_value = LLMResponse(content='{"A": "desc1", "B": "desc2", "C": "desc3"}')
        col_descriptions = client.generate_column_descriptions(sample_df)
        assert isinstance(col_descriptions, dict)
        assert all(col in col_descriptions for col in sample_df.columns)
        assert all(isinstance(desc, str) for desc in col_descriptions.values())
        
        # Test that descriptions can be set via property
        df = create_super_dataframe(sample_df)
        df.super.set_column_descriptions(col_descriptions)
        assert df.super.column_descriptions == col_descriptions
        assert df.super.column_descriptions is not col_descriptions  # Should be a copy
        
        # Test generate_df_name
        mock_model.return_value = LLMResponse(content="test_name")
        name = client.generate_df_name(sample_df)
        assert isinstance(name, str)
        assert name == "test_name"

    def test_available_providers(self):
        """Test getting available LLM providers"""
        providers = LLMClient.available_providers()
        assert isinstance(providers, list)
        # Check that providers list is not empty
        assert len(providers) > 0

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing"""
        return pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c'],
            'C': [1.1, 2.2, 3.3]
        }) 