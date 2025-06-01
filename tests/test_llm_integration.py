import pytest
from superpandas import LLMClient, create_super_dataframe, SuperPandasConfig
from superpandas.llm_client import DummyLLMClient, LLMResponse
from unittest.mock import Mock, patch
from smolagents import Model
import pandas as pd

class TestLLMIntegration:
    """Test LLM integration functionality"""
    
    def test_auto_describe_functionality(self, sample_df):
        """Test auto_describe functionality using DummyLLMClient"""
        # Test with pandas DataFrame input
        df = create_super_dataframe(sample_df)
        
        # Create test config
        test_config = SuperPandasConfig()
        test_config.model = DummyLLMClient()
        
        # Test auto_describe
        df.super.auto_describe(
            config=test_config,
            generate_name=True,
            generate_description=True,
            generate_column_descriptions=True
        )
        
        assert df.super.name != ''
        assert df.super.description != ''
        assert len(df.super.get_column_descriptions()) > 0
        
        # Test with existing metadata
        df2 = create_super_dataframe(
            sample_df,
            name='test',
            description='',
            column_descriptions={}
        )
        
        # Test auto_describe
        result = df2.super.auto_describe(
            config=test_config,
            generate_name=True,
            generate_description=True,
            generate_column_descriptions=True
        )
        
        assert result.super.name != ''
        assert result.super.description != ''
        assert len(result.super.get_column_descriptions()) > 0
    
    def test_llm_client_error_handling(self):
        """Test error handling in LLMClient"""
        client = LLMClient(model=DummyLLMClient())  # Use DummyLLMClient as a mock model
        
        # Test query with no model
        client.model = None
        with pytest.raises(RuntimeError, match="No LLM provider available"):
            client.query(prompt="test prompt")
    
    def test_llm_client_methods(self, sample_df):
        """Test LLMClient methods with dummy client"""
        client = DummyLLMClient()
        
        # Test generate_df_description
        description = client.generate_df_description(sample_df)
        assert isinstance(description, str)
        assert "This is a dummy response" in description
        
        # Test generate_column_descriptions
        col_descriptions = client.generate_column_descriptions(sample_df)
        assert isinstance(col_descriptions, dict)
        assert len(col_descriptions) > 0
        for col in sample_df.columns:
            assert col in col_descriptions
            assert isinstance(col_descriptions[col], str)

    def test_available_providers(self):
        """Test getting available LLM providers"""
        providers = LLMClient.available_providers()
        assert isinstance(providers, dict)
        # Check some common providers that should be available
        expected_providers = ['LiteLLMModel', 'OpenAIServerModel', 'HfApiModel']
        for provider in expected_providers:
            if provider in providers:
                assert issubclass(providers[provider], Model)

    def test_query_with_different_inputs(self):
        """Test query method with different input types"""
        client = DummyLLMClient()

        # Test with simple string
        response = client.query(prompt="Simple query")
        assert isinstance(response, LLMResponse)

        # Test with multi-line string
        response = client.query(prompt="""
        Multi-line
        query
        text
        """)
        assert isinstance(response, LLMResponse)

        # Test with empty string
        response = client.query(prompt="")
        assert isinstance(response, LLMResponse)

    def test_generate_df_name_scenarios(self, sample_super_df):
        """Test generate_df_name with different scenarios"""
        client = DummyLLMClient()

        # Test with empty dataframe
        empty_df = pd.DataFrame()
        name = client.generate_df_name(empty_df)
        assert isinstance(name, str)

        # Test with single column dataframe
        single_col_df = pd.DataFrame({'values': [1, 2, 3]})
        name = client.generate_df_name(single_col_df)
        assert isinstance(name, str)

        # Test with dataframe that already has a name
        name = client.generate_df_name(sample_super_df)
        assert isinstance(name, str)

    def test_llm_client_with_kwargs(self):
        """Test LLMClient initialization with various kwargs"""
        # Test with temperature
        client = LLMClient(model=None, temperature=0.7)
        assert hasattr(client, 'model')

        # Test with max_tokens
        client = LLMClient(model=None, max_tokens=100)
        assert hasattr(client, 'model')

        # Test with multiple kwargs
        client = LLMClient(
            model=None,
            temperature=0.7,
            max_tokens=100,
            top_p=0.9,
            frequency_penalty=0.1
        )
        assert hasattr(client, 'model')

    def test_model_validation(self):
        """Test model validation during initialization"""
        # Test with invalid model string but valid provider
        mock_provider = Mock(spec=type)
        mock_provider.side_effect = ValueError("Invalid model")
        
        client = LLMClient(model="invalid_model", provider_class=mock_provider)
        assert client.model is not None  # Should fall back to default model

        # TODO: Test with both invalid model and provider
        # with patch.object(LLMClient, 'DEFAULT_LLM', None):
        #     client = LLMClient(model="invalid_model", provider_class=mock_provider)
        #     assert client.model is None 