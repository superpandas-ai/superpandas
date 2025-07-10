import pytest
from superpandas import LLMClient, create_super_dataframe, SuperPandasConfig
from superpandas.llm_client import LLMResponse, LLMMessage
from unittest.mock import Mock, patch
from smolagents import Model
import pandas as pd
from test_llm_client import MockModel


class TestLLMIntegration:
    """Test LLM integration functionality"""
    
    def test_auto_describe_functionality(self, sample_df):
        """Test auto_describe functionality using mock model"""
        # Test with pandas DataFrame input
        df = create_super_dataframe(sample_df)
        
        # Create test config with mock model
        test_config = SuperPandasConfig.get_default_config()
        mock_model = MockModel()
        test_config.model = mock_model
        df.super.config = test_config
        # Create a new LLMClient with the mock model
        df.super.llm_client = LLMClient(model=mock_model)
        # Test auto_describe
        df.super.auto_describe(
            generate_name=True,
            generate_description=True,
            generate_column_descriptions=True
        )
        assert df.super.name != ''
        assert df.super.description != ''
        assert len(df.super.column_descriptions) > 0
        
        # Test with existing metadata
        df2 = create_super_dataframe(
            sample_df,
            name='test',
            description='',
            column_descriptions={}
        )
        df2.super.config = test_config
        df2.super.llm_client = LLMClient(model=mock_model)
        # Test auto_describe
        result = df2.super.auto_describe(
            generate_name=True,
            generate_description=True,
            generate_column_descriptions=True
        )
        assert result.super.name != ''
        assert result.super.description != ''
        assert len(result.super.column_descriptions) > 0
    
    def test_llm_client_error_handling(self):
        """Test error handling in LLMClient"""
        mock_model = MockModel()
        client = LLMClient(model=mock_model)
        # Test query with no model
        client.model = None
        with pytest.raises(RuntimeError, match="No LLM provider available"):
            client.query("test prompt")
    
    def test_llm_client_methods(self, sample_df):
        """Test LLMClient methods with mock model"""
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
        assert len(col_descriptions) > 0
        for col in sample_df.columns:
            assert col in col_descriptions
            assert isinstance(col_descriptions[col], str)

    def test_query_with_different_inputs(self):
        """Test query method with different input types"""
        mock_model = MockModel()
        client = LLMClient(model=mock_model)
        # Test with simple string
        response = client.query("Simple query")
        assert isinstance(response, LLMResponse)
        assert response.content == "test response"
        # Test with multi-line string
        response = client.query("""
        Multi-line
        query
        text
        """)
        assert isinstance(response, LLMResponse)
        # Test with empty string
        response = client.query("")
        assert isinstance(response, LLMResponse)
        # Test with LLMMessage
        message = LLMMessage(role='user', content="test message")
        response = client.query(message)
        assert isinstance(response, LLMResponse)
        # Test with list of messages
        messages = [
            LLMMessage(role='system', content="system message"),
            LLMMessage(role='user', content="user message")
        ]
        response = client.query(messages)
        assert isinstance(response, LLMResponse)

    def test_generate_df_name_scenarios(self, sample_df):
        """Test generate_df_name with different scenarios"""
        mock_model = MockModel()
        mock_model.return_value = LLMResponse(content="test_name")
        client = LLMClient(model=mock_model)
        # Test with empty dataframe
        empty_df = pd.DataFrame()
        name = client.generate_df_name(empty_df)
        assert isinstance(name, str)
        assert name == "test_name"
        # Test with single column dataframe
        single_col_df = pd.DataFrame({'values': [1, 2, 3]})
        name = client.generate_df_name(single_col_df)
        assert isinstance(name, str)
        assert name == "test_name"
        # Test with dataframe that already has a name
        name = client.generate_df_name(sample_df)
        assert isinstance(name, str)
        assert name == "test_name"

    def test_llm_client_with_kwargs(self):
        """Test LLMClient initialization with various kwargs"""
        # Test with temperature
        client = LLMClient(temperature=0.7)
        assert hasattr(client, 'model')
        # Test with max_tokens
        client = LLMClient(max_tokens=100)
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
        # Test with invalid provider
        with pytest.raises(RuntimeError, match="LLM provider invalid_provider not available"):
            LLMClient(model="test-model", provider="invalid_provider")
        
        # Test with valid provider but invalid model - mock the provider to raise an error
        from unittest.mock import Mock
        mock_provider_class = Mock()
        mock_provider_class.side_effect = Exception("Invalid model")
        with patch('superpandas.llm_client.available_providers', {'hf': mock_provider_class}):
            with pytest.raises(RuntimeError, match="Error initializing model"):
                LLMClient(model="invalid_model", provider="hf")

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing"""
        return pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c'],
            'C': [1.1, 2.2, 3.3]
        }) 