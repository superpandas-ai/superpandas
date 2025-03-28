import pytest
from superpandas import (
    SuperDataFrame, auto_describe_dataframe, 
    LLMClient, DummyLLMClient
)

class TestLLMIntegration:
    """Test LLM integration functionality"""
    
    def test_dummy_llm_client(self, sample_df):
        """Test the DummyLLMClient"""
        client = DummyLLMClient()
        
        # Test query
        response = client.query("Describe this dataframe")
        assert isinstance(response, str)
        assert "This is a dummy response" in response
        
        # Test analyze_dataframe
        sdf = SuperDataFrame(sample_df)
        response = client.analyze_dataframe(sdf, "What are the trends?")
        assert isinstance(response, str)
        assert "This is a dummy response" in response
    
    def test_dummy_llm_client_descriptions(self, sample_df):
        """Test DummyLLMClient description generation"""
        client = DummyLLMClient()
        sdf = SuperDataFrame(sample_df)
        
        # Test DataFrame description
        df_description = client.generate_df_description(sdf)
        assert isinstance(df_description, str)
        assert "This is a dummy response" in df_description
        
        # Test column descriptions
        col_descriptions = client.generate_column_descriptions(sdf)
        assert isinstance(col_descriptions, dict)
        assert all(col in col_descriptions for col in sdf.columns)
        assert all(isinstance(desc, str) for desc in col_descriptions.values())
    
    def test_llm_client_initialization(self):
        """Test LLMClient initialization with different inputs"""
        # Test with no model (should use default provider)
        client = LLMClient()
        assert hasattr(client, 'model')
        
        # Test with invalid model string
        client = LLMClient("invalid_model")
        assert client.model is not None  # Should fall back to default provider
        
        # Test with invalid provider class
        client = LLMClient("some_model", provider_class=None)
        assert client.model is not None  # Should fall back to default provider
    
    def test_auto_describe_dataframe(self, sample_df):
        """Test auto_describe_dataframe function"""
        # Test with pandas DataFrame input
        sdf = auto_describe_dataframe(
            sample_df,
            model=None,  # Will use default provider
            generate_df_description=True,
            generate_column_descriptions=True
        )
        
        assert isinstance(sdf, SuperDataFrame)
        assert hasattr(sdf, 'description')
        assert hasattr(sdf, 'column_descriptions')
        
        # Test with existing SuperDataFrame
        sdf2 = SuperDataFrame(sample_df)
        result = auto_describe_dataframe(
            sdf2,
            model=None,
            generate_df_description=True,
            generate_column_descriptions=True
        )
        
        assert isinstance(result, SuperDataFrame)
        assert result.description != ""
        assert len(result.column_descriptions) > 0
    
    def test_llm_client_error_handling(self):
        """Test error handling in LLMClient"""
        client = LLMClient(model=None)  # Use default provider
        
        # Test query with no model
        client.model = None
        with pytest.raises(RuntimeError, match="No LLM provider available"):
            client.query("test prompt")
        
        # Test analyze_dataframe with no model
        sdf = SuperDataFrame({'a': [1, 2, 3]})
        with pytest.raises(RuntimeError, match="No LLM provider available"):
            client.analyze_dataframe(sdf, "test query")
    
    def test_llm_client_methods(self, sample_df):
        """Test LLMClient methods with dummy client"""
        client = DummyLLMClient()
        sdf = SuperDataFrame(sample_df)
        
        # Test generate_df_description
        description = client.generate_df_description(sdf)
        assert isinstance(description, str)
        assert "This is a dummy response" in description
        
        # Test generate_column_descriptions
        col_descriptions = client.generate_column_descriptions(sdf)
        assert isinstance(col_descriptions, dict)
        assert all(isinstance(v, str) for v in col_descriptions.values())
        
        # Test analyze_dataframe
        analysis = client.analyze_dataframe(sdf, "Describe the trends")
        assert isinstance(analysis, str)
        assert "This is a dummy response" in analysis 