"""
Tests for the LangGraph agent functionality
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from superpandas.langgraph_agent import LangGraphAgent, AgentState
from superpandas.config import SuperPandasConfig


class TestLangGraphAgent:
    """Test the LangGraph agent functionality"""
    
    def test_create_langgraph_agent(self):
        """Test that the agent can be created successfully"""
        config = SuperPandasConfig()
        agent = LangGraphAgent(config=config)
        
        # Check that the agent has the expected attributes
        assert hasattr(agent, 'graph')
        assert hasattr(agent, 'run')
        assert callable(agent.run)
    
    def test_agent_state_structure(self):
        """Test that AgentState has the correct structure"""
        state = AgentState(
            messages=[],
            current_query="test query",
            dataframe=pd.DataFrame({'A': [1, 2, 3]}),
            generated_code="",
            result=None,
            error="",
            iterations=0,
            formatted_response="",
            fig=None
        )
        
        assert isinstance(state, dict)
        assert "messages" in state
        assert "current_query" in state
        assert "dataframe" in state
        assert "generated_code" in state
        assert "result" in state
        assert "error" in state
        assert "iterations" in state
        assert "formatted_response" in state
        assert "fig" in state
    
    @patch('superpandas.langgraph_agent.LLMClient')
    def test_run_agent_basic(self, mock_llm_client):
        """Test running the agent with a simple query"""
        # Mock the LLM client
        mock_client = Mock()
        mock_client.query.return_value = Mock(content="result = df.head()")
        mock_llm_client.return_value = mock_client
        
        # Create a simple DataFrame
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        
        # Create agent and run
        agent = LangGraphAgent(max_iterations=1)
        result = agent.run(
            query="Show the first few rows",
            dataframe=df
        )
        
        # Check that the result contains the expected keys
        assert isinstance(result, dict)
        assert "result" in result
        assert "error" in result
        assert "formatted_response" in result
    
    def test_code_blob_output_parser(self):
        """Test the CodeBlobOutputParser"""
        from superpandas.utils import CodeBlobOutputParser
        
        parser = CodeBlobOutputParser()
        
        # Test with code blocks
        text_with_blocks = "Here's the code:\n```python\nresult = df.head()\n```"
        result = parser.parse(text_with_blocks)
        assert result == "result = df.head()"
        
        # Test without code blocks
        text_without_blocks = "result = df.head()"
        result = parser.parse(text_without_blocks)
        assert result == "result = df.head()"
    
    def test_superdataframe_agent_integration(self):
        """Test that the agent works with SuperDataFrame accessor"""
        # Create a DataFrame with super accessor
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        df.super.name = "Test DataFrame"
        df.super.description = "A test DataFrame"
        
        # Test that the analyze_with_agent method exists
        assert hasattr(df.super, 'analyze_with_agent')
        assert callable(df.super.analyze_with_agent)
    
    def test_agent_with_error_handling(self):
        """Test that the agent handles errors gracefully"""
        # Create a DataFrame that might cause issues
        df = pd.DataFrame({'A': [1, 2, 3]})
        
        # This should not raise an exception even if the LLM is not available
        try:
            agent = LangGraphAgent(max_iterations=1)
            result = agent.run(
                query="Invalid query that might fail",
                dataframe=df
            )
            assert isinstance(result, dict)
        except Exception as e:
            # If there's an error, it should be related to LLM configuration
            assert "LLM" in str(e) or "model" in str(e).lower() 