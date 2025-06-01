import pytest
from superpandas import config, LLMClient
from smolagents import Model

class TestConfig:
    """Test configuration functionality"""
    
    def test_config_singleton(self):
        """Test that config is a singleton"""
        from superpandas.config import SuperPandasConfig
        
        config2 = SuperPandasConfig()
        assert config2 is config
        
    def test_llm_settings(self):
        """Test setting and getting LLM configuration"""
        # Test initial values
        assert config.llm_model == "meta-llama/Llama-3.2-3B-Instruct"
        assert config.llm_provider == "HfApiModel"
        assert 'existing_values' in config.llm_kwargs
        assert config.llm_kwargs['existing_values'] == 'warn'
        
        # Test setting values
        class DummyProvider(Model):
            def query(self, prompt): return "dummy"
        
        config.llm_model = "test-model"
        config.llm_provider = DummyProvider
        config.llm_kwargs.update({'temperature': 0.7})
        
        assert config.llm_model == "test-model"
        assert config.llm_provider == DummyProvider
        assert config.llm_kwargs['temperature'] == 0.7
        assert config.llm_kwargs['existing_values'] == 'warn'
        
    def test_configure_llm(self):
        """Test configure_llm method"""
        class DummyProvider(Model):
            def query(self, prompt): return "dummy"
            
        config.configure_llm(
            provider_class=DummyProvider,
            model="test-model",
            temperature=0.5,
            existing_values='skip'
        )
        
        assert config.llm_model == "test-model"
        assert config.llm_provider == DummyProvider
        assert config.llm_kwargs['temperature'] == 0.5
        assert config.llm_kwargs['existing_values'] == 'skip'
    
    def test_existing_values_validation(self):
        """Test validation of existing_values setting"""
        # Test invalid value in configure_llm
        with pytest.raises(ValueError):
            config.configure_llm(existing_values='invalid')
        
        # Test invalid value in llm_kwargs
        with pytest.raises(ValueError):
            config.llm_kwargs = {'existing_values': 'invalid'}
        
        # Test valid values
        for value in ['warn', 'skip', 'overwrite']:
            config.configure_llm(existing_values=value)
            assert config.llm_kwargs['existing_values'] == value 