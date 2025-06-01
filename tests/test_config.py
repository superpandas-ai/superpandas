import pytest
from smolagents import Model
import os
import json
from superpandas.config import SuperPandasConfig

class TestConfig:
    """Test configuration functionality"""
    
    def test_config_singleton(self):
        """Test that config instances are independent"""
        from superpandas.config import SuperPandasConfig
        
        config1 = SuperPandasConfig()
        config2 = SuperPandasConfig()
        assert config1 is not config2  # Instances should be different
        assert config1.model == config2.model  # But should have same default values
        config1.model = "new-model"
        assert config1.model != config2.model  # Changes to one shouldn't affect the other
        
    def test_llm_settings(self):
        """Test setting and getting LLM configuration"""
        config = SuperPandasConfig()
        
        # Test initial values
        assert config.model == "meta-llama/Llama-3.2-3B-Instruct"
        assert config.provider == "HfApiModel"
        assert 'existing_values' in config.llm_kwargs
        assert config.llm_kwargs['existing_values'] == 'warn'
        
        # Test setting values
        class DummyProvider(Model):
            def query(self, prompt): return "dummy"
        
        config.model = "test-model"
        config.provider = DummyProvider
        config.llm_kwargs.update({'temperature': 0.7})
        
        assert config.model == "test-model"
        assert config.provider == DummyProvider
        assert config.llm_kwargs['temperature'] == 0.7
        assert config.llm_kwargs['existing_values'] == 'warn'
    
    def test_existing_values_validation(self):
        """Test validation of existing_values setting"""
        config = SuperPandasConfig()
        
        # Test invalid value in llm_kwargs
        with pytest.raises(ValueError):
            config.llm_kwargs = {'existing_values': 'invalid'}
        
        # Test valid values
        for value in ['warn', 'skip', 'overwrite']:
            config.llm_kwargs['existing_values'] = value
            assert config.llm_kwargs['existing_values'] == value

@pytest.fixture
def temp_config_path(tmp_path):
    """Create a temporary directory for config files."""
    return str(tmp_path / "test_config.json")

@pytest.fixture
def sample_config():
    """Create a sample configuration dictionary."""
    return {
        'provider': 'TestProvider',
        'model': 'test-model',
        'llm_kwargs': {'existing_values': 'skip'},
        'system_template': 'test system template',
        'user_template': 'test user template'
    }

def test_config_to_dict():
    """Test converting config to dictionary."""
    config = SuperPandasConfig()
    config_dict = config.to_dict()
    
    assert isinstance(config_dict, dict)
    assert 'provider' in config_dict
    assert 'model' in config_dict
    assert 'llm_kwargs' in config_dict
    assert 'system_template' in config_dict
    assert 'user_template' in config_dict

def test_config_from_dict(sample_config):
    """Test loading config from dictionary."""
    config = SuperPandasConfig()
    config.from_dict(sample_config)
    
    assert config.provider == sample_config['provider']
    assert config.model == sample_config['model']
    assert config.llm_kwargs == sample_config['llm_kwargs']
    assert config.system_template == sample_config['system_template']
    assert config.user_template == sample_config['user_template']

def test_config_save_load(temp_config_path, sample_config):
    """Test saving and loading config to/from file."""
    config = SuperPandasConfig()
    config.from_dict(sample_config)
    
    # Save config
    config.save(temp_config_path)
    assert os.path.exists(temp_config_path)
    
    # Verify file contents
    with open(temp_config_path, 'r') as f:
        saved_config = json.load(f)
    assert saved_config == sample_config
    
    # Create new config instance and load
    new_config = SuperPandasConfig()
    new_config.load(temp_config_path)
    
    # Verify loaded values
    assert new_config.provider == sample_config['provider']
    assert new_config.model == sample_config['model']
    assert new_config.llm_kwargs == sample_config['llm_kwargs']
    assert new_config.system_template == sample_config['system_template']
    assert new_config.user_template == sample_config['user_template']

def test_config_default_path():
    """Test that default config path is set correctly."""
    config = SuperPandasConfig()
    expected_path = os.path.expanduser("~/.cache/superpandas/config.json")
    assert config.DEFAULT_CONFIG_PATH == expected_path

def test_config_load_nonexistent_file():
    """Test loading from non-existent file."""
    config = SuperPandasConfig()
    original_provider = config.provider
    
    # Try to load from non-existent file
    config.load("/nonexistent/path/config.json")
    
    # Config should remain unchanged
    assert config.provider == original_provider

def test_config_partial_load(sample_config):
    """Test loading config with partial data."""
    config = SuperPandasConfig()
    original_provider = config.provider
    
    # Create partial config
    partial_config = {
        'model': 'new-model',
        'llm_kwargs': {'existing_values': 'overwrite'}
    }
    
    config.from_dict(partial_config)
    
    # Only specified values should change
    assert config.provider == original_provider  # Unchanged
    assert config.model == 'new-model'  # Changed
    assert config.llm_kwargs == {'existing_values': 'overwrite'}  # Changed

def test_config_save_creates_directory(temp_config_path):
    """Test that save creates necessary directories."""
    config = SuperPandasConfig()
    
    # Create a path with non-existent parent directories
    deep_path = os.path.join(os.path.dirname(temp_config_path), 'deep', 'path', 'config.json')
    
    # Save should create all necessary directories
    config.save(deep_path)
    assert os.path.exists(deep_path) 