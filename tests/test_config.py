import pytest
from smolagents import Model
import os
import json
from superpandas.config import SuperPandasConfig, available_providers
import superpandas as spd

class TestConfig:
    """Test configuration functionality"""
    
    def test_config_singleton(self):
        """Test that config instances are independent"""
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
        assert config.provider == "hf"
        assert config.llm_kwargs == {}
        
        # Test setting values
        
        config.model = "meta-llama/Llama-3.2-3B-Instruct"
        config.provider = "tf"
        config.llm_kwargs.update({'temperature': 0.7})
        
        assert config.model == "meta-llama/Llama-3.2-3B-Instruct"
        assert config.provider == "tf"
        assert config.llm_kwargs['temperature'] == 0.7
    
    def test_existing_values_validation(self):
        """Test validation of existing_values setting"""
        config = SuperPandasConfig()
        
        # Test invalid value
        with pytest.raises(ValueError):
            config.existing_values = 'invalid'
        
        # Test valid values
        for value in ['warn', 'skip', 'overwrite']:
            config.existing_values = value
            assert config.existing_values == value

@pytest.fixture
def temp_config_path(tmp_path):
    """Create a temporary directory for config files."""
    return str(tmp_path / "test_config.json")

@pytest.fixture
def sample_config():
    """Create a sample configuration dictionary."""
    return {
        'provider': 'openai',
        'model': 'test-model',
        'llm_kwargs': {'temperature': 0.7},
        'existing_values': 'skip',
        'system_template': 'You are a helpful AI assistant.',
        'user_template': 'Question: {question}\n\nContext: {schema}',
        'schema_template': 'DataFrame: {name}\nDescription: {description}\nShape: {shape}\nColumns:\n{column_info}'
    }

def test_config_to_dict():
    """Test converting config to dictionary."""
    config = SuperPandasConfig()
    config_dict = config.to_dict()
    
    assert isinstance(config_dict, dict)
    assert 'provider' in config_dict
    assert 'model' in config_dict
    assert 'llm_kwargs' in config_dict
    assert 'existing_values' in config_dict
    assert 'system_template' in config_dict
    assert 'user_template' in config_dict
    assert 'schema_template' in config_dict

def test_config_from_dict(sample_config):
    """Test loading config from dictionary."""
    config = SuperPandasConfig()
    config.from_dict(sample_config)
    
    assert config.provider == sample_config['provider']
    assert config.model == sample_config['model']
    assert config.llm_kwargs == sample_config['llm_kwargs']
    assert config.existing_values == sample_config['existing_values']
    assert config.system_template == sample_config['system_template']
    assert config.user_template == sample_config['user_template']
    assert config.schema_template == sample_config['schema_template']

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
    assert new_config.existing_values == sample_config['existing_values']
    assert new_config.system_template == sample_config['system_template']
    assert new_config.user_template == sample_config['user_template']
    assert new_config.schema_template == sample_config['schema_template']

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
        'llm_kwargs': {'temperature': 0.7}
    }
    
    config.from_dict(partial_config)
    
    # Only specified values should change
    assert config.provider == original_provider  # Unchanged
    assert config.model == 'new-model'  # Changed
    assert config.llm_kwargs == {'temperature': 0.7}  # Changed

def test_config_save_creates_directory(temp_config_path):
    """Test that save creates necessary directories."""
    config = SuperPandasConfig()
    
    # Create a path with non-existent parent directories
    deep_path = os.path.join(os.path.dirname(temp_config_path), 'deep', 'path', 'config.json')
    
    # Save should create all necessary directories
    config.save(deep_path)
    assert os.path.exists(deep_path)

def test_default_config_initialization():
    """Test that default config is properly initialized"""
    # Check that default_config is initialized
    assert spd.default_config is not None
    assert isinstance(spd.default_config, SuperPandasConfig)
    
    # Check that it's the same instance as the class-level default
    assert spd.default_config is SuperPandasConfig.get_default_config()

def test_set_default_config():
    """Test setting a new default config"""
    # Create a new config with different settings
    new_config = SuperPandasConfig()
    new_config.model = "test-model"
    new_config.provider = "openai"
    
    # Store original config for comparison
    original_config = spd.default_config
    
    # Set new default config
    spd.set_default_config(new_config)
    
    # Verify the change
    assert spd.default_config is new_config
    assert spd.default_config.model == "test-model"
    assert spd.default_config.provider == "openai"
    
    # Verify class-level default is also updated
    assert SuperPandasConfig.get_default_config() is new_config

def test_super_accessor_uses_default_config():
    """Test that SuperDataFrameAccessor uses default config when none is set"""
    import pandas as pd
    
    # Create a DataFrame
    df = pd.DataFrame({'A': [1, 2, 3]})
    
    # Verify super accessor uses default config
    assert df.super.config is spd.default_config
    
    # Create new config
    new_config = SuperPandasConfig()
    new_config.model = "test-model"
    
    # Set as default
    spd.set_default_config(new_config)
    
    # Create new DataFrame and verify it uses new default config
    df2 = pd.DataFrame({'B': [4, 5, 6]})
    assert df2.super.config is new_config

def test_super_accessor_config_override():
    """Test that SuperDataFrameAccessor can override default config"""
    import pandas as pd
    
    # Create a DataFrame
    df = pd.DataFrame({'A': [1, 2, 3]})
    
    # Create custom config
    custom_config = SuperPandasConfig()
    custom_config.model = "custom-model"
    
    # Set custom config for this DataFrame
    df.super.config = custom_config
    
    # Verify it uses custom config instead of default
    assert df.super.config is custom_config
    assert df.super.config is not spd.default_config

def test_default_config_persistence():
    """Test that default config persists across multiple imports"""
    import importlib
    
    # Set a custom default config
    custom_config = SuperPandasConfig()
    custom_config.model = "persistent-model"
    spd.set_default_config(custom_config)
    
    # Reload the module
    importlib.reload(spd)
    
    # Verify the default config persists
    assert spd.default_config.model == "persistent-model"

def test_default_config_save_load():
    """Test that default config can be saved and loaded"""
    # Create and set a custom config
    custom_config = SuperPandasConfig()
    custom_config.model = "saved-model"
    custom_config.provider = "openai"
    spd.set_default_config(custom_config)
    
    # Save the config
    custom_config.save()
    
    # Create a new config instance and load
    new_config = SuperPandasConfig()
    new_config.load()
    
    # Verify loaded config matches saved config
    assert new_config.model == "saved-model"
    assert new_config.provider == "openai"

def test_multiple_dataframes_config():
    """Test that multiple DataFrames can share or have different configs"""
    import pandas as pd
    
    # Create two DataFrames
    df1 = pd.DataFrame({'A': [1, 2, 3]})
    df2 = pd.DataFrame({'B': [4, 5, 6]})
    
    # Initially both should use default config
    assert df1.super.config is spd.default_config
    assert df2.super.config is spd.default_config
    
    # Set custom config for df1
    custom_config = SuperPandasConfig()
    custom_config.model = "df1-model"
    df1.super.config = custom_config
    
    # Verify df1 uses custom config while df2 still uses default
    assert df1.super.config is custom_config
    assert df2.super.config is spd.default_config 