"""
SuperPandas: Enhanced pandas DataFrames with metadata and LLM integration
"""

from .superdataframe import create_super_dataframe, read_csv, read_pickle
from .llm_client import LLMClient
from .config import SuperPandasConfig

# Initialize default config, which will load from saved file if it exists
default_config = SuperPandasConfig.get_default_config()

def set_default_config(config: SuperPandasConfig):
    """
    Set the default configuration instance for the library.
    
    Parameters:
    -----------
    config : SuperPandasConfig
        The configuration instance to use as default
    """
    SuperPandasConfig.set_default_config(config)
    global default_config
    default_config = config

# Re-export pandas functionality, but exclude read_csv to avoid conflict
import pandas as pd
for name in pd.__all__:
    if name != 'read_csv' and name != 'read_pickle':  # Skip read_csv to avoid overriding our version
        globals()[name] = getattr(pd, name)

__version__ = '0.1.0'

__all__ = [
    'create_super_dataframe',
    'LLMClient',
    'read_csv',
    'read_pickle',
    'SuperPandasConfig',
    'default_config',
    'set_default_config',
] 