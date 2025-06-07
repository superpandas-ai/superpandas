"""
SuperPandas: Enhanced pandas DataFrames with metadata and LLM integration
"""

from .superdataframe import create_super_dataframe, read_csv, read_pickle
from .llm_client import LLMClient
from .config import SuperPandasConfig

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
] 