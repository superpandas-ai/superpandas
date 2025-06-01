"""
SuperPandas: Enhanced pandas DataFrames with metadata and LLM integration
"""

from .superdataframe import create_super_dataframe, SuperDataFrameAccessor, read_csv
from .llm_client import LLMClient, DummyLLMClient
from .config import config

# Re-export pandas functionality, but exclude read_csv to avoid conflict
import pandas as pd
for name in pd.__all__:
    if name != 'read_csv':  # Skip read_csv to avoid overriding our version
        globals()[name] = getattr(pd, name)

__version__ = '0.1.0'

__all__ = [
    'create_super_dataframe',
    'LLMClient',
    'DummyLLMClient',
    'SuperDataFrameAccessor',
    'read_csv',
    'config',
] 