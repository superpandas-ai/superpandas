"""
SuperPandas: Enhanced pandas DataFrames with metadata and LLM integration
"""

from .superdataframe import create_super_dataframe, SuperDataFrameAccessor, read_csv
from .llm_client import auto_describe_dataframe, LLMClient, DummyLLMClient

# Re-export pandas functionality, but exclude read_csv to avoid conflict
import pandas as pd
for name in pd.__all__:
    if name != 'read_csv':  # Skip read_csv to avoid overriding our version
        globals()[name] = getattr(pd, name)

# Add LLM-powered description functionality
describe_with_llm = auto_describe_dataframe

__version__ = '0.1.0'

__all__ = [
    'create_super_dataframe',
    'LLMClient',
    'DummyLLMClient',
    'SuperDataFrameAccessor',
    'describe_with_llm',
    'read_csv',
] 