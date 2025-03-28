"""
SuperPandas: Enhanced pandas DataFrames with metadata and LLM integration
"""

from .superdataframe import SuperDataFrame, create_super_dataframe
from .llm_client import auto_describe_dataframe, LLMClient, DummyLLMClient

# Re-export all pandas functionality
from pandas import *

# Override pandas DataFrame with SuperDataFrame
DataFrame = SuperDataFrame

# Use SuperDataFrame I/O methods directly
read_csv = SuperDataFrame.read_csv
read_json = SuperDataFrame.read_json
read_pickle = SuperDataFrame.read_pickle

# Add LLM-powered description functionality
describe_with_llm = auto_describe_dataframe

__version__ = '0.1.0'

__all__ = [
    'SuperDataFrame',
    'create_super_dataframe',
    'LLMClient',
    'DummyLLMClient',
    'read_csv',
    'read_json',
    'read_pickle',
] 