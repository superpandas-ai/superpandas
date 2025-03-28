"""
SuperPandas: Enhanced pandas DataFrames with metadata and LLM integration
"""

from .superdataframe import SuperDataFrame, create_super_dataframe
from . import _pandas_wrapper
from .llm_client import auto_describe_dataframe, LLMClient, DummyLLMClient

# Re-export all pandas functionality
from pandas import *

# Override pandas DataFrame with SuperDataFrame
DataFrame = SuperDataFrame

# Add custom read methods
read_csv = _pandas_wrapper.read_csv
read_excel = _pandas_wrapper.read_excel
read_json = _pandas_wrapper.read_json
read_sql = _pandas_wrapper.read_sql
read_parquet = _pandas_wrapper.read_parquet

# Add LLM-powered description functionality
describe_with_llm = auto_describe_dataframe

__version__ = '0.1.0'

__all__ = [
    'SuperDataFrame',
    'create_super_dataframe',
    'LLMClient',
    'DummyLLMClient',
    'read_csv',
    'read_excel',
    'read_sql',
    'read_parquet',
    'read_json',
] 