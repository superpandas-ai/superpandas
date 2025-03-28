import pytest
import pandas as pd
import numpy as np
from superpandas import SuperDataFrame
import os

@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing"""
    return pd.DataFrame({
        'int_col': [1, 2, 3, 4, 5],
        'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
        'str_col': ['a', 'b', 'c', 'd', 'e'],
        'bool_col': [True, False, True, False, True],
        'mixed_col': [1, 2, 'three', 4.0, True],
        'date_col': pd.date_range('2023-01-01', periods=5),
        'null_col': [None, None, None, None, None]
    })

@pytest.fixture
def sample_super_df(sample_df):
    """Create a sample SuperDataFrame for testing"""
    return SuperDataFrame(
        sample_df,
        name="Test DataFrame",
        description="A test dataframe with various column types",
        column_descriptions={
            'int_col': 'Integer column',
            'float_col': 'Float column',
            'str_col': 'String column'
        }
    )

@pytest.fixture
def titanic_csv_path():
    """Path to the titanic.csv file for testing file reading functions"""
    # This assumes titanic.csv is in the tests directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, 'titanic.csv') 