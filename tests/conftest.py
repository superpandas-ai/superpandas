import pytest
import pandas as pd
import numpy as np
import os
from superpandas import create_super_dataframe

@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing"""
    df = pd.DataFrame({
        'int_col': [1, 2, 3, 4, 5],
        'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
        'str_col': ['a', 'b', 'c', 'd', 'e'],
        'bool_col': [True, False, True, False, True],
        'mixed_col': [1, 2, 'three', 4.0, True],
        'date_col': pd.date_range('2023-01-01', periods=5),
        'null_col': [None, None, None, None, None]
    })
    return df

@pytest.fixture
def sample_super_df(sample_df):
    """Create a sample DataFrame with super metadata for testing"""
    return create_super_dataframe(
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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, 'titanic.csv') 