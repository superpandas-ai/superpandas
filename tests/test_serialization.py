import pytest
import pandas as pd
import json
import os
from superpandas import SuperDataFrame
import numpy as np

@pytest.fixture
def sample_df():
    """Create a sample SuperDataFrame with various data types and metadata"""
    data = {
        'int_col': [1, 2, 3],
        'float_col': [1.1, 2.2, 3.3],
        'str_col': ['a', 'b', 'c'],
        'bool_col': [True, False, True],
        'date_col': pd.date_range('2023-01-01', periods=3)
    }
    
    df = SuperDataFrame(
        data,
        name='test_df',
        description='A test dataframe',
        column_descriptions={
            'int_col': 'Integer column',
            'float_col': 'Float column',
            'str_col': 'String column',
            'bool_col': 'Boolean column',
            'date_col': 'Date column'
        }
    )
    return df

@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test files"""
    return tmp_path

class TestSerialization:
    def test_pickle_serialization(self, sample_df, temp_dir):
        """Test pickle serialization and deserialization"""
        # Save to pickle
        pickle_path = temp_dir / 'test.pkl'
        sample_df.to_pickle(pickle_path)
        
        # Load from pickle
        loaded_df = SuperDataFrame.read_pickle(pickle_path)
        
        # Check data equality
        pd.testing.assert_frame_equal(sample_df, loaded_df)
        
        # Check metadata
        assert loaded_df.name == sample_df.name
        assert loaded_df.description == sample_df.description
        assert loaded_df.column_descriptions == sample_df.column_descriptions
        assert loaded_df.column_types == sample_df.column_types

    def test_json_serialization(self, sample_df, temp_dir):
        """Test JSON serialization and deserialization"""
        # Reset index before saving to ensure consistent state
        sample_df = sample_df.reset_index(drop=True)
        
        # Save to JSON
        json_path = temp_dir / 'test.json'
        sample_df.to_json(str(json_path))
        
        # Verify JSON file is valid
        with open(json_path) as f:
            json_data = json.load(f)
        assert 'dataframe' in json_data
        assert 'metadata' in json_data
        
        # Load from JSON
        loaded_df = SuperDataFrame.read_json(str(json_path))
        
        # Check data equality (convert datetime columns to same type for comparison)
        sample_df_copy = sample_df.copy()
        loaded_df_copy = loaded_df.copy()
        sample_df_copy['date_col'] = pd.to_datetime(sample_df_copy['date_col'])
        loaded_df_copy['date_col'] = pd.to_datetime(loaded_df_copy['date_col'])
        pd.testing.assert_frame_equal(sample_df_copy, loaded_df_copy)
        
        # Check metadata
        assert loaded_df.name == sample_df.name
        assert loaded_df.description == sample_df.description
        assert loaded_df.column_descriptions == sample_df.column_descriptions
        assert loaded_df.column_types == sample_df.column_types

    def test_csv_serialization(self, sample_df, temp_dir):
        """Test CSV serialization and deserialization"""
        # Reset index before saving to ensure consistent state
        sample_df = sample_df.reset_index(drop=True)
        
        # Save to CSV with metadata
        csv_path = temp_dir / 'test.csv'
        sample_df.to_csv(str(csv_path), index=False)  # Explicitly set index=False
        
        # Verify metadata file exists
        metadata_path = temp_dir / 'test_metadata.json'
        assert metadata_path.exists()
        
        # Load from CSV
        loaded_df = SuperDataFrame.read_csv(str(csv_path))
        
        # Check data equality (convert datetime columns to same type for comparison)
        sample_df_copy = sample_df.copy()
        loaded_df_copy = loaded_df.copy()
        sample_df_copy['date_col'] = pd.to_datetime(sample_df_copy['date_col'])
        loaded_df_copy['date_col'] = pd.to_datetime(loaded_df_copy['date_col'])
        pd.testing.assert_frame_equal(sample_df_copy, loaded_df_copy)
        
        # Check metadata
        assert loaded_df.name == sample_df.name
        assert loaded_df.description == sample_df.description
        assert loaded_df.column_descriptions == sample_df.column_descriptions
        assert loaded_df.column_types == sample_df.column_types

    def test_csv_without_metadata(self, sample_df, temp_dir):
        """Test CSV serialization without metadata"""
        # Reset index before saving to ensure consistent state
        sample_df = sample_df.reset_index(drop=True)
        
        # Save to CSV without metadata
        csv_path = temp_dir / 'test_no_metadata.csv'
        sample_df.to_csv(str(csv_path), include_metadata=False, index=False)
        
        # Verify metadata file doesn't exist
        metadata_path = temp_dir / 'test_no_metadata_metadata.json'
        assert not metadata_path.exists()
        
        # Load from CSV
        loaded_df = SuperDataFrame.read_csv(str(csv_path))
        
        # Check data equality
        sample_df_copy = sample_df.copy()
        loaded_df_copy = loaded_df.copy()
        sample_df_copy['date_col'] = pd.to_datetime(sample_df_copy['date_col'])
        loaded_df_copy['date_col'] = pd.to_datetime(loaded_df_copy['date_col'])
        pd.testing.assert_frame_equal(sample_df_copy, loaded_df_copy)
        
        # Check metadata is empty/default
        assert loaded_df.name == ''
        assert loaded_df.description == ''
        assert loaded_df.column_descriptions == {}
        assert isinstance(loaded_df.column_types, dict)

    def test_edge_cases(self, temp_dir):
        """Test edge cases and special data types"""
        # Create DataFrame with special values
        data = {
            'nulls': [None, np.nan, None],
            'mixed': ['1', 'two', '3.0'],  # Make all values strings for consistent type
            'empty': [np.nan] * 3,
            # Convert complex numbers to strings for JSON compatibility
            'complex_str': [f"{c.real}+{c.imag}j" for c in [complex(1, 2), complex(3, 4), complex(5, 6)]]
        }
        
        df = SuperDataFrame(
            data,
            name='edge_case_df',
            description='Testing edge cases'
        )
        
        # Test pickle (can handle complex numbers directly)
        pickle_data = {
            'nulls': [None, np.nan, None],
            'mixed': ['1', 'two', '3.0'],  # Keep consistent with main data
            'empty': [np.nan] * 3,
            'complex': [complex(1, 2), complex(3, 4), complex(5, 6)]  # Keep complex for pickle test
        }
        df_pickle = SuperDataFrame(
            pickle_data,
            name='edge_case_df',
            description='Testing edge cases'
        )
        pickle_path = temp_dir / 'edge_case.pkl'
        df_pickle.to_pickle(pickle_path)
        loaded_pickle = SuperDataFrame.read_pickle(pickle_path)
        pd.testing.assert_frame_equal(df_pickle, loaded_pickle)
        
        # Test JSON (with string representation of complex numbers)
        json_path = temp_dir / 'edge_case.json'
        df.to_json(str(json_path))
        loaded_json = SuperDataFrame.read_json(str(json_path))
        pd.testing.assert_frame_equal(df, loaded_json)
        
        # Test CSV
        csv_path = temp_dir / 'edge_case.csv'
        df.to_csv(str(csv_path), index=False)
        loaded_csv = SuperDataFrame.read_csv(str(csv_path))
        pd.testing.assert_frame_equal(df, loaded_csv)

    def test_error_handling(self, temp_dir):
        """Test error handling for serialization methods"""
        # Test reading non-existent file
        with pytest.raises(FileNotFoundError):
            SuperDataFrame.read_pickle(temp_dir / 'nonexistent.pkl')
        
        with pytest.raises(FileNotFoundError):
            SuperDataFrame.read_json(temp_dir / 'nonexistent.json')
        
        # Test reading invalid JSON
        invalid_json_path = temp_dir / 'invalid.json'
        with open(invalid_json_path, 'w') as f:
            f.write('{"invalid": json')
        
        with pytest.raises(json.JSONDecodeError):
            SuperDataFrame.read_json(invalid_json_path) 