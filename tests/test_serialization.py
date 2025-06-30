import pytest
import pandas as pd
import json
import os
from superpandas import create_super_dataframe
import numpy as np
import superpandas as spd

@pytest.fixture
def sample_df():
    """Create a sample DataFrame with various data types and metadata"""
    data = {
        'int_col': [1, 2, 3],
        'float_col': [1.1, 2.2, 3.3],
        'str_col': ['a', 'b', 'c'],
        'bool_col': [True, False, True],
        'date_col': pd.date_range('2023-01-01', periods=3)
    }
    
    df = create_super_dataframe(
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
        sample_df.super.to_pickle(pickle_path)
        
        # Load from pickle
        loaded_df = pd.read_pickle(pickle_path)
        
        # Check data equality
        pd.testing.assert_frame_equal(pd.DataFrame(sample_df), pd.DataFrame(loaded_df))
        
        # Check metadata
        assert loaded_df.super.name == sample_df.super.name
        assert loaded_df.super.description == sample_df.super.description
        assert loaded_df.super.column_descriptions == sample_df.super.column_descriptions
        assert loaded_df.super.column_types == sample_df.super.column_types

    def test_csv_serialization(self, sample_df, temp_dir):
        """Test CSV serialization and deserialization"""
        # Reset index before saving to ensure consistent state
        sample_df = sample_df.reset_index(drop=True)
        
        # Save to CSV with metadata
        csv_path = temp_dir / 'test.csv'
        sample_df.super.to_csv(str(csv_path), index=False)  # Explicitly set index=False
        
        # Verify metadata file exists
        metadata_path = temp_dir / 'test_metadata.json'
        assert metadata_path.exists()
        
        # Load from CSV using standalone read_csv function with date parsing
        loaded_df = spd.read_csv(str(csv_path), parse_dates=['date_col'])
        
        # Check data equality (no need for datetime conversion anymore)
        pd.testing.assert_frame_equal(sample_df, loaded_df)
        
        # Check metadata
        assert loaded_df.super.name == sample_df.super.name
        assert loaded_df.super.description == sample_df.super.description
        assert loaded_df.super.column_descriptions == sample_df.super.column_descriptions
        assert loaded_df.super.column_types == sample_df.super.column_types

    def test_csv_serialization_no_metadata(self, sample_df, temp_dir):
        """Test CSV serialization without metadata and requiring metadata"""
        # Reset index before saving to ensure consistent state
        sample_df = sample_df.reset_index(drop=True)
        
        # Save to CSV without metadata
        csv_path = temp_dir / 'test_no_metadata.csv'
        sample_df.super.to_csv(str(csv_path), include_metadata=False, index=False)
        
        # Verify metadata file doesn't exist
        metadata_path = temp_dir / 'test_no_metadata_metadata.json'
        assert not metadata_path.exists()
        
        # Test that reading with include_metadata=True raises error
        with pytest.raises(FileNotFoundError, match="Metadata file not found"):
            spd.read_csv(str(csv_path), include_metadata=True)
        
        # Test reading without requiring metadata
        loaded_df = spd.read_csv(str(csv_path), include_metadata=False, parse_dates=['date_col'])
        
        # Check data equality
        pd.testing.assert_frame_equal(sample_df, loaded_df)
        
        # Check metadata is empty/default
        assert loaded_df.super.name == ''
        assert loaded_df.super.description == ''
        assert loaded_df.super.column_descriptions == {}
        assert isinstance(loaded_df.super.column_types, dict)

    # def test_edge_cases(self, temp_dir):
    #     """Test edge cases and special data types"""
    #     # Create DataFrame with special values
    #     data = {
    #         'nulls': [None, np.nan, None],
    #         'mixed': ['1', 'two', '3.0'],
    #         'empty': [np.nan] * 3,
    #         'complex_str': [f"{c.real}+{c.imag}j" for c in [complex(1, 2), complex(3, 4), complex(5, 6)]]
    #     }
        
    #     df = create_super_dataframe(
    #         data,
    #         name='edge_case_df',
    #         description='Testing edge cases'
    #     )
        
    #     # Test JSON serialization with edge cases
    #     json_path = temp_dir / 'edge_case.json'
    #     df.super.to_json(str(json_path))
        
    #     loaded_df = pd.DataFrame()
    #     loaded_df.super.read_json(str(json_path))
        
    #     # Check data equality
    #     pd.testing.assert_frame_equal(df, loaded_df)
        
    #     # Check metadata
    #     assert loaded_df.super.name == df.super.name
    #     assert loaded_df.super.description == df.super.description
    #     assert loaded_df.super.column_descriptions == df.super.column_descriptions
    #     assert loaded_df.super.column_types == df.super.column_types
    

    # def test_error_handling(self, temp_dir):
    #     """Test error handling for serialization methods"""
    #     df = pd.DataFrame({'a': [1, 2, 3]})
    #     df.attrs['super'] = {
    #         'name': 'test',
    #         'description': '',
    #         'column_descriptions': {},
    #         'column_types': {}
    #     }
        
    #     # Test reading non-existent file
    #     with pytest.raises(FileNotFoundError):
    #         df.super.read_pickle(temp_dir / 'nonexistent.pkl')
        
    #     with pytest.raises(FileNotFoundError):
    #         df.super.read_json(temp_dir / 'nonexistent.json')
        
    #     # Test reading invalid JSON
    #     invalid_json_path = temp_dir / 'invalid.json'
    #     with open(invalid_json_path, 'w') as f:
    #         f.write('{"invalid": json')
        
    #     # Should handle invalid JSON gracefully by initializing empty metadata
    #     df_invalid = pd.DataFrame()
    #     df_invalid.super.read_json(str(invalid_json_path))
    #     assert df_invalid.super.name == ''
    #     assert df_invalid.super.description == '' 

    def test_pickle_serialization_new_interface(self, temp_dir):
        """Test pickle serialization and deserialization with the new interface"""
        # Create test data
        df = create_super_dataframe(
            {'A': [1, 2, 3], 'B': ['x', 'y', 'z']},
            name="Test DF",
            description="Test description",
            column_descriptions={
                'A': 'Numbers',
                'B': 'Letters'
            }
        )
        
        # Save to pickle
        pickle_path = os.path.join(temp_dir, "test.pkl")
        df.super.to_pickle(pickle_path)
        
        # Read it back
        loaded_df = pd.read_pickle(pickle_path)
        
        # Verify data and metadata
        assert 'super' in loaded_df.attrs
        assert loaded_df.super.name == "Test DF"
        assert loaded_df.super.description == "Test description"
        assert loaded_df.super.column_descriptions == df.super.column_descriptions
        assert list(loaded_df.columns) == ['A', 'B']
        pd.testing.assert_frame_equal(pd.DataFrame(df), pd.DataFrame(loaded_df))

    def test_pickle_serialization_new_interface_empty(self, temp_dir):
        """Test pickle serialization and deserialization with the new interface when column descriptions are empty"""
        # Create test data
        df = create_super_dataframe(
            {'A': [1, 2, 3], 'B': ['x', 'y', 'z']},
            name="Test DF",
            description="Test description"
        )
        
        # Save to pickle
        pickle_path = os.path.join(temp_dir, "test.pkl")
        df.super.to_pickle(pickle_path)
        
        # Read it back
        loaded_df = spd.read_pickle(pickle_path)
        
        # Verify data and metadata
        assert 'super' in loaded_df.attrs
        assert loaded_df.super.name == "Test DF"
        assert loaded_df.super.description == "Test description"
        assert loaded_df.super.column_descriptions == {'A': '', 'B': ''}
        assert list(loaded_df.columns) == ['A', 'B']
        pd.testing.assert_frame_equal(pd.DataFrame(df), pd.DataFrame(loaded_df))

    def test_read_pickle_with_metadata(self, sample_df, temp_dir):
        """Test reading a pickle file with existing metadata"""
        # Save to pickle
        pickle_path = temp_dir / 'test.pkl'
        sample_df.super.to_pickle(pickle_path)
        
        # Read using the new read_pickle function
        loaded_df = spd.read_pickle(pickle_path)
        
        # Check data equality
        pd.testing.assert_frame_equal(pd.DataFrame(sample_df), pd.DataFrame(loaded_df))
        
        # Check metadata
        assert loaded_df.super.name == sample_df.super.name
        assert loaded_df.super.description == sample_df.super.description
        assert loaded_df.super.column_descriptions == sample_df.super.column_descriptions
        assert loaded_df.super.column_types == sample_df.super.column_types

    def test_read_pickle_without_metadata(self, temp_dir):
        """Test reading a pickle file without metadata"""
        # Create a regular pandas DataFrame without metadata
        df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
        
        # Save to pickle
        pickle_path = temp_dir / 'test.pkl'
        df.to_pickle(pickle_path)
        
        # Read using the new read_pickle function
        loaded_df = spd.read_pickle(pickle_path)
        
        # Check data equality
        pd.testing.assert_frame_equal(df, loaded_df)
        
        # Check that metadata was initialized
        assert 'super' in loaded_df.attrs
        assert loaded_df.super.name == ''
        assert loaded_df.super.description == ''
        assert loaded_df.super.column_descriptions == {'A': '', 'B': ''}
        assert loaded_df.super.column_types == {'A': 'int64', 'B': 'str'}

    # def test_read_pickle_invalid_format(self, temp_dir): # TODO: Check it later
    #     """Test reading a pickle file with invalid format"""
    #     # Create invalid data
    #     invalid_data = {'invalid': 'data'}
        
    #     # Save invalid data to pickle
    #     pickle_path = temp_dir / 'test.pkl'
    #     pd.to_pickle(invalid_data, pickle_path)
        
    #     # Test that reading invalid format raises ValueError
    #     with pytest.raises(ValueError, match="Invalid pickle file format"):
    #         spd.read_pickle(pickle_path)

    def test_read_pickle_nonexistent_file(self, temp_dir):
        """Test reading a non-existent pickle file"""
        # Test that reading non-existent file raises FileNotFoundError
        with pytest.raises(FileNotFoundError):
            spd.read_pickle(temp_dir / 'nonexistent.pkl') 