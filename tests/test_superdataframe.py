import pytest
import pandas as pd
import numpy as np
from superpandas import create_super_dataframe
import json
import os
from superpandas import config
from superpandas.llm_client import DummyLLMClient

class TestSuperDataFrameBasics:
    """Test basic super accessor functionality"""
    
    def test_creation(self, sample_df):
        """Test creating a DataFrame with super metadata"""
        df = create_super_dataframe(sample_df)
        assert 'super' in df.attrs
        assert isinstance(df, pd.DataFrame)
        
        # Test with metadata
        df = create_super_dataframe(
            sample_df,
            name="Test DF",
            description="Test description",
            column_descriptions={'int_col': 'Integer column'}
        )
        assert df.super.name == "Test DF"
        assert df.super.description == "Test description"
        assert df.super.get_column_description('int_col') == 'Integer column'
    
    def test_create_super_dataframe_helper(self, sample_df):
        """Test the create_super_dataframe helper function"""
        df = create_super_dataframe(sample_df)
        assert 'super' in df.attrs
        
        # Test creating from scratch
        df = create_super_dataframe({
            'a': [1, 2, 3],
            'b': ['x', 'y', 'z']
        }, name="New DF")
        assert 'super' in df.attrs
        assert df.super.name == "New DF"
    
    def test_metadata_properties(self, sample_super_df):
        """Test metadata property getters and setters"""
        # Test getters
        assert sample_super_df.super.name == "Test DataFrame"
        assert sample_super_df.super.description == "A test dataframe with various column types"
        assert sample_super_df.super.column_descriptions['int_col'] == 'Integer column'
        
        # Test setters
        sample_super_df.super.name = "Updated Name"
        sample_super_df.super.description = "Updated description"
        sample_super_df.super.set_column_description('bool_col', 'Boolean column')
        
        assert sample_super_df.super.name == "Updated Name"
        assert sample_super_df.super.description == "Updated description"
        assert sample_super_df.super.get_column_description('bool_col') == 'Boolean column'
        
        # Test set_column_descriptions
        sample_super_df.super.set_column_descriptions({
            'mixed_col': 'Mixed types column',
            'date_col': 'Date column'
        })
        assert sample_super_df.super.get_column_description('mixed_col') == 'Mixed types column'
        assert sample_super_df.super.get_column_description('date_col') == 'Date column'
        
        # Test error on non-existent column
        with pytest.raises(ValueError):
            sample_super_df.super.set_column_description('non_existent', 'This should fail')

class TestColumnTypeInference:
    """Test column type inference functionality"""
    
    def test_infer_column_types(self, sample_super_df):
        """Test column type inference"""
        column_types = sample_super_df.super.column_types
        
        assert column_types['int_col'] == 'int64'
        assert column_types['float_col'] == 'float64'
        assert column_types['str_col'] == 'str'
        assert column_types['bool_col'] == 'bool'
        assert 'mixed' in column_types['mixed_col']
        assert column_types['null_col'] == 'empty'
    
    def test_refresh_column_types(self, sample_super_df):
        """Test refreshing column types"""
        # Modify the dataframe
        sample_super_df['int_col'] = sample_super_df['int_col'].astype(float)
        sample_super_df['new_col'] = ['x', 'y', 'z', 'a', 'b']
        
        # Refresh types
        updated_types = sample_super_df.super.refresh_column_types()
        
        assert updated_types['int_col'] == 'float64'
        assert updated_types['new_col'] == 'str'
        assert sample_super_df.super.column_types == updated_types

class TestSchemaAndLLMFormat:
    """Test schema generation and LLM formatting"""
    
    def test_schema_generation(self, sample_super_df):
        """Test schema generation"""
        schema = sample_super_df.super.schema()
        
        # Check that schema contains key information
        assert "Test DataFrame" in schema
        assert "A test dataframe with various column types" in schema
        assert "int_col" in schema
        assert "Integer column" in schema
        assert "float_col" in schema
        assert "str_col" in schema
        assert "Shape: (5, 7)" in schema or f"Shape: {sample_super_df.shape}" in schema
    
    def test_schema_with_custom_template(self, sample_super_df):
        """Test schema generation with custom template"""
        template = """
        Name: {name}
        Desc: {description}
        Size: {shape}
        Columns:
        {columns}
        """
        
        schema = sample_super_df.super.schema(template=template)
        
        assert "Name: Test DataFrame" in schema
        assert "Desc: A test dataframe with various column types" in schema
        assert "Size: (5, 7)" in schema or f"Size: {sample_super_df.shape}" in schema
    
    def test_to_llm_format_json(self, sample_super_df):
        """Test JSON format for LLMs"""
        json_str = sample_super_df.super.to_llm_format(format_type='json')
        data = json.loads(json_str)
        
        # Check structure
        assert 'metadata' in data
        assert 'data' in data
        
        # Check metadata
        assert data['metadata']['name'] == "Test DataFrame"
        assert data['metadata']['description'] == "A test dataframe with various column types"
        assert 'columns' in data['metadata']
        assert 'int_col' in data['metadata']['columns']
        
        # Check data
        assert isinstance(data['data'], list)
        assert len(data['data']) == 5  # All rows should be included as we have only 5
    
    def test_to_llm_format_markdown(self, sample_super_df):
        """Test Markdown format for LLMs"""
        md = sample_super_df.super.to_llm_format(format_type='markdown')
        
        assert "# DataFrame: Test DataFrame" in md
        assert "**Description**: A test dataframe with various column types" in md
        assert "## Columns" in md
        assert "## Data Sample" in md
        assert "int_col" in md
    
    def test_to_llm_format_text(self, sample_super_df):
        """Test text format for LLMs"""
        text = sample_super_df.super.to_llm_format(format_type='text')
        
        assert "DataFrame: Test DataFrame" in text
        assert "Description: A test dataframe with various column types" in text
        assert "Columns:" in text
        assert "Data Sample:" in text
    
    def test_to_llm_format_invalid(self, sample_super_df):
        """Test invalid format type"""
        with pytest.raises(ValueError):
            sample_super_df.super.to_llm_format(format_type='invalid_format')

class TestDataFrameOperations:
    """Test pandas operations with super metadata"""
    
    def test_copy(self, sample_super_df):
        """Test copying a DataFrame with super metadata"""
        copied = sample_super_df.copy()
        
        assert 'super' in copied.attrs
        assert copied.super.name == sample_super_df.super.name
        assert copied.super.description == sample_super_df.super.description
        assert copied.super.column_descriptions == sample_super_df.super.column_descriptions
        
        # Modify the copy and check that original is unchanged
        copied.super.name = "Modified Copy"
        assert sample_super_df.super.name == "Test DataFrame"
    
    def test_basic_operations(self, sample_super_df):
        """Test basic pandas operations preserve super metadata"""
        # Test filtering
        filtered = sample_super_df[sample_super_df['int_col'] > 2]
        assert 'super' in filtered.attrs
        assert filtered.super.name == sample_super_df.super.name
        
        # Test selecting columns
        selected = sample_super_df[['int_col', 'str_col']]
        assert 'super' in selected.attrs
        assert selected.super.name == sample_super_df.super.name
        
        # Test adding a column
        with_new_col = sample_super_df.copy()
        with_new_col['new_col'] = [10, 20, 30, 40, 50]
        assert 'super' in with_new_col.attrs
        assert with_new_col.super.name == sample_super_df.super.name

class TestSuperDataFrameIO:
    """Test I/O methods with super metadata"""
    
    def test_read_csv(self, titanic_csv_path):
        """Test read_csv with super metadata"""
        # Basic read
        df = pd.read_csv(titanic_csv_path)
        assert isinstance(df, pd.DataFrame)
        
        # Create test CSV with metadata
        df = create_super_dataframe(
            pd.read_csv(titanic_csv_path),
            name="Titanic Dataset",
            description="Passenger data from the Titanic",
            column_descriptions={
                'Survived': 'Whether the passenger survived (1) or not (0)'
            }
        )
        
        # Save with metadata
        test_path = os.path.join(os.path.dirname(titanic_csv_path), "test_titanic.csv")
        df.super.to_csv(test_path, include_metadata=True)
        
        # Read back and verify metadata preserved
        loaded_df = pd.read_csv(test_path)
        loaded_df.super.read_metadata(test_path)  # Load metadata from companion file
        assert loaded_df.super.name == "Titanic Dataset"
        assert loaded_df.super.description == "Passenger data from the Titanic"
        assert loaded_df.super.get_column_description('Survived') == 'Whether the passenger survived (1) or not (0)'
        
        # Cleanup
        if os.path.exists(test_path):
            os.remove(test_path)
            os.remove(test_path.replace('.csv', '_metadata.json'))
    
    # def test_read_json(self, tmp_path):
    #     """Test read_json and to_json with super metadata"""
    #     # Create test data
    #     original_df = create_super_dataframe(
    #         {'A': [1, 2, 3], 'B': ['x', 'y', 'z']},
    #         name="JSON Test",
    #         description="Test JSON file",
    #         column_descriptions={'A': 'Numbers', 'B': 'Letters'}
    #     )
        
    #     # Save to JSON
    #     json_path = os.path.join(tmp_path, "test.json")
    #     original_df.super.to_json(json_path)
        
    #     # Read it back
    #     loaded_df = pd.read_json(json_path)
        
    #     # Verify data and metadata
    #     assert 'super' in loaded_df.attrs
    #     assert loaded_df.super.name == "JSON Test"
    #     assert loaded_df.super.description == "Test JSON file"
    #     assert loaded_df.super.get_column_description('A') == 'Numbers'
    #     assert loaded_df.super.get_column_description('B') == 'Letters'
    #     assert list(loaded_df.columns) == ['A', 'B']
    #     pd.testing.assert_frame_equal(pd.DataFrame(original_df), pd.DataFrame(loaded_df))
    
    def test_read_pickle(self, tmp_path):
        """Test read_pickle and to_pickle with super metadata"""
        # Create test data
        original_df = create_super_dataframe(
            {'A': [1, 2, 3], 'B': ['x', 'y', 'z']},
            name="Pickle Test",
            description="Test Pickle file",
            column_descriptions={'A': 'Numbers', 'B': 'Letters'}
        )
        
        # Save to pickle
        pickle_path = os.path.join(tmp_path, "test.pkl")
        original_df.super.to_pickle(pickle_path)
        
        # Read it back
        loaded_df = pd.read_pickle(pickle_path)
        
        # Verify data and metadata
        assert 'super' in loaded_df.attrs
        assert loaded_df.super.name == "Pickle Test"
        assert loaded_df.super.description == "Test Pickle file"
        assert loaded_df.super.get_column_description('A') == 'Numbers'
        assert loaded_df.super.get_column_description('B') == 'Letters'
        assert list(loaded_df.columns) == ['A', 'B']
        pd.testing.assert_frame_equal(pd.DataFrame(original_df), pd.DataFrame(loaded_df)) 

class TestAutoDescribe:
    """Test auto_describe functionality"""
    
    def test_auto_describe_with_config(self, sample_df):
        """Test auto_describe using config settings"""
        from superpandas import create_super_dataframe
        
        # Create test DataFrame with super accessor
        df = create_super_dataframe(sample_df)
        
        # Configure global settings
        config.configure_llm(
            model=DummyLLMClient(),
            existing_values='warn'
        )
        
        # Test with default config settings
        df.super.auto_describe()
        
        # Test with override
        df.super.auto_describe(existing_values='overwrite')
        
        # Test with existing values
        df.super.name = "Existing Name"
        df.super.description = "Existing Description"
        df.super.set_column_descriptions({'int_col': 'Existing description'})
        
        # Should warn but not overwrite with default settings
        with pytest.warns(UserWarning):
            df.super.auto_describe()
        assert df.super.name == "Existing Name"
        
        # Should skip silently
        df.super.auto_describe(existing_values='skip')
        assert df.super.name == "Existing Name"
        
        # Should overwrite
        df.super.auto_describe(existing_values='overwrite')
        assert df.super.name != "Existing Name"

    def test_auto_describe_partial_metadata(self, sample_df):
        """Test auto_describe with partial existing metadata"""
        from superpandas import create_super_dataframe
        
        df = create_super_dataframe(sample_df)
        
        # Set some metadata but leave others empty
        df.super.name = "Existing Name"
        # Leave description empty
        df.super.set_column_descriptions({'int_col': 'Integer column'})
        
        # Configure to skip existing values
        config.configure_llm(existing_values='skip',
                             model=DummyLLMClient())
        
        # Should only generate missing metadata
        df.super.auto_describe()
        
        assert df.super.name == "Existing Name"  # Should not change
        assert df.super.description != ""  # Should be generated
        assert df.super.get_column_description('int_col') == 'Integer column'  # Should not change
        assert df.super.get_column_description('float_col') != ""  # Should be generated

    def test_auto_describe_empty_dataframe(self):
        """Test auto_describe with empty DataFrame"""
        import pandas as pd
        from superpandas import create_super_dataframe
        
        # Create empty DataFrame
        df = create_super_dataframe(pd.DataFrame())
        
        # Should handle empty DataFrame gracefully
        df.super.auto_describe()
        
        assert isinstance(df.super.name, str)
        assert isinstance(df.super.description, str)
        assert isinstance(df.super.column_descriptions, dict) 