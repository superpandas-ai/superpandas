import pytest
import pandas as pd
import numpy as np
from superpandas import SuperDataFrame, create_super_dataframe
import json

class TestSuperDataFrameBasics:
    """Test basic SuperDataFrame functionality"""
    
    def test_creation(self, sample_df):
        """Test creating a SuperDataFrame"""
        sdf = SuperDataFrame(sample_df)
        assert isinstance(sdf, SuperDataFrame)
        assert isinstance(sdf, pd.DataFrame)
        
        # Test with metadata
        sdf = SuperDataFrame(
            sample_df,
            name="Test DF",
            description="Test description",
            column_descriptions={'int_col': 'Integer column'}
        )
        assert sdf.name == "Test DF"
        assert sdf.description == "Test description"
        assert sdf.get_column_description('int_col') == 'Integer column'
    
    def test_create_super_dataframe_helper(self, sample_df):
        """Test the create_super_dataframe helper function"""
        sdf = create_super_dataframe(sample_df)
        assert isinstance(sdf, SuperDataFrame)
        
        # Test creating from scratch
        sdf = create_super_dataframe({
            'a': [1, 2, 3],
            'b': ['x', 'y', 'z']
        }, name="New DF")
        assert isinstance(sdf, SuperDataFrame)
        assert sdf.name == "New DF"
    
    def test_from_pandas(self, sample_df):
        """Test creating a SuperDataFrame from a pandas DataFrame"""
        sdf = SuperDataFrame.from_pandas(
            sample_df,
            name="Converted DF",
            description="Converted from pandas",
            column_descriptions={'int_col': 'Integer column'}
        )
        assert isinstance(sdf, SuperDataFrame)
        assert sdf.name == "Converted DF"
        assert sdf.description == "Converted from pandas"
        assert sdf.get_column_description('int_col') == 'Integer column'
    
    def test_metadata_properties(self, sample_super_df):
        """Test metadata property getters and setters"""
        # Test getters
        assert sample_super_df.name == "Test DataFrame"
        assert sample_super_df.description == "A test dataframe with various column types"
        assert sample_super_df.column_descriptions['int_col'] == 'Integer column'
        
        # Test setters
        sample_super_df.name = "Updated Name"
        sample_super_df.description = "Updated description"
        sample_super_df.set_column_description('bool_col', 'Boolean column')
        
        assert sample_super_df.name == "Updated Name"
        assert sample_super_df.description == "Updated description"
        assert sample_super_df.get_column_description('bool_col') == 'Boolean column'
        
        # Test set_column_descriptions
        sample_super_df.set_column_descriptions({
            'mixed_col': 'Mixed types column',
            'date_col': 'Date column'
        })
        assert sample_super_df.get_column_description('mixed_col') == 'Mixed types column'
        assert sample_super_df.get_column_description('date_col') == 'Date column'
        
        # Test error on non-existent column
        with pytest.raises(ValueError):
            sample_super_df.set_column_description('non_existent', 'This should fail')

class TestColumnTypeInference:
    """Test column type inference functionality"""
    
    def test_infer_column_types(self, sample_super_df):
        """Test column type inference"""
        column_types = sample_super_df.column_types
        
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
        updated_types = sample_super_df.refresh_column_types()
        
        assert updated_types['int_col'] == 'float64'
        assert updated_types['new_col'] == 'str'
        assert sample_super_df.column_types == updated_types

class TestSchemaAndLLMFormat:
    """Test schema generation and LLM formatting"""
    
    def test_schema_generation(self, sample_super_df):
        """Test schema generation"""
        schema = sample_super_df.schema()
        
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
        
        schema = sample_super_df.schema(template=template)
        
        assert "Name: Test DataFrame" in schema
        assert "Desc: A test dataframe with various column types" in schema
        assert "Size: (5, 7)" in schema or f"Size: {sample_super_df.shape}" in schema
    
    def test_to_llm_format_json(self, sample_super_df):
        """Test JSON format for LLMs"""
        json_str = sample_super_df.to_llm_format(format_type='json')
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
        md = sample_super_df.to_llm_format(format_type='markdown')
        
        assert "# DataFrame: Test DataFrame" in md
        assert "**Description**: A test dataframe with various column types" in md
        assert "## Columns" in md
        assert "## Data Sample" in md
        assert "int_col" in md
    
    def test_to_llm_format_text(self, sample_super_df):
        """Test text format for LLMs"""
        text = sample_super_df.to_llm_format(format_type='text')
        
        assert "DataFrame: Test DataFrame" in text
        assert "Description: A test dataframe with various column types" in text
        assert "Columns:" in text
        assert "Data Sample:" in text
    
    def test_to_llm_format_invalid(self, sample_super_df):
        """Test invalid format type"""
        with pytest.raises(ValueError):
            sample_super_df.to_llm_format(format_type='invalid_format')

class TestDataFrameOperations:
    """Test pandas operations with SuperDataFrame"""
    
    def test_copy(self, sample_super_df):
        """Test copying a SuperDataFrame"""
        copied = sample_super_df.copy()
        
        assert isinstance(copied, SuperDataFrame)
        assert copied.name == sample_super_df.name
        assert copied.description == sample_super_df.description
        assert copied.column_descriptions == sample_super_df.column_descriptions
        
        # Modify the copy and check that original is unchanged
        copied.name = "Modified Copy"
        assert sample_super_df.name == "Test DataFrame"
    
    def test_basic_operations(self, sample_super_df):
        """Test basic pandas operations preserve SuperDataFrame type"""
        # Test filtering
        filtered = sample_super_df[sample_super_df['int_col'] > 2]
        assert isinstance(filtered, SuperDataFrame)
        assert filtered.name == sample_super_df.name
        
        # Test selecting columns
        selected = sample_super_df[['int_col', 'str_col']]
        assert isinstance(selected, SuperDataFrame)
        assert selected.name == sample_super_df.name
        
        # Test adding a column
        with_new_col = sample_super_df.copy()
        with_new_col['new_col'] = [10, 20, 30, 40, 50]
        assert isinstance(with_new_col, SuperDataFrame)
        assert with_new_col.name == sample_super_df.name
    
    def test_concat(self, sample_super_df):
        """Test concatenation of SuperDataFrames"""
        import pandas as pd
        
        # Create a second dataframe
        df2 = sample_super_df.copy()
        df2.name = "Second DF"
        
        # Concatenate
        result = pd.concat([sample_super_df, df2])
        
        assert isinstance(result, SuperDataFrame)
        # Should inherit metadata from first dataframe
        assert result.name == "Test DataFrame"
        
        # Test with axis=1 (column concat)
        result_cols = pd.concat([sample_super_df, df2], axis=1)
        assert isinstance(result_cols, SuperDataFrame)
    
    def test_merge(self, sample_super_df):
        """Test merging SuperDataFrames"""
        # Create a second dataframe
        df2 = SuperDataFrame({
            'int_col': [1, 2, 3],
            'extra_col': ['x', 'y', 'z']
        }, name="Second DF")
        
        # Merge
        result = sample_super_df.merge(df2, on='int_col')
        
        assert isinstance(result, SuperDataFrame)
        # Should inherit metadata from left dataframe
        assert result.name == "Test DataFrame" 