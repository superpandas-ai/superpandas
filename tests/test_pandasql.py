"""
Tests for the SQL accessor functionality
"""

import pytest
import pandas as pd
import numpy as np
from superpandas import pandasql


class TestSQLAccessor:
    """Test cases for the SQL accessor"""
    
    def test_sql_accessor_registration(self):
        """Test that the SQL accessor is properly registered"""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        assert hasattr(df, 'sql')
        assert callable(df.sql.query)
    
    def test_basic_query(self):
        """Test basic SQL query functionality"""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        result = df.sql.query("SELECT * FROM df")
        pd.testing.assert_frame_equal(result, df)
    
    def test_filtered_query(self):
        """Test SQL query with WHERE clause"""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        result = df.sql.query("SELECT * FROM df WHERE x > 1")
        expected = pd.DataFrame({"x": [2, 3], "y": [5, 6]})
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))
    
    def test_aggregation_query(self):
        """Test SQL query with aggregation"""
        df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [10, 20, 30, 40]})
        result = df.sql.query("SELECT SUM(x) as sum_x, AVG(y) as avg_y FROM df")
        assert len(result) == 1
        assert result.iloc[0]['sum_x'] == 10
        assert result.iloc[0]['avg_y'] == 25.0
    
    def test_query_with_env(self):
        """Test SQL query with additional tables in env"""
        df1 = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        df2 = pd.DataFrame({"a": [5, 6], "b": [7, 8]})
        env = {"df2": df2}
        
        result = df1.sql.query("SELECT * FROM df2", env=env)
        pd.testing.assert_frame_equal(result, df2)
    
    def test_join_query(self):
        """Test SQL query with JOIN"""
        df1 = pd.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]})
        df2 = pd.DataFrame({"id": [1, 2, 4], "value": [100, 200, 400]})
        env = {"df2": df2}
        
        result = df1.sql.query("""
            SELECT df.id, df.name, df2.value 
            FROM df 
            JOIN df2 ON df.id = df2.id
        """, env=env)
        
        expected = pd.DataFrame({
            "id": [1, 2],
            "name": ["A", "B"],
            "value": [100, 200]
        })
        pd.testing.assert_frame_equal(result, expected)
    
    def test_invalid_query(self):
        """Test that invalid SQL queries raise appropriate errors"""
        df = pd.DataFrame({"x": [1, 2, 3]})
        
        with pytest.raises(RuntimeError):
            df.sql.query("SELECT * FROM nonexistent_table")
    
    def test_empty_query(self):
        """Test that empty queries raise ValueError"""
        df = pd.DataFrame({"x": [1, 2, 3]})
        
        with pytest.raises(ValueError):
            df.sql.query("")
        
        with pytest.raises(ValueError):
            df.sql.query("   ")
    
    def test_invalid_env_type(self):
        """Test that invalid env types raise TypeError"""
        df = pd.DataFrame({"x": [1, 2, 3]})
        
        with pytest.raises(TypeError):
            df.sql.query("SELECT * FROM df", env="not_a_dict")
    
    def test_invalid_env_values(self):
        """Test that invalid env values raise TypeError"""
        df = pd.DataFrame({"x": [1, 2, 3]})
        
        with pytest.raises(TypeError):
            df.sql.query("SELECT * FROM df", env={"table": "not_a_dataframe"})
    
    def test_invalid_table_names(self):
        """Test that invalid table names raise TypeError"""
        df = pd.DataFrame({"x": [1, 2, 3]})
        
        with pytest.raises(TypeError):
            df.sql.query("SELECT * FROM df", env={123: df})
    
    def test_numeric_data(self):
        """Test SQL queries with numeric data"""
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "mixed_col": [1, 2.5, "text"]
        })
        
        result = df.sql.query("SELECT AVG(int_col) as avg_int FROM df")
        assert result.iloc[0]['avg_int'] == 2.0
    
    def test_string_data(self):
        """Test SQL queries with string data"""
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35]
        })
        
        result = df.sql.query("SELECT * FROM df WHERE name LIKE '%a%'")
        assert len(result) == 2  # Alice and Charlie
        assert "Alice" in result['name'].values
        assert "Charlie" in result['name'].values
    
    def test_null_values(self):
        """Test SQL queries with null values"""
        df = pd.DataFrame({
            "x": [1, None, 3],
            "y": [4, 5, None]
        })
        
        result = df.sql.query("SELECT COUNT(*) as count FROM df WHERE x IS NOT NULL")
        assert result.iloc[0]['count'] == 2
    
    def test_custom_database_uri(self):
        """Test using a custom database URI"""
        df = pd.DataFrame({"x": [1, 2, 3]})
        
        # This should work with the default in-memory database
        result = df.sql.query("SELECT * FROM df", db_uri="sqlite:///:memory:")
        pd.testing.assert_frame_equal(result, df)


if __name__ == "__main__":
    pytest.main([__file__])
