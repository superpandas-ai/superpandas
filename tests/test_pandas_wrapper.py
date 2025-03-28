import pytest
import pandas as pd
import os
from superpandas import (
    SuperDataFrame, read_csv, read_excel, read_json, read_sql, read_parquet
)

class TestPandasWrappers:
    """Test pandas I/O wrapper functions"""
    
    def test_read_csv(self, titanic_csv_path):
        """Test read_csv wrapper"""
        # Basic read
        df = read_csv(titanic_csv_path)
        assert isinstance(df, SuperDataFrame)
        
        # With metadata
        df = read_csv(
            titanic_csv_path,
            name="Titanic Dataset",
            description="Passenger data from the Titanic",
            column_descriptions={
                'Survived': 'Whether the passenger survived (1) or not (0)'
            }
        )
        
        assert df.name == "Titanic Dataset"
        assert df.description == "Passenger data from the Titanic"
        assert df.get_column_description('Survived') == 'Whether the passenger survived (1) or not (0)'
        
        # With pandas options
        df = read_csv(titanic_csv_path, usecols=['PassengerId', 'Survived', 'Name'])
        assert list(df.columns) == ['PassengerId', 'Survived', 'Name']
    
    def test_read_excel(self, tmp_path):
        """Test read_excel wrapper"""
        # Create a temporary Excel file
        excel_path = os.path.join(tmp_path, "test.xlsx")
        pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z']
        }).to_excel(excel_path, index=False)
        
        # Read it
        df = read_excel(
            excel_path,
            name="Excel Test",
            description="Test Excel file"
        )
        
        assert isinstance(df, SuperDataFrame)
        assert df.name == "Excel Test"
        assert df.description == "Test Excel file"
        assert list(df.columns) == ['A', 'B']
    
    def test_read_json(self, tmp_path):
        """Test read_json wrapper"""
        # Create a temporary JSON file
        json_path = os.path.join(tmp_path, "test.json")
        pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z']
        }).to_json(json_path, orient='records')
        
        # Read it
        df = read_json(
            json_path,
            name="JSON Test",
            description="Test JSON file"
        )
        
        assert isinstance(df, SuperDataFrame)
        assert df.name == "JSON Test"
        assert df.description == "Test JSON file"
        assert list(df.columns) == ['A', 'B']
    
    @pytest.mark.skip(reason="Requires database connection")
    def test_read_sql(self):
        """Test read_sql wrapper - skipped by default as it requires a DB"""
        # This would need a database connection to test
        pass
    
    def test_read_parquet(self, tmp_path):
        """Test read_parquet wrapper"""
        # Create a temporary parquet file
        parquet_path = os.path.join(tmp_path, "test.parquet")
        pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z']
        }).to_parquet(parquet_path)
        
        # Read it
        df = read_parquet(
            parquet_path,
            name="Parquet Test",
            description="Test Parquet file"
        )
        
        assert isinstance(df, SuperDataFrame)
        assert df.name == "Parquet Test"
        assert df.description == "Test Parquet file"
        assert list(df.columns) == ['A', 'B'] 