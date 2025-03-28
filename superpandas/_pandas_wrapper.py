import pandas as pd
from typing import Optional, Dict, Any
from .superdataframe import SuperDataFrame, create_super_dataframe

def _wrap_pandas_reader(func):
    """
    Wraps pandas reader functions to return SuperDataFrame
    """
    def wrapper(*args, name: str = "", description: str = "", column_descriptions: Dict[str, str] = None, **kwargs):
        df = func(*args, **kwargs)
        return SuperDataFrame.from_pandas(
            df,
            name=name,
            description=description,
            column_descriptions=column_descriptions or {}
        )
    return wrapper

# Wrap common pandas reader functions
read_csv = _wrap_pandas_reader(pd.read_csv)
read_excel = _wrap_pandas_reader(pd.read_excel)
read_json = _wrap_pandas_reader(pd.read_json)
read_sql = _wrap_pandas_reader(pd.read_sql)
read_parquet = _wrap_pandas_reader(pd.read_parquet)

# def read_csv(*args, **kwargs) -> SuperDataFrame:
#     """
#     Read a CSV file into a SuperDataFrame.
    
#     This is a wrapper around pd.read_csv that returns a SuperDataFrame.
#     All arguments are passed to pd.read_csv.
#     """
#     # Extract SuperDataFrame-specific kwargs
#     name = kwargs.pop('name', '')
#     description = kwargs.pop('description', '')
#     column_descriptions = kwargs.pop('column_descriptions', {})
    
#     # Read the CSV using pandas
#     df = pd.read_csv(*args, **kwargs)
    
#     # Convert to SuperDataFrame
#     return SuperDataFrame(
#         df, 
#         name=name,
#         description=description,
#         column_descriptions=column_descriptions
#     )


# def read_excel(*args, **kwargs) -> SuperDataFrame:
#     """
#     Read an Excel file into a SuperDataFrame.
    
#     This is a wrapper around pd.read_excel that returns a SuperDataFrame.
#     All arguments are passed to pd.read_excel.
#     """
#     # Extract SuperDataFrame-specific kwargs
#     name = kwargs.pop('name', '')
#     description = kwargs.pop('description', '')
#     column_descriptions = kwargs.pop('column_descriptions', {})
    
#     # Read the Excel file using pandas
#     df = pd.read_excel(*args, **kwargs)
    
#     # Convert to SuperDataFrame
#     return SuperDataFrame(
#         df, 
#         name=name,
#         description=description,
#         column_descriptions=column_descriptions
#     )


# def read_sql(*args, **kwargs) -> SuperDataFrame:
#     """
#     Read a SQL query or database table into a SuperDataFrame.
    
#     This is a wrapper around pd.read_sql that returns a SuperDataFrame.
#     All arguments are passed to pd.read_sql.
#     """
#     # Extract SuperDataFrame-specific kwargs
#     name = kwargs.pop('name', '')
#     description = kwargs.pop('description', '')
#     column_descriptions = kwargs.pop('column_descriptions', {})
    
#     # Read the SQL data using pandas
#     df = pd.read_sql(*args, **kwargs)
    
#     # Convert to SuperDataFrame
#     return SuperDataFrame(
#         df, 
#         name=name,
#         description=description,
#         column_descriptions=column_descriptions
#     )


# Add more pandas I/O functions as needed 