"""
SQL Accessor for pandas DataFrames

This module provides a SQL accessor that allows you to execute SQL queries
on pandas DataFrames using SQLite as the backend engine.
"""

import pandas as pd
from pandas.io.sql import to_sql, read_sql
from sqlalchemy import create_engine
from sqlalchemy.exc import DatabaseError, ResourceClosedError
from sqlalchemy.pool import NullPool
from contextlib import contextmanager
import re
from warnings import catch_warnings, filterwarnings
from typing import Dict, Optional, Union

@pd.api.extensions.register_dataframe_accessor("sql")
class SQLAccessor:
    """
    A pandas DataFrame accessor that adds SQL query capabilities.
    
    This accessor allows you to execute SQL queries on pandas DataFrames
    using SQLite as the backend engine. The main DataFrame is available
    as table 'df', and additional DataFrames can be provided via the env parameter.
    
    Examples:
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    >>> result = df.sql.query("SELECT * FROM df WHERE x > 1")
    
    >>> # With additional tables
    >>> df2 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    >>> env = {"df2": df2}
    >>> result = df.sql.query("SELECT * FROM df2", env=env)
    """
    
    def __init__(self, pandas_obj):
        """Initialize the accessor with a pandas DataFrame"""
        self._obj = pandas_obj
        self._validate(pandas_obj)

    def _validate(self, obj):
        """Validate the pandas object"""
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Can only use .sql accessor with pandas DataFrame objects")

    def query(self, query: str, db_uri: str = "sqlite:///:memory:", 
              env: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """
        Execute SQL query using the DataFrame as a table named 'df'.
        
        Parameters:
        -----------
        query : str
            The SQL query to execute
        db_uri : str, default "sqlite:///:memory:"
            SQLAlchemy database URI. Defaults to in-memory SQLite database.
        env : dict, optional
            Dictionary mapping table names to DataFrames. These will be available
            as tables in addition to the main DataFrame (which is always 'df').
            
        Returns:
        --------
        pd.DataFrame
            The result of the SQL query
            
        Raises:
        -------
        RuntimeError
            If there's an error executing the SQL query
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")
            
        engine = create_engine(db_uri, poolclass=NullPool)
        with engine.connect() as conn:
            # Write this DF as the 'df' table
            self._write_table(self._obj, "df", conn)
            
            # Write any other tables provided in env
            if env:
                if not isinstance(env, dict):
                    raise TypeError("env must be a dictionary")
                    
                for tbl_name, tbl_df in env.items():
                    if not isinstance(tbl_name, str):
                        raise TypeError("Table names in env must be strings")
                    if not isinstance(tbl_df, pd.DataFrame):
                        raise TypeError("Values in env must be pandas DataFrames")
                    if tbl_name != "df":
                        self._write_table(tbl_df, tbl_name, conn)
                        
            try:
                result = read_sql(query, conn)
                return result
            except DatabaseError as ex:
                raise RuntimeError(f"SQL query failed: {ex}")
            except ResourceClosedError:
                raise RuntimeError("Database connection was closed unexpectedly")
            except Exception as ex:
                raise RuntimeError(f"Unexpected error executing SQL query: {ex}")

    def _write_table(self, df: pd.DataFrame, tablename: str, conn) -> None:
        """
        Write a DataFrame to the database as a table.
        
        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame to write
        tablename : str
            The name of the table in the database
        conn : sqlalchemy.engine.Connection
            The database connection
        """
        with catch_warnings():
            filterwarnings('ignore',
                message=f"The provided table name '{tablename}' is not found exactly as such in the database")
            to_sql(df, name=tablename, con=conn,
                   index=not any(name is None for name in df.index.names))

# Example usage:
if __name__ == "__main__":
    import pandas as pd
    df = pd.DataFrame({"x": range(5), "y": range(5)})
    # Simple query: uses 'df' as table name
    print(df.sql.query("SELECT * FROM df"))
    # With environment
    df2 = pd.DataFrame({"x": range(3), "y": range(3)})
    env = {"df2": df2}
    print(df.sql.query("SELECT * FROM df2", env=env))