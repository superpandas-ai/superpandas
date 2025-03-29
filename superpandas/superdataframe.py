import warnings
import pandas as pd
import json
from typing import Dict, Literal, Optional


class SuperDataFrame(pd.DataFrame):
    """
    An extension of pandas DataFrame with additional metadata capabilities:
    - dataframe name
    - dataframe description
    - column descriptions
    - column data types (with refined types for object columns)
    - schema serialization for LLMs
    """
    
    _metadata = ['_df_name', '_df_description', '_column_descrSiptions', '_column_types'] + pd.DataFrame._metadata
    
    def __init__(self, *args, **kwargs):
        """
        Initialize a SuperDataFrame with optional name, description and column descriptions.
        
        Parameters:
        -----------
        name : str, optional
            A name for the dataframe
        description : str, optional
            A description of the dataframe
        column_descriptions : dict, optional
            A dictionary mapping column names to their descriptions
        *args, **kwargs : passed to pandas DataFrame constructor
        """
        name = kwargs.pop('name', '')
        description = kwargs.pop('description', '')
        column_descriptions = kwargs.pop('column_descriptions', {})
        
        # Initialize the pandas DataFrame
        super().__init__(*args, **kwargs)
        
        # Set the additional attributes
        self._df_name = name
        self._df_description = description
        self._column_descriptions = column_descriptions
        self._column_types = {}
        
        # Infer column types
        self._infer_column_types()
    
    def _infer_column_types(self):
        """Infer detailed column types, especially for object columns"""
        for col in self.columns:
            pandas_dtype = self.dtypes[col]
            
            # For object dtype, try to determine more specific type
            if str(pandas_dtype) == 'object':
                self._column_types[col] = self._infer_object_column_type(col)
            else:
                self._column_types[col] = str(pandas_dtype)
    
    def _infer_object_column_type(self, column: str) -> str:
        """
        Infer the actual data type of an object column by sampling values.
        
        Returns a string description of the inferred type.
        """
        # Skip empty columns
        if self[column].isna().all():
            return 'empty'
        
        # Get non-null values for sampling
        non_null_values = self[column].dropna()
        if len(non_null_values) == 0:
            return 'empty'
            
        # Sample values (all if less than 100, otherwise sample 100)
        sample_size = min(100, len(non_null_values))
        sample = non_null_values.sample(sample_size) if len(non_null_values) > sample_size else non_null_values
        
        # Check types of sampled values
        types = set(type(x).__name__ for x in sample)
        
        # Check if all values are of the same type
        if len(types) == 1:
            return next(iter(types))
        
        # Check for common mixed types
        if types.issubset({'str', 'int', 'float', 'bool'}):
            if types == {'int', 'float'}:
                return 'numeric'
            if 'str' in types:
                return 'mixed_with_text'
        
        # If we have multiple types, return them joined
        return f"mixed({', '.join(sorted(types))})"
    
    @property
    def _constructor(self):
        """Return the constructor for this class"""
        return SuperDataFrame
    
    @property
    def name(self) -> str:
        """Get the dataframe name"""
        return self._df_name
    
    @name.setter
    def name(self, value: str):
        """Set the dataframe name"""
        self._df_name = value
    
    @property
    def description(self) -> str:
        """Get the dataframe description"""
        return self._df_description
    
    @description.setter
    def description(self, value: str):
        """Set the dataframe description"""
        self._df_description = value
    
    @property
    def column_descriptions(self) -> Dict[str, str]:
        """Get all column descriptions"""
        return self._column_descriptions
    
    @property
    def column_types(self) -> Dict[str, str]:
        """Get refined column data types"""
        return self._column_types
    
    def refresh_column_types(self):
        """Refresh the inferred column types"""
        self._infer_column_types()
        return self._column_types
    
    def get_column_description(self, column: str) -> str:
        """Get the description for a specific column"""
        return self._column_descriptions.get(column, '')
    
    def set_column_description(self, column: str, description: str, errors: Literal['raise', 'ignore', 'warn'] = 'raise'):
        """Set the description for a specific column with error handling options
        
        Parameters:
        -----------
        column : str
            The name of the column to set the description for
        description : str
            The description to set for the column
        errors : str, default 'raise'
            The error handling option ('raise', 'ignore', 'warn')
        """
        if column not in self.columns:
            if errors == 'raise':
                raise ValueError(f"Column '{column}' does not exist in the dataframe")
            elif errors == 'ignore':
                return
            elif errors == 'warn':
                warnings.warn(f"Column '{column}' does not exist in the dataframe")
        self._column_descriptions[column] = description
    
    def set_column_descriptions(self, descriptions: Dict[str, str], errors: Literal['raise', 'ignore', 'warn'] = 'raise'):
        """Set descriptions for multiple columns at once with error handling options
        
        Parameters:
        -----------
        descriptions : dict
            A dictionary mapping column names to their descriptions
        errors : str, default 'raise'
            The error handling option ('raise', 'ignore', 'warn')
        """
        for column, description in descriptions.items():
            if column not in self.columns:
                if errors == 'raise':
                    raise ValueError(f"Column '{column}' does not exist in the dataframe")
                elif errors == 'ignore':
                    continue
                elif errors == 'warn':
                    warnings.warn(f"Column '{column}' does not exist in the dataframe")

            self._column_descriptions[column] = description
    
    def schema(self, template: Optional[str] = None) -> str:
        """
        Generate a schema representation of the dataframe for use with LLMs.
        
        Parameters:
        -----------
        template : str, optional
            A template string with placeholders for formatting the schema.
            Available placeholders:
            - {name}: The dataframe name
            - {description}: The dataframe description
            - {columns}: The formatted column information
            - {dtypes}: The dataframe dtypes
            - {shape}: The dataframe shape
            
        Returns:
        --------
        str
            A formatted schema representation
        """
        if template is None:
            template = """
DataFrame Name: {name}
DataFrame Description: {description}

Shape: {shape}

Columns:
{columns}
"""
        
        # Format column information
        columns_info = []
        for col in self.columns:
            refined_type = self._column_types.get(col, str(self.dtypes[col]))
            desc = self._column_descriptions.get(col, '')
            columns_info.append(f"- {col} ({refined_type}): {desc}")
        
        columns_str = "\n".join(columns_info)
        
        # Format the schema using the template
        schema_str = template.format(
            name=self._df_name,
            description=self._df_description,
            columns=columns_str,
            dtypes=str(self.dtypes),
            shape=str(self.shape)
        )
        
        return schema_str
    
    def to_llm_format(self, format_type: str = 'json', max_rows: int = 5) -> str:
        """
        Convert the dataframe to a format suitable for LLM consumption.
        
        Parameters:
        -----------
        format_type : str, default 'json'
            The format to convert to ('json', 'markdown', 'text')
        max_rows : int, default 5
            Maximum number of rows to include
        
        Returns:
        --------
        str
            The formatted representation
        """
        # Sample the dataframe if needed
        sample_df = self.head(max_rows) if len(self) > max_rows else self
        
        if format_type == 'json':
            # Create a dictionary with metadata and data
            result = {
                "metadata": {
                    "name": self._df_name,
                    "description": self._df_description,
                    "shape": self.shape,
                    "columns": {
                        col: {
                            "pandas_dtype": str(self.dtypes[col]),
                            "refined_type": self._column_types.get(col, str(self.dtypes[col])),
                            "description": self._column_descriptions.get(col, "")
                        } for col in self.columns
                    }
                },
                # Convert to records using pandas to_json
                "data": json.loads(pd.DataFrame(sample_df).reset_index(drop=True).to_json(orient='records', date_format='iso'))
            }
            return json.dumps(result, indent=2)
            
        elif format_type == 'markdown':
            # Create a markdown representation
            md = f"# DataFrame: {self._df_name}\n\n"
            if self._df_description:
                md += f"**Description**: {self._df_description}\n\n"
            md += f"Shape: {self.shape}\n\n"
            
            # Add column descriptions
            md += "## Columns\n\n"
            for col in self.columns:
                refined_type = self._column_types.get(col, str(self.dtypes[col]))
                desc = self._column_descriptions.get(col, "")
                md += f"- **{col}** ({refined_type}): {desc}\n"
            
            md += "\n## Data Sample\n\n"
            md += sample_df.to_markdown()
            return md
            
        elif format_type == 'text':
            # Create a plain text representation
            text = f"DataFrame: {self._df_name}\n"
            if self._df_description:
                text += f"Description: {self._df_description}\n"
            text += f"\nShape: {self.shape}\n\n"
            
            text += "Columns:\n"
            for col in self.columns:
                refined_type = self._column_types.get(col, str(self.dtypes[col]))
                desc = self._column_descriptions.get(col, "")
                text += f"- {col} ({refined_type}): {desc}\n"
            
            text += "\nData Sample:\n"
            text += sample_df.to_string()
            return text
            
        else:
            raise ValueError(f"Unsupported format type: {format_type}.")

    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        name: str = "",
        description: str = "",
        column_descriptions: Dict[str, str] = None,
        _is_auto_describing: bool = False,
    ) -> 'SuperDataFrame':
        """
        Create a SuperDataFrame from an existing pandas DataFrame
        
        Args:
            df: Pandas DataFrame to convert
            name: Name of the DataFrame
            description: Description of the DataFrame
            column_descriptions: Dictionary of column descriptions
            _is_auto_describing: Internal flag to indicate auto-describing process
        
        Returns:
            SuperDataFrame with metadata
        """
        result = cls(
            df,
            name=name,
            description=description,
            column_descriptions=column_descriptions or {},
        )
        
        # Ensure column types are inferred
        if not _is_auto_describing:
            result.refresh_column_types()
            
        return result

    def __finalize__(self, other, method=None, **kwargs):
        """
        Propagate metadata from other to self after operations
        """
        super().__finalize__(other, method=method, **kwargs)
        
        # Handle the case where other is None
        if other is None:
            return self
            
        # Copy metadata if other is a SuperDataFrame
        if isinstance(other, SuperDataFrame):
            self._df_name = getattr(other, '_df_name', '')
            self._df_description = getattr(other, '_df_description', '')
            self._column_descriptions = getattr(other, '_column_descriptions', {}).copy()
            self._column_types = getattr(other, '_column_types', {}).copy()
            
        # Special handling for concat/merge
        elif method in ['merge', 'concat']:
            # Try to find first SuperDataFrame in other's objects
            if hasattr(other, 'objs'):  # concat case
                for obj in other.objs:
                    if isinstance(obj, SuperDataFrame):
                        self._df_name = getattr(obj, '_df_name', '')
                        self._df_description = getattr(obj, '_df_description', '')
                        self._column_descriptions = getattr(obj, '_column_descriptions', {}).copy()
                        self._column_types = getattr(obj, '_column_types', {}).copy()
                        break
            elif hasattr(other, 'left'):  # merge case
                if isinstance(other.left, SuperDataFrame):
                    self._df_name = getattr(other.left, '_df_name', '')
                    self._df_description = getattr(other.left, '_df_description', '')
                    self._column_descriptions = getattr(other.left, '_column_descriptions', {}).copy()
                    self._column_types = getattr(other.left, '_column_types', {}).copy()
                
        # Refresh column types after operations that might change the structure
        if method in ['merge', 'concat', 'join']:
            self._infer_column_types()
                
        return self

    def copy(self, deep=True):
        """
        Make a copy of this object's indices and data.
        """
        data = super().copy(deep=deep)
        if deep:
            data._df_name = self._df_name
            data._df_description = self._df_description
            data._column_descriptions = self._column_descriptions.copy()
            data._column_types = self._column_types.copy()
        return data

    @classmethod
    def _concat(cls, objs, **kwargs):
        """
        Concatenate SuperDataFrames preserving metadata from the first object
        """
        # Convert all objects to DataFrame first to avoid pandas bugs
        dfs = [pd.DataFrame(obj) if not isinstance(obj, pd.DataFrame) else obj for obj in objs]
        
        # Do the concatenation
        df = pd.concat(dfs, **kwargs)
        
        # Create new SuperDataFrame
        result = cls(df)
        
        # Copy metadata from first SuperDataFrame if available
        for obj in objs:
            if isinstance(obj, cls):
                result._df_name = getattr(obj, '_df_name', '')
                result._df_description = getattr(obj, '_df_description', '')
                result._column_descriptions = getattr(obj, '_column_descriptions', {}).copy()
                result._column_types = getattr(obj, '_column_types', {}).copy()
                break
        
        # Refresh column types after concatenation
        result._infer_column_types()
                
        return result
    
    def to_pickle(self, path: str):
        """
        Save the SuperDataFrame to a pickle file, preserving all metadata.
        
        Parameters:
        -----------
        path : str
            Path where the pickle file will be saved
        """
        data = {
            'dataframe': pd.DataFrame(self),  # Convert to regular pandas DataFrame
            'metadata': {
                'name': self._df_name,
                'description': self._df_description,
                'column_descriptions': self._column_descriptions,
                'column_types': self._column_types
            }
        }
        pd.to_pickle(data, path)

    @classmethod
    def read_pickle(cls, path: str) -> 'SuperDataFrame':
        """
        Read a SuperDataFrame from a pickle file.
        
        Parameters:
        -----------
        path : str
            Path to the pickle file
        
        Returns:
        --------
        SuperDataFrame
            The loaded SuperDataFrame with all metadata
        """
        data = pd.read_pickle(path)
        
        # Create SuperDataFrame from the stored data
        df = cls(
            data['dataframe'],
            name=data['metadata']['name'],
            description=data['metadata']['description'],
            column_descriptions=data['metadata']['column_descriptions']
        )
        
        # Restore column types
        df._column_types = data['metadata']['column_types']
        
        return df

    def to_json(self, path: str):
        """
        Save the SuperDataFrame to JSON format, preserving all metadata.
        
        Parameters:
        -----------
        path : str
            Path where the JSON file will be saved
        """
        # Convert DataFrame to records with date handling
        records = self.copy()
        for col in records.select_dtypes(include=['datetime64[ns]']).columns:
            records[col] = records[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        data = {
            'dataframe': records.to_dict(orient='records'),
            'metadata': {
                'name': self._df_name,
                'description': self._df_description,
                'column_descriptions': self._column_descriptions,
                'column_types': self._column_types,
                'dtypes': {col: str(dtype) for col, dtype in self.dtypes.items()}
            }
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def read_json(cls, path: str) -> 'SuperDataFrame':
        """
        Read a SuperDataFrame from a JSON file.
        
        Parameters:
        -----------
        path : str
            Path to the JSON file
        
        Returns:
        --------
        SuperDataFrame
            The loaded SuperDataFrame with all metadata
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create DataFrame from the stored records
        df = pd.DataFrame(data['dataframe'])
        
        # Convert to SuperDataFrame with metadata
        result = cls(
            df,
            name=data['metadata']['name'],
            description=data['metadata']['description'],
            column_descriptions=data['metadata']['column_descriptions']
        )
        
        # Restore column types
        result._column_types = data['metadata']['column_types']
        
        return result

    def to_csv(self, path: str, include_metadata: bool = True, **kwargs):
        """
        Save the SuperDataFrame to CSV format with optional metadata.
        
        Parameters:
        -----------
        path : str
            Path where the CSV file will be saved
        include_metadata : bool, default True
            If True, saves an additional JSON file with metadata
        **kwargs : dict
            Additional arguments passed to pandas to_csv method
        """
        # Save the data as CSV
        super().to_csv(path, **kwargs)
        
        # If requested, save metadata separately
        if include_metadata:
            path_str = str(path)  # Convert Path object to string
            metadata_path = path_str.rsplit('.', 1)[0] + '_metadata.json'
            metadata = {
                'name': self._df_name,
                'description': self._df_description,
                'column_descriptions': self._column_descriptions,
                'column_types': self._column_types,
                'dtypes': {col: str(dtype) for col, dtype in self.dtypes.items()}
            }
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)

    @classmethod
    def read_csv(cls, path: str, load_metadata: bool = True, **kwargs) -> 'SuperDataFrame':
        """
        Read a SuperDataFrame from a CSV file with optional metadata.
        
        Parameters:
        -----------
        path : str
            Path to the CSV file
        load_metadata : bool, default True
            If True, loads metadata from the companion JSON file if it exists
        **kwargs : dict
            Additional arguments passed to pandas read_csv method
        
        Returns:
        --------
        SuperDataFrame
            The loaded SuperDataFrame with metadata if available
        """
        # Read the CSV data
        df = pd.read_csv(path, **kwargs)
        
        metadata = {}
        if load_metadata:
            # Convert path to string and handle metadata path
            path_str = str(path)
            metadata_path = path_str.rsplit('.', 1)[0] + '_metadata.json'
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except FileNotFoundError:
                print(f"Metadata file not found at {metadata_path}")
                pass

        # Create SuperDataFrame with metadata if available
        result = cls(
            df,
            name=metadata.get('name', ''),
            description=metadata.get('description', ''),
            column_descriptions=metadata.get('column_descriptions', {})
        )
        
        # Restore column types if available
        if 'column_types' in metadata:
            result._column_types = metadata['column_types']
        else:
            result._infer_column_types()
        
        return result

# Helper function to create a SuperDataFrame
def create_super_dataframe(*args, **kwargs) -> SuperDataFrame:
    """
    Create a SuperDataFrame with the given arguments.
    
    This is a convenience function that works like pd.DataFrame() but returns a SuperDataFrame.
    """
    return SuperDataFrame(*args, **kwargs) 