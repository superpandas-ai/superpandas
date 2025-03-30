import warnings
import pandas as pd
import json
import yaml
from typing import Dict, Literal, Optional, Union, Type
from smolagents import Model

@pd.api.extensions.register_dataframe_accessor("super")
class SuperDataFrameAccessor:
    """
    A pandas DataFrame accessor that adds metadata capabilities:
    - dataframe name
    - dataframe description
    - column descriptions
    - column data types (with refined types for object columns)
    - schema serialization for LLMs
    """
    
    def __init__(self, pandas_obj):
        """Initialize the accessor with a pandas DataFrame"""
        self._obj = pandas_obj
        self._validate(pandas_obj)
        self._initialize_metadata()

    def _validate(self, obj):
        """Validate the pandas object"""
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Can only use .super accessor with pandas DataFrame objects")

    def _initialize_metadata(self):
        """Initialize metadata in DataFrame.attrs if not present"""
        if 'super' not in self._obj.attrs:
            self._obj.attrs['super'] = {
                'name': '',
                'description': '',
                'column_descriptions': {},
                'column_types': {}
            }
        self._infer_column_types()

    def _infer_column_types(self):
        """Infer detailed column types, especially for object columns"""
        column_types = {}
        for col in self._obj.columns:
            pandas_dtype = self._obj.dtypes[col]
            
            # For object dtype, try to determine more specific type
            if str(pandas_dtype) == 'object':
                column_types[col] = self._infer_object_column_type(col)
            else:
                column_types[col] = str(pandas_dtype)
        
        self._obj.attrs['super']['column_types'] = column_types

    def _infer_object_column_type(self, column: str) -> str:
        """
        Infer the actual data type of an object column by sampling values.
        
        Returns a string description of the inferred type.
        """
        # Skip empty columns
        if self._obj[column].isna().all():
            return 'empty'
        
        # Get non-null values for sampling
        non_null_values = self._obj[column].dropna()
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
    def name(self) -> str:
        """Get the dataframe name"""
        return self._obj.attrs['super']['name']
    
    @name.setter
    def name(self, value: str):
        """Set the dataframe name"""
        self._obj.attrs['super']['name'] = value

    @property
    def description(self) -> str:
        """Get the dataframe description"""
        return self._obj.attrs['super']['description']
    
    @description.setter
    def description(self, value: str):
        """Set the dataframe description"""
        self._obj.attrs['super']['description'] = value

    @property
    def column_descriptions(self) -> Dict[str, str]:
        """Get all column descriptions"""
        return self._obj.attrs['super']['column_descriptions']

    @property
    def column_types(self) -> Dict[str, str]:
        """Get refined column data types"""
        return self._obj.attrs['super']['column_types']

    def refresh_column_types(self):
        """Refresh the inferred column types"""
        self._infer_column_types()
        return self._obj.attrs['super']['column_types']

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
            - {columns}: The formatted column information, including refined type and description
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
        for col in self._obj.columns:
            refined_type = self.column_types.get(col, str(self._obj.dtypes[col]))
            desc = self.column_descriptions.get(col, '')
            columns_info.append(f"- {col} ({refined_type}): {desc}")
        
        columns_str = "\n".join(columns_info)
        
        # Format the schema using the template
        schema_str = template.format(
            name=self.name,
            description=self.description,
            columns=columns_str,
            dtypes=str(self._obj.dtypes),
            shape=str(self._obj.shape)
        )
        
        return schema_str

    def to_llm_format(self, format_type: str = 'json', max_rows: int = 5) -> str:
        """
        Convert the dataframe to a format suitable for LLM consumption.
        
        Parameters:
        -----------
        format_type : str, default 'json'
            The format to convert to ('json', 'markdown', 'text', 'yaml')
        max_rows : int, default 5
            Maximum number of rows to include
        
        Returns:
        --------
        str
            The formatted representation
        """
        # Sample the dataframe if needed
        sample_df = self._obj.head(max_rows) if len(self._obj) > max_rows else self._obj
        
        if format_type == 'json':
            # Create a dictionary with metadata and data
            result = {
                    'metadata' : {
                    "name": self.name,
                    "description": self.description,
                    "shape": self._obj.shape,
                    "columns": {
                        col: {
                            "pandas_dtype": str(self._obj.dtypes[col]),
                            "refined_type": self.column_types.get(col, str(self._obj.dtypes[col])),
                            "description": self.column_descriptions.get(col, "")
                        } for col in self._obj.columns
                    }
                }
            }
            
            if max_rows > 0:
                result["data"] = json.loads(pd.DataFrame(sample_df).reset_index(drop=True).to_json(orient='records', date_format='iso'))
            return json.dumps(result, indent=2)
            
        elif format_type == 'markdown':
            # Create a markdown representation
            md = f"# DataFrame: {self.name}\n\n"
            if self.description:
                md += f"**Description**: {self.description}\n\n"
            md += f"Shape: {self._obj.shape}\n\n"
            
            # Add column descriptions
            md += "## Columns\n\n"
            for col in self._obj.columns:
                refined_type = self.column_types.get(col, str(self._obj.dtypes[col]))
                desc = self.column_descriptions.get(col, "")
                md += f"- **{col}** ({refined_type}): {desc}\n"
            
            md += "\n## Data Sample\n\n"
            md += sample_df.to_markdown()
            return md
            
        elif format_type == 'text':
            # Create a plain text representation
            text = f"DataFrame: {self.name}\n"
            if self.description:
                text += f"Description: {self.description}\n"
            text += f"\nShape: {self._obj.shape}\n\n"
            
            text += "Columns:\n"
            for col in self._obj.columns:
                refined_type = self.column_types.get(col, str(self._obj.dtypes[col]))
                desc = self.column_descriptions.get(col, "")
                text += f"- {col} ({refined_type}): {desc}\n"
            
            text += "\nData Sample:\n"
            text += sample_df.to_string()
            return text

        elif format_type == 'yaml':
            # Create a dictionary with metadata and data
            result = {
                'metadata': {
                    "name": self.name,
                    "description": self.description,
                    "shape": self._obj.shape,
                    "columns": {
                        col: {
                            "pandas_dtype": str(self._obj.dtypes[col]),
                            "refined_type": self.column_types.get(col, str(self._obj.dtypes[col])),
                            "description": self.column_descriptions.get(col, "")
                        } for col in self._obj.columns
                    }
                }
            }
            
            if max_rows > 0:
                result["data"] = json.loads(pd.DataFrame(sample_df).reset_index(drop=True).to_json(orient='records', date_format='iso'))
            
            return yaml.dump(result, default_flow_style=False)
            
        else:
            raise ValueError(f"Unsupported format type: {format_type}.")

    def get_column_description(self, column: str) -> str:
        """Get the description for a specific column"""
        return self.column_descriptions.get(column, '')
    
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
        if column not in self._obj.columns:
            if errors == 'raise':
                raise ValueError(f"Column '{column}' does not exist in the dataframe")
            elif errors == 'ignore':
                return
            elif errors == 'warn':
                warnings.warn(f"Column '{column}' does not exist in the dataframe")     
        self._obj.attrs['super']['column_descriptions'][column] = description
    
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
            if column not in self._obj.columns:
                if errors == 'raise':
                    raise ValueError(f"Column '{column}' does not exist in the dataframe")
                elif errors == 'ignore':
                    continue
                elif errors == 'warn':
                    warnings.warn(f"Column '{column}' does not exist in the dataframe")

            self._obj.attrs['super']['column_descriptions'][column] = description
    
    def __finalize__(self, other, method=None, **kwargs):
        """
        Propagate metadata from other to self after operations
        """
        if other is None:
            return self._obj
            
        # Copy metadata if other is a SuperDataFrameAccessor
        if isinstance(other, SuperDataFrameAccessor):
            self._obj.attrs['super']['name'] = getattr(other, '_obj').attrs['super']['name']
            self._obj.attrs['super']['description'] = getattr(other, '_obj').attrs['super']['description']
            self._obj.attrs['super']['column_descriptions'] = getattr(other, '_obj').attrs['super']['column_descriptions'].copy()
            self._obj.attrs['super']['column_types'] = getattr(other, '_obj').attrs['super']['column_types'].copy()
            
        # Special handling for concat/merge
        elif method in ['merge', 'concat']:
            # Try to find first SuperDataFrameAccessor in other's objects
            if hasattr(other, 'objs'):  # concat case
                for obj in other.objs:
                    if isinstance(obj, SuperDataFrameAccessor):
                        self._obj.attrs['super']['name'] = getattr(obj, '_obj').attrs['super']['name']
                        self._obj.attrs['super']['description'] = getattr(obj, '_obj').attrs['super']['description']
                        self._obj.attrs['super']['column_descriptions'] = getattr(obj, '_obj').attrs['super']['column_descriptions'].copy()
                        self._obj.attrs['super']['column_types'] = getattr(obj, '_obj').attrs['super']['column_types'].copy()
                        break
            elif hasattr(other, 'left'):  # merge case
                if isinstance(other.left, SuperDataFrameAccessor):
                    self._obj.attrs['super']['name'] = getattr(other.left, '_obj').attrs['super']['name']
                    self._obj.attrs['super']['description'] = getattr(other.left, '_obj').attrs['super']['description']
                    self._obj.attrs['super']['column_descriptions'] = getattr(other.left, '_obj').attrs['super']['column_descriptions'].copy()
                    self._obj.attrs['super']['column_types'] = getattr(other.left, '_obj').attrs['super']['column_types'].copy()
                
        # Refresh column types after operations that might change the structure
        if method in ['merge', 'concat', 'join']:
            self._infer_column_types()
                
        return self._obj

    def copy(self, deep=True):
        """
        Make a copy of this object's indices and data.
        """
        data = self._obj.copy(deep=deep)
        if deep:
            data.attrs['super']['name'] = self._obj.attrs['super']['name']
            data.attrs['super']['description'] = self._obj.attrs['super']['description']
            data.attrs['super']['column_descriptions'] = self._obj.attrs['super']['column_descriptions'].copy()
            data.attrs['super']['column_types'] = self._obj.attrs['super']['column_types'].copy()
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
        
        # Create new SuperDataFrameAccessor
        result = cls(df)
        
        # Copy metadata from first SuperDataFrameAccessor if available
        for obj in objs:
            if isinstance(obj, cls):
                result._obj.attrs['super']['name'] = getattr(obj, '_obj').attrs['super']['name']
                result._obj.attrs['super']['description'] = getattr(obj, '_obj').attrs['super']['description']
                result._obj.attrs['super']['column_descriptions'] = getattr(obj, '_obj').attrs['super']['column_descriptions'].copy()
                result._obj.attrs['super']['column_types'] = getattr(obj, '_obj').attrs['super']['column_types'].copy()
                break
        
        # Refresh column types after concatenation
        result._infer_column_types()
                
        return result
    
    def to_pickle(self, path: str):
        """Save DataFrame to pickle with metadata preserved"""
        # Save DataFrame with metadata in attrs
        self._obj.to_pickle(path)

    def read_pickle(self, path: str):
        """Read DataFrame from pickle with metadata"""
        df = pd.read_pickle(path)
        if isinstance(df, pd.DataFrame):
            if 'super' in df.attrs:
                self._obj.attrs['super'] = df.attrs['super']
            else:
                self._initialize_metadata()
        elif isinstance(df, dict) and 'dataframe' in df and 'metadata' in df:
            # Handle legacy format where data was stored as dict
            self._obj = df['dataframe']
            self._obj.attrs['super'] = df['metadata']
        else:
            self._initialize_metadata()

    def to_csv(self, path, include_metadata: bool = True, **kwargs):
        """
        Save DataFrame to CSV with optional metadata in a companion file.
        
        Parameters:
        -----------
        path : str
            Path to save the CSV file
        include_metadata : bool, default True
            If True, saves metadata to a companion JSON file
        **kwargs : dict
            Additional arguments passed to pandas to_csv method
        """
        # Save the data as CSV
        self._obj.to_csv(path, **kwargs)
        
        # If requested, save metadata separately
        if include_metadata:
            path_str = str(path)  # Convert Path object to string
            metadata_path = path_str.rsplit('.', 1)[0] + '_metadata.json'
            metadata = {
                'name': self._obj.attrs['super']['name'],
                'description': self._obj.attrs['super']['description'],
                'column_descriptions': self._obj.attrs['super']['column_descriptions'],
                'column_types': self._obj.attrs['super']['column_types'],
                'dtypes': {col: str(dtype) for col, dtype in self._obj.dtypes.items()}
            }
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)

    def read_metadata(self, csv_path: str):
        """
        Read metadata from a companion JSON file for a CSV file.
        
        Parameters:
        -----------
        csv_path : str
            Path to the CSV file (metadata file path will be derived from this)
        """
        path_str = str(csv_path)
        metadata_path = path_str.rsplit('.', 1)[0] + '_metadata.json'
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                self._obj.attrs['super'] = {
                    'name': metadata.get('name', ''),
                    'description': metadata.get('description', ''),
                    'column_descriptions': metadata.get('column_descriptions', {}),
                    'column_types': metadata.get('column_types', {})
                }
        except FileNotFoundError:
            # Initialize empty metadata if no metadata file exists
            self._obj.attrs['super'] = {
                'name': '',
                'description': '',
                'column_descriptions': {},
                'column_types': {}
            }
            self._infer_column_types()

    def auto_describe(self,
                     model: Optional[Union[str, Model]] = None,
                     provider_class: Optional[Type[Model]] = None,
                     generate_name: bool = True,
                     generate_description: bool = True,
                     generate_column_descriptions: bool = True,
                     existing_values: Optional[Literal['warn', 'skip', 'overwrite']] = None,
                     **model_kwargs) -> pd.DataFrame:
        """
        Automatically generate descriptions for this DataFrame using LLMs.
        
        Args:
            model: Model name or instance of Model class. If None, uses the model from config
            provider_class: Class to use for model provider. If None, uses the provider from config
            generate_name: Whether to generate DataFrame name
            generate_description: Whether to generate overall DataFrame description
            generate_column_descriptions: Whether to generate column descriptions
            existing_values: How to handle existing metadata values. If None, uses value from config:
                - 'warn': Warn and skip if value exists
                - 'skip': Silently skip if value exists
                - 'overwrite': Replace existing values
            **model_kwargs: Additional arguments to pass to the model provider
        
        Returns:
            self._obj: The DataFrame with updated descriptions
            
        Example:
            >>> df = pd.DataFrame(...)
            >>> df.super.auto_describe()  # Uses config settings
            >>> df.super.auto_describe(model="gpt-4")  # Uses specific model
            >>> # Skip existing values without warning
            >>> df.super.auto_describe(existing_values='skip')
            >>> # Overwrite any existing values
            >>> df.super.auto_describe(existing_values='overwrite')
        """
        from .llm_client import LLMClient
        from .config import config
        
        # Use config values if no explicit values provided
        if model is None:
            model = config.llm_model
        if provider_class is None:
            provider_class = config.llm_provider_class
        
        # Merge config kwargs with provided kwargs, handling existing_values specially
        merged_kwargs = {**config.llm_kwargs, **model_kwargs}
        if existing_values is not None:
            merged_kwargs['existing_values'] = existing_values
        existing_values = merged_kwargs.pop('existing_values', 'warn')
        
        llm_client = LLMClient(model=model, provider_class=provider_class, **merged_kwargs)
        
        if generate_name:
            if self.name and existing_values != 'overwrite':
                if existing_values == 'warn':
                    warnings.warn(f"DataFrame already has a name: '{self.name}'. Skipping name generation.")
            else:
                self.name = llm_client.generate_df_name(self._obj)
        
        if generate_description:
            if self.description and existing_values != 'overwrite':
                if existing_values == 'warn':
                    warnings.warn("DataFrame already has a description. Skipping description generation.")
            else:
                self.description = llm_client.generate_df_description(self._obj)
        
        if generate_column_descriptions:
            existing_cols = {col: desc for col, desc in self.column_descriptions.items() if desc}
            if existing_cols and existing_values != 'overwrite':
                if existing_values == 'warn':
                    warnings.warn(f"Some columns already have descriptions: {list(existing_cols.keys())}. "
                                "Skipping those columns.")
                # Only generate descriptions for columns without existing descriptions
                cols_to_describe = set(self._obj.columns) - set(existing_cols.keys())
                if cols_to_describe:
                    new_descriptions = llm_client.generate_column_descriptions(self._obj)
                    # Merge new descriptions with existing ones
                    merged_descriptions = {**existing_cols, 
                                        **{k: v for k, v in new_descriptions.items() 
                                           if k in cols_to_describe}}
                    self.set_column_descriptions(merged_descriptions)
            else:
                # Generate descriptions for all columns
                column_descriptions = llm_client.generate_column_descriptions(self._obj)
                self.set_column_descriptions(column_descriptions)
        
        return self._obj

    def __str__(self):
        return (f"SuperDataFrameAccessor("
                f"name={self.name}, "
                f"description={self.description}, "
                f"column_descriptions={self.column_descriptions}, "
                f"column_types={self.column_types})")

    def __repr__(self):
        return self.__str__()

    def query(self, question: str, schema: Optional[str] = None, system_template: Optional[str] = None, user_template: Optional[str] = None) -> str:
        """
        Query the DataFrame using an LLM with a given question and optional templates.
        
        Parameters:
        -----------
        question : str
            The question to ask about the DataFrame.
        schema : str, optional
            The schema of the DataFrame.
        system_template : str, optional
            A system template string for formatting the query message.
        user_template : str, optional
            A user template string for formatting the query message.
        
        Returns:
        --------
        str
            The response from the LLM.
        """
        from .llm_client import LLMClient
        from .config import config

        # Use the system template from config if no system template is provided
        if system_template is None:
            system_template = config.system_template

        # Use the user template from config if no user template is provided
        if user_template is None:
            from .templates import user_template as default_user_template
            user_template = default_user_template

        # Initialize the LLM client
        llm_client = LLMClient(model=config.llm_model, provider_class=config.llm_provider_class, **config.llm_kwargs)

        # Format the message using the provided templates
        system_message = system_template.format(schema=schema)
        user_message = user_template.format(schema=schema, question=question)

        # Query the LLM client
        response = llm_client.query(system_message=system_message, user_message=user_message)

        return response

def read_csv(path: str, require_metadata: bool = True, **kwargs) -> pd.DataFrame:
    """
    Read a CSV file into a DataFrame with super accessor metadata.
    
    Parameters:
    -----------
    path : str
        Path to the CSV file
    require_metadata : bool, default True
        If True, raises FileNotFoundError when metadata file is not found
        If False, initializes empty metadata when metadata file is not found
    **kwargs : dict
        Additional arguments passed to pandas read_csv method
    Returns:
    --------
    pd.DataFrame
        DataFrame with initialized super accessor metadata
        
    Raises:
    -------
    FileNotFoundError
        If the CSV file is not found, or if require_metadata=True and metadata file is not found
        
    Examples:
    --------
    >>> import superpandas as spd
    >>> # Read CSV with metadata (will raise error if metadata file not found)
    >>> df = spd.read_csv('data.csv')
    >>> 
    >>> # Read CSV without requiring metadata
    >>> df = spd.read_csv('data.csv', require_metadata=False)
    >>> 
    >>> # Pass pandas read_csv arguments
    >>> df = spd.read_csv('data.csv', index_col=0, parse_dates=['date_column'])
    """
    # Try to read metadata
    path_str = str(path)
    metadata_path = path_str.rsplit('.', 1)[0] + '_metadata.json'

    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            # Add parse_dates for datetime columns based on metadata
            if 'dtypes' in metadata:
                date_columns = [col for col, dtype in metadata['dtypes'].items() 
                              if 'datetime' in dtype.lower()]
                if date_columns:
                    kwargs.setdefault('parse_dates', date_columns)
    except FileNotFoundError:
        if require_metadata:
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    # Read the CSV data
    df = pd.read_csv(path, **kwargs)
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            df.attrs['super'] = {
                'name': metadata.get('name', ''),
                'description': metadata.get('description', ''),
                'column_descriptions': metadata.get('column_descriptions', {}),
                'column_types': metadata.get('column_types', {})
            }
    except FileNotFoundError:
        if require_metadata:
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        else:
            # Initialize empty metadata if metadata file doesn't exist
            df.attrs['super'] = {
                'name': '',
                'description': '',
                'column_descriptions': {},
                'column_types': {}
            }
            df.super._infer_column_types()
            
    return df

# Helper function to create a SuperDataFrameAccessor
def create_super_dataframe(*args, **kwargs) -> pd.DataFrame:
    """
    Create a DataFrame with initialized super accessor metadata.
    
    This is a convenience function that works like pd.DataFrame() but initializes
    the super accessor metadata.
    """
    name = kwargs.pop('name', '')
    description = kwargs.pop('description', '')
    column_descriptions = kwargs.pop('column_descriptions', {})
    
    df = pd.DataFrame(*args, **kwargs)
    df.attrs['super'] = {
        'name': name,
        'description': description,
        'column_descriptions': column_descriptions,
        'column_types': {}
    }
    df.super._infer_column_types()
    return df 