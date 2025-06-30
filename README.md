# SuperPandas

<div align="center">
  <img src="https://avatars.githubusercontent.com/u/192782649" alt="SuperPandas Logo" width="200"/>
</div>

<div align="center">
  <strong>⚠️ Beta Version - Breaking Changes Expected</strong><br>
  This package is currently in beta. Please expect errors and breaking changes in future releases.
</div>


### Introduction

SuperPandas is a lightweight Python package that extends the well known Pandas library with functionality for AI-powered data analytics. It is a barebones wrapper using Pandas Dataframe [accessors](https://pandas.pydata.org/docs/development/extending.html) feature to add namespace 'super' which adds metadata and methods to dataframes for use with LLMs. 

## Key Features

### Enhanced DataFrame with Metadata
- **Rich Metadata Support**: Add names, descriptions, and detailed column information to Pandas DataFrames, which allows for better representation of Pandas DataFrames in LLM applications.
- **Automatically generate metadata**: Automatically generate metadata for DataFrames using LLMs.
- **Support for multiple LLM providers** Support for multiple LLM providers using `smolagents` library
- **Intelligent Type Inference**: Automatically extracts detailed column types, especially for object columns. (e.g use str instead of object)
- **Schema Generation**: Generate clear schema representations for using as context in LLM applications.
- **Serialization**: Save and load DataFrames with all metadata intact.
- **Drop-in replacement for Pandas DataFrames**: SuperPandas is a lightweight wrapper around pandas DataFrames, so you can use it as a drop-in replacement for your existing code.
- **Templated Prompt Generation** Easily store and use templates for system and user prompts.

## Installation

You can install SuperPandas using pip:

```bash
pip install superpandas
```

Or install the latest development version from GitHub:

```bash
pip install git+https://github.com/superpandas-ai/superpandas.git
```

## Usage

### Creating SuperDataFrame

```python
import pandas as pd
import numpy as np

# Create a sample DataFrame
df = pd.DataFrame({
    'date': pd.date_range(start='2023-01-01', periods=6, freq='ME'),
    'region': ['North', 'South', 'East', 'West', 'North', 'South'],
    'revenue': np.random.randint(10000, 100000, 6),
    'units_sold': np.random.randint(100, 1000, 6)
})

# Method 1: Create a SuperDataFrame explicitly with metadata
import superpandas as spd

sdf = spd.create_super_dataframe(df,
    name="sales_data",
    description="Monthly sales data by region",
    column_descriptions={
        "revenue": "Monthly revenue in USD",
        "region": "Sales region code"
    }
)

# Access metadata
print(sdf.super.name)  # "sales_data"
print(sdf.super.description) # 'Monthly sales data by region'
print(sdf.super.get_column_descriptions())
print(sdf.super.column_types)  # Shows refined type information
```

```python
# Method 2: Explicitly add metadata
import superpandas # adds 'super' namespace to pandas without changing existing code

# Using df from above
print(df.super.name) # yields empty string
df.super.name = "sales data"
df.super.description = "Monthly sales data by region"
df.super.set_column_descriptions({
    "revenue": "Monthly revenue in USD",
    "region": "Sales region code"
})
print(df.super.name) # prints 'sales data'
print(df.super.description) # 'Monthly sales data by region'
print(df.super.get_column_descriptions())
print(df.super.column_types) # produces a dict of column names and data types ('object' data type is converted to more finegraned datatype)
```

### Core Methods

#### Metadata Management
```python
# Get/Set DataFrame name and description
df.super.name = "my_dataframe"
df.super.description = "Description of my dataframe"

# Get/Set column descriptions
df.super.set_column_description("column_name", "Description of column")
df.super.set_column_descriptions({
    "col1": "Description 1",
    "col2": "Description 2"
}, errors='raise')  # errors can be 'raise', 'ignore', or 'warn'

# Get column information
description = df.super.get_column_description("column_name")
all_descriptions = df.super.get_column_descriptions()
column_types = df.super.column_types

# Refresh column type inference
df.super.refresh_column_types()
```

#### Schema Generation
```python
# Generate schema in different formats
schema = df.super.get_schema(
    template=None,  # Optional custom template
    format_type='text',  # Options: 'json', 'markdown', 'text', 'yaml'
    max_rows=5  # Number of sample rows to include
)

# Custom schema template
template = """
# {name}
{description}

## Data Structure
Rows: {shape[0]}
Columns: {shape[1]}

## Columns
{columns}
"""
schema = df.super.get_schema(template=template)
```

### LLM Integration

SuperPandas supports multiple LLM providers through the `smolagents` package:

- OpenAI API (`OpenAIServerModel`)
- Hugging Face API (`HfApiModel`)
- LiteLLM (`LiteLLMModel`)
- Azure OpenAI (`AzureOpenAIServerModel`)
- VLLM (`VLLMModel`)
- MLX (`MLXModel`)
- Local Transformers (`TransformersModel`)

```python
from superpandas import SuperPandasConfig, LLMClient
# List available providers
providers = LLMClient.available_providers()
print(providers.keys())

# Initialize LLM config
config = SuperPandasConfig()
config.provider = 'HfApiModel'  # Available providers: LiteLLMModel, OpenAIServerModel, HfApiModel, TransformersModel, VLLMModel, MLXModel, AzureOpenAIServerModel
config.model = "meta-llama/Llama-3.2-3B-Instruct"

# Configure at the DataFrame level
df.super.config = config

# Access and configure the LLM client directly
df.super.llm_client = LLMClient(
    model="gpt-3.5-turbo",
    provider=providers['OpenAIServerModel']
)

# Auto-describe your DataFrame
df.super.auto_describe(
    config=None,  # Optional SuperPandasConfig instance
    generate_name=True,
    generate_description=True,
    generate_column_descriptions=True,
    existing_values='warn',  # Options: 'warn', 'skip', 'overwrite'
    **model_kwargs  # Additional arguments for the model provider
)

# Query the DataFrame
response = df.super.query(
    "What are the key trends in this data?",
    system_template=None,  # Optional custom system template
    user_template=None  # Optional custom user template
)
```

### Serialization

#### CSV with Metadata
```python
# Save with metadata
df.super.to_csv("data.csv", include_metadata=True, index=False)
# This will save all the metadata into a file data_metadata.json alongwith the actual data in data.csv.

# Load with metadata (Note it overloads pandas read_csv instead)
df = spd.read_csv("data.csv", include_metadata=True)  # Raises FileNotFoundError if metadata not found
df = spd.read_csv("data.csv", include_metadata=False)  # Initializes empty metadata if not found

# Read metadata separately
df.super.read_metadata("data.csv")
```
#### Pickle
```python
# Save to pickle
df.super.to_pickle("data.pkl")

# Read from pickle
df = spd.read_pickle("data.pkl")
```

### Configuration

The `SuperPandasConfig` class manages global configuration settings:

```python
from superpandas import SuperPandasConfig

# Create a new configuration
config = SuperPandasConfig()

# Available settings
config.provider = 'HfApiModel'  # LLM provider
config.model = "meta-llama/Llama-3.2-3B-Instruct"  # Model name
config.llm_kwargs = {'existing_values': 'warn'}  # Additional LLM arguments
config.system_template = "..."  # System prompt template
config.user_template = "..."  # User prompt template

# Set as default configuration for the library
import superpandas as spd
spd.set_default_config(config)

# Save/load configuration
config.save()  # Saves to ~/.cache/superpandas/config.json
config.load()  # Loads from default path
```

The default configuration is automatically loaded when the library is imported. You can:
1. Create a new configuration and set it as default using `spd.set_default_config()`
2. Modify the existing default configuration directly
3. Save and load configurations to/from disk

The default configuration persists across module reloads and is shared across all DataFrames unless explicitly overridden.

### Error Handling

- Column description methods (`set_column_description`, `set_column_descriptions`) support error handling options:
  - `'raise'`: Raise ValueError for non-existent columns (default)
  - `'ignore'`: Silently skip non-existent columns
  - `'warn'`: Warn and skip non-existent columns

- CSV reading with metadata:
  - `include_metadata=True`: Raises FileNotFoundError if metadata file not found
  - `include_metadata=False`: Initializes empty metadata if metadata file not found

## Future Features / Roadmap

*   **Core DataFrame Enhancements:**
    *   Advanced data validation and schema enforcement.
    *   Support for more complex data types (e.g., nested structures, geospatial data).
    *   Enhanced time-series data handling.
*   **LLM Integration Improvements:**
    *   More sophisticated automated analysis and insight generation.
    *   Support for fine-tuning LLMs on specific datasets.
    *   Integration with a wider range of LLM providers and models.
*   **Expanded Serialization Options:**
    *   Support for additional storage formats (e.g., Parquet, Avro) with metadata.
    *   Integration with data versioning systems (e.g., DVC).
*   **Visualization and Reporting:**
    *   Automatic generation of charts and reports based on LLM analysis.
    *   Integration with popular visualization libraries.
*   **Community and Ecosystem:**
    *   Development of a plugin system for extending functionality.
    *   Improved documentation and tutorials.
    *   Building a strong community around the project.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Documentation

For more detailed information on SuperPandas, please refer to the [API documentation (Work in Progress)](https://superpandas.readthedocs.io/en/latest/).


