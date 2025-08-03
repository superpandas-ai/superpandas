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

### LangGraph Agent for Code Execution
- **Intelligent Code Generation**: Automatically generates Python code based on natural language queries
- **Safe Code Execution**: Executes generated code in a controlled environment
- **Error Reflection**: Analyzes errors and provides suggestions for fixes
- **Iterative Improvement**: Retries failed code with improved suggestions
- **Response Formatting**: Formats results into user-friendly responses

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

sdf = spd.create_super_dataframe(
    df=df,
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
print(df.super.column_descriptions)
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
all_descriptions = df.super.column_descriptions
column_types = df.super.column_types

# Refresh column type inference
df.super.refresh_column_types()
```

#### Schema Generation
```python
# Generate schema in different formats
schema = df.super.get_schema(
    format_type='text',  # Options: 'json', 'markdown', 'text', 'yaml'
    max_rows=5  # Number of sample rows to include
)

# Custom schema template
template = """
DataFrame Name: {name}
DataFrame Description: {description}

Shape: {shape}

Columns:
{column_info}
"""
schema = df.super.get_schema(template=template)
```

**Required template elements:**
- `{name}`: DataFrame name
- `{description}`: DataFrame description  
- `{shape}`: DataFrame shape as string
- `{column_info}`: Formatted column information

**Optional template elements:**
- `{columns}`: Raw column list (alternative to `{column_info}`)

### LLM Integration and Usage

SuperPandas supports multiple LLM providers through the `smolagents` package:

```python
from superpandas import available_providers

# List available providers
providers = available_providers
print(providers.keys())
```

Available providers:
- `lite`: LiteLLM (`LiteLLMModel`)
- `openai`: OpenAI API (`OpenAIServerModel`)
- `hf`: Hugging Face API (`HfApiModel`)
- `tf`: Local Transformers (`TransformersModel`)
- `vllm`: VLLM (`VLLMModel`)
- `mlx`: MLX (`MLXModel`)
- `openai_az`: Azure OpenAI (`AzureOpenAIServerModel`)
- `bedrock`: Amazon Bedrock (`AmazonBedrockServerModel`)

```python
from superpandas import SuperPandasConfig

# Initialize config
config = SuperPandasConfig()
config.provider = 'hf'  # Available providers: 'lite', 'openai', 'hf', 'tf', 'vllm', 'mlx', 'openai_az', 'bedrock'
config.model = "meta-llama/Llama-3.2-3B-Instruct"

# Configure at the DataFrame level
df.super.config = config

# Auto-describe your DataFrame
df.super.auto_describe(
    generate_name=True,
    generate_description=True,
    generate_column_descriptions=True,
    existing_values='warn'  # Options: 'warn', 'skip', 'overwrite'
)

# Query the DataFrame
response = df.super.query("What are the key trends in this data?")
```

#### LangGraph Agent for Code Execution

The LangGraph agent provides intelligent code execution for data analysis:

```python
# Basic usage with the agent
result = df.super.analyze_with_agent(
    query="Calculate the average salary by department",
    max_iterations=5
)

print(result['formatted_response'])
print(result['result'])

# Direct agent usage
from superpandas.langgraph_agent import run_agent

result = run_agent(
    query="Find employees with salary above the median",
    dataframe=df,
    max_iterations=3
)
```

The agent workflow:
1. **Code Generation**: Generates Python code based on your query
2. **Code Execution**: Safely executes the code
3. **Error Handling**: If errors occur, reflects and tries again
4. **Response Formatting**: Provides clear explanations of results


### Serialization

#### CSV with Metadata
```python
# Save with metadata
df.super.to_csv("data.csv", include_metadata=True, index=False)
# This will save all the metadata into a file data_metadata.json alongwith the actual data in data.csv.

# Load with metadata (it overloads pandas read_csv)
df = spd.read_csv("data.csv", include_metadata=True)  # Raises FileNotFoundError if metadata not found
df = spd.read_csv("data.csv", include_metadata=False)  # Initializes empty metadata if not found

# Read metadata separately
df.super.read_metadata("data.csv")
```
#### Pickle
```python
# Save to pickle with metadata
df.super.to_pickle("data.pkl")

# Read from pickle (it overloads pandas read_pickle)
df = spd.read_pickle("data.pkl")
```

### Configuration

The `SuperPandasConfig` class manages global configuration settings. The configuration is automatically saved to `~/.cache/superpandas/config.json` and persists across sessions, allowing you to set global defaults like `provider` and `model` once for your entire installation.

```python
from superpandas import SuperPandasConfig

# Create a new configuration
config = SuperPandasConfig()

# Available settings
config.provider = 'hf'  # LLM provider ('lite', 'openai', 'hf', 'tf', 'vllm', 'mlx', 'openai_az', 'bedrock')
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

#### Configuration Hierarchy and Override Options

The configuration system supports multiple levels of customization:

1. **Global Default Configuration** (persists across sessions):
   ```python
   # Set global defaults that apply to all DataFrames
   config = SuperPandasConfig()
   config.provider = 'openai'
   config.model = 'gpt-4'
   spd.set_default_config(config)  # Saves to ~/.cache/superpandas/config.json
   ```

2. **Per-DataFrame Configuration** (overrides global for specific DataFrame):
   ```python
   # Override config for a specific DataFrame
   df.super.config = SuperPandasConfig()
   df.super.config.provider = 'hf'
   df.super.config.model = 'meta-llama/Llama-3.2-3B-Instruct'
   ```

3. **Configuration at Creation Time**:
   ```python
   # Set config when creating a SuperDataFrame
   custom_config = SuperPandasConfig()
   custom_config.provider = 'vllm'
   custom_config.model = 'llama-3.2-3b'
   
   sdf = spd.create_super_dataframe(
       df=df,
       name="my_data",
       config=custom_config
   )
   ```

4. **Temporary Configuration Override**:
   ```python
   # Temporarily change config for specific operations
   original_config = df.super.config
   df.super.config.provider = 'openai'
   df.super.config.model = 'gpt-3.5-turbo'
   
   # Perform operations with temporary config
   df.super.auto_describe()
   
   # Restore original config
   df.super.config = original_config
   ```

The default configuration is automatically loaded when the library is imported. You can:
1. Create a new configuration and set it as default using `spd.set_default_config()`
2. Modify the existing default configuration directly
3. Save and load configurations to/from disk
4. Set the config when creating a superdataframe using `create_super_dataframe()`

The default/loaded configuration persists across module reloads and is shared across all DataFrames unless explicitly overridden.

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
    *   Extending metadata merging to DataFrame methods like `concat`, `merge`, `join`, etc.
    *   Extending metadata to methods like checking equality, copy, etc.
*   **LLM Integration Improvements:**
    *   More sophisticated automated analysis and insight generation using agents.
    *   Add Code Execution.
    *   Integration with a wider range of LLM providers and models.
    *   Adding support for chat history in LLM interactions.
*   **Multi-DataFrame and Database Support:**
    *   Extending SuperDataFrame to have multiple DataFrames with foreign key support.
    *   Providing AI interface to RDBMS systems.
    *   Support for complex relational data operations with metadata preservation.
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


