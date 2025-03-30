# SuperPandas

SuperPandas is a Python package that extends pandas DataFrame functionality with rich metadata to allow for AI-powered analysis on your Pandas DataFrames.

## Key Features

### Enhanced DataFrame with Metadata
- **Rich Metadata Support**: Add names, descriptions, and detailed column information to your DataFrames, which allows for better representation of Pandas DataFrames in LLM applications.
- **Automatically generate metadata**: Automatically generate metadata for your DataFrames using LLMs from various providers like OpenAI, Hugging Face, and more.
- **Intelligent Type Inference**: Automatically detects and tracks detailed column types, especially for object columns.
- **Schema Generation**: Generate clear schema representations for documentation or LLM analysis.
- **Serialization**: Save and load DataFrames with all metadata intact.
- **Drop-in replacement for Pandas DataFrames**: SuperPandas is a lightweight wrapper around pandas DataFrames, so you can use it as a drop-in replacement for your existing code.

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

```python
import superpandas as spd

# Create a SuperDataFrame with metadata
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
print(sdf.super.description)
print(sdf.super.column_descriptions)
print(sdf.super.column_types)  # Shows refined type information
```

### LLM Integration
- **Built-in LLM Support**: Seamless integration with various LLM providers using the `smolagents` package.
- **Automated Analysis**: Get AI-powered insights about your data.
- **Auto-Documentation**: Generate descriptions and documentation automatically.
- **Flexible Provider System**: Support for multiple LLM providers through a unified interface.

```python
from superpandas import LLMClient

# Initialize LLM client
llm = LLMClient(model="meta-llama/Llama-3.2-3B-Instruct")

# Auto-describe your DataFrame
df.super.auto_describe(
    model="meta-llama/Llama-3.2-3B-Instruct",
    generate_name=True,
    generate_description=True,
    generate_column_descriptions=True
)

# Get AI analysis
analysis = llm.analyze_dataframe(df, "What are the key trends in this data?")

# Generate individual components
df_name = llm.generate_df_name(df)
df_description = llm.generate_df_description(df)
column_descriptions = llm.generate_column_descriptions(df)
```

### Enhanced Serialization
- **Metadata Preservation**: Save and load DataFrames with all metadata intact.
- **Multiple Format Support**: Export to CSV and pickle with metadata. (JSON format coming soon)
- **Backwards Compatibility**: Works seamlessly with standard pandas operations.

```python
# Save with metadata
df.super.to_csv("data.csv", include_metadata=True, index=False)
df.super.to_pickle("data.pkl")

# Load with metadata
df = spd.read_csv("data.csv", require_metadata=True)
df = spd.read_pickle("data.pkl")
```

### LLM Provider Support
SuperPandas supports multiple LLM providers through the `smolagents` package:

- OpenAI API (`OpenAIServerModel`)
- Hugging Face API (`HfApiModel`)
- LiteLLM (`LiteLLMModel`)
- Azure OpenAI (`AzureOpenAIServerModel`)
- VLLM (`VLLMModel`)
- MLX (`MLXModel`)
- Local Transformers (`TransformersModel`)
- Custom model implementations

```python
# List available providers
providers = LLMClient.available_providers()
print(providers.keys())

# Use specific provider
llm = LLMClient(
    model="gpt-3.5-turbo",
    provider_class=providers['OpenAIServerModel']
)

# Use default provider (Hugging Face with Llama 3.2)
llm = LLMClient()
```

### Auto-Documentation Features

The LLMClient provides several methods for automatic documentation:

```python
# Generate comprehensive DataFrame description
df_description = llm.generate_df_description(df)

# Generate column-level descriptions
column_descriptions = llm.generate_column_descriptions(df)

# Generate a concise name for the DataFrame
df_name = llm.generate_df_name(df)

# Automatically generate all metadata at once
df.super.auto_describe(
    model="meta-llama/Llama-3.2-3B-Instruct",
    generate_name=True,
    generate_description=True,
    generate_column_descriptions=True
)
```

### Custom LLM Integration

You can create custom LLM clients by extending the base `LLMClient` class:

```python
from superpandas import LLMClient

class CustomLLMClient(LLMClient):
    def query(self, prompt: str, **kwargs) -> str:
        # Implement custom query logic
        return your_custom_logic(prompt)
```

## Advanced Usage

### Custom Schema Templates
```python
# Define custom schema template
template = """
# {name}
{description}

## Data Structure
Rows: {shape[0]}
Columns: {shape[1]}

## Columns
{columns}
"""

# Generate schema with custom template
schema = df.super.schema(template=template)
```

### LLM Format Options
```python
# Convert to LLM-friendly formats
json_format = df.super.to_llm_format(format_type='json')
markdown_format = df.super.to_llm_format(format_type='markdown')
text_format = df.super.to_llm_format(format_type='text')
```

### LLM Analysis Options

```python
# Basic dataframe analysis
analysis = llm.analyze_dataframe(df, "What are the key trends in this data?")

# Custom analysis with additional parameters
analysis = llm.analyze_dataframe(
    df,
    "Identify outliers in the sales data",
    temperature=0.7,
    max_tokens=500
)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Documentation

For more detailed information on SuperPandas, please refer to the [API documentation (Work in Progress)](https://superpandas.readthedocs.io/en/latest/).

