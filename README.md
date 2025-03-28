# SuperPandas

SuperPandas is a Python package that extends pandas DataFrame functionality with rich metadata, LLM-powered analysis, and enhanced data manipulation capabilities.

## Installation

You can install SuperPandas using pip:

```bash
pip install superpandas
```

## Key Features

### Enhanced DataFrame with Metadata
- **Rich Metadata Support**: Add names, descriptions, and detailed column information to your DataFrames
- **Intelligent Type Inference**: Automatically detects and tracks detailed column types
- **Schema Generation**: Generate clear schema representations for documentation or LLM analysis

```python
import superpandas as spd

# Create a SuperDataFrame with metadata
df = spd.SuperDataFrame(data, 
    name="sales_data",
    description="Monthly sales data by region",
    column_descriptions={
        "revenue": "Monthly revenue in USD",
        "region": "Sales region code"
    }
)

# Access metadata
print(df.name)  # "sales_data"
print(df.description)
print(df.column_descriptions)
print(df.column_types)  # Shows refined type information
```

### LLM Integration
- **Built-in LLM Support**: Seamless integration with various LLM providers
- **Automated Analysis**: Get AI-powered insights about your data
- **Auto-Documentation**: Generate descriptions and documentation automatically

```python
from superpandas import LLMClient

# Initialize LLM client
llm = LLMClient(model="meta-llama/Llama-3.2-3B-Instruct")

# Auto-describe your DataFrame
df = spd.auto_describe_dataframe(
    df,
    model="meta-llama/Llama-3.2-3B-Instruct",
    generate_df_name=True,
    generate_df_description=True,
    generate_column_descriptions=True
)

# Get AI analysis
analysis = llm.analyze_dataframe(df, "What are the key trends in this data?")
```

### Enhanced Serialization
- **Metadata Preservation**: Save and load DataFrames with all metadata intact
- **Multiple Format Support**: Export to CSV, JSON, and pickle with metadata
- **Backwards Compatibility**: Works seamlessly with standard pandas operations

```python
# Save with metadata
df.to_json("data.json")
df.to_csv("data.csv", include_metadata=True)
df.to_pickle("data.pkl")

# Load with metadata
df = spd.SuperDataFrame.read_json("data.json")
df = spd.SuperDataFrame.read_csv("data.csv", load_metadata=True)
df = spd.SuperDataFrame.read_pickle("data.pkl")
```

### LLM Provider Support
SuperPandas supports multiple LLM providers through the `smolagents` package:
- OpenAI API
- Hugging Face API
- Local LLaMA models
- Azure OpenAI
- VLLM
- MLX
- Custom model implementations

```python
# Use different LLM providers
llm = LLMClient(
    model="gpt-3.5-turbo",
    provider_class=LLMClient.available_providers()['OpenAIServerModel']
)
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
schema = df.schema(template=template)
```

### LLM Format Options
```python
# Convert to LLM-friendly formats
json_format = df.to_llm_format(format_type='json')
markdown_format = df.to_llm_format(format_type='markdown')
text_format = df.to_llm_format(format_type='text')
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Features

- Enhanced data filtering and sorting
- Advanced groupby operations
- Custom aggregation functions
- Simplified time series analysis
- Extended visualization capabilities

## Quick Start

```python
import superpandas as spd

# Load data from CSV
df = spd.read_csv('data.csv')

# Apply transformations
df = df.groupby('category').agg({'value': 'sum'})
```

## Documentation

For more detailed information on SuperPandas, please refer to the [API documentation](https://superpandas.readthedocs.io/en/latest/).

