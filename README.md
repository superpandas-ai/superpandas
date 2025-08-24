# SuperPandas

<div align="center">
  <img src="https://avatars.githubusercontent.com/u/192782649" alt="SuperPandas Logo" width="200"/>
</div>

<div align="center">
  <strong>⚠️ Beta Version - Breaking Changes Expected</strong><br>
  This package is currently in beta. Please expect errors and breaking changes in future releases.
</div>


### Introduction

SuperPandas is a lightweight Python package that extends the well known Pandas library with functionality for AI-powered data analytics and SQL querying. It is a barebones wrapper using Pandas Dataframe [accessors](https://pandas.pydata.org/docs/development/extending.html) feature to add namespaces 'super' and 'sql' which add metadata, LLM integration, and SQL query capabilities to dataframes. 

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

### SQL Query Accessor
- **In-memory SQLite Database**: Execute SQL queries on pandas DataFrames using SQLite as the backend engine
- **Multiple Table Support**: Join and query multiple DataFrames using the `env` parameter
- **Full SQL Support**: Complete SQL functionality including SELECT, WHERE, JOIN, GROUP BY, HAVING, ORDER BY, and more
- **Type Safety**: Comprehensive error handling and validation for robust query execution
- **Custom Database URIs**: Option to use persistent databases instead of in-memory storage
- **Seamless Integration**: Works with both regular pandas DataFrames and SuperDataFrames

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

## Quick Start

### SQL Querying (New!)
```python
import pandas as pd
import superpandas  # Registers SQL accessor

# Create DataFrames
df1 = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})
df2 = pd.DataFrame({"id": [1, 2], "dept": ["Engineering", "Sales"]})

# Basic SQL query
result = df1.sql.query("SELECT * FROM df WHERE id > 1")

# Join multiple DataFrames
env = {"employees": df1, "departments": df2}
result = df1.sql.query("""
    SELECT e.name, d.dept 
    FROM df e 
    JOIN departments d ON e.id = d.id
""", env=env)
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

### SQL Querying with DataFrames

SuperPandas provides a powerful SQL accessor that allows you to query pandas DataFrames using familiar SQL syntax:

```python
import pandas as pd
import superpandas  # This registers the SQL accessor

# Basic SQL operations
df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
    "salary": [50000, 60000, 70000]
})

# Simple filtering
result = df.sql.query("SELECT * FROM df WHERE age > 28")

# Aggregations
result = df.sql.query("SELECT AVG(salary) as avg_salary, COUNT(*) as count FROM df")

# String operations
result = df.sql.query("SELECT UPPER(name) as upper_name, LENGTH(name) as name_length FROM df")

# Working with multiple DataFrames
df2 = pd.DataFrame({
    "name": ["Alice", "Bob"],
    "department": ["Engineering", "Sales"]
})

env = {"employees": df, "departments": df2}
result = df.sql.query("""
    SELECT e.name, e.salary, d.department
    FROM df e
    JOIN departments d ON e.name = d.name
    ORDER BY e.salary DESC
""", env=env)
```

### Core Methods

#### SQL Query Accessor
SuperPandas includes a powerful SQL accessor that allows you to execute SQL queries on pandas DataFrames using SQLite as the backend engine. This feature brings the power of SQL to pandas DataFrames, enabling complex data operations with familiar SQL syntax.

**Basic Usage:**
```python
import pandas as pd
import superpandas  # This registers the SQL accessor

# Create sample DataFrames
df1 = pd.DataFrame({
    "id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35]
})

df2 = pd.DataFrame({
    "id": [1, 2, 4],
    "department": ["Engineering", "Sales", "Marketing"],
    "salary": [80000, 70000, 75000]
})

# Basic SQL query
result = df1.sql.query("SELECT * FROM df WHERE age > 28")

# Query with additional tables
env = {"employees": df1, "departments": df2}
result = df1.sql.query("""
    SELECT e.name, e.age, d.department, d.salary
    FROM df e
    JOIN departments d ON e.id = d.id
    WHERE e.age > 25
""", env=env)

# Aggregation queries
result = df1.sql.query("SELECT AVG(age) as avg_age, COUNT(*) as count FROM df")

# Complex queries with multiple tables
result = df1.sql.query("""
    SELECT 
        d.department,
        AVG(e.age) as avg_age,
        SUM(d.salary) as total_salary
    FROM df e
    JOIN departments d ON e.id = d.id
    GROUP BY d.department
    HAVING AVG(e.age) > 25
""", env=env)
```

**Advanced SQL Features:**
```python
# String functions and pattern matching
result = df.sql.query("""
    SELECT 
        name,
        UPPER(name) as upper_name,
        LENGTH(name) as name_length
    FROM df 
    WHERE name LIKE '%a%'
""")

# Date functions
result = df.sql.query("""
    SELECT 
        name,
        created_date,
        STRFTIME('%Y-%m', created_date) as year_month
    FROM df 
    ORDER BY created_date
""")

# Conditional logic with CASE statements
result = df.sql.query("""
    SELECT 
        name,
        score,
        CASE 
            WHEN score >= 90 THEN 'Excellent'
            WHEN score >= 80 THEN 'Good'
            WHEN score >= 70 THEN 'Average'
            ELSE 'Needs Improvement'
        END as grade
    FROM df 
    ORDER BY score DESC
""")

# Custom database URI for persistent storage
result = df.sql.query(
    "SELECT * FROM df WHERE x > 1",
    db_uri="sqlite:///my_database.db"
)
```

**Key Features:**
- **In-memory SQLite**: Uses SQLite in-memory database for fast queries
- **Multiple Tables**: Support for joining multiple DataFrames via the `env` parameter
- **Full SQL Support**: Supports all standard SQL operations (SELECT, WHERE, JOIN, GROUP BY, HAVING, ORDER BY, etc.)
- **Type Safety**: Comprehensive error handling and validation
- **Custom Database**: Option to use custom database URIs for persistent storage
- **String & Date Functions**: Full support for SQLite string and date manipulation functions
- **Conditional Logic**: CASE statements and complex WHERE clauses
- **Aggregations**: GROUP BY, HAVING, and all standard aggregation functions

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
*   **SQL Accessor Enhancements:**
    *   Support for additional SQL dialects (PostgreSQL, MySQL, etc.).
    *   Advanced query optimization and performance improvements.
    *   Integration with external databases and data warehouses.
    *   Query result caching and optimization.
    *   Support for stored procedures and complex SQL functions.
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

### Third-Party Licenses

The SQL accessor functionality in SuperPandas is inspired by and builds upon concepts from the [pandasql](https://github.com/yhat/pandasql) project, which is also licensed under the MIT License. We acknowledge and thank the pandasql contributors for their work.

## Documentation

For more detailed information on SuperPandas, please refer to the [API documentation (Work in Progress)](https://superpandas.readthedocs.io/en/latest/).


