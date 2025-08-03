# LangGraph Agent for Code Execution

SuperPandas includes a LangGraph agent for intelligent code execution and data analysis. This agent can generate, execute, and debug Python code to analyze DataFrames.

## Overview

The LangGraph agent provides:
- **Code Generation**: Automatically generates Python code based on natural language queries
- **Code Execution**: Safely executes generated code in a controlled environment
- **Error Reflection**: Analyzes errors and provides suggestions for fixes
- **Response Formatting**: Formats results into user-friendly responses
- **Iterative Improvement**: Retries failed code with improved suggestions

## Basic Usage

### Using the SuperDataFrame Accessor

```python
import pandas as pd
import superpandas as spd

# Create a DataFrame with metadata
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [50000, 60000, 75000]
})

# Add metadata
df.super.name = "Employee Data"
df.super.description = "Sample employee data"

# Use the agent for analysis
result = df.super.analyze_with_agent(
    query="Calculate the average salary by age group",
    max_iterations=5
)

print(result['formatted_response'])
print(result['result'])
```

### Direct Agent Usage

```python
from superpandas.langgraph_agent import run_agent

result = run_agent(
    query="Find employees with salary above the median",
    dataframe=df,
    max_iterations=3
)
```

## Agent Workflow

The agent follows this workflow:

1. **Code Generation**: Generates Python code based on the query and DataFrame schema
2. **Code Execution**: Executes the code in a safe environment
3. **Error Handling**: If errors occur, reflects on them and generates improved code
4. **Response Formatting**: Formats the final result into a user-friendly response

## Configuration

You can configure the agent using the SuperPandasConfig:

```python
from superpandas.config import SuperPandasConfig

config = SuperPandasConfig()
config.model = "gpt-4"  # Use a different model
config.provider = "openai"  # Use a different provider

# Use the config with the agent
result = df.super.analyze_with_agent(
    query="Your query here",
    max_iterations=5
)
```

## Available Variables

The agent provides these variables in the execution environment:

- `df`: The pandas DataFrame to analyze
- `pd`: pandas library
- `json`: JSON library
- `result`: Variable to store the analysis result
- `fig`: Variable to store matplotlib figures

## Error Handling

The agent includes robust error handling:

- **Syntax Errors**: Automatically detects and fixes syntax issues
- **Import Errors**: Handles missing imports
- **DataFrame Errors**: Manages column and data type issues
- **Logic Errors**: Provides suggestions for logical improvements

## Examples

### Basic Analysis

```python
result = df.super.analyze_with_agent("Show the first 5 rows")
```

### Statistical Analysis

```python
result = df.super.analyze_with_agent("Calculate mean, median, and standard deviation of numeric columns")
```

### Visualization

```python
result = df.super.analyze_with_agent("Create a histogram of the age distribution")
```

### Complex Queries

```python
result = df.super.analyze_with_agent(
    "Find employees with salary above the 75th percentile and age below 30"
)
```

## Advanced Usage

### Custom Agent Creation

```python
from superpandas.langgraph_agent import create_langgraph_agent

agent = create_langgraph_agent(
    config=your_config,
    max_iterations=10
)

# Use the agent directly
initial_state = {
    "messages": [],
    "current_query": "Your query",
    "dataframe": df,
    "generated_code": "",
    "result": None,
    "error": "",
    "iterations": 0,
    "formatted_response": "",
    "fig": None
}

final_state = agent.invoke(initial_state)
```

### State Structure

The agent state contains:

- `messages`: Conversation history
- `current_query`: The user's query
- `dataframe`: The DataFrame to analyze
- `generated_code`: The generated Python code
- `result`: The analysis result
- `error`: Any error messages
- `iterations`: Number of attempts made
- `formatted_response`: User-friendly response
- `fig`: Matplotlib figure (if created)

## Best Practices

1. **Provide Clear Queries**: Be specific about what you want to analyze
2. **Use Metadata**: Add descriptions to your DataFrame and columns for better results
3. **Handle Errors**: Check the `error` field in results for debugging
4. **Iteration Limits**: Set appropriate `max_iterations` based on complexity
5. **Model Selection**: Use appropriate models for your use case

## Limitations

- Requires LLM API access (OpenAI, etc.)
- Code execution is limited to safe operations
- Complex queries may require multiple iterations
- Results depend on the quality of the LLM model used 