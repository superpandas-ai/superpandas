# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SuperPandas is a Python library that extends pandas DataFrames with AI-powered metadata capabilities. It uses a pandas accessor pattern to add a `.super` namespace to DataFrames, enabling rich metadata, LLM integration, and enhanced serialization.

## Development Commands

### Testing
```bash
# Run all tests with coverage
pytest

# Run tests with verbose output
pytest -v

# Run specific test file
pytest tests/test_superdataframe.py

# Run tests without slow tests
pytest -m "not slow"
```

### Build and Package
```bash
# Build package
python -m build

# Install in development mode
pip install -e .

# Install with test dependencies
pip install -e ".[test]"
```

## Architecture

### Core Components

1. **SuperDataFrameAccessor** (`superdataframe.py`): The main pandas accessor that adds `.super` namespace
   - Manages metadata (name, description, column descriptions)
   - Provides LLM integration methods (`query`, `auto_describe`)
   - Handles schema generation and serialization

2. **SuperPandasConfig** (`config.py`): Configuration management
   - Global settings for LLM providers and models
   - Template management for prompts
   - Persistent configuration storage in `~/.cache/superpandas/config.json`

3. **LLMClient** (`llm_client.py`): LLM integration layer
   - Supports multiple providers via smolagents (OpenAI, HuggingFace, LiteLLM, etc.)
   - Handles provider-specific initialization and error handling
   - Provides unified interface for different LLM backends

4. **Templates** (`templates.py`): Prompt templates for LLM interactions
   - System template for data science tasks
   - User template for query formatting
   - Schema template for data representation

5. **PandasSQL** (`pandasql.py`): SQL query interface for DataFrames
   - Adds `.sql` accessor to DataFrames
   - Supports SQL queries on DataFrame data
   - Query execution with caching capabilities

6. **LangGraph Agent** (`langgraph_agent.py`): Code generation agent
   - Agent-based approach for complex data operations
   - Multi-step reasoning and code execution
   - Integration with LangGraph framework

7. **Providers** (`providers.py`): LLM provider management
   - Dynamic provider discovery and initialization
   - Provider-specific configuration handling
   - Error handling for missing dependencies

### Key Design Patterns

- **Pandas Accessor Pattern**: Uses `@pd.api.extensions.register_dataframe_accessor("super")` and `@pd.api.extensions.register_dataframe_accessor("sql")` to extend DataFrame functionality
- **Metadata Storage**: Stores metadata in `DataFrame.attrs['super']` for persistence
- **Configuration Singleton**: Global default configuration with ability to override per DataFrame
- **Provider Abstraction**: Unified interface for different LLM providers through smolagents
- **Agent Pattern**: LangGraph-based agent for complex multi-step operations
- **Code Execution**: Safe code parsing and execution with utility modules

### Data Flow

1. DataFrame created → SuperDataFrameAccessor initialized → metadata container created in `df.attrs['super']`
2. User calls `df.super.auto_describe()` → LLMClient invoked → metadata populated
3. User calls `df.super.query()` → schema generated → LLM called with templates → response returned
4. User calls `df.sql("SELECT * FROM df")` → SQL query executed on DataFrame data
5. User calls agent-based operations → LangGraph agent processes multi-step tasks → code generated and executed
6. Serialization preserves metadata alongside data (CSV + JSON metadata, or pickle)

## File Structure

```
superpandas/
├── __init__.py          # Main exports and pandas re-exports
├── superdataframe.py    # Core SuperDataFrameAccessor class
├── config.py           # Configuration management
├── llm_client.py       # LLM integration layer
├── templates.py        # Prompt templates
├── pandasql.py         # SQL accessor for DataFrames
├── langgraph_agent.py  # Code generation agent
├── providers.py        # LLM provider management
└── utils/              # Utility modules
    ├── __init__.py
    ├── codeparser.py   # Code parsing utilities
    └── executors.py    # Code execution utilities
```

## Key Implementation Details

- **Column Type Inference**: Automatically refines pandas 'object' dtypes to more specific types (str, mixed, etc.)
- **Error Handling**: Supports 'raise', 'warn', 'ignore' strategies for handling non-existent columns
- **Metadata Serialization**: CSV files saved with companion `_metadata.json` files
- **Provider Discovery**: Dynamically imports available LLM providers to handle optional dependencies
- **Configuration Persistence**: Automatic loading of saved configurations on import
- **SQL Query Execution**: Direct SQL queries on DataFrame data using pandasql backend
- **Agent-based Processing**: Multi-step reasoning and code generation for complex operations
- **Code Safety**: Parsing and execution utilities with proper error handling

## Testing Notes

- Tests use pytest with coverage reporting
- Test data includes various formats (CSV, JSON, Excel, pickle)
- Slow tests are marked and can be skipped with `-m "not slow"`
- Mock LLM responses used for integration tests to avoid external API calls