.. _usage_guide:

Usage Guide
===========

This guide provides an overview of how to use SuperPandas, from creating enhanced DataFrames to leveraging its AI capabilities.

Creating SuperDataFrame
-----------------------

You can create a SuperDataFrame in the following two ways (For more information on SuperDataFrames, see :ref:`api`):

**Method 1: Create a SuperDataFrame explicitly with metadata**

.. code-block:: python

   import pandas as pd
   import numpy as np
   import superpandas as spd

   # Create a sample DataFrame
   df = pd.DataFrame({
       'date': pd.date_range(start='2023-01-01', periods=6, freq='ME'),
       'region': ['North', 'South', 'East', 'West', 'North', 'South'],
       'revenue': np.random.randint(10000, 100000, 6),
       'units_sold': np.random.randint(100, 1000, 6)
   })

   sdf = spd.create_super_dataframe(df,
       name="sales_data",
       description="Monthly sales data by region",
       column_descriptions={
           "revenue": "Monthly revenue in USD",
           "region": "Sales region code"
       }
   )

   # Access metadata
   print(sdf.super.name)
   print(sdf.super.description)
   print(sdf.super.column_descriptions)
   print(sdf.super.column_types)

**Method 2: Explicitly add metadata to an existing DataFrame**

.. code-block:: python

   import pandas as pd
   import numpy as np
   import superpandas # adds 'super' namespace to pandas without changing the existing code(see :ref:`api`)

   # Any dataframe from existing code
   df = pd.DataFrame({
       'date': pd.date_range(start='2023-01-01', periods=6, freq='ME'),
       'region': ['North', 'South', 'East', 'West', 'North', 'South'],
       'revenue': np.random.randint(10000, 100000, 6),
       'units_sold': np.random.randint(100, 1000, 6)
   })

   print(df.super.name) # yields empty string
   df.super.name = "sales_data"
   df.super.description = "Monthly sales data by region"
   df.super.set_column_descriptions({
       "revenue": "Monthly revenue in USD",
       "region": "Sales region code"
   })
   print(df.super.name)
   print(df.super.description)
   print(df.super.column_descriptions)
   print(df.super.column_types)

Core Methods
------------

Metadata Management
~~~~~~~~~~~~~~~~~~~

Manage and access metadata associated with your DataFrame.

.. code-block:: python

   # Assuming df is a SuperDataFrame or a Pandas DataFrame with the .super accessor
   # from previous examples.

   # Get/Set DataFrame name and description
   df.super.name = "my_dataframe"
   df.super.description = "Description of my dataframe"

   # Get/Set column descriptions
   df.super.set_column_description("revenue", "Total revenue in USD") # Example for a specific column
   df.super.set_column_descriptions({
       "region": "Geographical sales region",
       "units_sold": "Number of units sold"
   }, errors='raise')  # errors can be 'raise', 'ignore', or 'warn'

   # Get column information
   description = df.super.get_column_description("revenue")
   all_descriptions = df.super.get_column_descriptions()
   column_types = df.super.column_types

   # Refresh column type inference
   df.super.refresh_column_types()

Schema Generation
~~~~~~~~~~~~~~~~~

Generate a schema representation of your DataFrame to be used as context for LLMs.

.. code-block:: python

   # Assuming sdf is a SuperDataFrame or a Pandas DataFrame with the .super accessor

   # Generate schema in different formats
   schema_text = sdf.super.get_schema(
       template=None,  # Optional custom template
       format_type='text',  # Options: 'json', 'markdown', 'text', 'yaml' (default: 'text')
       max_rows=5  # Optional: Number of sample rows to include
   )
   print(schema_text)

   # Custom schema template example
   custom_template = """
   Dataset Name: {name}
   Description: {description}

   Shape: {shape[0]} rows, {shape[1]} columns

   Columns:
   {columns}
   """
   schema_custom = sdf.super.get_schema(template=custom_template)
   print(schema_custom)

LLM Integration
---------------

SuperPandas integrates with various LLM providers via the `smolagents` package.

Supported providers include:

- OpenAI API (`OpenAIServerModel`)
- Hugging Face API (`HfApiModel`)
- LiteLLM (`LiteLLMModel`)
- Azure OpenAI (`AzureOpenAIServerModel`)
- VLLM (`VLLMModel`)
- MLX (`MLXModel`)
- Local Transformers (`TransformersModel`)

.. code-block:: python

   from superpandas import SuperPandasConfig, LLMClient
   # Assuming df is a SuperDataFrame or a Pandas DataFrame with the .super accessor

   # List available providers
   providers = LLMClient.available_providers()
   print(providers) 

   # Initialize LLM config
   config = SuperPandasConfig()
   # Ensure you have the necessary API keys/environment variables set for your chosen provider
   config.provider = 'HfApiModel'  # Example provider
   config.model = "meta-llama/Llama-3.2-3B-Instruct" # Example model

   # Configure at the DataFrame level
   df.super.config = config

   # Access and configure the LLM client directly (alternative)
   # df.super.llm_client = LLMClient(
   #     model="gpt-3.5-turbo", # Example model
   #     provider=providers['OpenAIServerModel'] # Example provider
   # )

   # Auto-describe your DataFrame (requires LLM client to be configured)
   # This operation can be costly and time-consuming depending on the LLM and data size.
   # Ensure your LLM provider and model are correctly set up.
   # df.super.auto_describe(
   #     generate_name=True,
   #     generate_description=True,
   #     generate_column_descriptions=True,
   #     existing_values='warn'  # Options: 'warn', 'skip', 'overwrite'
   #     # **model_kwargs  # Additional arguments for the model provider
   # )
   # print(df.super.name)
   # print(df.super.description)
   # print(df.super.get_column_descriptions())


   # Query the DataFrame (requires LLM client to be configured)
   # Ensure your LLM provider and model are correctly set up.
   # response = df.super.query(
   #     "What are the key trends in this data?",
   #     system_template=None,  # Optional custom system template
   #     user_template=None  # Optional custom user template
   # )
   # print(response)

Serialization
-------------

Save and load SuperDataFrames with their metadata.

CSV
~~~

.. code-block:: python

   import superpandas as spd
   # Assuming sdf is a SuperDataFrame or a Pandas DataFrame with the .super accessor

   # Save with metadata
   sdf.super.to_csv("data.csv", include_metadata=True, index=False)
   # This saves metadata to data_metadata.json alongside data.csv.

   # Load with metadata (overloads pandas.read_csv)
   sdf_loaded_csv = spd.read_csv("data.csv", include_metadata=True)

   # Load without metadata (initializes empty metadata)
   sdf_loaded_no_meta = spd.read_csv("data.csv", include_metadata=False)

Pickle
~~~~~~

.. code-block:: python

   import superpandas as spd
   # Assuming sdf is a SuperDataFrame or a Pandas DataFrame with the .super accessor

   # Save to pickle
   sdf.super.to_pickle("data.pkl")

   # Read from pickle
   sdf_loaded_pkl = spd.read_pickle("data.pkl")
   # print(sdf_loaded_pkl.super.name)

Configuration
-------------

Manage configuration settings using `SuperPandasConfig`.

.. code-block:: python

   from superpandas import SuperPandasConfig
   import superpandas as spd

   # Create a new configuration
   config = SuperPandasConfig()

   # Available settings
   config.provider = 'HfApiModel'  # LLM provider
   config.model = "meta-llama/Llama-3.2-3B-Instruct"  # Model name
   config.llm_kwargs = {'existing_values': 'warn'}  # Additional LLM arguments
   config.system_template = "Your default system prompt template..."
   config.user_template = "Your default user prompt template for {query} on {name}..."

   # Set as default configuration for the library
   spd.set_default_config(config)

   # Save/load configuration
   config.save()  # Saves to ~/.cache/superpandas/config.json
   config.load()  # Loads from default path
   print(f"Loaded provider: {config.provider}")

The default configuration is automatically loaded when the library is imported. You can:

1. Create a new configuration and set it as default using ``spd.set_default_config()``
2. Modify the existing default configuration directly
3. Save and load configurations to/from disk

The default configuration persists across module reloads and is shared across all DataFrames unless explicitly overridden.

Error Handling
--------------

SuperPandas provides options for handling errors in certain operations:

- Column description methods (`set_column_description`, `set_column_descriptions`):

  - ``'raise'``: Raise `ValueError` for non-existent columns (default).
  - ``'ignore'``: Silently skip non-existent columns.
  - ``'warn'``: Warn and skip non-existent columns.

- CSV reading with metadata (`read_csv` from `superpandas`):

  - `include_metadata=True`: Raises `FileNotFoundError` if the corresponding metadata file (`*_metadata.json`) is not found.
  - `include_metadata=False`: Initializes empty metadata if the metadata file is not found (reads only the CSV).
