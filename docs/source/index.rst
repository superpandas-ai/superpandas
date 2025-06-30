.. SuperPandas documentation master file, created by
   sphinx-quickstart on Mon Jul 22 13:18:55 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: _static/spd_logo.png
   :alt: SuperPandas Logo
   :align: center
   :width: 200px


Welcome to SuperPandas's documentation!
========================================

SuperPandas is a lightweight Python package that enhances the popular Pandas library with AI-powered data analytics capabilities. It introduces a minimalistic wrapper using Pandas DataFrame `accessors <https://pandas.pydata.org/docs/development/extending.html>`_ to add a custom super namespace, enabling metadata integration and specialized methods designed for seamless interaction with large language models (LLMs).

Key Goals:
-----------
* **Enhanced DataFrame with Metadata**: Add rich metadata support to facilitate LLM-based data analytics, including dataframe name, description, column types, column descriptions and schema templates. Support of automatically inferring metadata from the dataframe.
* **Drop-in replacement for Pandas DataFrames**: SuperPandas can be used as a drop-in replacement for Pandas DataFrames without changing existing code.
* **Templated Prompt Generation**: Easily store and use templates for system and user prompts. This allows for easy reuse of prompts and for creating custom prompts for specific use cases.
* **Multiple LLM Providers**: Support multiple LLM providers like OpenAI, Anthropic, Google, and more. This allows for easy integration with different LLM providers and models. 
* **MIT License**: SuperPandas is released under the MIT License, allowing for free use and modification.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   api
   changelog
   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
