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

SuperPandas is a lightweight Python package that extends the well known Pandas library with functionality for AI-powered data analytics. It is a barebones wrapper using Pandas Dataframe `accessors <https://pandas.pydata.org/docs/development/extending.html>`_ feature to add namespace 'super' which adds metadata and methods to dataframes for use with LLMs.

Key Goals:
-----------
* **Enhanced DataFrame with Metadata**: Add rich metadata support, automatically generate metadata, support multiple LLM providers, infer column types, generate schemas, and enable serialization.
* **Drop-in replacement for Pandas DataFrames**: Use SuperPandas as a lightweight wrapper without changing existing code.
* **Templated Prompt Generation**: Easily store and use templates for system and user prompts.

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
