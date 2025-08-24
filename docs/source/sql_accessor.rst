.. _sql_accessor:

SQL Accessor
============

The SQL accessor in SuperPandas provides a powerful way to execute SQL queries on pandas DataFrames using SQLite as the backend engine. This feature brings the power of SQL to pandas DataFrames, enabling complex data operations with familiar SQL syntax.

.. note::
   The SQL accessor functionality is inspired by and builds upon concepts from the `pandasql <https://github.com/yhat/pandasql>`_ project. We acknowledge and thank the pandasql contributors for their work. See :ref:`license` for more information about third-party licenses.

Overview
--------

The SQL accessor is automatically registered when you import SuperPandas and is available on all pandas DataFrames as the `.sql` namespace. It provides:

- **In-memory SQLite database** for fast queries
- **Multiple table support** via the `env` parameter
- **Full SQL support** including SELECT, WHERE, JOIN, GROUP BY, HAVING, ORDER BY, etc.
- **Type safety** with comprehensive error handling and validation
- **Custom database URIs** for persistent storage
- **String & date functions** for data manipulation
- **Conditional logic** with CASE statements

Basic Usage
-----------

Getting Started
~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   import superpandas  # This registers the SQL accessor

   # Create a simple DataFrame
   df = pd.DataFrame({
       "id": [1, 2, 3, 4, 5],
       "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
       "age": [25, 30, 35, 28, 32],
       "salary": [50000, 60000, 70000, 55000, 65000]
   })

   # Basic query
   result = df.sql.query("SELECT * FROM df WHERE age > 30")
   print(result)

Simple Filtering
~~~~~~~~~~~~~~~

.. code-block:: python

   # Filter by condition
   result = df.sql.query("SELECT name, age FROM df WHERE salary > 60000")

   # Multiple conditions
   result = df.sql.query("""
       SELECT * FROM df 
       WHERE age > 25 AND salary > 55000
       ORDER BY salary DESC
   """)

Aggregations
~~~~~~~~~~~~

.. code-block:: python

   # Basic aggregations
   result = df.sql.query("""
       SELECT 
           AVG(age) as avg_age,
           MAX(salary) as max_salary,
           MIN(salary) as min_salary,
           COUNT(*) as total_employees
       FROM df
   """)

   # Group by aggregations
   result = df.sql.query("""
       SELECT 
           age_group,
           COUNT(*) as count,
           AVG(salary) as avg_salary
       FROM (
           SELECT 
               CASE 
                   WHEN age < 30 THEN 'Young'
                   WHEN age < 40 THEN 'Middle'
                   ELSE 'Senior'
               END as age_group,
               salary
           FROM df
       )
       GROUP BY age_group
       ORDER BY avg_salary DESC
   """)

Working with Multiple Tables
---------------------------

The SQL accessor supports querying multiple DataFrames by providing them in the `env` parameter.

Basic Joins
~~~~~~~~~~~

.. code-block:: python

   # Create multiple DataFrames
   employees = pd.DataFrame({
       "id": [1, 2, 3, 4],
       "name": ["Alice", "Bob", "Charlie", "Diana"],
       "department_id": [1, 1, 2, 3]
   })

   departments = pd.DataFrame({
       "id": [1, 2, 3],
       "name": ["Engineering", "Sales", "Marketing"],
       "budget": [500000, 300000, 200000]
   })

   # Join tables
   env = {"departments": departments}
   result = employees.sql.query("""
       SELECT 
           e.name as employee_name,
           d.name as department_name,
           d.budget
       FROM df e
       JOIN departments d ON e.department_id = d.id
       ORDER BY d.budget DESC
   """, env=env)

Complex Joins
~~~~~~~~~~~~~

.. code-block:: python

   # Create additional tables
   salaries = pd.DataFrame({
       "employee_id": [1, 2, 3, 4],
       "amount": [80000, 90000, 70000, 75000],
       "year": [2023, 2023, 2023, 2023]
   })

   projects = pd.DataFrame({
       "id": [1, 2, 3],
       "name": ["Project A", "Project B", "Project C"],
       "department_id": [1, 1, 2]
   })

   # Complex multi-table join
   env = {
       "departments": departments,
       "salaries": salaries,
       "projects": projects
   }

   result = employees.sql.query("""
       SELECT 
           e.name as employee_name,
           d.name as department_name,
           s.amount as salary,
           COUNT(p.id) as project_count
       FROM df e
       JOIN departments d ON e.department_id = d.id
       JOIN salaries s ON e.id = s.employee_id
       LEFT JOIN projects p ON d.id = p.department_id
       WHERE s.year = 2023
       GROUP BY e.id, e.name, d.name, s.amount
       ORDER BY s.amount DESC
   """, env=env)

Advanced SQL Features
---------------------

String Functions
~~~~~~~~~~~~~~~

.. code-block:: python

   # String manipulation
   result = df.sql.query("""
       SELECT 
           name,
           UPPER(name) as upper_name,
           LOWER(name) as lower_name,
           LENGTH(name) as name_length,
           SUBSTR(name, 1, 3) as name_prefix,
           REPLACE(name, 'a', 'A') as replaced_name
       FROM df
       WHERE name LIKE '%a%'
   """)

Date Functions
~~~~~~~~~~~~~~

.. code-block:: python

   # Create DataFrame with dates
   df_dates = pd.DataFrame({
       "name": ["Alice", "Bob", "Charlie"],
       "hire_date": pd.date_range("2023-01-01", periods=3, freq="M"),
       "salary": [50000, 60000, 70000]
   })

   # Date operations
   result = df_dates.sql.query("""
       SELECT 
           name,
           hire_date,
           STRFTIME('%Y-%m', hire_date) as hire_month,
           STRFTIME('%W', hire_date) as week_number,
           STRFTIME('%w', hire_date) as day_of_week,
           JULIANDAY('2024-01-01') - JULIANDAY(hire_date) as days_since_hire
       FROM df
       ORDER BY hire_date
   """)

Conditional Logic
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # CASE statements
   result = df.sql.query("""
       SELECT 
           name,
           age,
           salary,
           CASE 
               WHEN age < 30 THEN 'Young'
               WHEN age < 40 THEN 'Middle'
               ELSE 'Senior'
           END as age_group,
           CASE 
               WHEN salary < 55000 THEN 'Low'
               WHEN salary < 65000 THEN 'Medium'
               ELSE 'High'
           END as salary_level
       FROM df
       ORDER BY salary DESC
   """)

Custom Database URIs
-------------------

By default, the SQL accessor uses an in-memory SQLite database. You can specify a custom database URI for persistent storage.

.. code-block:: python

   # Use persistent database
   result = df.sql.query(
       "SELECT * FROM df WHERE age > 30",
       db_uri="sqlite:///my_database.db"
   )

   # Use different SQLite database
   result = df.sql.query(
       "SELECT * FROM df WHERE age > 30",
       db_uri="sqlite:///path/to/another/database.db"
   )

Error Handling
--------------

The SQL accessor provides comprehensive error handling and validation:

.. code-block:: python

   # Empty query
   try:
       result = df.sql.query("")
   except ValueError as e:
       print(f"Error: {e}")

   # Invalid table reference
   try:
       result = df.sql.query("SELECT * FROM nonexistent_table")
   except RuntimeError as e:
       print(f"Error: {e}")

   # Invalid environment parameter
   try:
       result = df.sql.query("SELECT * FROM df", env="not_a_dict")
   except TypeError as e:
       print(f"Error: {e}")

   # Invalid table names in environment
   try:
       result = df.sql.query("SELECT * FROM df", env={123: df})
   except TypeError as e:
       print(f"Error: {e}")

Performance Considerations
-------------------------

- **In-memory operations**: The default in-memory SQLite database provides fast query execution
- **Large DataFrames**: For very large DataFrames, consider using persistent databases
- **Multiple queries**: Reuse the same DataFrame for multiple queries to avoid repeated data loading
- **Index optimization**: SQLite automatically creates indexes for better performance on repeated queries

Best Practices
--------------

1. **Use meaningful table aliases**: When joining multiple tables, use clear aliases for better readability
2. **Validate data types**: Ensure your DataFrames have appropriate data types for SQL operations
3. **Handle null values**: Use IS NULL and IS NOT NULL in your queries when dealing with missing data
4. **Use parameterized queries**: For dynamic queries, consider building SQL strings carefully to avoid injection
5. **Test complex queries**: Break down complex queries into smaller parts for easier debugging

Examples
--------

Complete Example
~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   import superpandas

   # Sample data
   employees = pd.DataFrame({
       "id": [1, 2, 3, 4, 5],
       "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
       "age": [25, 30, 35, 28, 32],
       "department": ["Engineering", "Engineering", "Sales", "Marketing", "Sales"],
       "salary": [80000, 90000, 70000, 75000, 72000],
       "hire_date": pd.date_range("2023-01-01", periods=5, freq="M")
   })

   departments = pd.DataFrame({
       "name": ["Engineering", "Sales", "Marketing"],
       "budget": [500000, 300000, 200000],
       "location": ["Building A", "Building B", "Building C"]
   })

   # Complex analysis query
   env = {"departments": departments}
   result = employees.sql.query("""
       SELECT 
           d.name as department_name,
           d.location,
           COUNT(e.id) as employee_count,
           AVG(e.age) as avg_age,
           AVG(e.salary) as avg_salary,
           SUM(e.salary) as total_salary,
           ROUND(SUM(e.salary) * 100.0 / d.budget, 2) as salary_percent_of_budget,
           STRFTIME('%Y-%m', MIN(e.hire_date)) as earliest_hire_month
       FROM df e
       JOIN departments d ON e.department = d.name
       GROUP BY d.name, d.location, d.budget
       HAVING AVG(e.salary) > 70000
       ORDER BY avg_salary DESC
   """, env=env)

   print(result)
