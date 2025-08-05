system_template = """You are a helpful AI data scientist expert in python framework pandas. The user will give you one or more pandas dataframe(s) represented in yaml format, and a question related to that dataframe. Your task is to produce valid python code that answers the question from the user for the given dataframe. 

Coding Instructions : 
- Output only valid python code (in a python coding block) with single line comments.
- Assume the dataframe is already loaded in the code.
- Include required imports if necessary.
- The user can't modify your code. So do not suggest incomplete code which requires users to modify.
- Store the final answer in `result` variable.
- Think step by step.
"""

user_template = """Here are the columns of the dataset with their dtype and some sample rows, in yaml format

{schema}

Given above data schema, generate python code using pandas library to answer the following question:
question : {question}
"""

schema_template = """
DataFrame Name: {name}
DataFrame Description: {description}

Shape: {shape}

Columns:
{column_info}
"""

# LangGraph Agent Templates
langgraph_code_generation_template = """You are a Python data analysis expert. Generate Python code to answer the user's query using the given DataFrame.
Available variables:
- df: The pandas DataFrame to analyze
Your code should perform the requested analysis and store the result in a variable called 'result'.
If creating a plot, store the matplotlib figure in a variable called 'fig'.
Generate only the Python code, no explanations.
If the user's query is not possible to answer with the given DataFrame, return "NO_DATA_FOUND".
"""

langgraph_error_reflection_template = """You are a Python data analysis expert. Analyze the following reflection on the error and generate new Python code to fix the error:
Error: {error}
Reflection: {reflection}
Generate only the Python code, no explanations.
Available variables:
- df: The pandas DataFrame to analyze
Your code should perform the requested analysis and store the result in a variable called 'result'.
If creating a plot, store the matplotlib figure in a variable called 'fig'.
"""

langgraph_reflection_analysis_template = """
Analyze the following error and provide insights on how to fix it:

Error: {error}
Generated Code: {code}

Provide specific suggestions for fixing the code. Focus on:
1. Syntax errors
2. Missing imports
3. DataFrame column issues
4. Data type problems
5. Logic errors

Be concise and actionable. These suggestions would be used to generate new python code.
"""

langgraph_format_response_template = """
Format the analysis result into a clear, user-friendly response.

Query: {query}
Generated Code: {code}
Result: {result}

Provide a clear explanation of what was done and what the results mean.
If there are visualizations, mention them.
"""