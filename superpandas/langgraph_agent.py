"""
LangGraph agent for code execution in SuperPandas
"""
from typing import Dict, List, Optional, Any, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
import pandas as pd
import json
import traceback

from superpandas.llm_client import LLMClient, LLMMessage, LLMResponse
from superpandas.config import SuperPandasConfig
from superpandas.utils import CodeExecutor, CodeBlobOutputParser
from superpandas.templates import (
    langgraph_code_generation_template,
    langgraph_error_reflection_template,
    langgraph_reflection_analysis_template,
    langgraph_format_response_template
)

class AgentState(TypedDict):
    """State for the LangGraph agent"""
    messages: List[BaseMessage]
    current_query: str
    dataframe: pd.DataFrame
    generated_code: str
    result: Any
    error: str
    iterations: int
    formatted_response: str
    fig: Optional[Any]  # For matplotlib figures


def create_langgraph_agent(
    config: Optional[SuperPandasConfig] = None,
    max_iterations: int = 5
) -> StateGraph:
    """
    Create a LangGraph agent for data analysis

    Parameters:
    -----------
    config : SuperPandasConfig, optional
        Configuration for the agent. If None, uses default config.
    max_iterations : int, default 5
        Maximum number of iterations for the agent to try fixing errors.

    Returns:
    --------
    StateGraph: The compiled LangGraph agent
    """
    
    if config is None:
        config = SuperPandasConfig.get_default_config()
    
    # Initialize LLM client
    llm_client = LLMClient(config=config)
    
    # Create output parsers
    code_parser = CodeBlobOutputParser()
    
    executor = CodeExecutor(
            executor_type="local",
            code_block_tags = "markdown",
            additional_authorized_imports=[
                "pandas", "numpy", "matplotlib", "seaborn"
            ]
        )

    # Define the nodes for our graph
    # Node 1: Generate code
    def generate_code(state: AgentState) -> AgentState:
        """Generate code based on the current query and dataframe schema"""
        # Get the current query and dataframe schema
        current_query = state["current_query"]
        df = state["dataframe"]
        
        # Generate schema for the DataFrame
        if hasattr(df, 'super'):
            schema = df.super.get_schema(format_type='text')
        else:
            # Fallback schema generation
            schema = f"DataFrame with shape {df.shape}\nColumns: {list(df.columns)}\nTypes: {df.dtypes.to_dict()}"
        
        # Get the messages
        messages = state["messages"]

        # If we are generating code after reflection, we need to add the reflection to the messages
        if messages[-1].role == "assistant": 
            reflection = messages[-1].content
            error = state["error"]
            system_msg = langgraph_error_reflection_template.format(
                error=error,
                reflection=reflection
            )
        else:
            # Create messages for the LLM
            system_msg = langgraph_code_generation_template
        user_msg = f"DataFrame schema:\n{schema}\n\nUser query: {current_query}"
        
        llm_messages = [
            LLMMessage(role='system', content=system_msg),
            LLMMessage(role='user', content=user_msg)
        ]
        # Generate the code
        try:    
            response = llm_client.query(messages=llm_messages)
            generated_code = code_parser.parse(response.content)
        except Exception as e:
            generated_code = ""
            state["error"] = str(e)
            state["messages"].append(
                {"role": "assistant", "content": f"Error generating code: {str(e)}"}
            )

        # Update the state
        state["generated_code"] = generated_code
        state["iterations"] += 1

        return state

    # Node 2: Execute code
    def execute_code(state: AgentState) -> AgentState:
        """Execute the generated code safely"""
        generated_code = state["generated_code"]
        df = state["dataframe"]

        df_var = df.attrs['super']['name'] if df.attrs['super']['name']!='' else 'df'

        local_env = {
            df_var: df,
        }

        if generated_code == "NO_DATA_FOUND" or not generated_code.strip():
            state["error"] = "NO_DATA_FOUND"
            return state
        
        try:
            # Execute the code in a safe environment
            exec(generated_code, local_env)
            
            if "result" in local_env:
                state["result"] = local_env["result"]
            else:
                state["error"] = "Dataframe not set in the generated code."

            if "fig" in local_env:
                state["fig"] = local_env["fig"] 

        except Exception as e:
            state["error"] = str(e)
            state["messages"].append(
                {"role": "assistant", "content": f"Error executing code: {str(e)}"}
            )

        return state
    
    # Node 3: Reflect on errors
    def reflect(state: AgentState) -> AgentState:
        """Reflect on errors and provide insights on how to fix them"""
        # Get the error and code
        error = state["error"]
        code = state["generated_code"]

        # Generate reflection
        reflection_prompt_text = langgraph_reflection_analysis_template.format(
            error=error,
            code=code
        )
        
        llm_messages = [
            LLMMessage(role='system', content='You are a Python debugging expert.'),
            LLMMessage(role='user', content=reflection_prompt_text)
        ]
        
        response = llm_client.query(messages=llm_messages)
        reflection = response.content

        # Add the reflection to the messages
        state["messages"].append(
            {"role": "assistant", "content": f"Reflection on the error: {reflection}"}
        )

        # Clear the error
        state["error"] = ""

        return state

    # Node 4: Format response
    def format_response(state: AgentState) -> AgentState:
        """Format the response into proper text form"""
        # Get the response and current query
        result = state["result"]
        generated_code = state["generated_code"]

        # Format the response
        format_prompt_text = langgraph_format_response_template.format(
            query=state["current_query"],
            code=generated_code,
            result=result
        )
        
        llm_messages = [
            LLMMessage(role='system', content='You are a data analysis expert who explains results clearly.'),
            LLMMessage(role='user', content=format_prompt_text)
        ]
        
        response = llm_client.query(messages=llm_messages)
        formatted_response = response.content

        # Update the state
        state["formatted_response"] = formatted_response

        return state

    # Node 5: Check for execution errors
    def check_execution_errors(state: AgentState) -> str:
        """Check if there are any errors in the execution"""
        # If there's an error, we need to fix it
        if state["error"]:
            # If we've reached the maximum number of iterations, end
            if state["iterations"] >= max_iterations or state["error"] == "NO_DATA_FOUND":
                return "end"
            else:
                # Otherwise, reflect on the error and try to fix it
                return "reflect"

        # If there's no error, format the response
        return "format"
    
    # Node 6: Check for code generation errors
    def check_codegen_errors(state: AgentState) -> str:
        """Check if there are any errors in the code generation"""
        # If there's an error, we need to fix it
        if state["error"]:
            # If we've reached the maximum number of iterations, end
            if state["iterations"] >= max_iterations:
                return "end"
            else:
                # Otherwise, reflect on the error and try to fix it
                return "reflect"

        # If there's no error, execute the code
        return "execute_code"

    # Create the graph
    workflow = StateGraph(AgentState)

    # Add the nodes
    workflow.add_node("generate_code", generate_code)
    workflow.add_node("execute_code", execute_code)
    workflow.add_node("reflect", reflect)
    workflow.add_node("format", format_response)

    # Add the edges
    workflow.add_edge("generate_code", "execute_code")
    workflow.add_conditional_edges(
        "execute_code",
        check_execution_errors,
        {"reflect": "reflect", "format": "format", "end": END},
    )
    workflow.add_conditional_edges(
        "generate_code",
        check_codegen_errors,
        {"reflect": "reflect", "execute_code": "execute_code", "end": END},
    )
    workflow.add_edge("reflect", "generate_code")
    workflow.add_edge("format", END)

    # Set the entry point
    workflow.set_entry_point("generate_code")

    # Compile the graph
    agent = workflow.compile()

    return agent


def run_agent(
    query: str,
    dataframe: pd.DataFrame,
    config: Optional[SuperPandasConfig] = None,
    max_iterations: int = 5
) -> Dict[str, Any]:
    """
    Run the LangGraph agent to analyze a DataFrame
    
    Parameters:
    -----------
    query : str
        The analysis query to perform
    dataframe : pd.DataFrame
        The DataFrame to analyze
    config : SuperPandasConfig, optional
        Configuration for the agent
    max_iterations : int, default 5
        Maximum number of iterations
        
    Returns:
    --------
    Dict[str, Any]
        The agent's final state including results and any errors
    """
    
    # Create the agent
    agent = create_langgraph_agent(config=config, max_iterations=max_iterations)
    
    # Initialize the state
    initial_state = {
        "messages": [],
        "current_query": query,
        "dataframe": dataframe,
        "generated_code": "",
        "result": None,
        "error": "",
        "iterations": 0,
        "formatted_response": "",
        "fig": None
    }
    
    # Run the agent
    final_state = agent.invoke(initial_state)
    
    return final_state 

if __name__ == "__main__":
    import superpandas as spd
    sdf = spd.read_csv("/home/haris/git/superpandas/tests/titanic_sdf.csv")
    output = run_agent(query="what's the age distribution for different genders", dataframe=sdf)
    print(output)