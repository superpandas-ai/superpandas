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


class LangGraphAgent:
    """LangGraph agent for code execution in SuperPandas"""
    
    def __init__(self, config: Optional[SuperPandasConfig] = None, max_iterations: int = 5):
        """
        Initialize the LangGraph agent.
        
        Parameters:
        -----------
        config : SuperPandasConfig, optional
            Configuration for the agent. If None, uses default config.
        max_iterations : int, default 5
            Maximum number of iterations for the agent to try fixing errors.
        """
        self.config = config if config is not None else SuperPandasConfig.get_default_config()
        self.max_iterations = max_iterations
        
        # Initialize components
        self.llm_client = LLMClient(config=self.config)
        self.code_parser = CodeBlobOutputParser()
        self.executor = CodeExecutor(
            executor_type="local",
            code_block_tags="markdown",
            additional_authorized_imports=[
                "pandas", "numpy", "matplotlib", "seaborn"
            ]
        )
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _generate_code(self, state: AgentState) -> AgentState:
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
            response = self.llm_client.query(messages=llm_messages)
            generated_code = self.code_parser.parse(response.content)
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

    def _execute_code(self, state: AgentState) -> AgentState:
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
    
    def _reflect(self, state: AgentState) -> AgentState:
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
        
        response = self.llm_client.query(messages=llm_messages)
        reflection = response.content

        # Add the reflection to the messages
        state["messages"].append(
            {"role": "assistant", "content": f"Reflection on the error: {reflection}"}
        )

        # Clear the error
        state["error"] = ""

        return state

    def _format_response(self, state: AgentState) -> AgentState:
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
        
        response = self.llm_client.query(messages=llm_messages)
        formatted_response = response.content

        # Update the state
        state["formatted_response"] = formatted_response

        return state

    def _check_execution_errors(self, state: AgentState) -> str:
        """Check if there are any errors in the execution"""
        # If there's an error, we need to fix it
        if state["error"]:
            # If we've reached the maximum number of iterations, end
            if state["iterations"] >= self.max_iterations or state["error"] == "NO_DATA_FOUND":
                return "end"
            else:
                # Otherwise, reflect on the error and try to fix it
                return "reflect"

        # If there's no error, format the response
        return "format"
    
    def _check_codegen_errors(self, state: AgentState) -> str:
        """Check if there are any errors in the code generation"""
        # If there's an error, we need to fix it
        if state["error"]:
            # If we've reached the maximum number of iterations, end
            if state["iterations"] >= self.max_iterations:
                return "end"
            else:
                # Otherwise, reflect on the error and try to fix it
                return "reflect"

        # If there's no error, execute the code
        return "execute_code"

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)

        # Add the nodes
        workflow.add_node("generate_code", self._generate_code)
        workflow.add_node("execute_code", self._execute_code)
        workflow.add_node("reflect", self._reflect)
        workflow.add_node("format", self._format_response)

        # Add the edges
        workflow.add_edge("generate_code", "execute_code")
        workflow.add_conditional_edges(
            "execute_code",
            self._check_execution_errors,
            {"reflect": "reflect", "format": "format", "end": END},
        )
        workflow.add_conditional_edges(
            "generate_code",
            self._check_codegen_errors,
            {"reflect": "reflect", "execute_code": "execute_code", "end": END},
        )
        workflow.add_edge("reflect", "generate_code")
        workflow.add_edge("format", END)

        # Set the entry point
        workflow.set_entry_point("generate_code")

        # Compile the graph
        return workflow.compile()

    def run(self, query: str, dataframe: pd.DataFrame) -> Dict[str, Any]:
        """
        Run the LangGraph agent to analyze a DataFrame
        
        Parameters:
        -----------
        query : str
            The analysis query to perform
        dataframe : pd.DataFrame
            The DataFrame to analyze
            
        Returns:
        --------
        Dict[str, Any]
            The agent's final state including results and any errors
        """
        message = LLMMessage(role='user', content=query)
        
        # Initialize the state
        initial_state = {
            "messages": [message],
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
        final_state = self.graph.invoke(initial_state)
        
        return final_state


if __name__ == "__main__":
    import superpandas as spd
    sdf = spd.read_csv("/home/haris/git/superpandas/tests/titanic_sdf.csv")
    agent = LangGraphAgent()
    output = agent.run(query="what's the age distribution for different genders", dataframe=sdf)
    print(output)