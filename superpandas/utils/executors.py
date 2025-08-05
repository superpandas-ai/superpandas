from smolagents import LocalPythonExecutor, DockerExecutor
from smolagents.utils import parse_code_blobs, AgentParsingError, AgentExecutionError
from smolagents.local_python_executor import BASE_BUILTIN_MODULES
from smolagents.monitoring import AgentLogger, LogLevel

class CodeExecutor:
    """A class to execute code snippets in different environments.

    This class provides a unified interface for executing Python code in different
    environments like local Python interpreter or Docker containers. It handles code
    parsing, execution, and error handling.

    Attributes:
        additional_authorized_imports (list): Additional modules allowed for import.
        authorized_imports (list): Combined list of builtin and additional allowed imports.
        code_block_tags (tuple): Tags used to identify code blocks in text.
        logger (AgentLogger): Logger instance for execution logs.
        executor: The underlying executor instance (LocalPythonExecutor or DockerExecutor).

    Example:
        >>> executor = CodeExecutor(executor_type="local")
        >>> output, logs, is_final = executor.execute("print('hello')")
    """

    def __init__(self, executor_type: str = "local", 
                 docker_image: str = "python:3.10", 
                 additional_authorized_imports: list = [],
                 code_block_tags: str | tuple[str, str] | None = None,
                 logger: AgentLogger | None = None,
                 verbosity_level: LogLevel = LogLevel.INFO,):
        """Initialize the code executor.

        Args:
            executor_type (str, optional): Type of executor to use. Defaults to "local".
                Valid values are "local" or "docker".
            docker_image (str, optional): Docker image to use for execution. Defaults to "python:3.10".
            additional_authorized_imports (list, optional): Additional modules to allow importing.
                Defaults to [].
            code_block_tags (Union[str, tuple[str, str], None], optional): Tags to identify code blocks.
                Can be a string ("markdown") or a tuple of start/end tags. Defaults to None.
            logger (AgentLogger, optional): Logger instance to use. Defaults to None.
            verbosity_level (LogLevel, optional): Logging verbosity level. Defaults to LogLevel.INFO.

        Raises:
            ValueError: If an unsupported executor_type is provided.
        """
        if executor_type not in {"local", "e2b", "docker", "wasm"}:
            raise ValueError(f"Unsupported executor type: {executor_type}")
        
        self.additional_authorized_imports = additional_authorized_imports if additional_authorized_imports else []
        self.authorized_imports = sorted(set(BASE_BUILTIN_MODULES) | set(self.additional_authorized_imports))

        self.code_block_tags = (
            code_block_tags
            if isinstance(code_block_tags, tuple)
            else ("```python", "```")
            if code_block_tags == "markdown"
            else ("<code>", "</code>")
        )

        if logger is None:
            self.logger = AgentLogger(level=verbosity_level)
        else:
            self.logger = logger
        
        if executor_type == "local":
            self.executor = LocalPythonExecutor(additional_authorized_imports=self.authorized_imports)
        elif executor_type == "docker":
            self.executor = DockerExecutor(additional_imports=self.authorized_imports, logger=self.logger)
        else:
            raise ValueError(f"Invalid executor type: {executor_type}")

    def __enter__(self):
        """Context manager entry point.

        Returns:
            CodeExecutor: The executor instance.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit point.

        Args:
            exc_type: The type of the exception that was raised.
            exc_value: The instance of the exception that was raised.
            traceback: The traceback of the exception that was raised.
        """
        self.cleanup()

    def cleanup(self):
        """Clean up resources used by the executor.

        This method ensures proper cleanup of resources, particularly for remote
        Python executors or Docker containers.
        """
        if hasattr(self.executor, "cleanup"):
            self.executor.cleanup()

    def _parse_code(self, llm_output: str) -> str:
        """Parse code blocks from LLM output text.

        Args:
            llm_output (str): The raw text output from the LLM containing code blocks.

        Returns:
            str: The extracted code, ready for execution.

        Raises:
            AgentParsingError: If code parsing fails or no valid code blocks are found.
        """
        try:
            code = parse_code_blobs(llm_output)
            if isinstance(code, dict) and "code" in code:
                code = code["code"]
            self.logger.log_code(title="Executing parsed code:", content=code, level=LogLevel.INFO)
        except Exception as e:
            error_msg = f"Error in code parsing:\n{e}\nMake sure to provide correct code blobs."
            raise AgentParsingError(error_msg, self.logger)
        
        return code

    def execute(self, llm_output: str, state: dict = {}) -> tuple[str, str, bool]:
        """Execute code from LLM output with the configured executor.

        This method parses code blocks from the LLM output, executes them in the
        configured environment (local or Docker), and returns the execution results.

        Args:
            llm_output (str): The raw text output from the LLM containing code blocks.
            state (dict, optional): Variables to inject into the execution environment.
                Defaults to {}.

        Returns:
            tuple[str, str, bool]: A tuple containing:
                - output: The execution output or result
                - logs: Execution logs and stderr output
                - is_final_answer: Whether this is considered a final answer

        Raises:
            AgentParsingError: If code parsing fails.
            AgentExecutionError: If code execution fails.
        """

        code = self._parse_code(llm_output)
        self.executor.send_variables(state)
        self.executor.send_tools({})

        try:
            output, logs, is_final_answer = self.executor(code)
        except Exception as e:
            error_msg = str(e)
            if "Import of " in error_msg and " is not allowed" in error_msg:
                self.logger.log(
                    "[bold red]Warning to user: Code execution failed due to an unauthorized import - Consider passing said import under `additional_authorized_imports` when initializing your CodeAgent.",
                    level=LogLevel.INFO,
                )
            raise AgentExecutionError(error_msg, self.logger)
        
        return output, logs, is_final_answer