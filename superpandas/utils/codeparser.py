import re
import ast
from textwrap import dedent
from langchain_core.output_parsers.string import StrOutputParser
from typing import Literal, Dict
from pydantic import Field

class CodeBlobOutputParser(StrOutputParser):
    """A custom LangChain output parser to extract code blocks from LLM output.

    This parser extends LangChain's StrOutputParser to handle code block extraction
    from LLM outputs. It supports different programming languages/frameworks and
    can handle both markdown-style code blocks and direct code snippets.

    Attributes:
        framework (str): The programming language/framework to parse ("python" or "sql").
        patterns (Dict[str, str]): Regex patterns for different code block formats.

    Example:
        >>> parser = CodeBlobOutputParser(framework="python")
        >>> code = parser.parse("```python\nprint('hello')\n```")
    """

    framework: Literal["python", "sql"] = Field(default="python", description="The programming language/framework to parse")
    patterns: Dict[str, str] = Field(
        default={
            "python": r"```(?:py|python)\n(.*?)\n```",
            "sql": r"```(?:sql)?\n(.*?)\n```"
        },
        description="Regex patterns for different code block formats"
    )

    def __init__(self, framework: Literal["python", "sql"] = "python"):
        """Initialize the parser with the specified framework.

        Args:
            framework (Literal["python", "sql"], optional): The programming language/framework
                to parse. Valid values are "python" or "sql". Defaults to "python".

        Example:
            >>> parser = CodeBlobOutputParser("python")
            >>> parser.framework
            'python'
        """
        super().__init__(framework=framework)

    def parse(self, text: str) -> str:
        """Extract and validate code blocks from LLM output text.

        This method attempts to extract code blocks from the given text using regex patterns.
        If the text is already a valid code block (can be parsed as Python code or contains
        SQL keywords), it returns the text directly. For Python code, it validates the syntax
        using ast.parse().

        Args:
            text (str): The text to parse, which may contain code blocks or be a code block itself.

        Returns:
            str: The extracted and validated code block. Returns "NO_DATA_FOUND" for special cases.

        Raises:
            ValueError: If no valid code block is found and the text cannot be parsed as code.

        Example:
            >>> parser = CodeBlobOutputParser("python")
            >>> code = parser.parse("```python\nprint('hello')\n```")
            >>> print(code)
            print('hello')
        """
        # Check for "no data" message first
        if "Required information is not available in the given database." in text:
            return "NO_DATA_FOUND"
            
        pattern = self.patterns[self.framework]
        matches = re.findall(pattern, text, re.DOTALL)

        if matches:
            return "\n\n".join(match.strip() for match in matches)

        # Maybe the LLM outputted a code blob directly
        if self.framework == "python":
            try:
                ast.parse(text)
                return text
            except SyntaxError:
                pass
        elif self.framework == "sql":
            # Basic SQL validation - check for common SQL keywords
            sql_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP"]
            if any(keyword.lower() in text.lower() for keyword in sql_keywords):
                return text

        raise ValueError(
            dedent(
                f"""
                Your code snippet is invalid, because the regex pattern {pattern} was not found in it.
                Here is the generated code:
                {text}
                Make sure to include code with the correct pattern, for instance:
                Code:
                ```{self.framework}
                # Your {self.framework} code here
                ```
                """
            ).strip()
        )
