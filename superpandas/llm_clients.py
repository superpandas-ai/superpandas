from openai import OpenAI
from huggingface_hub import InferenceClient
# from vllm import LLM, SamplingParams
from typing import Any, List
from abc import ABC, abstractmethod
import os
from .utils import valid_url


class LLMClient(ABC):

    @abstractmethod
    def __init__(self,
                 ) -> None:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError

    def _verify_messages(self, messages):
        """
        Verify the messages format for the chat API. The messages should be of the form:
        [
            {
                "role": "system",
                "content": "Hello, how can I help you today?"
            },
            {
                "role": "user",
                "content": "I would like to book a flight."
            },
            {
                "role": "assistant",
                "content": "Sure, where would you like to fly to?"
            }
        ]
        """
        if not isinstance(messages, list):
            raise ValueError("messages must be a list of dictionaries")
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("messages must be a list of dictionaries")
            if "role" not in message:
                raise ValueError("Each message must have a 'role' key")
            if "content" not in message:
                raise ValueError("Each message must have a 'content' key")
            if message["role"] not in ["system", "user", "assistant"]:
                raise ValueError(
                    "Each message must have a 'role' key with value 'system', 'user' or 'assistant'")


class OpenAIClient(LLMClient):
    """
    OpenAIClient class for interfacing with OpenAI API. 

    This class provides a convenient interface for interacting with the OpenAI API, specifically for text generation tasks.
    It supports different models and allows sending messages to the API for generating responses.

    Args:
        api_key (str, optional): The API key for accessing the OpenAI API. If not provided, it will be fetched from the environment variable OPENAI_API_KEY or HF_API_KEY. Defaults to None.
        base_url (str, optional): The base URL for the API endpoint. Required for the Text Generation Interface Messages API. Defaults to None.
        model (str, optional): The model to use for text generation. Defaults to "gpt-3.5-turbo".

    Usage:
        To use with HuggingFace Text Generation Interface Messages API, use model='tgi' and provide the base_url for the TGI Server.
        To use with HuggingFace Inference Endpoints, provide the HF API starting with 'hf_' or set HF_API_KEY as environment variable.

    Raises:
        AssertionError: For OpenAI, if the provided model is not found in the available models.
        AssertionError: If the API key is not provided either as an argument or as an environment variable.

    """

    def __init__(self,
                 api_key: str = None,
                 base_url: str = None,
                 model: str = "gpt-3.5-turbo",
                 ) -> None:

        if model == 'tgi':
            print('Using Text Generation Interface Messages API')
            assert base_url is not None, "Please provide a base_url for the Text Generation Interface Messages API"
            api_key = "-"
        else:
            if base_url is not None and api_key.startswith('hf_'):
                print('Using HuggingFace Inference Endpoints')
                env_api_key = os.environ.get("HF_API_KEY")
            else:
                print('Using OpenAI Client')
                env_api_key = os.environ.get("OPENAI_API_KEY")

            if api_key is None and env_api_key is not None:
                print("Using API key from environment variable.")

            api_key = api_key if api_key is not None else env_api_key
            assert api_key is not None, "Please provide an API key either as an argument or as an environment variable : OPENAI_API_KEY/HF_API_KEY"

        self.client = OpenAI(api_key=api_key,
                             base_url=base_url,
                             )
        if model != 'tgi':
            available_models = [c.id for c in self.client.models.list()]
            assert model in available_models, f"Model {model} not found in available models: {available_models}"
            print("Initialized OpenAI client with model:", model)

        self.model = model

    def __call__(self, messages: List[dict], stream=False) -> Any:
        """
        Generate completions for the given messages.

        Args:
            messages (List[dict]): A list of messages to generate completions for.
            stream (bool, optional): Whether to stream the response. Defaults to False.
        """

        self._verify_messages(messages)

        output = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            stream=stream
        )

        return output


class TGIClient(LLMClient):

    def __init__(self,
                 api_key: str = None,
                 timeout: float = None,
                 model: str = "HuggingFaceH4/zephyr-7b-beta",
                 ) -> None:

        # For self hosted TGI server, we don't need the api key.
        if not valid_url(model):
            env_api_key = os.environ.get("HF_API_KEY")
            if api_key is None and env_api_key is not None:
                print("Using HuggingFace API key from environment variable.")

            api_key = api_key if api_key is not None else env_api_key
            assert api_key is not None, "Please provide a HuggingFace API key either as an argument or as an environment variable : HF_API_KEY"

        self.client = InferenceClient(model=model,
                                      token=api_key,
                                      timeout=timeout)

    def __call__(self,
                 messages,
                 stream=False,
                 max_tokens=100) -> Any:

        self._verify_messages(messages)

        # TODO : Add functionality for other tasks (https://huggingface.co/docs/huggingface_hub/main/en/guides/inference#supported-tasks)
        # TODO : Replace these and other optional parameters with a config object like SamplingParams

        output = self.client.chat_completion(messages=messages,
                                             stream=stream,
                                             max_tokens=max_tokens,
                                             )

        return output


# class vLLMClient(LLMClient):

#     def __init__(self,
#                  api_key: str,
#                  model: str,
#                  # TODO Replace all optional parameters with a config object like ModelParams
#                  tokenizer: str | None = None,
#                  tokenizer_mode: str = "auto",
#                  trust_remote_code: bool = False,
#                  tensor_parallel_size: int = 1,
#                  dtype: str = "auto",
#                  quantization: str | None = None,
#                  revision: str | None = None,
#                  tokenizer_revision: str | None = None,
#                  seed: int = 0,
#                  gpu_memory_utilization: float = 0.9,
#                  swap_space: int = 4,
#                  temperature: float = 1.0,
#                  top_p: float = 1.0,
#                  ) -> None:
#         super().__init__(api_key, model)

#         self.client = LLM(model=model,
#                           revision=revision,
#                           tokenizer=tokenizer,
#                           tokenizer_mode=tokenizer_mode,
#                           trust_remote_code=trust_remote_code,
#                           tensor_parallel_size=tensor_parallel_size,
#                           dtype=dtype,
#                           quantization=quantization,
#                           gpu_memory_utilization=gpu_memory_utilization,
#                           tokenizer_revision=tokenizer_revision,
#                           seed=seed,
#                           swap_space=swap_space
#                           )
#         self.sampling_params = SamplingParams(temperature=temperature,
#                                               top_p=top_p)

#     def __call__(self, query, sampling_params: SamplingParams = None) -> Any:

#         sampling_params = self.sampling_params if sampling_params is None else sampling_params

#         output = self.client.generate(query, sampling_params)

#         return output
