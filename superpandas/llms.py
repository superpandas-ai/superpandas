from openai import OpenAI
from huggingface_hub import InferenceClient
# from vllm import LLM, SamplingParams
from typing import Any
from abc import ABC, abstractmethod
import os


class LLMClient(ABC):

    @abstractmethod
    def __init__(self,
                 ) -> None:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError


class OpenAIClient(LLMClient):

    def __init__(self,
                 api_key: str,
                 base_url: str = None,
                 max_retries: int = 2,
                 model: str = "gpt-3.5-turbo",
                 ) -> None:

        env_api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is None and env_api_key is not None:
            print("Using OpenAI API key from environment variable.")

        api_key = api_key if api_key is not None else env_api_key
        assert api_key is not None, "Please provide an OpenAI API key either as an argument or as an environment variable : OPENAI_API_KEY"

        self.client = OpenAI(api_key=api_key,
                             base_url=base_url,
                            #  max_retries=max_retries
                             )
        self.model = model

    def __call__(self, query, role='user') -> Any:

        output = self.client.chat.completions.create(
            messages=[
                {
                    "role": role,
                    "content": query,
                }
            ],
            model=self.model,
        )
        return output


class TGIClient(LLMClient):

    def __init__(self,
                 api_key: str,
                 timeout: float = None,
                 model: str = "HuggingFaceH4/zephyr-7b-beta",
                 ) -> None:

        env_api_key = os.environ.get("HF_API_KEY")
        if api_key is None and env_api_key is not None:
            print("Using HuggingFace API key from environment variable.")
            
        api_key = api_key if api_key is not None else env_api_key
        assert api_key is not None, "Please provide a HuggingFace API key either as an argument or as an environment variable : HF_API_KEY"

        self.client = InferenceClient(model=model,
                                      token=api_key,
                                      timeout=timeout)

    def __call__(self,
                 prompt,
                 details=False,  # TODO Replace all optional parameters with a config object like SamplingParams
                 stream=False,
                 max_new_tokens=100) -> Any:

        output = self.client.text_generation(prompt=prompt,
                                             details=details,
                                             stream=stream,
                                             max_new_tokens=max_new_tokens)

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

