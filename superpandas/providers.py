"""
Provider management for SuperPandas LLM clients.
"""
# Import individual providers
available_providers = {}

try:
    from smolagents import LiteLLMModel
    available_providers['lite'] = LiteLLMModel
except ImportError:
    pass

try:
    from smolagents import OpenAIServerModel
    available_providers['openai'] = OpenAIServerModel
except ImportError:
    pass

try:
    from smolagents import InferenceClientModel
    available_providers['hf'] = InferenceClientModel
except ImportError:
    pass

try:
    from smolagents import TransformersModel
    available_providers['tf'] = TransformersModel
except ImportError:
    pass

try:
    from smolagents import VLLMModel
    available_providers['vllm'] = VLLMModel
except ImportError:
    pass

try:
    from smolagents import MLXModel
    available_providers['mlx'] = MLXModel
except ImportError:
    pass

try:
    from smolagents import AzureOpenAIServerModel
    available_providers['openai_az'] = AzureOpenAIServerModel
except ImportError:
    pass

try:
    from smolagents import AmazonBedrockServerModel
    available_providers['bedrock'] = AmazonBedrockServerModel
except ImportError:
    pass 