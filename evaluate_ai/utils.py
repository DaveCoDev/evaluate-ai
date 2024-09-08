from enum import Enum
from typing import Any

from not_again_ai.llm.gh_models.azure_ai_client import azure_ai_client
from not_again_ai.llm.openai_api.openai_client import openai_client
from not_again_ai.local_llm.ollama.ollama_client import ollama_client


class Provider(Enum):
    AZURE_OPENAI = "azure_openai"
    OLLAMA = "ollama"
    OPENAI = "openai_api"
    GH_MODELS = "gh_models"


def get_llm_client(provider_name: str) -> Any:
    if provider_name == Provider.OLLAMA:
        llm_client = ollama_client()
    elif provider_name == Provider.OPENAI:
        llm_client = openai_client()
    elif provider_name == Provider.AZURE_OPENAI:
        llm_client = openai_client(api_type="azure_openai")
    elif provider_name == Provider.GH_MODELS:
        llm_client = azure_ai_client()
    else:
        raise ValueError(f"Provider {provider_name} is not supported.")

    return llm_client
