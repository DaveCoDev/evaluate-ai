import os

from not_again_ai.llm.chat_completion.providers.ollama_api import ollama_client
from not_again_ai.llm.chat_completion.providers.openai_api import openai_client

OLLAMA_CLIENT = ollama_client()
OPENAI_CLIENT = openai_client(api_type="openai", api_key=os.getenv("OPENAI_API_KEY"))
AZURE_OPENAI_CLIENT = openai_client(api_type="azure_openai", api_key=os.getenv("AZURE_OPENAI_PAI_KEY"))
