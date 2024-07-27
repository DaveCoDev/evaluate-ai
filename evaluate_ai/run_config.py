from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger
from not_again_ai.llm.openai_api.openai_client import openai_client
from not_again_ai.local_llm.ollama.ollama_client import ollama_client
from pydantic import BaseModel, ValidationError
import yaml


class Provider(Enum):
    OPENAI = "openai_api"
    OLLAMA = "ollama"


def get_llm_client(provider_name: str) -> Any:
    if provider_name == Provider.OLLAMA:
        llm_client = ollama_client()
    elif provider_name == Provider.OPENAI:
        llm_client = openai_client()
    else:
        raise ValueError(f"Provider {provider_name} is not supported.")

    return llm_client


class RunConfig(BaseModel):
    """Defines the possible configurations for running this script.
    This config is expected to be loaded from a YAML file."""

    # Defines the models that each evaluation will be run on.
    # Must be a model supported by not_again_ai.llm.chat_completion
    models: dict[Provider, list[str]]

    # For any evaluations that require an LLM, this determines which model that is
    evaluation_model: str
    evaluation_provider: Provider

    @classmethod
    def load(cls, path: Path) -> "RunConfig":
        """Class method to load, validate, and handle errors for the configuration from a YAML file.

        Args:
            path (Path): The path to the YAML configuration file.

        Returns:
            ModelConfig: The validated configuration if successful; None if an error occurs.

        Notes:
            Logs error messages instead of raising exceptions to allow execution to continue.
        """
        try:
            with path.open() as file:
                data = yaml.safe_load(file)
                config_data = {}
                config_data["models"] = data.get("models", {})
                config_data["evaluation_model"] = data.get("evaluation_model", None)
                config_data["evaluation_provider"] = data.get("evaluation_provider", None)
            return cls(**config_data)
        except FileNotFoundError:
            logger.error(f"The file {path} does not exist.")
        except ValidationError as e:
            logger.error(f"Validation error for the YAML file {path}: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading the file {path}: {e}")
        return None
