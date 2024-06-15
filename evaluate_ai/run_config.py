from pathlib import Path

from loguru import logger
from pydantic import BaseModel, ValidationError
import yaml

from evaluate_ai.evaluation import Provider


class RunConfig(BaseModel):
    """Defines the possible configurations for running this script.
    This config is expected to be loaded from a YAML file."""

    # Defines the models that each evaluation will be run on.
    # Must be a model supported by not_again_ai.llm.chat_completion
    models: dict[Provider, list[str]]

    # Which external model responses are coming from for manual evaluation
    external_models: list[str]
    # Which evaluations do we want to run on the external model
    external_evaluations: list[str]

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
                external_evals_data = data.get("evaluate_external", None)
                if external_evals_data:
                    config_data["external_models"] = external_evals_data.get("models", [])
                    config_data["external_evaluations"] = external_evals_data.get("evaluation_names", [])
            return cls(**config_data)
        except FileNotFoundError:
            logger.error(f"The file {path} does not exist.")
        except ValidationError as e:
            logger.error(f"Validation error for the YAML file {path}: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading the file {path}: {e}")
        return None
