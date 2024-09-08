"""Defines abstractions for implementing custom evaluations."""

from abc import ABC, abstractmethod
import importlib
from pathlib import Path
from typing import Literal

from loguru import logger
import pendulum
from pydantic import BaseModel, Field, ValidationError
from pydantic_extra_types.pendulum_dt import DateTime
from rich.progress import Progress
from tinydb import TinyDB
import yaml

from evaluate_ai.tinydb_helpers.db_path import TINYDB_PATH
from evaluate_ai.utils import Provider


class EvaluationRunConfig(BaseModel):
    """Defines general configuration parameters for this evaluation.

    The evaluation must define an evaluation_type to uniquely identify it.
    Models can be defined, but the evaluation can determine how to use this information.
    """

    module_name: str = Field(metadata={"description": "The name of the module the evaluation is in."})
    class_name: str = Field(metadata={"description": "The name of the class in the module."})
    # Defines the models that each evaluation will be run on.
    # It is up to the evaluation to determine how to use this information.
    models: dict[Provider, list[str]] | None


class EvaluationInstance(BaseModel):
    """Defines an individual evaluation instance within this evaluation.

    Each instance must have a name.
    Other parameters can be extended as needed depending on the evaluation.
    """

    name: str = Field(
        None,
        metadata={
            "description": "The name of the evaluation used to identify the evaluation, which can be anything, but should be unique within this evaluation."
        },
    )


class EvaluationConfig(BaseModel):
    """Defines the overall configuration for this evaluation."""

    run_config: EvaluationRunConfig
    evaluation_instances: list[EvaluationInstance]

    @classmethod
    def load(cls, path: Path) -> "EvaluationRunConfig":
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
                config_data["run_config"] = data.get("run_config", {})
                config_data["evaluation_instances"] = data.get("evaluation_instances", [])
            return cls(**config_data)
        except FileNotFoundError:
            logger.error(f"The file {path} does not exist.")
        except ValidationError as e:
            logger.error(f"Validation error for the YAML file {path}: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading the file {path}: {e}")
        return None


class EvaluationBaseOutput(BaseModel):
    """Base class for all evaluation output types."""

    output_type: Literal["instance", "overall"] = Field(description="The type of output.")
    module_name: str = Field(metadata={"description": "The name of the module the evaluation is in."})
    class_name: str = Field(metadata={"description": "The name of the evaluation output class in the module."})
    name_model: str = Field(None, description="Name of the mode being evaluated.")
    provider: Provider = Field(None, description="The provider of the model, such as OpenAI or Ollama.")
    score: float = Field(None, description="The score out of 100 for this evaluation instance.")
    execution_date: DateTime = Field(
        default_factory=pendulum.now, description="The datetime the evaluation instance was executed."
    )

    def save_to_db(self) -> None:
        """Insert the current state of EvaluationData into a TinyDB database."""
        db = TinyDB(TINYDB_PATH)

        evaluation_data_dict = self.to_dict(self)
        db.insert(evaluation_data_dict)

    @staticmethod
    def to_dict(instance: BaseModel) -> dict:
        """Convert an instance of EvaluationData to a dictionary, handling special types."""
        instance_dict = instance.model_dump(mode="json")
        return instance_dict

    @staticmethod
    def load_class(module_name: str, class_name: str, data: dict) -> "EvaluationBaseOutput":
        """Load the class dynamically from the module and class name."""
        module = importlib.import_module(f"evaluate_ai.evaluations.{module_name}")
        class_ = getattr(module, class_name)
        evaluation_output_class: EvaluationBaseOutput = class_(**data)
        return evaluation_output_class


class EvaluationInstanceOutput(EvaluationBaseOutput):
    """The data, outputs, and metrics corresponding to the individual evaluation instance execution.
    evaluation_instance should always be overwritten with the specific evaluation instance class."""

    output_type: Literal["instance"] = Field(default="instance", description="The type of output.")
    evaluation_instance: EvaluationInstance = Field(description="The evaluation instance that was executed.")


class EvaluationOverallOutput(EvaluationBaseOutput):
    """The data, outputs, and metrics corresponding to the overall evaluation run.
    evaluation_instances should always be overwritten with the specific evaluation instance output class.
    NOTE: Currently unused anywhere.
    """

    output_type: Literal["overall"] = Field(default="overall", description="The type of output.")
    evaluation_instances: list[EvaluationInstanceOutput] = Field(description="The list of evaluation instances.")


class Evaluation(ABC):
    @abstractmethod
    def __init__(self, config: EvaluationConfig) -> None:
        pass

    @abstractmethod
    def num_instances(self, keys_to_skip: tuple) -> int:
        """Used to estimate progress of evaluation runs.

        Args:
            keys_to_skip (tuple): A set of unique keys to skip when counting the number of instances.
                Each key consists of: (EvaluationInstanceOutput class name, model, provider, evaluation_instance_name).
        """
        pass

    @abstractmethod
    def execute(self, progress: Progress, keys_to_skip: tuple) -> None:
        """Execute the evaluation. Takes in a rich progress bar to update progress.
        NOTE: This should call progress.advance(0) each time one of the num_instances
        evaluation instances is completed

        Args:
            progress (Progress): The rich progress bar to update.
            keys_to_skip (tuple): A set of unique keys to skip when counting the number of instances.
                Each key consists of: (EvaluationInstanceOutput class name, model, provider, evaluation_instance_name).
        """
        pass

    @staticmethod
    def load_class(module_name: str, class_name: str, config: EvaluationConfig) -> "Evaluation":
        """Load the class dynamically from the module and class name."""
        module = importlib.import_module(f"evaluate_ai.evaluations.{module_name}")
        class_ = getattr(module, class_name)
        evaluation_class: Evaluation = class_(config=config)
        return evaluation_class
