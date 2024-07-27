import importlib
from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, field_validator

from evaluate_ai.evaluation import Evaluation
from evaluate_ai.evaluation_registry import EVALUATION_REGISTRY
from evaluate_ai.run_config import RunConfig


class EvaluationInstance(BaseModel):
    """Defines an evaluation. Each evaluation instance will be executed by this script and is defined in a YAML file.

    Args:
        name (str): The name of the evaluation used to identify the evaluation, which can be anything.
        type (str): One of the supported evaluation types (the keys in EVALUATION_REGISTRY).
        parameters (dict[str, Any]): The parameters to pass to the evaluation class.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: RunConfig

    name: str
    type: str
    parameters: dict[str, Any] = {}

    evaluation_class_instance: Evaluation | None = None

    # Validate if type is in type_to_class_map
    @field_validator("type")
    @classmethod
    def ensure_type_is_valid(cls, v):
        if v not in EVALUATION_REGISTRY:
            raise ValueError(f"Type {v} is not in the type_to_class_map")
        return v

    # Initialize the Evaluation class for the given evaluation_type by dynamically loading the module and then the class
    def model_post_init(self, __context: Any) -> None:
        module = importlib.import_module(f"evaluate_ai.evaluations.{self.type}")
        class_name = EVALUATION_REGISTRY[self.type]
        class_ = getattr(module, class_name)

        self.parameters["name"] = self.name
        self.evaluation_class_instance = class_(config=self.config, **self.parameters)

    @classmethod
    def initialize_evaluation_instances(cls, evaluations: list[dict], config: RunConfig) -> list["EvaluationInstance"]:
        """Load, validate, and handle errors for the evaluations from a YAML file.

        Args:
            evaluations (list[dict]): The list of evaluations to load.

        Returns:
            list[EvaluationInstance]: The list of validated evaluations if successful; an empty list if an error occurs.

        Notes:
            Logs error messages instead of raising exceptions to allow execution to continue.
        """
        evaluation_instances: list[EvaluationInstance] = []
        try:
            for evaluation in evaluations:
                evaluation_instance = cls(config=config, **evaluation)
                evaluation_instances.append(evaluation_instance)
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading the evaluation: {evaluation}")
            logger.error(f"Error: {e}")
        return evaluation_instances

    @classmethod
    def check_evaluation_instances(cls, evaluations: list[dict], config: RunConfig) -> None:
        """Load, validate, and handle errors for the evaluations from a YAML file.

        Args:
            evaluations (list[dict]): The list of evaluations to check.

        Raises:
            Exception: If an error occurs while loading an evaluation.
        """
        try:
            for evaluation in evaluations:
                cls(config=config, **evaluation)
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading the evaluation: {evaluation}")
            logger.error(f"Error: {e}")
            raise e
