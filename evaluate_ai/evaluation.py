from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from not_again_ai.llm.chat_completion import chat_completion
from tinydb import TinyDB

from evaluate_ai.evaluation_registry import EVALUATION_ENUM


class Provider(Enum):
    OPENAI = "openai_api"
    OLLAMA = "ollama"


@dataclass
class Metadata:
    evaluation_version: int | None = field(
        default=None, metadata={"description": "The version of the evaluation in case changes are made."}
    )
    model_output: list[str] = field(metadata={"description": "The raw output(s) from the model."}, default_factory=list)
    evaluation_parameters: dict[str, str] | None = field(
        default=None, metadata={"description": "The parameters used to setup the evaluation."}
    )
    evaluation_output: str | None = field(
        default=None, metadata={"description": "Any extra output from the evaluation"}
    )
    model_parameters: dict[str, str] | None = field(
        default=None, metadata={"description": "The parameters used for the model."}
    )
    non_api_model: bool = field(
        default=False, metadata={"description": "Whether the model is from a non-API source like Microsoft Copilot."}
    )
    model_provider: Provider = field(
        default=None, metadata={"description": "The provider of the model, such as OpenAI or Ollama."}
    )


@dataclass
class EvaluationData:
    name: str = field(
        default=None,
        metadata={"description": "The name of the evaluation used to identify the evaluation, which can be anything."},
    )
    # This should be the key added to the EVALUATION_REGISTRY
    type: EVALUATION_ENUM | None = field(
        default=None, metadata={"description": "One of the supported evaluation types"}
    )
    score: float | None = field(default=None, metadata={"description": "The score out of 100 for the evaluation."})
    model_name: str | None = field(default=None, metadata={"description": "Name of the model"})
    execution_date: datetime | None = field(
        default_factory=datetime.now, metadata={"description": "The datetime the evaluation was executed."}
    )
    metadata: Metadata = field(default_factory=Metadata, metadata={"description": "Any additional and optional info."})

    def save_to_db(self) -> None:
        """Insert the current state of EvaluationData into a TinyDB database."""
        db_path = Path(__file__).parent.parent / "data" / "tinydb.json"
        db = TinyDB(db_path)

        evaluation_data_dict = self.to_dict(self)
        db.insert(evaluation_data_dict)

    @staticmethod
    def to_dict(instance) -> dict:
        """Convert an instance of EvaluationData to a dictionary, handling special types."""
        # Convert the dataclass to a dictionary
        instance_dict = asdict(instance)
        # Handle the Category Enum and datetime specially
        instance_dict["type"] = instance.type if isinstance(instance.type, str) else instance.type.name
        instance_dict["execution_date"] = instance.execution_date.isoformat()
        if instance.metadata:
            instance_dict["metadata"] = asdict(instance.metadata)
            if instance_dict["metadata"].get("model_parameters"):
                instance_dict["metadata"]["model_parameters"] = dict(instance_dict["metadata"]["model_parameters"])
        return instance_dict


class Evaluation(ABC):
    def __init__(self):
        self._evaluation_data = EvaluationData()

    @property
    def evaluation_data(self) -> EvaluationData:
        return self._evaluation_data

    def task_as_string(self) -> str:
        """Defines the task as a string to be displayed to the user.
        Useful for external evaluations that have to get results manually."""
        return "No task string provided."

    @abstractmethod
    def get_result(self, model: str, llm_client: Any, *args, **kwargs) -> None:
        """Defines getting a result for evaluation using a model (can be used in any way and multiple times).
        The result must be stored in self.evaluation_data.metadata.model_output.

        Args:
            model (str): The model name to use.
            llm_client (Any): The client to use for the model. Must be supported by call_llm.
        """
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> None:
        """Defines an evaluation that will produce a score from 0 to 100.
        The score must be stored in self.evaluation_data.score.
        """
        pass

    def execute(self, model: str, llm_client: Any, provider: Provider, *args, **kwargs) -> None:
        self.get_result(model, llm_client, *args, **kwargs)
        self.evaluate(*args, **kwargs)
        self.evaluation_data.metadata.model_provider = provider.value
        self.evaluation_data.save_to_db()

    def evaluate_external(self, source: str, *args, **kwargs) -> None:
        self.evaluation_data.model_name = source

        manual_file_path = Path(__file__).parent.parent / "external_evaluation.txt"
        if not manual_file_path.exists():
            manual_file_path.touch()

        print(self.task_as_string())

        input(f"Copy and paste the response into {manual_file_path} and press Enter when you are ready to continue.")

        with Path.open(manual_file_path, "r") as file:
            text = file.read()
        self.evaluation_data.metadata.model_output.append(text)
        self.evaluation_data.metadata.non_api_model = True

        self.evaluate(*args, **kwargs)
        self.evaluation_data.save_to_db()

    def call_llm(self, model: str, llm_client: Any, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Helper function to call the LLM using not-again-ai's chat_completion function.
        Assumes the messages are in the correct format and the result is successful.
        """
        response = chat_completion(messages=messages, model=model, client=llm_client, **kwargs)["message"]
        return response
