from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

from not_again_ai.local_llm.chat_completion import chat_completion
from pydantic import BaseModel, Field
from tinydb import TinyDB

from evaluate_ai.run_config import Provider, RunConfig


class Metadata(BaseModel):
    evaluation_version: int | None = Field(None, description="The version of the evaluation in case changes are made.")
    output: list[Any] = Field(default_factory=list, description="The raw output(s) from the model.")
    response_durations: list[float] = Field(
        default_factory=list, description="The time taken for each response from the model."
    )
    prompt_tokens: list[int] = Field(default_factory=list, description="The tokens used in the prompt for the model.")
    completion_tokens: list[int] = Field(
        default_factory=list, description="The tokens used in the completion for the model."
    )
    evaluation_parameters: dict[str, Any] | None = Field(
        None, description="The parameters used to setup the evaluation."
    )
    evaluation_output: str | None = Field(None, description="Any extra output from the evaluation")
    output_parameters: dict[str, Any] | None = Field(
        None, description="The parameters used for generating model outputs."
    )
    non_api_model: bool = Field(False, description="Whether the model is from a non-API source like Microsoft Copilot.")
    provider: Provider | None = Field(None, description="The provider of the model, such as OpenAI or Ollama.")


class EvaluationData(BaseModel):
    name: str | None = Field(
        None,
        metadata={"description": "The name of the evaluation used to identify the evaluation, which can be anything."},
    )
    # This should be the key added to the EVALUATION_REGISTRY
    type: str | None = Field(None, metadata={"description": "One of the supported evaluation types"})
    score: float | None = Field(None, metadata={"description": "The score out of 100 for the evaluation."})
    name_model: str | None = Field(None, metadata={"description": "Name of the model"})
    execution_date: datetime | None = Field(
        default_factory=datetime.now, metadata={"description": "The datetime the evaluation was executed."}
    )
    metadata: Metadata = Field(default_factory=Metadata, metadata={"description": "Any additional and optional info."})

    def save_to_db(self) -> None:
        """Insert the current state of EvaluationData into a TinyDB database."""
        db_path = Path(__file__).parent.parent / "data" / "tinydb.json"
        db = TinyDB(db_path)

        evaluation_data_dict = self.to_dict(self)
        db.insert(evaluation_data_dict)

    @staticmethod
    def to_dict(instance: BaseModel) -> dict:
        """Convert an instance of EvaluationData to a dictionary, handling special types."""
        instance_dict = instance.model_dump(mode="json")
        return instance_dict


class Evaluation(ABC):
    def __init__(self, config: RunConfig):
        self.config = config

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
        Any time the LLM is called, it automatically is stored in self.evaluation_data.metadata.output in order of the LLM calls

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
        self.evaluation_data.name_model = model

        self.get_result(model, llm_client, *args, **kwargs)
        self.evaluate(*args, **kwargs)
        self.evaluation_data.metadata.provider = provider
        self.evaluation_data.save_to_db()

    def evaluate_external(self, source: str, *args, **kwargs) -> None:
        self.evaluation_data.name_model = source

        manual_file_path = Path(__file__).parent.parent / "external_evaluation.txt"
        if not manual_file_path.exists():
            manual_file_path.touch()

        print(self.task_as_string())

        input(f"Copy and paste the response into {manual_file_path} and press Enter when you are ready to continue.")

        with Path.open(manual_file_path, "r") as file:
            text = file.read()
        self.evaluation_data.metadata.output.append(text)
        self.evaluation_data.metadata.non_api_model = True

        self.evaluate(*args, **kwargs)
        self.evaluation_data.save_to_db()

    def call_llm(
        self,
        model: str,
        llm_client: Any,
        messages: list[dict[str, str]],
        log_to_evaluation_data: bool,
        **kwargs: Any,
    ) -> str:
        """Helper function to call the LLM using not-again-ai's chat_completion function.
        Assumes the messages are in the correct format and the result is successful.
        """
        response = chat_completion(messages=messages, model=model, client=llm_client, **kwargs)

        if log_to_evaluation_data:
            self.evaluation_data.metadata.output.append(response["message"])
            self.evaluation_data.metadata.prompt_tokens.append(response["prompt_tokens"])
            self.evaluation_data.metadata.completion_tokens.append(response["completion_tokens"])
            self.evaluation_data.metadata.response_durations.append(response["response_duration"])

        return response["message"]
