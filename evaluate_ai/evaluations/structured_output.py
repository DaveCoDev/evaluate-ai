from typing import Any

from jsonschema import validate
from jsonschema.exceptions import ValidationError

from evaluate_ai.evaluation import Evaluation


class EvaluationStructuredOutput(Evaluation):
    """Given a prompt that asks for output in a structured JSON format, validate the model output against a JSON Schema."""

    def __init__(self, name: str, prompt: str, schema: dict) -> None:
        """Initializes the evaluation with the name of the evaluation and the prompt and pattern to check for.

        Schema is a JSON Schema, see https://json-schema.org/understanding-json-schema/reference/enum#light-scheme-icon.
        The model must output a JSON object that is valid against the schema for a score of 100, otherwise 0.

        Args:
            name (str): The friendly name of the evaluation.
            prompt (str): The prompt to be provided as the user message.
            schema (dict): The schema to validate the model's output against.
        """
        super().__init__()

        self.prompt = prompt
        self.schema = schema

        self.evaluation_data.name = name
        self.evaluation_data.type = "structured_output"

        self.messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that is answering tasks in a structured JSON format.",
            },
            {
                "role": "user",
                "content": self.prompt,
            },
        ]

    def get_result(self, model: str, llm_client: Any) -> None:
        response = self.call_llm(
            model, messages=self.messages, llm_client=llm_client, max_tokens=1000, temperature=0.5, json_mode=True
        )

        self.evaluation_data.metadata.model_output.append(response)
        self.evaluation_data.model_name = model
        self.evaluation_data.metadata.model_parameters = {
            "max_tokens": 1000,
            "temperature": 0.5,
        }

    def evaluate(self) -> None:
        try:
            validate(instance=self.evaluation_data.metadata.model_output[0], schema=self.schema)
        except ValidationError:
            self.evaluation_data.score = 0
            return

        self.evaluation_data.score = 100

    def task_as_string(self) -> str:
        return super().task_as_string()