import json
from typing import Any

from jsonschema import validate
from jsonschema.exceptions import SchemaError, ValidationError
from loguru import logger

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
        self.call_llm(
            model, messages=self.messages, llm_client=llm_client, max_tokens=1000, temperature=0.5, json_mode=True
        )

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

    def test_schema(self, sample_model_response: str | dict) -> None:
        """A helper function to test if the jsonschema is validating responses as expected.

        Args:
            sample_model_response (str | dict): The sample model response to test against the schema.
                jsonschema validation requires a dictionary, so if a string is provided, we attempt to parsed into a dictionary.
        """
        if isinstance(sample_model_response, str):
            try:
                sample_model_response = json.loads(sample_model_response)
            except json.JSONDecodeError as e:
                logger.error(f"Could not parse the sample model response into a Python dictionary: {e}")
                return

        try:
            validate(instance=sample_model_response, schema=self.schema)
            logger.info("Schema validation succeeded.")
        except ValidationError as e:
            logger.info(f"Schema validation failed: {e}")
        # Catch an error with the schema itself
        except SchemaError as e:
            logger.info(f"Schema error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    # Sample test script to check if your schemas validate model responses as expected
    model_response = """{"itemType": "Pizza", "size": "large", "addedToppings": ["black olives"], "removedToppings": ["bananas"], "quantity": 1, "name": "Pepperoni"}"""

    schema = {
        "type": "object",
        "properties": {
            "itemType": {"type": "string", "const": "Pizza"},
            "size": {"type": "string", "const": "large"},
            "addedToppings": {
                "type": "array",
                "items": {"oneOf": [{"const": "black olives"}, {"const": "pepperoni"}]},
                "minItems": 1,
                "maxItems": 2,
            },
            "removedToppings": {"type": "array", "items": {"const": "mushrooms"}, "minItems": 1, "maxItems": 1},
            "quantity": {"type": "integer", "const": 1},
            "name": {"type": "string", "const": "Pepperoni"},
        },
        "required": ["itemType", "size", "addedToppings", "removedToppings", "quantity", "name"],
    }

    evaluation = EvaluationStructuredOutput(name="Not Applicable", prompt="Not Applicable", schema=schema)
    evaluation.test_schema(model_response)
