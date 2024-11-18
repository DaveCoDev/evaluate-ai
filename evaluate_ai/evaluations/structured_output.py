import json
from typing import Any

from jsonschema import validate
from jsonschema.exceptions import SchemaError, ValidationError
from loguru import logger
from not_again_ai.local_llm.chat_completion import chat_completion
from not_again_ai.local_llm.prompts import chat_prompt
from pydantic import Field
from rich.progress import Progress

from evaluate_ai.evaluation import Evaluation, EvaluationConfig, EvaluationInstance, EvaluationInstanceOutput
from evaluate_ai.utils import get_llm_client

STRUCTURED_OUTPUT_MESSAGES = [
    {
        "role": "system",
        "content": """You are a helpful assistant that is answering tasks in a structured JSON format.""",
    },
    {
        "role": "user",
        "content": """{{prompt}}""",
    },
]


class EvaluationInstanceStructuredOutput(EvaluationInstance):
    prompt: str = Field(
        description="The prompt to present to the model.",
    )
    json_schema: dict[str, Any] = Field(description="The schema to validate the model's output against.")


class EvaluationConfigStructuredOutput(EvaluationConfig):
    evaluation_instances: list[EvaluationInstanceStructuredOutput] = Field(
        description="The list of evaluation instances that will be executed.",
    )


class EvaluationInstanceOutputStructuredOutput(EvaluationInstanceOutput):
    evaluation_instance: EvaluationInstanceStructuredOutput
    message: str
    error_message: str | None
    prompt_tokens_total: int
    completion_tokens_total: int
    duration_sec_total: float


class EvaluationStructuredOutput(Evaluation):
    def __init__(self, config: EvaluationConfigStructuredOutput) -> None:
        self.config: EvaluationConfigStructuredOutput = EvaluationConfigStructuredOutput(**config)
        super().__init__(self.config)

    def _get_output_class(self) -> type[EvaluationInstanceOutput]:
        return EvaluationInstanceOutputStructuredOutput

    def execute(self, progress: Progress, keys_to_skip: set) -> None:
        for provider, model in self.models:
            for e_instance in self.config.evaluation_instances:
                key = (EvaluationInstanceOutputStructuredOutput.__name__, model, provider.value, e_instance.name)
                if key not in keys_to_skip:
                    response = self._get_response(
                        prompt=e_instance.prompt,
                        model=model,
                        llm_client=get_llm_client(provider),
                    )
                    message = response["message"]
                    score, error = self._evaluate(message, e_instance.json_schema)
                    instance_output = EvaluationInstanceOutputStructuredOutput(
                        module_name=self.config.run_config.module_name,
                        class_name=EvaluationInstanceOutputStructuredOutput.__name__,
                        name_model=model,
                        provider=provider,
                        evaluation_instance=e_instance,
                        message=str(message),
                        error_message=error,
                        score=score,
                        prompt_tokens_total=response["prompt_tokens"],
                        completion_tokens_total=response["completion_tokens"],
                        duration_sec_total=response["response_duration"],
                    )
                    instance_output.save_to_db()
                    progress.advance(0)

    def _get_response(self, prompt: str, model: str, llm_client: Any) -> str:
        messages = chat_prompt(
            messages_unformatted=STRUCTURED_OUTPUT_MESSAGES,
            variables={
                "prompt": prompt,
            },
        )
        response = chat_completion(
            messages, model=model, client=llm_client, temperature=0.5, max_tokens=2000, json_mode=True
        )
        return response

    def _evaluate(self, response: dict, json_schema: dict) -> tuple[float, str | None]:
        try:
            validate(instance=response, schema=json_schema)
            return (100, None)
        except ValidationError as e:
            return (0, str(e))

    @staticmethod
    def test_schema(sample_model_response: str | dict, json_schema: dict) -> None:
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
            validate(instance=sample_model_response, schema=json_schema)
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
    model_response = """{"itemType": "Pizza", "size": "large", "addedToppings": ["black olives"], "removedToppings": ["mushrooms"], "quantity": 1, "name": "Pepperoni"}"""

    json_schema = {
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

    EvaluationStructuredOutput.test_schema(model_response, json_schema)
