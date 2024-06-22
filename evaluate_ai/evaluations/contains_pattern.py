import datetime
import re
from typing import Any

from evaluate_ai.evaluation import Evaluation


class EvaluationContainsPattern(Evaluation):
    """Given a prompt, such as a question, and a regex pattern, check if the model output contains the pattern."""

    def __init__(self, name: str, prompt: str, pattern: str, system_prompt: str | None = None) -> None:
        """Initializes the evaluation with the friendly name of the evaluation and the prompt and pattern to check for.

        Args:
            name (str): The friendly name of the evaluation.
            prompt (str): The prompt to present to the model.
            pattern (str): The regex pattern to check for in the model output.
            system_prompt (str | None): The system prompt to present to the model. Defaults to None.
        """
        super().__init__()

        self.prompt = prompt
        self.pattern = pattern
        self.system_prompt = system_prompt

        self.evaluation_data.name = name
        self.evaluation_data.type = "contains_pattern"

        self.messages = [
            {
                "role": "system",
                "content": self.system_prompt
                or f"""- You are a helpful assistant.
- The current date is {datetime.datetime.now().strftime("%Y-%m-%d")}.
- Your should answer questions truthfully and accurately.""",
            },
            {
                "role": "user",
                "content": self.prompt,
            },
        ]

    def get_result(self, model: str, llm_client: Any) -> None:
        self.call_llm(model, messages=self.messages, llm_client=llm_client, max_tokens=1000, temperature=0.5)

        self.evaluation_data.metadata.model_parameters = {
            "max_tokens": 1000,
            "temperature": 0.5,
        }

    def evaluate(self) -> None:
        pattern = re.compile(self.pattern)
        success = bool(pattern.search(self.evaluation_data.metadata.model_output[0]))
        score = 100 if success else 0
        self.evaluation_data.score = score

    def task_as_string(self) -> str:
        return "\n".join([message["content"] for message in self.messages])
