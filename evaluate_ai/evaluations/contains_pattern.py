import re
from typing import Any

from not_again_ai.local_llm.chat_completion import chat_completion
from not_again_ai.local_llm.prompts import chat_prompt
import pendulum
from pydantic import Field
from rich.progress import Progress

from evaluate_ai.evaluation import Evaluation, EvaluationConfig, EvaluationInstance, EvaluationInstanceOutput
from evaluate_ai.utils import Provider, get_llm_client

CONTAINS_PATTERN_MESSAGES = [
    {
        "role": "system",
        "content": """{% if system_prompt %}{{ system_prompt }}{% else %}- You are a helpful assistant.
- The current date is {{datetime}}.
- You should answer questions truthfully and accurately.{% endif %}""",
    },
    {
        "role": "user",
        "content": """{{prompt}}""",
    },
]


class EvaluationInstanceContainsPattern(EvaluationInstance):
    prompt: str = Field(
        description="The prompt to present to the model.",
    )
    system_prompt: str | None = Field(
        default=None,
        description="The system prompt to present to the model.",
    )
    pattern: str = Field(
        description="The regex pattern to check for in the model output.",
    )


class EvaluationConfigContainsPattern(EvaluationConfig):
    evaluation_instances: list[EvaluationInstanceContainsPattern] = Field(
        description="The list of evaluation instances that will be executed.",
    )


class EvaluationInstanceOutputContainsPattern(EvaluationInstanceOutput):
    evaluation_instance: EvaluationInstanceContainsPattern
    message: str
    prompt_tokens_total: int
    completion_tokens_total: int
    duration_sec_total: float


class EvaluationContainsPattern(Evaluation):
    def __init__(self, config: EvaluationConfigContainsPattern) -> None:
        self.config: EvaluationConfigContainsPattern = EvaluationConfigContainsPattern(**config)

        # Get a tuple of (provider, model) for each model in the run config
        self.models: list[tuple[Provider, str]] = []
        for provider, models in self.config.run_config.models.items():
            for model in models:
                self.models.append((provider, model))

    def num_instances(self, keys_to_skip: set) -> int:
        num = 0
        for provider, model in self.models:
            for e_instance in self.config.evaluation_instances:
                key = (EvaluationInstanceOutputContainsPattern.__name__, model, provider.value, e_instance.name)
                if key not in keys_to_skip:
                    num += 1
        return num

    def execute(self, progress: Progress, keys_to_skip: set) -> None:
        for provider, model in self.models:
            for e_instance in self.config.evaluation_instances:
                key = (EvaluationInstanceOutputContainsPattern.__name__, model, provider.value, e_instance.name)
                if key not in keys_to_skip:
                    response = self._get_response(
                        system_prompt=e_instance.system_prompt,
                        prompt=e_instance.prompt,
                        model=model,
                        llm_client=get_llm_client(provider),
                    )
                    message = response["message"]
                    score = self._evaluate(message, e_instance.pattern)
                    instance_output = EvaluationInstanceOutputContainsPattern(
                        module_name=self.config.run_config.module_name,
                        class_name=EvaluationInstanceOutputContainsPattern.__name__,
                        name_model=model,
                        provider=provider,
                        evaluation_instance=e_instance,
                        message=message,
                        score=score,
                        prompt_tokens_total=response["prompt_tokens"],
                        completion_tokens_total=response["completion_tokens"],
                        duration_sec_total=response["response_duration"],
                    )
                    instance_output.save_to_db()
                    progress.advance(0)

    def _get_response(self, system_prompt: str, prompt: str, model: str, llm_client: Any) -> str:
        messages = chat_prompt(
            messages_unformatted=CONTAINS_PATTERN_MESSAGES,
            variables={
                "system_prompt": system_prompt,
                "prompt": prompt,
                "datetime": pendulum.now().strftime("%Y-%m-%d"),
            },
        )
        response = chat_completion(messages, model=model, client=llm_client, temperature=0.7)
        return response

    def _evaluate(self, response: str, pattern: str) -> float:
        pattern = re.compile(pattern)
        success = bool(pattern.search(response))
        score = 100 if success else 0
        return score
