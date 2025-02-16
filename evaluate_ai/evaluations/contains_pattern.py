import re

from not_again_ai.llm.chat_completion import chat_completion
from not_again_ai.llm.chat_completion.types import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    SystemMessage,
    UserMessage,
)
from not_again_ai.llm.prompting.compile_prompt import compile_messages
import pendulum
from pydantic import Field
from rich.progress import Progress

from evaluate_ai.evaluation import Evaluation, EvaluationConfig, EvaluationInstance, EvaluationInstanceOutput
from evaluate_ai.utils import get_llm_client, strip_thinking

CONTAINS_PATTERN_MESSAGES = [
    SystemMessage(
        content="""{% if system_prompt %}{{ system_prompt }}{% else %}- You are a helpful assistant.
- The current date is {{datetime}}.
- You should answer questions truthfully and accurately.{% endif %}""",
    ),
    UserMessage(
        content="""{{prompt}}""",
    ),
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
        super().__init__(self.config)

    def _get_output_class(self) -> type[EvaluationInstanceOutput]:
        return EvaluationInstanceOutputContainsPattern

    def execute(self, progress: Progress, keys_to_skip: set) -> None:
        for provider, model in self.models:
            for e_instance in self.config.evaluation_instances:
                key = (EvaluationInstanceOutputContainsPattern.__name__, model, provider.value, e_instance.name)
                if key not in keys_to_skip:
                    response = self._get_response(
                        system_prompt=e_instance.system_prompt,
                        prompt=e_instance.prompt,
                        model=model,
                        provider=provider.value,
                    )
                    message = response.choices[0].message.content
                    score = self._evaluate(message, e_instance.pattern)
                    instance_output = EvaluationInstanceOutputContainsPattern(
                        module_name=self.config.run_config.module_name,
                        class_name=EvaluationInstanceOutputContainsPattern.__name__,
                        name_model=model,
                        provider=provider,
                        evaluation_instance=e_instance,
                        message=message,
                        score=score,
                        prompt_tokens_total=response.prompt_tokens,
                        completion_tokens_total=response.completion_tokens,
                        duration_sec_total=response.response_duration,
                    )
                    instance_output.save_to_db()
                    progress.advance(0)

    def _get_response(self, system_prompt: str, prompt: str, model: str, provider: str) -> ChatCompletionResponse:
        messages = compile_messages(
            messages=CONTAINS_PATTERN_MESSAGES,
            variables={
                "system_prompt": system_prompt,
                "prompt": prompt,
                "datetime": pendulum.now().strftime("%Y-%m-%d"),
            },
        )
        request = ChatCompletionRequest(
            messages=messages,
            model=model,
            temperature=0.7,
            max_completion_tokens=3000,
            context_window=8000,
        )
        response = chat_completion(request, provider=provider, client=get_llm_client(provider))
        return response

    def _evaluate(self, response: str, pattern: str) -> float:
        response = strip_thinking(response)
        pattern = re.compile(pattern)
        success = bool(pattern.search(response))
        score = 100 if success else 0
        return score
