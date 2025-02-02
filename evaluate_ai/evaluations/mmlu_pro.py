"""Based on https://github.com/TIGER-AI-Lab/MMLU-Pro"""

import re

from not_again_ai.llm.chat_completion import chat_completion
from not_again_ai.llm.chat_completion.types import ChatCompletionRequest, ChatCompletionResponse, UserMessage
from pydantic import Field
from rich.progress import Progress

from evaluate_ai.evaluation import (
    Evaluation,
    EvaluationConfig,
    EvaluationInstance,
    EvaluationInstanceOutput,
    EvaluationRunConfig,
)
from evaluate_ai.utils import download_parquet, get_llm_client, strip_thinking


class EvaluationInstanceMMLUPro(EvaluationInstance):
    """Defines the parameters needed for each evaluation instance."""

    question: str
    options: list[str]
    category: str
    answer: str


class EvaluationRunConfigMMLUPro(EvaluationRunConfig):
    data_url: str = Field(
        description="The URL to the data file.",
    )


class EvaluationConfigMMLUPro(EvaluationConfig):
    """Define the configuration structure for this evaluation."""

    run_config: EvaluationRunConfigMMLUPro
    evaluation_instances: list[EvaluationInstanceMMLUPro] = Field(
        default_factory=list,
        description="The list of evaluation instances that will be executed.",
    )


class EvaluationInstanceOutputMMLUPro(EvaluationInstanceOutput):
    """Define the output structure for each evaluation instance."""

    evaluation_instance: EvaluationInstanceMMLUPro
    message: str
    prompt_tokens_total: int
    completion_tokens_total: int
    duration_sec_total: float


class MMLUProEvaluation(Evaluation):
    def __init__(self, config: EvaluationConfigMMLUPro) -> None:
        self.config: EvaluationConfigMMLUPro = EvaluationConfigMMLUPro(**config)
        super().__init__(self.config)

        # Extract the file name from the URL
        file_name = self.config.run_config.data_url.split("/")[-1]
        self.data = download_parquet(self.config.run_config.data_url, file_name)
        # Create the evaluation instances from the downloaded data
        for row in self.data:
            instance_data = {
                "name": str(row["question_id"]),
                "question": row["question"],
                "options": row["options"],
                "category": row["category"],
                "answer": row["answer"],
            }
            self.config.evaluation_instances.append(EvaluationInstanceMMLUPro(**instance_data))

    def _get_output_class(self) -> type[EvaluationInstanceOutput]:
        return EvaluationInstanceOutputMMLUPro

    def execute(self, progress: Progress, keys_to_skip: set) -> None:
        """Execute the evaluation for each model and instance."""
        for provider, model in self.models:
            for e_instance in self.config.evaluation_instances:
                key = (EvaluationInstanceOutputMMLUPro.__name__, model, provider.value, e_instance.name)
                if key not in keys_to_skip:
                    response = self._get_response(
                        e_instance.question,
                        e_instance.options,
                        e_instance.category,
                        model,
                        provider.value,
                    )

                    try:
                        score = self._evaluate(response.choices[0].message.content, e_instance.answer)
                    except Exception:
                        score = 0

                    instance_output = EvaluationInstanceOutputMMLUPro(
                        module_name=self.config.run_config.module_name,
                        class_name=EvaluationInstanceOutputMMLUPro.__name__,
                        name_model=model,
                        provider=provider.value,
                        evaluation_instance=e_instance,
                        message=response.choices[0].message.content,
                        score=score,
                        prompt_tokens_total=response.prompt_tokens,
                        completion_tokens_total=response.completion_tokens,
                        duration_sec_total=response.response_duration,
                    )
                    instance_output.save_to_db()
                    progress.advance(0)

    def _get_response(
        self, question: str, options: list[str], category: str, model: str, provider: str
    ) -> ChatCompletionResponse:
        prompt = f"""The following are multiple choice questions (with answers) about {category}. Think step by \
step and then output the answer in the format of "The answer is (X)" at the end.\n\n"""
        prompt += f"Question: {question}\nOptions:\n"

        choice_map = "ABCDEFGHIJ"
        for i, opt in enumerate(options):
            prompt += f"{choice_map[i]}. {opt}\n"

        prompt += "\nAnswer: Let's think step by step.\n"

        messages = [
            UserMessage(
                content=f"{prompt}",
            ),
        ]
        request = ChatCompletionRequest(
            messages=messages,
            model=model,
            temperature=0.7,
            mmax_completion_tokens=4000,
            context_window=8000,
        )
        response = chat_completion(request, provider=provider, client=get_llm_client(provider))
        return response

    def _evaluate(self, response: str, expected_value: str) -> float:
        """Extracts the answer from the model's response and compares it to the expected value."""
        response = strip_thinking(response)

        def extract_answer(text):
            pattern = r"answer is \(?([A-J])\)?"
            match = re.search(pattern, text)
            if match:
                return match.group(1)
            else:
                return extract_again(text)

        def extract_again(text):
            match = re.search(r".*[aA]nswer:\s*([A-J])", text)
            if match:
                return match.group(1)
            else:
                return extract_final(text)

        def extract_final(text):
            pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(0)
            else:
                return None

        response = response.replace("**", "")
        extracted_answer = extract_answer(response)

        if extracted_answer is None:
            return 0
        elif extracted_answer == expected_value:
            return 100
        else:
            return 0
