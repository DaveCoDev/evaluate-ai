"""Based on https://github.com/google-research/google-research/blob/master/instruction_following_eval/"""

from typing import Any

from not_again_ai.local_llm.chat_completion import chat_completion
from pydantic import Field
from rich.progress import Progress

from evaluate_ai.evaluation import (
    Evaluation,
    EvaluationConfig,
    EvaluationInstance,
    EvaluationInstanceOutput,
    EvaluationRunConfig,
)
from evaluate_ai.evaluations.instruction_following_eval import instructions_registry
from evaluate_ai.utils import download_jsonl, get_llm_client


class EvaluationInstanceIFEval(EvaluationInstance):
    """Defines the parameters needed for each evaluation instance."""

    prompt: str
    instruction_id_list: list[str]
    kwargs: list[dict[str, str | int | None | list[str]]]


class EvaluationRunConfigIFEval(EvaluationRunConfig):
    data_url: str = Field(
        description="The URL to the data file.",
    )
    first_n_instances: int = Field(
        default=None,
        description="The number of instances to run. If not provided, all instances will be run.",
    )


class EvaluationConfigIFEval(EvaluationConfig):
    """Define the configuration structure for this evaluation."""

    run_config: EvaluationRunConfigIFEval
    evaluation_instances: list[EvaluationInstanceIFEval] = Field(
        default_factory=list,
        description="The list of evaluation instances that will be executed.",
    )


class EvaluationInstanceOutputIFEval(EvaluationInstanceOutput):
    """Define the output structure for each evaluation instance."""

    evaluation_instance: EvaluationInstanceIFEval
    message: str
    prompt_tokens_total: int
    completion_tokens_total: int
    duration_sec_total: float


class IFEvalEvaluation(Evaluation):
    def __init__(self, config: EvaluationConfigIFEval) -> None:
        self.config: EvaluationConfigIFEval = EvaluationConfigIFEval(**config)
        super().__init__(self.config)

        # Extract the file name from the URL
        file_name = self.config.run_config.data_url.split("/")[-1]
        self.data = download_jsonl(self.config.run_config.data_url, file_name)
        # Create the evaluation instances from the downloaded data
        for row in self.data:
            instance_data = {
                "name": str(row["key"]),
                "prompt": row["prompt"],
                "instruction_id_list": row["instruction_id_list"],
                "kwargs": row["kwargs"],
            }
            self.config.evaluation_instances.append(EvaluationInstanceIFEval(**instance_data))

        # Sort the instances by name and take the first n instances if provided
        self.config.evaluation_instances.sort(key=lambda x: x.name)
        if self.config.run_config.first_n_instances:
            self.config.evaluation_instances = self.config.evaluation_instances[
                : self.config.run_config.first_n_instances
            ]

    def _get_output_class(self) -> type[EvaluationInstanceOutput]:
        return EvaluationInstanceOutputIFEval

    def execute(self, progress: Progress, keys_to_skip: set) -> None:
        """Execute the evaluation for each model and instance."""
        for provider, model in self.models:
            for e_instance in self.config.evaluation_instances:
                key = (EvaluationInstanceOutputIFEval.__name__, model, provider.value, e_instance.name)
                if key not in keys_to_skip:
                    response = self._get_response(
                        e_instance.prompt,
                        model,
                        get_llm_client(provider),
                    )

                    score = self._evaluate(
                        response["message"],
                        e_instance.instruction_id_list,
                        e_instance.kwargs,
                    )

                    instance_output = EvaluationInstanceOutputIFEval(
                        module_name=self.config.run_config.module_name,
                        class_name=EvaluationInstanceOutputIFEval.__name__,
                        name_model=model,
                        provider=provider,
                        evaluation_instance=e_instance,
                        message=response["message"],
                        score=score,
                        prompt_tokens_total=response["prompt_tokens"],
                        completion_tokens_total=response["completion_tokens"],
                        duration_sec_total=response["response_duration"],
                    )
                    instance_output.save_to_db()
                    progress.advance(0)

    def _get_response(self, prompt: str, model: str, llm_client: Any) -> dict[str, Any]:
        messages = [
            {
                "role": "user",
                "content": f"{prompt}",
            },
        ]
        response = chat_completion(messages, model=model, client=llm_client, temperature=0.5, max_tokens=2000)
        return response

    def _evaluate(
        self,
        response: str,
        instruction_id_list: list[str],
        kwargs: list[dict[str, str | int | None | list[str]]],
    ) -> float:
        """Tests response to see if instructions are followed."""
        is_following_list = []
        for index, instruction_id in enumerate(instruction_id_list):
            instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
            instruction = instruction_cls(instruction_id)
            # if kwargs contains a key "section_spliter", rename it to "section_splitter"
            if "section_spliter" in kwargs[index]:
                kwargs[index]["section_splitter"] = kwargs[index]["section_spliter"]
                del kwargs[index]["section_spliter"]
            instruction.build_description(**kwargs[index])
            args = instruction.get_instruction_args()
            if args and "prompt" in args:
                instruction.build_description(prompt="")

            if response.strip() and instruction.check_following(response):
                is_following_list.append(True)
            else:
                is_following_list.append(False)

        # Compute the score based on the number of instructions that are followed
        score = (sum(is_following_list) / len(is_following_list)) * 100
        return score
