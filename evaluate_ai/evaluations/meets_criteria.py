from typing import Any

from not_again_ai.local_llm.chat_completion import chat_completion
from not_again_ai.local_llm.prompts import chat_prompt
from pydantic import BaseModel, Field
from rich.progress import Progress

from evaluate_ai.evaluation import (
    Evaluation,
    EvaluationConfig,
    EvaluationInstance,
    EvaluationInstanceOutput,
    EvaluationRunConfig,
)
from evaluate_ai.utils import Provider, get_llm_client

RESPONSE_MESSAGES = [
    {
        "role": "system",
        "content": """You are a helpful assistant that is answering requests for a user.""",
    },
    {
        "role": "user",
        "content": """{{prompt}}""",
    },
]

EVALUATION_MESSAGES = [
    {
        "role": "system",
        "content": """You are evaluating how well a response meets a given criteria. Each can either be met or not met. You must make a choice.
First, for the criteria, first think step by step about if the response meets the criteria.
Then as your final answer write down true or false. true means the criteria is met and false means it is not met. \
Below is an example of the process.

RESPONSE:
There have been many incredible football games over the years, each memorable for different reasons. Here are a few that often come up in discussions:

1. **Super Bowl LI (2017)**: New England Patriots vs. Atlanta Falcons
   - The Patriots made a historic comeback from a 28-3 deficit to win 34-28 in overtime, the first Super Bowl to go into overtime.

2. **The "Tuck Rule" Game (2002)**: New England Patriots vs. Oakland Raiders
   - Known for the controversial "tuck rule" call, this game saw the Patriots win 16-13 in overtime, propelling them to their first Super Bowl victory.

3. **Super Bowl XXIII (1989)**: San Francisco 49ers vs. Cincinnati Bengals
   - Joe Montana led a 92-yard drive in the final minutes to secure a 20-16 victory for the 49ers.

CRITERIA:
The first two games should not both include the New England Patriots.

SAMPLE ANSWER:
The first game mentioned, Super Bowl LI, involves the Patriots. The second game, the Tuck Rule Game, also involves the Patriots. Therefore the criteria is NOT met. Therefore the final output is: false""",
    },
    {
        "role": "user",
        "content": """RESPONSE:
{{response}}

CRITERIA:
{{criteria}}

Now first think step by step if the response meets the criteria and then write either true or false as your final answer.""",
    },
]

CONVERT_RESPONSE_MESSAGES = [
    {
        "role": "system",
        "content": """You are a helpful, thoughtful, and meticulous assistant. You will be given text which you must parse out a true or false answer from. 
Your final output should be a very simple JSON object: { "answer" : true } or { "answer" : false }.

Below is the an example of the process.

RESPONSE:
The first game mentioned, Super Bowl LI, involves the Patriots. The second game, the Tuck Rule Game, also involves the Patriots. Therefore the criteria is NOT met. Therefore the final output is: false

SAMPLE JSON:
{ "answer" : false }""",
    },
    {
        "role": "user",
        "content": """RESPONSE:
{{response}}

Now convert that response into JSON following the format: { "answer" : true } or { "answer" : false }""",
    },
]


class EvaluationRunConfigMeetsCriteria(EvaluationRunConfig):
    evaluation_provider: Provider
    evaluation_model: str


class SemanticCriteria(BaseModel):
    criteria: str
    importance: int


class EvaluationInstanceMeetsCriteria(EvaluationInstance):
    prompt: str = Field(
        description="The prompt to present to the model.",
    )
    semantic_criteria: list[SemanticCriteria] = Field(
        description="The semantic criteria to evaluate the model's output against and how much to factor it into the score."
    )


class EvaluationConfigMeetsCriteria(EvaluationConfig):
    run_config: EvaluationRunConfigMeetsCriteria
    evaluation_instances: list[EvaluationInstanceMeetsCriteria] = Field(
        description="The list of evaluation instances that will be executed.",
    )


class EvaluationInstanceOutputMeetsCriteria(EvaluationInstanceOutput):
    evaluation_instance: EvaluationInstanceMeetsCriteria
    message: str
    prompt_tokens_total: int
    completion_tokens_total: int
    duration_sec_total: float


class EvaluationMeetsCriteria(Evaluation):
    def __init__(self, config: EvaluationConfigMeetsCriteria) -> None:
        self.config: EvaluationConfigMeetsCriteria = EvaluationConfigMeetsCriteria(**config)

        # Get a tuple of (provider, model) for each model in the run config
        self.models: list[tuple[Provider, str]] = []
        for provider, models in self.config.run_config.models.items():
            for model in models:
                self.models.append((provider, model))

    def num_instances(self, keys_to_skip: set) -> int:
        num = 0
        for provider, model in self.models:
            for e_instance in self.config.evaluation_instances:
                key = (EvaluationInstanceOutputMeetsCriteria.__name__, model, provider.value, e_instance.name)
                if key not in keys_to_skip:
                    num += 1
        return num

    def execute(self, progress: Progress, keys_to_skip: set) -> None:
        for provider, model in self.models:
            for e_instance in self.config.evaluation_instances:
                key = (EvaluationInstanceOutputMeetsCriteria.__name__, model, provider.value, e_instance.name)
                if key not in keys_to_skip:
                    response = self._get_response(
                        prompt=e_instance.prompt, model=model, llm_client=get_llm_client(provider)
                    )
                    message = response["message"]
                    score = self._evaluate(message, e_instance)
                    instance_output = EvaluationInstanceOutputMeetsCriteria(
                        module_name=self.config.run_config.module_name,
                        class_name=EvaluationInstanceOutputMeetsCriteria.__name__,
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

    def _get_response(self, prompt: str, model: str, llm_client: Any) -> str:
        messages = chat_prompt(
            messages_unformatted=RESPONSE_MESSAGES,
            variables={"prompt": prompt},
        )
        response = chat_completion(messages, model=model, client=llm_client, temperature=0.5, max_tokens=1500)
        return response

    def _evaluate(self, response_message: str, e_instance: EvaluationInstanceMeetsCriteria) -> float:
        score = 0
        # Normalize the importance values from parameters to sum to 100
        total_importance = sum([c.importance for c in e_instance.semantic_criteria])
        for crit in e_instance.semantic_criteria:
            importance = crit.importance
            importance = (importance / total_importance) * 100
            reasoning_messages = chat_prompt(
                messages_unformatted=EVALUATION_MESSAGES,
                variables={
                    "response": response_message,
                    "criteria": crit.criteria,
                },
            )
            reasoning = chat_completion(
                messages=reasoning_messages,
                model=self.config.run_config.evaluation_model,
                client=get_llm_client(self.config.run_config.evaluation_provider),
                max_tokens=1000,
                temperature=0.3,
            )

            # Convert the reasoning to JSON
            convert_messages = chat_prompt(
                messages_unformatted=CONVERT_RESPONSE_MESSAGES,
                variables={
                    "response": reasoning["message"],
                },
            )
            evaluation_result = chat_completion(
                messages=convert_messages,
                model=self.config.run_config.evaluation_model,
                client=get_llm_client(self.config.run_config.evaluation_provider),
                max_tokens=500,
                temperature=0,
                json_mode=True,
            )

            try:
                evaluation_result = evaluation_result["message"]["answer"]
                if evaluation_result:
                    score += importance
            except Exception:
                continue

        return score
