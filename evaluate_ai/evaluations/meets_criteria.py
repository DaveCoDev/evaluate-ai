from not_again_ai.llm.chat_completion import chat_completion
from not_again_ai.llm.chat_completion.types import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    SystemMessage,
    UserMessage,
)
from not_again_ai.llm.prompting.compile_prompt import compile_messages
from pydantic import BaseModel, Field
from rich.progress import Progress

from evaluate_ai.evaluation import (
    Evaluation,
    EvaluationConfig,
    EvaluationInstance,
    EvaluationInstanceOutput,
    EvaluationRunConfig,
)
from evaluate_ai.utils import Provider, get_llm_client, strip_thinking

RESPONSE_MESSAGES = [
    SystemMessage(
        role="system",
        content="""You are a helpful assistant that is answering requests for a user.""",
    ),
    UserMessage(
        role="user",
        content="""{{prompt}}""",
    ),
]

EVALUATION_MESSAGES = [
    SystemMessage(
        role="system",
        content="""You are evaluating how well a response meets a given criteria. Each can either be met or not met. You must make a choice.
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
    ),
    UserMessage(
        role="user",
        content="""RESPONSE:
{{response}}

CRITERIA:
{{criteria}}

Now first think step by step if the response meets the criteria and then write either true or false as your final answer.""",
    ),
]

CONVERT_RESPONSE_MESSAGES = [
    SystemMessage(
        role="system",
        content="""You are a helpful, thoughtful, and meticulous assistant. You will be given text which you must parse out a true or false answer from. 
Your final output should be a very simple JSON object: { "answer" : true } or { "answer" : false }.

Below is the an example of the process.

RESPONSE:
The first game mentioned, Super Bowl LI, involves the Patriots. The second game, the Tuck Rule Game, also involves the Patriots. Therefore the criteria is NOT met. Therefore the final output is: false

SAMPLE JSON:
{ "answer" : false }""",
    ),
    UserMessage(
        role="user",
        content="""RESPONSE:
{{response}}

Now convert that response into JSON following the format: { "answer" : true } or { "answer" : false }""",
    ),
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
        super().__init__(self.config)

    def _get_output_class(self) -> type[EvaluationInstanceOutput]:
        return EvaluationInstanceOutputMeetsCriteria

    def execute(self, progress: Progress, keys_to_skip: set) -> None:
        for provider, model in self.models:
            for e_instance in self.config.evaluation_instances:
                key = (EvaluationInstanceOutputMeetsCriteria.__name__, model, provider.value, e_instance.name)
                if key not in keys_to_skip:
                    response = self._get_response(prompt=e_instance.prompt, model=model, provider=provider.value)
                    message = response.choices[0].message.content
                    score = self._evaluate(message, e_instance)
                    instance_output = EvaluationInstanceOutputMeetsCriteria(
                        module_name=self.config.run_config.module_name,
                        class_name=EvaluationInstanceOutputMeetsCriteria.__name__,
                        name_model=model,
                        provider=provider.value,
                        evaluation_instance=e_instance,
                        message=message,
                        score=score,
                        prompt_tokens_total=response.prompt_tokens,
                        completion_tokens_total=response.completion_tokens,
                        duration_sec_total=response.response_duration,
                    )
                    instance_output.save_to_db()
                    progress.advance(0)

    def _get_response(self, prompt: str, model: str, provider: str) -> ChatCompletionResponse:
        messages = compile_messages(
            messages=RESPONSE_MESSAGES,
            variables={"prompt": prompt},
        )
        request = ChatCompletionRequest(
            messages=messages,
            model=model,
            temperature=0.5,
            max_completion_tokens=4000,
            context_window=8000,
        )
        response = chat_completion(request, provider=provider, client=get_llm_client(provider))
        return response

    def _evaluate(self, response_message: str, e_instance: EvaluationInstanceMeetsCriteria) -> float:
        response_message = strip_thinking(response_message)
        provider = self.config.run_config.evaluation_provider.value
        model = self.config.run_config.evaluation_model

        score = 0
        # Normalize the importance values from parameters to sum to 100
        total_importance = sum([c.importance for c in e_instance.semantic_criteria])
        for crit in e_instance.semantic_criteria:
            importance = crit.importance
            importance = (importance / total_importance) * 100
            reasoning_messages = compile_messages(
                messages=EVALUATION_MESSAGES,
                variables={
                    "response": response_message,
                    "criteria": crit.criteria,
                },
            )
            request = ChatCompletionRequest(
                messages=reasoning_messages,
                model=model,
                max_completion_tokens=3000,
                context_window=8000,
                temperature=0.3,
            )
            reasoning = chat_completion(request, provider=provider, client=get_llm_client(provider))

            # Convert the reasoning to JSON
            convert_messages = compile_messages(
                messages=CONVERT_RESPONSE_MESSAGES,
                variables={
                    "response": reasoning.choices[0].message.content,
                },
            )
            request = ChatCompletionRequest(
                messages=convert_messages,
                model=model,
                max_completion_tokens=3000,
                context_window=8000,
                temperature=0,
                json_mode=True,
            )
            evaluation_result = chat_completion(request, provider=provider, client=get_llm_client(provider))

            try:
                evaluation_result = evaluation_result.choices[0].json_message["answer"]
                if evaluation_result:
                    score += importance
            except Exception:
                continue

        return score
