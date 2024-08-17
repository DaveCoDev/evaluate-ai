from typing import Any

from loguru import logger
from not_again_ai.local_llm.prompts import chat_prompt

from evaluate_ai.evaluation import Evaluation
from evaluate_ai.run_config import RunConfig, get_llm_client


class EvaluationMeetsCriteria(Evaluation):
    """Assesses how well the answer meets defined criteria."""

    def __init__(self, config: RunConfig, name: str, prompt: str, semantic_criteria: list) -> None:
        """Initializes the evaluation with the friendly name of the evaluation and the prompt and pattern to check for.

        Args:
            name (str): The friendly name of the evaluation.
        """
        super().__init__(config=config)

        self.prompt = prompt
        self.semantic_criteria = semantic_criteria

        self.evaluation_data.name = name
        self.evaluation_data.type = "meets_criteria"

        self.response_messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that is answering requests for a user.",
            },
            {
                "role": "user",
                "content": self.prompt,
            },
        ]

        self.evaluation_messages = [
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

        self.convert_response = [
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

    def get_result(self, model: str, llm_client: Any) -> None:
        # First get a response for the prompt
        self.call_llm(
            model,
            messages=self.response_messages,
            llm_client=llm_client,
            log_to_evaluation_data=True,
            max_tokens=1000,
            temperature=0.5,
        )
        self.evaluation_data.metadata.output_parameters = {
            "max_tokens": 1000,
            "temperature": 0.5,
        }

    def evaluate(self) -> None:
        response = self.evaluation_data.metadata.output[0]
        model = self.config.evaluation_model
        llm_client = get_llm_client(self.config.evaluation_provider)

        score = 0
        # Normalize the importance values from parameters to sum to 100
        total_importance = sum([c["importance"] for c in self.semantic_criteria])
        for crit in self.semantic_criteria:
            crit["importance"] = (crit["importance"] / total_importance) * 100

            reasoning_messages = chat_prompt(
                messages_unformatted=self.evaluation_messages,
                variables={
                    "response": response,
                    "criteria": crit["criteria"],
                },
            )
            reasoning = self.call_llm(
                model,
                messages=reasoning_messages,
                llm_client=llm_client,
                log_to_evaluation_data=False,
                max_tokens=1000,
                temperature=0.3,
            )

            # Convert the reasoning to JSON
            convert_messages = chat_prompt(
                messages_unformatted=self.convert_response,
                variables={
                    "response": reasoning,
                },
            )
            evaluation_result = self.call_llm(
                model,
                messages=convert_messages,
                llm_client=llm_client,
                log_to_evaluation_data=False,
                max_tokens=500,
                temperature=0,
                json_mode=True,
            )

            try:
                evaluation_result = evaluation_result["answer"]
                if evaluation_result:
                    score += crit["importance"]
            except Exception as e:
                logger.error(f"Error parsing evaluation result: {evaluation_result}\n {e}")
                continue

        self.evaluation_data.score = round(score, 2)

    def task_as_string(self) -> str:
        return super().task_as_string()
