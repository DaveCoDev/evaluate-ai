"""Prints latest results for evaluation and each model to the console."""

import argparse
from collections import defaultdict

from rich.console import Console
from rich.padding import Padding
from tinydb import TinyDB

from evaluate_ai.evaluation import EvaluationBaseOutput
from evaluate_ai.tinydb_helpers.db_path import TINYDB_PATH


def main() -> None:
    console = Console(highlight=False)

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", default=True, help="Additionally print model outputs")
    args = parser.parse_args()

    db = TinyDB(TINYDB_PATH)
    documents = db.all()

    latest_results = defaultdict(dict)
    for evaluation_data in documents:
        evaluation_output_class = EvaluationBaseOutput.load_class(
            evaluation_data["module_name"], evaluation_data["class_name"], evaluation_data
        )
        evaluation_key = f"{evaluation_output_class.module_name} ({evaluation_output_class.evaluation_instance.name})"
        model = evaluation_output_class.name_model
        provider = evaluation_output_class.provider
        model_key = f"{provider.value}.{model}"

        if evaluation_key in latest_results and model_key in latest_results[evaluation_key]:
            latest_result: EvaluationBaseOutput = latest_results.get(evaluation_key).get(model_key)
            latest_result_time = latest_result.execution_date
            evaluation_data_time = evaluation_output_class.execution_date
            if evaluation_data_time > latest_result_time:
                latest_results[evaluation_key][model_key] = evaluation_output_class
        else:
            latest_results[evaluation_key][model_key] = evaluation_output_class

    for evaluation_name, results in latest_results.items():
        console.print(f"[bold]{evaluation_name}[/bold]")
        for model_key, result in results.items():
            console.print(f"[italic]{model_key}[/italic]: [blue]{result.score}[/blue]")
            if args.verbose and hasattr(result, "message"):
                model_output = result.message
                console.print(Padding(f"[bright_black]{model_output}[/bright_black]", (0, 0, 0, 2)))
        console.print()

    # Aggregate the results by model
    model_results = defaultdict(list)
    for _, results in latest_results.items():
        for model_key, result in results.items():
            model_results[model_key].append(result.score)
    console.print("[bold]Model Averages[/bold]")
    for model, scores in model_results.items():
        average_score = sum(scores) / len(scores)
        number_of_evals = len(scores)
        console.print(f"[italic]{model}[/italic] ({number_of_evals} Evals): [blue]{average_score:.2f}[/blue]")


if __name__ == "__main__":
    main()
