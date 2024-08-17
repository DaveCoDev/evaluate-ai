"""Prints latest results for evaluation and each model to the console."""

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.padding import Padding
from tinydb import TinyDB

TINYDB_PATH = Path(__file__).parent.parent / "data" / "tinydb.json"


def main() -> None:
    console = Console(highlight=False)

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", default=True, help="Additionally print model outputs")
    args = parser.parse_args()

    db = TinyDB(TINYDB_PATH)
    documents = db.all()

    latest_results = defaultdict(dict)
    for evaluation_data in documents:
        evaluation_key = f"{evaluation_data['name']} ({evaluation_data['type']})"
        model = evaluation_data.get("name_model")

        if evaluation_key in latest_results and model in latest_results[evaluation_key]:
            latest_result = latest_results.get(evaluation_key).get(model)
            latest_result_time = datetime.fromisoformat(latest_result.get("execution_date"))
            evaluation_data_time = datetime.fromisoformat(evaluation_data.get("execution_date"))
            if evaluation_data_time > latest_result_time:
                latest_results[evaluation_key][model] = evaluation_data
        else:
            latest_results[evaluation_key][model] = evaluation_data

    for evaluation_name, results in latest_results.items():
        console.print(f"[bold]{evaluation_name}[/bold]")
        for model, result in results.items():
            console.print(f"[italic]{model}[/italic]: [blue]{result['score']}[/blue]")
            if args.verbose:
                model_outputs = result.get("metadata", {}).get("output", "")
                for idx, model_output in enumerate(model_outputs):
                    padding_amount = 0
                    if len(model_outputs) > 1:
                        console.print(f"[bright_black][italic]Output {idx + 1}:[/bright_black][/italic]")
                        padding_amount = 2
                    console.print(Padding(f"[bright_black]{model_output}[/bright_black]", (0, 0, 0, padding_amount)))
        console.print()

    # Aggregate the results by model
    model_results = defaultdict(list)
    for _, results in latest_results.items():
        for model, result in results.items():
            model_results[model].append(result["score"])
    console.print("[bold]Model Averages[/bold]")
    for model, scores in model_results.items():
        average_score = sum(scores) / len(scores)
        number_of_evals = len(scores)
        console.print(f"[italic]{model}[/italic] ({number_of_evals} Evals): [blue]{average_score:.2f}[/blue]")


if __name__ == "__main__":
    main()
