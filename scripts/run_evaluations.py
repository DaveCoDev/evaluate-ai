"""Run this script to execute all evaluations defined in data/evaluations/*yaml (except config.yaml)."""

import argparse
from pathlib import Path

from rich.progress import Progress
import yaml

from evaluate_ai.evaluation_instance import EvaluationInstance
from evaluate_ai.run_config import RunConfig, get_llm_client
from evaluate_ai.tinydb_helpers.evaluation_data import get_executed_evaluations

evaluations_folder = Path(__file__).parent.parent / "data" / "evaluations"


def main() -> None:
    # By default assume all .yaml files in the evaluations folder are to be used except config.yaml
    default_files = [str(file) for file in evaluations_folder.glob("*.yaml") if file.name != "config.yaml"]

    parser = argparse.ArgumentParser(description="Execute evaluations defined in specified .yaml files.")
    parser.add_argument(
        "-f",
        "--files",
        nargs="*",
        default=default_files,  # ["./data/evaluations/03_meets_criteria.yaml"],
        help="List of .yaml files to include in the evaluation. If not provided, all .yaml files in the evaluations folder will be used.",
    )
    parser.add_argument(
        "--only_new",
        action="store_true",
        default=True,
        help="If set, only evaluations not already in the database will be executed.",
    )
    args = parser.parse_args()
    evaluation_files = [Path(file) for file in args.files]

    # Assumes there is a config.yaml in the evaluations folder
    yaml_file = evaluations_folder / "config.yaml"
    config = RunConfig.load(yaml_file)

    evaluations = []
    for evaluation_file in evaluation_files:
        with Path.open(evaluation_file, "r") as file:
            evaluation = yaml.safe_load(file)
        evaluations.extend(evaluation)

    EvaluationInstance.check_evaluation_instances(evaluations, config)

    if args.only_new:
        unique_evals = get_executed_evaluations()

    # Gather tuples of (evaluation_instance, model, llm_client) so we can track progress
    evaluations_to_run = []
    for evaluation in evaluations:
        for provider, models in config.models.items():
            for model in models:
                llm_client = get_llm_client(provider)
                evaluation_instance = EvaluationInstance(config=config, **evaluation)
                if args.only_new:
                    curr_evaluation_name = evaluation_instance.evaluation_class_instance.evaluation_data.name
                    curr_evaluation_identifier = (curr_evaluation_name, model, provider.value)
                    if curr_evaluation_identifier in unique_evals:
                        continue
                evaluations_to_run.append((evaluation_instance, model, llm_client))

    with Progress() as progress:
        task = progress.add_task("Evaluations Progress", total=len(evaluations_to_run))
        for evaluation_instance, model, llm_client in evaluations_to_run:
            progress.console.print(
                f"Running evaluation [bold magenta]{evaluation_instance.name}[/bold magenta] for model [bold orange3]{model}[/bold orange3]",
                highlight=False,
            )
            evaluation_instance.evaluation_class_instance.execute(model, llm_client=llm_client, provider=provider)
            progress.advance(task)


if __name__ == "__main__":
    main()
