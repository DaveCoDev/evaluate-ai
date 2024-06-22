"""Run this script to execute all evaluations defined in data/evaluations/*yaml (except config.yaml)."""

import argparse
from pathlib import Path

from not_again_ai.llm.openai_api.openai_client import openai_client
from not_again_ai.local_llm.ollama.ollama_client import ollama_client
import yaml

from evaluate_ai.evaluation import Provider
from evaluate_ai.evaluation_instance import EvaluationInstance
from evaluate_ai.run_config import RunConfig

evaluations_folder = Path(__file__).parent.parent / "data" / "evaluations"


def main() -> None:
    # By default assume all .yaml files in the evaluations folder are to be used except config.yaml
    default_files = [str(file) for file in evaluations_folder.glob("*.yaml") if file.name != "config.yaml"]

    parser = argparse.ArgumentParser(description="Execute evaluations defined in specified .yaml files.")
    parser.add_argument(
        "-f",
        "--files",
        nargs="*",
        default=default_files,
        help="List of .yaml files to include in the evaluation. If not provided, all .yaml files in the evaluations folder will be used.",
    )
    evaluation_files = [Path(file) for file in parser.parse_args().files]

    # Assume there is a config.yaml in the evaluations folder
    yaml_file = evaluations_folder / "config.yaml"
    config = RunConfig.load(yaml_file)

    evaluations = []
    for evaluation_file in evaluation_files:
        with Path.open(evaluation_file, "r") as file:
            evaluation = yaml.safe_load(file)
        evaluations.extend(evaluation)

    client_openai = openai_client()
    client_ollama = ollama_client()

    # Check if the evaluations are valid
    EvaluationInstance.check_evaluation_instances(evaluations)
    # Iterate over each evaluation and execute it once for each model
    for evaluation in evaluations:
        for provider, models in config.models.items():
            if provider == Provider.OPENAI:
                llm_client = client_openai
            elif provider == Provider.OLLAMA:
                llm_client = client_ollama

            for model in models:
                evaluation_instance = EvaluationInstance(**evaluation)
                evaluation_instance.evaluation_class_instance.execute(model, llm_client=llm_client, provider=provider)


if __name__ == "__main__":
    main()
