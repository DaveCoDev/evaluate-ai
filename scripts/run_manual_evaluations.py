"""Run this script to execute all external evaluations defined in data/evaluations/*yaml (except config.yaml).
These evaluations require manual input and are not part of the automated evaluation pipeline.
"""

from pathlib import Path

import yaml

from evaluate_ai.evaluation_instance import EvaluationInstance
from evaluate_ai.run_config import RunConfig

evaluations_folder = Path(__file__).parent.parent / "data" / "evaluations"


def main() -> None:
    # Assume there is a config.yaml in the evaluations folder
    yaml_file = evaluations_folder / "config.yaml"
    config = RunConfig.load(yaml_file)

    # Assume the rest of the files are evaluations and load them all into a list
    evaluations = []
    for evaluation_file in evaluations_folder.glob("*.yaml"):
        if evaluation_file.name != "config.yaml":
            with Path.open(evaluation_file, "r") as file:
                evaluation = yaml.safe_load(file)
                evaluations.extend(evaluation)

    # For each evaluation, load the corresponding class and execute it
    evaluation_instances: list[EvaluationInstance] = EvaluationInstance.initialize_evaluation_instances(evaluations)
    for evaluation_instance in evaluation_instances:
        for model in config.external_models:
            if evaluation_instance.name in config.external_evaluations:
                evaluation_instance.evaluation_class_instance.evaluate_external(model)


if __name__ == "__main__":
    main()
