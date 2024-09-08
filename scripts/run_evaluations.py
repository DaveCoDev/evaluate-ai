import argparse
from pathlib import Path
import sys

from loguru import logger
from rich.progress import Progress
import yaml

from evaluate_ai.evaluation import Evaluation
from evaluate_ai.tinydb_helpers.evaluation_data import get_executed_evaluations

evaluations_folder = Path(__file__).parent.parent / "data" / "evaluations"


if __name__ == "__main__":
    # By default assume all .yaml files in the evaluations folder are to be used except config.yaml
    default_files = [str(file) for file in evaluations_folder.glob("*.yaml")]
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
    evaluation_paths = [Path(file) for file in args.files]

    executed_evaluations = get_executed_evaluations() if args.only_new else set()

    # Load each evaluation using the module and class names
    evaluation_classes = []
    evaluations_to_run = 0
    for evaluation_path in evaluation_paths:
        try:
            with Path.open(evaluation_path) as file:
                config = yaml.safe_load(file)
                module_name = config["run_config"]["module_name"]
                class_name = config["run_config"]["class_name"]
                evaluation_class: Evaluation = Evaluation.load_class(module_name, class_name, config)
                evaluations_to_run += evaluation_class.num_instances(keys_to_skip=executed_evaluations)
                evaluation_classes.append(evaluation_class)
        except FileNotFoundError:
            logger.error(f"The file {evaluation_path} does not exist.")
            sys.exit(1)
        except KeyError as e:
            logger.error(f"Missing key {e} in the evaluation file at {evaluation_path}.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"An unexpected error {e} occurred while loading the evaluation file at {evaluation_path}.")
            sys.exit(1)

    with Progress() as progress:
        task = progress.add_task("Evaluations Progress", total=evaluations_to_run)
        for evaluation_class in evaluation_classes:
            evaluation_class.execute(progress, keys_to_skip=executed_evaluations)
