from tinydb import TinyDB

from evaluate_ai.tinydb_helpers.db_path import TINYDB_PATH


def get_executed_evaluations() -> set[tuple[str, str, str, str]]:
    """Returns a set of keys corresponding to evaluations already in the database.
    The key is a tuple of (evaluation_name, model, provider, evaluation_instance_name)
    for any evaluation outputs where the output_type is 'instance'.

    Returns:
        set[tuple[str, str, str, str]]: A set of tuples of (evaluation_name, model, provider, evaluation_instance_name).
    """
    db = TinyDB(TINYDB_PATH)
    documents = db.all()

    unique_evals = set()

    # Iterate over each document and add the evaluation name, model, provider pair to the set
    for doc in documents:
        if doc["output_type"] == "instance":
            unique_evals.add(
                (
                    doc["class_name"],
                    doc["name_model"],
                    doc["provider"],
                    doc["evaluation_instance"]["name"],
                )
            )

    return unique_evals


get_executed_evaluations()
