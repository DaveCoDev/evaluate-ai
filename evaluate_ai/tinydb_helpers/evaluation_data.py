from tinydb import TinyDB

from evaluate_ai.tinydb_helpers.db_path import TINYDB_PATH


def get_executed_evaluations() -> list[tuple[str, str, str]]:
    db = TinyDB(TINYDB_PATH)
    documents = db.all()

    unique_evals = set()

    # Iterate over each document and add the evaluation name, model, provider pair to the set
    for doc in documents:
        unique_evals.add((doc["name"], doc["model_name"], doc["metadata"]["model_provider"]))

    return list(unique_evals)
