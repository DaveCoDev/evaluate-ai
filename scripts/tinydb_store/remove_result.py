"""Prints all documents in the database (id and execution_data fields only)
and prompts the user to select a document ID to delete.
"""

from tinydb import TinyDB

from evaluate_ai.tinydb_helpers.db_path import TINYDB_PATH

if __name__ == "__main__":
    db = TinyDB(TINYDB_PATH)

    documents = db.all()
    for doc in documents:
        print(f"ID: {doc.doc_id}, Content: {doc['execution_date']}")

    # Prompt the user to select a document ID to delete
    try:
        selected_id = int(input("Enter the ID of the document you want to delete: "))
    except ValueError:
        print("Please enter a valid integer.")
        exit()

    # Check if the selected ID exists in the database
    if any(doc.doc_id == selected_id for doc in documents):
        db.remove(doc_ids=[selected_id])
        print(f"Document with ID {selected_id} has been deleted.")
    else:
        print(f"No document found with ID {selected_id}.")
