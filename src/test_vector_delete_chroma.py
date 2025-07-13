import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Define a directory to store the ChromaDB data
#persist_directory = "c:/pydatacroma"
persist_directory = "c:/pydatachroma"
# Initialize your embedding model (must be the same one used for creation)
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# --- Retrieve the existing ChromaDB ---
# It's crucial to load the vectorstore from your persisted directory
# before you can perform any operations on it.
try:
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    print(f"ChromaDB loaded from {persist_directory}.")
except Exception as e:
    print(f"Error loading ChromaDB: {e}")
    print("Please ensure the ChromaDB exists at the specified path and was created with the same embedding function.")
    exit() # Exit if we can't load the vectorstore

# Example: Deleting a specific document by ID
print("\n--- Deleting 'doc_A' ---")
# First, retrieve the IDs of the chunks associated with 'doc_A'.
ids_to_delete = vectorstore.get(where={"document_id": "doc_A"})['ids']
#ids_to_delete = vectorstore.get(where={"document_id": "doc_B"})['ids']

if ids_to_delete:
    vectorstore.delete(ids=ids_to_delete)
    print(f"Deleted {len(ids_to_delete)} chunks associated with 'doc_A'.")
    # It's good practice to persist changes after deletion
    vectorstore.persist()
else:
    print("'doc_A' not found for deletion or has no associated chunks.")

# Verify deletion
print("\n--- Verifying deletion of 'doc_A' ---")
results_doc_A_after_delete = vectorstore.get(where={"document_id": "doc_A"})
print(f"Found {len(results_doc_A_after_delete['documents'])} chunks for 'doc_A' after deletion.")
# Verify deletion
print("\n--- Verifying deletion of 'doc_A' ---")
results_doc_A_after_delete = vectorstore.get(where={"document_id": "doc_A"})
print(f"Found {len(results_doc_A_after_delete['documents'])} chunks for 'doc_A' after deletion.")
