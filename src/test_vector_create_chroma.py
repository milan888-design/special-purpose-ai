import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

#documents_raw = [
#    {"id": "doc_A", "title":"To reduce the cost of education", "content": "Universities should control non education related expenses."},
#    {"id": "doc_A", "title":"To reduce the cost of education", "content": "Compress college degree period from 4 years to 3 years."},
#    {"id": "doc_A","title": "To reduce the cost of education", "content": "Promote trade and skill paths as an alternative to four year degree."},
#    {"id": "doc_A", "title":"To reduce the cost of education", "content": "Provide lower rates for student loan."},
#    {"id": "doc_A","title": "To reduce the cost of education", "content": "Provide loan payback by service to country or charity."},
#    {"id": "doc_B", "title":"PUBLIC education benefits", "content": "PUBLIC EDUCATION has done many good things such as affordability"},
#    {"id": "doc_B","title": "PUBLIC education benefits", "content": "PUBLIC EDUCATION has done many good things such as common education across multiple economic level."}
#]

#documents_raw = [
#    {"id": "doc_A", "title":"To reduce the cost of education", "content": "Universities should control non education related expenses. Compress college degree period from 4 years to 3 years. Promote trade and skill paths as an alternative to four year degree. Provide lower rates for student loan. Provide loan payback by service to country or charity."},
#    {"id": "doc_B", "title":"Public education benefits", "content": "Public education has done many good things such as affordability. Public education has done many good things such as common education across multiple economic level."},
#]

documents_raw = [
    {"id": "doc_A", "title":"To reduce crime", "content": "Government is needed to reduce crime. Less corrupt government is more effective to reduce crime."},
    {"id": "doc_B", "title":"Benefits of FREE PRESS", "content": "Government corruption can be reduced by FREE PRESS. Less corrupt government can reduce crime."}
]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = []
metadatas = []

for doc in documents_raw:
    # Split each document's content into chunks
    doc_chunks = text_splitter.split_text(doc["content"])
    for chunk_text in doc_chunks:
        chunks.append(chunk_text)
        # Associate metadata with each chunk
        metadatas.append({"document_id": doc["id"], "document_title": doc["title"]})

# Initialize your embedding model
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# --- End of Dummy Data ---

# Define a directory to store the ChromaDB data
#persist_directory = "./chroma_db_segregated"
#persist_directory = "c:/pydatacroma"
persist_directory = "c:/pydatachroma"

# Create the directory if it doesn't exist
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

# Option 1: Create and persist (first time or if you want to rebuild)
# Pass the chunks and their corresponding metadatas
vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=embedding_model,
    metadatas=metadatas,  # <--- Pass the list of metadatas here
    persist_directory=persist_directory
)
print(f"ChromaDB persisted to {persist_directory} with segregated data.")

# Remember to call persist() if you modify the DB after creation/loading
vectorstore.persist()

# --- How to use the segregation for operations ---
# Option 2: Load an existing persisted ChromaDB (for subsequent runs)
# (Commented out for now, but this is how you would load it)
# vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
# print(f"ChromaDB loaded from {persist_directory}")

# Example: Retrieve chunks related to a specific document ID
print("\n--- Retrieving chunks for 'doc_A' ---")
results_doc_A = vectorstore.get(
    where={"document_id": "doc_A"}
)
# Note: .get() returns a dictionary with 'ids', 'embeddings', 'documents', 'metadatas'
print(f"Found {len(results_doc_A['documents'])} chunks for 'doc_A'.")
for i, doc_content in enumerate(results_doc_A['documents']):
    print(f"Chunk {i+1} (ID: {results_doc_A['ids'][i]}, Metadata: {results_doc_A['metadatas'][i]}): {doc_content[:100]}...")


# Example: Retrieve chunks related to a specific document title
print("\n--- Retrieving chunks for 'Document B' ---")
results_doc_B = vectorstore.get(
    where={"document_title": "To reduce the cost of education"}
)
print(f"Found {len(results_doc_B['documents'])} chunks for 'Document B'.")
for i, doc_content in enumerate(results_doc_B['documents']):
    print(f"Chunk {i+1} (ID: {results_doc_B['ids'][i]}, Metadata: {results_doc_B['metadatas'][i]}): {doc_content[:100]}...")

