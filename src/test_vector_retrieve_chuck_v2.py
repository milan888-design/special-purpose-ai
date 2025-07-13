from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import torch
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline

# --- ONE-TIME SETUP (Steps 1, 2, 3, and part of 4) ---
# These components are loaded only once and reused for all questions.

print("Starting one-time setup: Loading model, tokenizer, and creating pipelines...")

# 1. Load your Gemma model and tokenizer
# Specify the path to your locally downloaded Gemma model
#model_name = "C:/Users/milan/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767"
#model_name = "C:/Users/milan/.cache/huggingface/hub/models--google--gemma-3-1b-it/snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752"
#model_name = "C:/Users/milan/.cache/huggingface/hub/models--google--gemma-3-12b-it/.no_exist/96b6f1eccf38110c56df3a15bffe176da04bfd80"
model_name = "google/gemma-3-1b-it"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16, # Use bfloat16 for better performance on compatible hardware
    # Add trust_remote_code=True if needed for custom model architectures
)

# 2. Create a text generation pipeline (raw transformers pipeline)
hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=500, # Adjust as needed for response length
    temperature=1.0,    # Controls randomness: higher = more creative, lower = more focused
    do_sample=True,     # Enables sampling (otherwise greedy decoding)
    top_k=50,           # Considers only the top_k most probable tokens
    top_p=0.95,         # Considers tokens whose cumulative probability exceeds top_p
    # Important for generation to avoid warnings and ensure consistent output for models like Gemma
    pad_token_id=tokenizer.eos_token_id
)

# 3. Wrap the transformers pipeline with LangChain's HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# 4. Create embeddings and load the vector database (also part of one-time setup)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#persist_directory = "c:/pydatacroma"
persist_directory = "c:/pydatachroma"

# Load the existing Chroma vectorstore from the specified directory
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
print(f"ChromaDB loaded from {persist_directory}")

# If you modify the DB after creation/loading, remember to call persist()
# vectorstore.persist()

# Configure the retriever to fetch a specified number of relevant documents
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

print("One-time setup complete.")

# --- RAG CHAIN DEFINITION (Step 5 & 6) ---
# This part also needs to be defined only once.

# 5. Define the RAG prompt template
# This template structures the input for the LLM, including context and question.
rag_prompt_template = """<start_of_turn>user
Answer the question based *only* on the following context. If you cannot answer the question from the context, please say "I don't have enough information to answer that."

Context:
{context}

Question: {question}<end_of_turn>
<start_of_turn>model
"""
rag_prompt = PromptTemplate.from_template(rag_prompt_template)

# 6. Build the RAG chain
# This chain orchestrates the retrieval and generation steps.
rag_chain = (
    # First, retrieve relevant context documents based on the question
    {"context": retriever, "question": RunnablePassthrough()}
    # Then, format the retrieved context and question into the defined prompt
    | rag_prompt
    # Pass the formatted prompt to the Language Model (LLM) for generation
    | llm # Use the wrapped LangChain LLM object initialized above
    # Finally, parse the LLM's output into a simple string
    | StrOutputParser()
)

# --- INTERACTIVE QUESTION ASKING LOOP (Step 7 - modified) ---
# This part allows continuous interaction.

print("\nReady to answer questions. Type 'exit' or 'quit' to end the session.")

while True:
    question = input("\nYour Question: ")
    if question.lower() in ["exit", "quit"]:
        print("Ending session.")
        break
    
    # Get a RAG-augmented answer for the user's question
    response = rag_chain.invoke(question)
    print(f"Answer: {response}")