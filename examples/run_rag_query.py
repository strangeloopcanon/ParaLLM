# examples/run_rag_query.py

import sys
import os
from pathlib import Path

# Add the project root to the Python path to allow importing 'parallm'
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Now import the necessary functions
# Make sure model_query can handle potential missing API keys gracefully or has dotenv setup
# For this example, we might need to handle potential LLM query errors.
from parallm.model_query import query_model 
from parallm.rag.pipeline import run_pipeline, setup_retriever

# --- Configuration ---
CONFIG_PATH = "examples/rag_config.yaml"
USER_QUERY = "What are the benefits of Bodo?"
# Choose the LLM model for the final answer generation
LLM_MODEL = "gemini-2.0-flash" 

# --- Main Script Logic ---
def main():
    print("--- ParaLLM RAG Example --- \n")

    # 1. Build the RAG Index (runs the pipeline defined in the YAML)
    #    This will ingest, chunk, embed, and store in ChromaDB.
    print("Step 1: Building RAG Index...")
    try:
        run_pipeline(CONFIG_PATH)
    except Exception as e:
        print(f"\nError during RAG pipeline execution: {e}")
        print("Please ensure all dependencies are installed (PyYAML, pandas, sentence-transformers, chromadb) and the config is valid.")
        return
    print("\nIndex building process completed.")

    # 2. Setup the Retriever 
    print("\nStep 2: Setting up the retriever...")
    try:
        retriever = setup_retriever(CONFIG_PATH)
    except Exception as e:
        print(f"\nError setting up retriever: {e}")
        return
    print("Retriever setup complete.")

    # 3. Retrieve Context for the Query
    print(f"\nStep 3: Retrieving context for query: '{USER_QUERY}'")
    try:
        retrieved_docs = retriever(USER_QUERY)
        print(f"Retrieved {len(retrieved_docs)} context chunks.")
        if not retrieved_docs:
             print("Warning: No relevant context found in the vector store for this query.")
             # Decide whether to proceed without context or stop
             # context_text = "No context found."
        else:
            # Display retrieved docs (optional)
            print("--- Retrieved Context Chunks ---")
            for i, doc in enumerate(retrieved_docs):
                print(f"Chunk {i+1} (Score: {doc.get('score', 'N/A'):.4f}):")
                print(f"  Text: {doc.get('chunk_text', '')[:150]}...") # Show snippet
                print(f"  Source: {doc.get('metadata', {}).get('filename', 'N/A')}")
            print("-----------------------------")

    except Exception as e:
        print(f"\nError during context retrieval: {e}")
        return

    # 4. Format Context and Augment Prompt
    print("\nStep 4: Formatting context and augmenting prompt...")
    if retrieved_docs:
        context_text = "\n\n---\n\n".join([doc['chunk_text'] for doc in retrieved_docs if doc.get('chunk_text')])
    else:
        context_text = "No relevant context was found."

    prompt_template = f"""Context information is below.
---------------------
{context_text}
---------------------

Given the context information and not prior knowledge, answer the query.
Query: {USER_QUERY}
Answer:"""

    print("Augmented prompt prepared.")
    # print("--- Augmented Prompt ---")
    # print(prompt_template)
    # print("------------------------")

    # 5. Call ParaLLM Query Model
    print(f"\nStep 5: Sending augmented prompt to LLM ({LLM_MODEL})...")
    try:
        # Ensure API keys (e.g., OPENAI_API_KEY) are set in your environment
        # or through a .env file loaded by parallm.model_query
        response = query_model(prompt_template, model_given=LLM_MODEL)
        print("\n--- LLM Response ---")
        print(response)
        print("--------------------")
    except ImportError as e:
         print(f"\nError: Missing dependency for model querying: {e}")
         print("Please install required LLM client libraries (e.g., 'pip install llm-gpt')")
    except Exception as e:
        print(f"\nError querying LLM ({LLM_MODEL}): {e}")
        print("Please ensure API keys are configured correctly and the model name is valid.")

    print("\n--- RAG Example Finished ---")

if __name__ == "__main__":
    main() 