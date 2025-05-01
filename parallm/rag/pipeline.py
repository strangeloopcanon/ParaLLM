from typing import Dict, Any, List, Callable
import pandas as pd
from functools import partial

from sentence_transformers import SentenceTransformer

from .config import load_config
from .ingestion import load_documents
from .chunking import chunk_dataframe
from .embedding import generate_embeddings
from .storage import save_to_vector_store
from .retrieval import retrieve_vector, retrieve_keyword, retrieve_hybrid
from .indexing import create_bm25_index, load_bm25_index

def run_pipeline(config_path: str):
    """Runs the RAG indexing pipeline based on the configuration file.

    Args:
        config_path: Path to the YAML configuration file.
    """
    print(f"Loading configuration from: {config_path}")
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        return

    print("Configuration loaded successfully.")
    pipeline_steps: List[Dict[str, Any]] = config.get('pipeline', [])

    processed_data: pd.DataFrame | None = None

    print("Starting RAG pipeline...")
    for i, step in enumerate(pipeline_steps):
        step_name = step.get('name')
        step_params = step.get('params', {})
        print(f"\n--- Step {i+1}: {step_name} ---")
        print(f"Params: {step_params}")

        try:
            if step_name == 'ingest':
                print("Executing ingestion step...")
                processed_data = load_documents(**step_params)
                if processed_data is not None:
                    print(f"Ingested {len(processed_data)} documents.")
                else:
                     print("Ingestion step returned no data.")
            elif step_name == 'chunk':
                print("Executing chunking step...")
                if processed_data is None or processed_data.empty:
                     raise ValueError("Cannot chunk without valid data from prior step.")
                processed_data = chunk_dataframe(processed_data, **step_params)
                if processed_data is not None:
                    print(f"Chunked into {len(processed_data)} chunks.")
                else:
                    print("Chunking step returned no data.")
            elif step_name == 'embed':
                print("Executing embedding step...")
                if processed_data is None or processed_data.empty:
                     raise ValueError("Cannot embed without valid data from prior step.")
                if 'embedding_model_name' not in step_params:
                     raise ValueError("Missing 'embedding_model_name' parameter for embed step.")
                processed_data = generate_embeddings(processed_data, **step_params)
                if processed_data is not None and 'embedding' in processed_data.columns:
                    print(f"Generated embeddings for {len(processed_data)} chunks.")
                else:
                    print("Embedding step failed or returned no data.")
            elif step_name == 'index_vector':
                print("Executing vector indexing step...")
                if processed_data is None or processed_data.empty or 'embedding' not in processed_data.columns:
                     raise ValueError("Cannot index vectors without valid data including embeddings from prior step.")
                if 'vector_store' not in step_params:
                    raise ValueError("Missing 'vector_store' type in index_vector params.")
                
                vector_store_type = step_params['vector_store']
                save_to_vector_store(processed_data, vector_store_type=vector_store_type, **step_params)
                print(f"Vector store indexing complete for '{vector_store_type}'.")
            elif step_name == 'index_keyword':
                print("Executing keyword indexing step...")
                if processed_data is None or processed_data.empty or 'chunk_text' not in processed_data.columns:
                    raise ValueError("Cannot create keyword index without valid chunked text data.")
                if 'index_path' not in step_params:
                    raise ValueError("Missing 'index_path' parameter for index_keyword step.")
                create_bm25_index(processed_data, **step_params)
                print(f"Keyword index creation complete.")
            elif step_name == 'save_intermediate':
                # This step is defined in the plan but not fully implemented yet.
                # Requires uncommenting/implementing save_intermediate_parquet in storage.py
                # and handling its call here based on params['output_path'].
                print("Warning: 'save_intermediate' step is defined but not implemented. Skipping.")
                pass 
            else:
                print(f"Warning: Unknown step name '{step_name}'. Skipping.")

        except Exception as e:
            print(f"Error during step '{step_name}': {e}")
            print("Pipeline execution aborted.")
            return

    print("\n--- RAG pipeline finished successfully! ---")

def setup_retriever(config_path: str) -> Callable[[str], List[Dict[str, Any]]]:
    """Loads configuration and components to set up the retrieval function.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A callable retrieval function that takes a query string and returns
        a list of retrieved documents (dictionaries).

    Raises:
        ValueError: If configuration is invalid or required components cannot be loaded.
    """
    print(f"Setting up retriever based on config: {config_path}")
    config = load_config(config_path)
    retrieval_config = config.get('retrieval', {})
    pipeline_config = config.get('pipeline', [])

    strategy = retrieval_config.get('strategy', 'vector')
    print(f"Retrieval strategy: {strategy}")

    # Load Embedding Model
    embedding_model_name = None
    for step in pipeline_config:
        if step.get('name') == 'embed':
            embedding_model_name = step.get('params', {}).get('embedding_model_name')
            break 
    if strategy in ['vector', 'hybrid'] and not embedding_model_name:
         raise ValueError("Could not find 'embedding_model_name' in pipeline config for vector/hybrid retrieval.")
    
    embedding_model = None
    if embedding_model_name:
        try:
            print(f"Loading embedding model for retriever: {embedding_model_name}")
            embedding_model = SentenceTransformer(embedding_model_name)
            print("Embedding model loaded.")
        except Exception as e:
            print(f"Error loading embedding model '{embedding_model_name}': {e}")
            raise

    # Get Vector Store Params 
    vector_store_params = {}
    for step in pipeline_config:
        if step.get('name') == 'index_vector':
            vector_store_params = step.get('params', {})
            break
    if strategy in ['vector', 'hybrid'] and not vector_store_params.get('collection_name'):
        raise ValueError("Could not find 'index_vector' params (collection_name) in pipeline config.")

    # Load Keyword Index 
    bm25_index = None
    chunk_mapping = None 
    keyword_index_path = None
    for step in pipeline_config:
         if step.get('name') == 'index_keyword':
              keyword_index_path = step.get('params', {}).get('index_path')
              break
              
    if strategy in ['keyword', 'hybrid']:
        if not keyword_index_path:
             print(f"Warning: Keyword index path not found in config for strategy '{strategy}'. Keyword retrieval disabled.")
             if strategy == 'keyword': 
                 raise ValueError("Keyword index path required for 'keyword' strategy.")
        else:
             try:
                bm25_index, chunk_mapping = load_bm25_index(keyword_index_path)
                if bm25_index is None or chunk_mapping is None:
                    print(f"Warning: Failed to load valid BM25 index/mapping from {keyword_index_path}")
                    if strategy == 'keyword': 
                        raise ValueError("Could not load BM25 index required for 'keyword' strategy.")
                else:
                    print(f"Successfully loaded BM25 index from {keyword_index_path}.")
             except Exception as e:
                 print(f"Error loading BM25 index from {keyword_index_path}: {e}. Keyword retrieval disabled.")
                 if strategy == 'keyword': raise 
                 bm25_index, chunk_mapping = None, None 

    # Configure the Retriever Function
    retriever_func: Callable[[str], List[Dict[str, Any]]]

    if strategy == 'vector':
        if not embedding_model or not vector_store_params.get('collection_name'):
            raise ValueError("Missing embedding model or vector store config for 'vector' strategy.")
        
        retriever_func = partial(retrieve_vector,
                                 embedding_model=embedding_model,
                                 vector_store_type=vector_store_params.get('vector_store', 'chromadb'),
                                 top_k=retrieval_config.get('top_k_vector', 5), 
                                 collection_name=vector_store_params.get('collection_name'),
                                 persist_directory=vector_store_params.get('persist_directory')
                                )
        print("Configured for VECTOR retrieval.")

    elif strategy == 'keyword':
        if not bm25_index or not chunk_mapping:
            raise ValueError("BM25 index/mapping could not be loaded for 'keyword' strategy.") 
        print("Configuring for KEYWORD retrieval...") 
        retriever_func = partial(retrieve_keyword,
                                 bm25_index=bm25_index,
                                 chunk_mapping=chunk_mapping,
                                 top_k=retrieval_config.get('top_k_keyword', 5)
                                )

    elif strategy == 'hybrid':
        vector_ready = embedding_model and vector_store_params.get('collection_name')
        keyword_ready = bm25_index and chunk_mapping
        
        if not vector_ready and not keyword_ready:
            raise ValueError("Neither vector nor keyword components could be loaded for 'hybrid' strategy.")
            
        if not vector_ready:
             print("Warning: Missing embedding model or vector store config for 'hybrid' strategy. Vector component disabled.")
        if not keyword_ready: 
             print("Warning: Missing or failed to load BM25 index/mapping for 'hybrid' strategy. Keyword component disabled.")
            
        print("Configuring for HYBRID retrieval...") 
        retriever_func = partial(retrieve_hybrid,
                                 embedding_model=embedding_model, 
                                 vector_store_type=vector_store_params.get('vector_store', 'chromadb'),
                                 vector_collection_name=vector_store_params.get('collection_name'),
                                 vector_persist_directory=vector_store_params.get('persist_directory'),
                                 bm25_index=bm25_index, 
                                 chunk_mapping=chunk_mapping,
                                 top_k_vector=retrieval_config.get('top_k_vector', 5),
                                 top_k_keyword=retrieval_config.get('top_k_keyword', 5)
                                )

    else:
        raise ValueError(f"Unknown retrieval strategy in config: {strategy}")

    print("Retriever setup complete.")
    return retriever_func
