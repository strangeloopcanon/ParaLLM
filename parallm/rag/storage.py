import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from typing import Dict, Any, List
import logging
import json

logger = logging.getLogger(__name__)

def _serialize_metadata(metadata_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure metadata values are JSON serializable strings for ChromaDB.

    ChromaDB metadata values must be strings, integers, floats, or booleans.
    This function attempts to convert other types to JSON strings as a fallback.

    Args:
        metadata_dict: The original metadata dictionary.

    Returns:
        A new dictionary with values converted to supported types.
    """
    serialized = {}
    for k, v in metadata_dict.items():
        if isinstance(v, (str, int, float, bool)):
            serialized[k] = v
        else:
            try:
                # Attempt to serialize complex types as JSON strings
                serialized[k] = json.dumps(v) 
                logger.debug(f"Serialized metadata key '{k}' to JSON string.")
            except TypeError:
                # Fallback: convert to string if JSON serialization fails
                str_v = str(v)
                serialized[k] = str_v
                logger.warning(
                    f"Metadata key '{k}' had non-serializable value '{v}'. "
                    f"Converted to basic string: '{str_v}'"
                )
    return serialized

def save_to_vector_store(df: pd.DataFrame, vector_store_type: str, **kwargs):
    """Saves document chunks, embeddings, and metadata to a vector store.

    Args:
        df: DataFrame with 'chunk_id', 'chunk_text', 'embedding', 'metadata' columns.
        vector_store_type: The type of vector store (currently only 'chromadb').
        **kwargs: Vector store specific parameters.
            For ChromaDB: 'collection_name', 'persist_directory' (optional).

    Raises:
        ValueError: If required parameters are missing or invalid.
        ImportError: If chromadb is not installed.
        Exception: For errors during ChromaDB operations.
    """
    if vector_store_type.lower() != 'chromadb':
        raise ValueError(f"Unsupported vector store type: {vector_store_type}. Currently only 'chromadb' is supported.")

    if df.empty:
        print("Warning: Input DataFrame for vector store saving is empty. Skipping.")
        return

    required_cols = ['chunk_id', 'chunk_text', 'embedding', 'metadata']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Input DataFrame missing one or more required columns: {required_cols}")

    collection_name = kwargs.get('collection_name')
    persist_directory = kwargs.get('persist_directory')

    if not collection_name:
        raise ValueError("Missing required parameter for ChromaDB: 'collection_name'")

    print(f"Connecting to ChromaDB...")
    if persist_directory:
        print(f"Using persistent storage at: {persist_directory}")
        client = chromadb.PersistentClient(path=persist_directory)
    else:
        print("Using in-memory ChromaDB storage.")
        client = chromadb.Client() 

    # We rely on providing embeddings directly during upsert.
    collection = client.get_or_create_collection(name=collection_name)
    print(f"Using ChromaDB collection: '{collection_name}'")

    batch_size = 100 # Adjust as needed
    num_batches = (len(df) + batch_size - 1) // batch_size
    print(f"Preparing to upsert {len(df)} items in {num_batches} batches of size {batch_size}...")

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx]

        ids = batch_df['chunk_id'].tolist()
        embeddings = batch_df['embedding'].tolist() 
        documents = batch_df['chunk_text'].tolist()
        
        metadatas = []
        for _, row in batch_df.iterrows():
            meta = row.get('metadata', {}).copy() if isinstance(row.get('metadata'), dict) else {}
            if 'doc_id' not in meta and 'doc_id' in row:
                 meta['doc_id'] = row['doc_id'] 
            metadatas.append(_serialize_metadata(meta))
            
        embeddings_list = [emb.tolist() if hasattr(emb, 'tolist') else emb for emb in embeddings]

        print(f"Upserting batch {i+1}/{num_batches} ({len(ids)} items)...")
        try:
            collection.upsert(
                ids=ids,
                embeddings=embeddings_list,
                documents=documents,
                metadatas=metadatas
            )
        except Exception as e:
            print(f"Error upserting batch {i+1} to ChromaDB collection '{collection_name}': {e}")
            raise 

    print(f"Successfully upserted {len(df)} items to ChromaDB collection '{collection_name}'.")
    if persist_directory:
         print(f"ChromaDB data updates persisted to {persist_directory}")

# Placeholder for the function to save intermediate parquet files if needed
# def save_intermediate_parquet(df: pd.DataFrame, output_path: str):
#     # ... implementation ...
#     pass
