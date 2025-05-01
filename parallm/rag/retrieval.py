import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from .indexing import _tokenize_text # Import the same tokenizer used for indexing

def retrieve_vector(
    query_text: str, 
    embedding_model: SentenceTransformer, # Pass loaded model directly
    vector_store_type: str, 
    top_k: int, 
    collection_name: str, 
    persist_directory: str | None = None, 
    **kwargs # For potential future args like filtering
) -> List[Dict[str, Any]]:
    """Retrieves relevant document chunks based on vector similarity.

    Args:
        query_text: The user query string.
        embedding_model: The loaded Sentence Transformer model instance.
        vector_store_type: The type of vector store (currently only 'chromadb').
        top_k: The number of top similar documents to retrieve.
        collection_name: The name of the ChromaDB collection.
        persist_directory: The path to the ChromaDB persistence directory (if used).
        **kwargs: Additional arguments (currently unused).

    Returns:
        A list of dictionaries, where each dictionary represents a retrieved chunk
        and contains keys like 'chunk_id', 'chunk_text', 'metadata', 'score'.

    Raises:
        ValueError: If vector_store_type is not 'chromadb' or collection_name is missing.
        Exception: For errors during ChromaDB connection or query.
    """
    if vector_store_type.lower() != 'chromadb':
        raise ValueError(f"Unsupported vector store type: {vector_store_type}. Currently only 'chromadb' is supported.")
    if not collection_name:
        raise ValueError("Missing required parameter: 'collection_name'")

    # 1. Embed the query
    print(f"Generating embedding for query: '{query_text[:100]}...'")
    try:
        query_embedding = embedding_model.encode([query_text])[0]
        # Convert to list for ChromaDB compatibility if it's numpy array
        query_embedding_list = query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        raise
    print("Query embedding generated.")

    # 2. Connect to ChromaDB
    print(f"Connecting to ChromaDB collection '{collection_name}'...")
    try:
        if persist_directory:
            client = chromadb.PersistentClient(path=persist_directory)
        else:
            client = chromadb.Client()
        
        collection = client.get_collection(name=collection_name)
        # Note: We assume the collection exists and was created with a compatible
        # embedding function or just stores pre-computed embeddings.
        print("Connected successfully.")
    except Exception as e:
        print(f"Error connecting to ChromaDB collection '{collection_name}': {e}")
        raise

    # 3. Query the vector store
    print(f"Querying collection '{collection_name}' for top {top_k} results...")
    try:
        results = collection.query(
            query_embeddings=[query_embedding_list], # Must be list of lists/embeddings
            n_results=top_k,
            include=['documents', 'metadatas', 'distances'] # Request necessary fields
        )
        print(f"Query successful. Found results for {len(results.get('ids', [[]])[0])} items.")
    except Exception as e:
        print(f"Error querying ChromaDB collection '{collection_name}': {e}")
        raise

    # 4. Format and return results
    formatted_results: List[Dict[str, Any]] = []
    if results and results.get('ids') and results['ids'][0]:
        ids = results['ids'][0]
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0]
        
        for i, doc_id in enumerate(ids):
            formatted_results.append({
                'chunk_id': doc_id,
                # Ensure documents, metadatas, distances lists are not shorter than ids list
                'chunk_text': documents[i] if i < len(documents) else None, 
                'metadata': metadatas[i] if i < len(metadatas) else None,
                'score': 1.0 - distances[i] if i < len(distances) and distances[i] is not None else None # Convert distance to similarity score (assuming lower distance is better)
            })
            
    return formatted_results

def retrieve_keyword(
    query_text: str, 
    bm25_index: Any, # BM25Okapi object 
    chunk_mapping: Dict[int, Dict], # Maps internal index to chunk_id, text, metadata
    top_k: int
) -> List[Dict[str, Any]]:
    """Retrieves relevant document chunks based on BM25 keyword matching.

    Args:
        query_text: The user query string.
        bm25_index: The loaded BM25Okapi index object.
        chunk_mapping: Dictionary mapping internal BM25 index IDs (0, 1, ...) 
                       to original chunk data (chunk_id, chunk_text, metadata).
        top_k: The number of top relevant documents to retrieve.

    Returns:
        A list of dictionaries, where each dictionary represents a retrieved chunk
        and contains keys like 'chunk_id', 'chunk_text', 'metadata', 'score'.
    """
    if bm25_index is None or chunk_mapping is None:
        print("Error: BM25 index or chunk mapping is not loaded. Cannot perform keyword retrieval.")
        return []
        
    print(f"Tokenizing query for BM25: '{query_text[:100]}...'")
    tokenized_query = _tokenize_text(query_text)
    
    print(f"Querying BM25 index for top {top_k} results...")
    try:
        # Get scores for all documents in the corpus
        doc_scores = bm25_index.get_scores(tokenized_query)
        
        # Get the top N indices and their scores
        # Sort scores descending and get indices
        top_n_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
        
        # Get the scores corresponding to the top indices
        top_n_scores = [doc_scores[i] for i in top_n_indices]
        
        print(f"BM25 query successful. Found {len(top_n_indices)} results.")
        
    except Exception as e:
        print(f"Error querying BM25 index: {e}")
        return []

    # Format results using the chunk mapping
    formatted_results: List[Dict[str, Any]] = []
    for i, score in zip(top_n_indices, top_n_scores):
        if i in chunk_mapping:
            chunk_data = chunk_mapping[i]
            formatted_results.append({
                'chunk_id': chunk_data.get('chunk_id'),
                'chunk_text': chunk_data.get('chunk_text'),
                'metadata': chunk_data.get('metadata'),
                'score': score 
            })
        else:
            print(f"Warning: BM25 index returned index {i}, but it was not found in the chunk mapping.")
            
    return formatted_results

def retrieve_hybrid(
    query_text: str, 
    # Vector params
    embedding_model: Any, # SentenceTransformer or None
    vector_store_type: str | None, 
    vector_collection_name: str | None,
    vector_persist_directory: str | None,
    top_k_vector: int,
    # Keyword params
    bm25_index: Any | None, # BM25 object or None
    chunk_mapping: Dict | None,
    top_k_keyword: int,
    # Future: Add fusion/reranking options
) -> List[Dict[str, Any]]:
    """Retrieves document chunks using a hybrid of vector and keyword search.

    Combines results from vector and keyword searches, removing duplicates.

    Args:
        query_text: The user query string.
        embedding_model: Loaded Sentence Transformer model (or None if vector disabled).
        vector_store_type: Type of vector store ('chromadb').
        vector_collection_name: Name of the ChromaDB collection.
        vector_persist_directory: Path to ChromaDB persistence directory.
        top_k_vector: Number of vector results to retrieve.
        bm25_index: Loaded BM25 index object (or None if keyword disabled).
        chunk_mapping: Mapping for BM25 results.
        top_k_keyword: Number of keyword results to retrieve.

    Returns:
        A combined list of unique dictionaries representing retrieved chunks.
        Results currently preserve original scores and are not re-ranked.
    """
    vector_results: List[Dict[str, Any]] = []
    keyword_results: List[Dict[str, Any]] = []

    # 1. Perform Vector Search (if components available)
    if embedding_model and vector_store_type and vector_collection_name:
        print("Performing vector search component of hybrid retrieval...")
        try:
            vector_results = retrieve_vector(
                query_text=query_text,
                embedding_model=embedding_model,
                vector_store_type=vector_store_type,
                top_k=top_k_vector,
                collection_name=vector_collection_name,
                persist_directory=vector_persist_directory
            )
            print(f"Vector search returned {len(vector_results)} results.")
        except Exception as e:
            print(f"Error during vector search in hybrid retrieval: {e}")
            # Continue without vector results
    else:
         print("Skipping vector search component (missing config/components)." )

    # 2. Perform Keyword Search (if components available)
    if bm25_index and chunk_mapping:
        print("Performing keyword search component of hybrid retrieval...")
        try:
            keyword_results = retrieve_keyword(
                query_text=query_text,
                bm25_index=bm25_index,
                chunk_mapping=chunk_mapping,
                top_k=top_k_keyword
            )
            print(f"Keyword search returned {len(keyword_results)} results.")
        except Exception as e:
            print(f"Error during keyword search in hybrid retrieval: {e}")
            # Continue without keyword results
    else:
         print("Skipping keyword search component (missing config/components)." )

    # 3. Combine and Deduplicate Results
    print("Combining and deduplicating hybrid results...")
    combined_results: Dict[str, Dict[str, Any]] = {}

    # Add results, prioritizing based on order (e.g., vector first)
    # A more sophisticated approach would use scores (RRF), but BM25 and vector scores aren't directly comparable.
    for result in vector_results + keyword_results:
        chunk_id = result.get('chunk_id')
        if chunk_id and chunk_id not in combined_results:
            combined_results[chunk_id] = result
            
    final_results = list(combined_results.values())
    print(f"Combined hybrid search yields {len(final_results)} unique results.")
    
    # Optional: Sort results? (Currently unsorted relative to each other)
    # Optional: Limit total number of results?
    
    return final_results
