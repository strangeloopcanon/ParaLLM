import pandas as pd
from typing import List, Dict, Any

# Semantic chunking dependency
try:
    import nltk
    # Ensure punkt is downloaded (attempt during setup or first use)
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("NLTK 'punkt' tokenizer not found. Downloading...")
        nltk.download('punkt')
    nltk_installed = True
except ImportError:
    print("Warning: NLTK not installed. Semantic chunking will not be available.")
    nltk_installed = False

# Placeholder for potential future imports like NLTK, spaCy, langchain splitters

def _chunk_text_fixed_size(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Helper function to chunk a single text string into fixed sizes with overlap."""
    if not text:
        return []
    if chunk_size <= overlap:
        raise ValueError("Chunk size must be greater than overlap.")

    chunks = []
    start_index = 0
    while start_index < len(text):
        end_index = start_index + chunk_size
        chunks.append(text[start_index:end_index])
        start_index += chunk_size - overlap
        # Ensure we don't overshoot if the last chunk is smaller or next overlap is past end
        if start_index >= len(text) and len(chunks[-1]) < chunk_size:
             break 
        elif start_index + overlap >= len(text):
             break 
            
    return chunks

def _chunk_text_semantic(text: str, sentences_per_chunk: int = 3) -> List[str]:
    """Chunks text by grouping sentences using NLTK (requires 'punkt' data)."""
    if not nltk_installed:
        raise ImportError("NLTK is required for semantic chunking.")
    if not text:
        return []
        
    sentences = nltk.sent_tokenize(text)
    
    chunks = []
    current_chunk_sentences = []
    for sentence in sentences:
        current_chunk_sentences.append(sentence)
        if len(current_chunk_sentences) >= sentences_per_chunk:
            chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = [] # Reset for next chunk
            
    # Add any remaining sentences as the last chunk
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))
        
    return chunks

def chunk_dataframe(df: pd.DataFrame, strategy: str, **kwargs) -> pd.DataFrame:
    """Chunks the text in a DataFrame according to the specified strategy.

    Args:
        df: DataFrame containing 'doc_id', 'text', and 'metadata' columns.
        strategy: The chunking strategy to use ('fixed_size', 'semantic').
        **kwargs: Strategy-specific parameters.
            For fixed_size: 'chunk_size' (int), 'overlap' (int).
            For semantic: 'sentences_per_chunk' (int, default: 3). Requires NLTK.

    Returns:
        A new DataFrame with columns: 'doc_id', 'chunk_id', 'chunk_text', 'metadata'.

    Raises:
        ValueError: If the strategy is unknown or required parameters are missing/invalid.
        ImportError: If required library for a strategy is missing.
    """
    if df.empty:
        print("Warning: Input DataFrame for chunking is empty.")
        return pd.DataFrame(columns=['doc_id', 'chunk_id', 'chunk_text', 'metadata'])
        
    if 'text' not in df.columns or 'doc_id' not in df.columns or 'metadata' not in df.columns:
        raise ValueError("Input DataFrame missing required columns: 'doc_id', 'text', 'metadata'")

    all_chunks_data: List[Dict[str, Any]] = []

    print(f"Applying chunking strategy: {strategy}")

    for index, row in df.iterrows():
        doc_id = row['doc_id']
        text = str(row['text']) if pd.notna(row['text']) else '' # Ensure text is string
        metadata = row['metadata']
        chunks: List[str] = []

        if strategy == 'fixed_size':
            chunk_size = kwargs.get('chunk_size')
            overlap = kwargs.get('overlap', 0)
            if not isinstance(chunk_size, int) or chunk_size <= 0:
                raise ValueError("'chunk_size' parameter must be a positive integer for fixed_size strategy.")
            if not isinstance(overlap, int) or overlap < 0:
                raise ValueError("'overlap' parameter must be a non-negative integer for fixed_size strategy.")
            chunks = _chunk_text_fixed_size(text, chunk_size, overlap)
        
        elif strategy == 'semantic':
            if not nltk_installed:
                 raise ImportError("NLTK is required for semantic chunking strategy.")
            # Use default from helper if not provided in kwargs
            default_sentences = 3 
            sentences_per_chunk = kwargs.get('sentences_per_chunk', default_sentences)
            if not isinstance(sentences_per_chunk, int) or sentences_per_chunk <= 0:
                 raise ValueError(f"'sentences_per_chunk' must be a positive integer for semantic strategy (default: {default_sentences}).")
            chunks = _chunk_text_semantic(text, sentences_per_chunk)
                 
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

        # Create rows for the new DataFrame
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            all_chunks_data.append({
                'doc_id': doc_id,
                'chunk_id': chunk_id,
                'chunk_text': chunk_text,
                'metadata': metadata 
            })

    chunked_df = pd.DataFrame(all_chunks_data)
    return chunked_df
