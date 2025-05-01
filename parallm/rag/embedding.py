import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import numpy as np

def generate_embeddings(df: pd.DataFrame, embedding_model_name: str) -> pd.DataFrame:
    """Generates embeddings for the 'chunk_text' column of a DataFrame.

    Args:
        df: DataFrame containing 'chunk_id', 'chunk_text' columns.
        embedding_model_name: The name of the Sentence Transformer model to use 
                                (e.g., 'all-MiniLM-L6-v2').

    Returns:
        The original DataFrame with an added 'embedding' column containing
        numpy arrays.

    Raises:
        ValueError: If the input DataFrame is empty or missing 'chunk_text'.
        ImportError: If sentence-transformers is not installed.
        Exception: If the specified model cannot be loaded.
    """
    if df.empty:
        print("Warning: Input DataFrame for embedding is empty.")
        df['embedding'] = pd.Series(dtype='object') 
        return df

    if 'chunk_text' not in df.columns:
        raise ValueError("Input DataFrame missing required column: 'chunk_text'")

    try:
        print(f"Loading sentence transformer model: {embedding_model_name}")
        model = SentenceTransformer(embedding_model_name)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading Sentence Transformer model '{embedding_model_name}': {e}")
        raise

    texts_to_embed: List[str] = df['chunk_text'].tolist()
    
    print(f"Generating embeddings for {len(texts_to_embed)} text chunks...")
    try:
        embeddings = model.encode(texts_to_embed, show_progress_bar=True)
        print("Embeddings generated successfully.")
        
        # Store embeddings as a list of arrays to ensure DataFrame compatibility
        df['embedding'] = list(embeddings) 
        
    except Exception as e:
        print(f"Error during embedding generation: {e}")
        raise
        
    return df
