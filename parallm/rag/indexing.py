import pandas as pd
import pickle
from typing import Tuple, Dict, List, Any

# BM25 dependency
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("Warning: rank_bm25 not installed. BM25 indexing/retrieval will not be available.")
    BM25Okapi = None

# Tokenizer dependency (using simple split for now, consider NLTK/spaCy later)
# try:
#     import nltk
#     nltk.data.find('tokenizers/punkt')
# except (ImportError, LookupError):
#     print("Warning: NLTK or punkt tokenizer not found. Using simple whitespace split for BM25.")
#     print("Install NLTK ('pip install nltk') and download punkt ('python -m nltk.downloader punkt') for better tokenization.")
#     nltk = None

def _tokenize_text(text: str) -> List[str]:
    """Basic tokenizer for BM25 using simple whitespace split."""
    # if nltk:
    #     return nltk.word_tokenize(text.lower())
    # else:
    # Very basic fallback
    return text.lower().split()

def create_bm25_index(df: pd.DataFrame, index_path: str):
    """Creates and saves a BM25 index and chunk mapping from a DataFrame.

    Args:
        df: DataFrame containing 'chunk_id' and 'chunk_text' columns.
        index_path: Path to save the pickled index file.

    Raises:
        ValueError: If input DataFrame is invalid.
        ImportError: If rank_bm25 is not installed.
        Exception: For errors during index creation or saving.
    """
    if BM25Okapi is None:
        raise ImportError("rank_bm25 library is required for BM25 indexing.")
        
    if df.empty:
        print("Warning: Input DataFrame for BM25 indexing is empty. Skipping.")
        return

    required_cols = ['chunk_id', 'chunk_text']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Input DataFrame missing required columns for BM25: {required_cols}")

    print(f"Preparing data for BM25 index (Tokenizing {len(df)} chunks)...")
    tokenized_corpus = [ _tokenize_text(text) for text in df['chunk_text'].fillna('').astype(str) ]
    
    # Create a mapping from the internal index (0, 1, 2...) back to chunk data
    chunk_mapping = {
        i: {'chunk_id': row['chunk_id'], 'chunk_text': row['chunk_text'], 'metadata': row.get('metadata')} 
        for i, (_, row) in enumerate(df.iterrows())
    }

    print("Building BM25 index...")
    try:
        bm25 = BM25Okapi(tokenized_corpus)
    except Exception as e:
        print(f"Error creating BM25 index: {e}")
        raise
    print("BM25 index built successfully.")

    print(f"Saving BM25 index and chunk mapping to {index_path}...")
    try:
        with open(index_path, "wb") as f_out:
            pickle.dump({'bm25_index': bm25, 'chunk_mapping': chunk_mapping}, f_out)
        print("BM25 data saved successfully.")
    except Exception as e:
        print(f"Error saving BM25 data to {index_path}: {e}")
        raise

def load_bm25_index(index_path: str) -> Tuple[Any | None, Dict | None]:
    """Loads a pickled BM25 index and chunk mapping from a file.

    Args:
        index_path: Path to the pickled index file.

    Returns:
        A tuple containing (bm25_index_object, chunk_mapping_dict).
        Returns (None, None) if the file is not found or fails to load.
        
    Raises:
         ImportError: If rank_bm25 is not installed.
    """
    if BM25Okapi is None:
         raise ImportError("rank_bm25 library is required to load BM25 index.")
         
    print(f"Loading BM25 index and mapping from {index_path}...")
    try:
        with open(index_path, "rb") as f_in:
            data = pickle.load(f_in)
            bm25_index = data.get('bm25_index')
            chunk_mapping = data.get('chunk_mapping')
            if bm25_index and chunk_mapping is not None:
                 if not hasattr(bm25_index, 'get_scores'):
                      print(f"Warning: Loaded object from {index_path} does not appear to be a valid BM25 index.")
                      return None, None
                 print(f"BM25 data loaded successfully. Index covers {len(chunk_mapping)} chunks.")
                 return bm25_index, chunk_mapping
            else:
                 print(f"Error: File {index_path} did not contain expected 'bm25_index' or 'chunk_mapping' keys.")
                 return None, None
    except FileNotFoundError:
        print(f"Error: BM25 index file not found at {index_path}. Cannot perform keyword search.")
        return None, None
    except pickle.UnpicklingError as e:
        print(f"Error unpickling BM25 data from {index_path}: {e}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred loading BM25 data from {index_path}: {e}")
        raise
