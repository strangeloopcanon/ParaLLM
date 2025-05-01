# tests/test_rag/test_embedding.py
import pytest
import pandas as pd
import numpy as np

# Assume SentenceTransformer is installed for type hinting, but we'll mock it
from sentence_transformers import SentenceTransformer 
from unittest.mock import MagicMock # Use unittest.mock via pytest-mock

from parallm.rag.embedding import generate_embeddings

# --- Fixtures --- 

@pytest.fixture
def chunked_dataframe() -> pd.DataFrame:
    data = [
        {
            'chunk_id': 'doc1_chunk_0',
            'chunk_text': 'This is the first chunk.',
            'metadata': {'filename': 'doc1.txt'}
        },
        {
            'chunk_id': 'doc1_chunk_1',
            'chunk_text': 'This is the second chunk.',
            'metadata': {'filename': 'doc1.txt'}
        },
         {
            'chunk_id': 'doc2_chunk_0',
            'chunk_text': 'Another document chunk.',
            'metadata': {'filename': 'doc2.txt'}
        }
    ]
    return pd.DataFrame(data)

@pytest.fixture
def mock_sentence_transformer(mocker) -> MagicMock:
    """Mocks the SentenceTransformer class and its encode method."""
    # Mock the class constructor to return a mock object
    mock_model_instance = MagicMock(spec=SentenceTransformer)
    
    # Define what the mock encode method should return 
    # Let's return dummy embeddings of a fixed dimension (e.g., 3)
    dummy_embeddings = [
        np.array([0.1, 0.2, 0.3]), 
        np.array([0.4, 0.5, 0.6]), 
        np.array([0.7, 0.8, 0.9])
    ] 
    # Make encode return the right number of embeddings based on input length
    def mock_encode_func(texts, show_progress_bar=False):
         return np.array(dummy_embeddings[:len(texts)])
         
    mock_model_instance.encode.side_effect = mock_encode_func
    
    # Patch the class in the module where it's imported/used
    mock_class = mocker.patch('parallm.rag.embedding.SentenceTransformer', return_value=mock_model_instance)
    
    # Return the mock *instance* so we can assert calls on it if needed
    # Also return the mock class if needed to assert constructor calls
    return mock_class, mock_model_instance 

# --- Tests --- 

def test_generate_embeddings_success(chunked_dataframe, mock_sentence_transformer):
    """Test successful embedding generation using mocks."""
    mock_st_class, mock_st_instance = mock_sentence_transformer
    model_name = "mock-model"
    
    result_df = generate_embeddings(chunked_dataframe.copy(), embedding_model_name=model_name)
    
    # Check DataFrame structure
    assert 'embedding' in result_df.columns
    assert len(result_df) == len(chunked_dataframe)
    assert result_df.iloc[0]['chunk_id'] == 'doc1_chunk_0' # Check original data preserved
    
    # Check that the mock model was loaded
    mock_st_class.assert_called_once_with(model_name)
    
    # Check that encode was called with the correct texts
    expected_texts = chunked_dataframe['chunk_text'].tolist()
    mock_st_instance.encode.assert_called_once_with(expected_texts, show_progress_bar=True)
    
    # Check the embeddings added (should match our dummy embeddings)
    assert len(result_df['embedding'].iloc[0]) == 3 # Check dimension
    np.testing.assert_array_almost_equal(result_df['embedding'].iloc[0], np.array([0.1, 0.2, 0.3]))
    np.testing.assert_array_almost_equal(result_df['embedding'].iloc[1], np.array([0.4, 0.5, 0.6]))
    np.testing.assert_array_almost_equal(result_df['embedding'].iloc[2], np.array([0.7, 0.8, 0.9]))

def test_generate_embeddings_empty_df():
    """Test generating embeddings with an empty DataFrame."""
    empty_df = pd.DataFrame(columns=['chunk_id', 'chunk_text', 'metadata'])
    result_df = generate_embeddings(empty_df, embedding_model_name="any-model")
    
    assert 'embedding' in result_df.columns
    assert result_df.empty

def test_generate_embeddings_missing_column():
    """Test generating embeddings when 'chunk_text' is missing."""
    bad_df = pd.DataFrame([{'chunk_id': 'c1'}])
    with pytest.raises(ValueError, match="missing required column: 'chunk_text'"):
        generate_embeddings(bad_df, embedding_model_name="any-model")

def test_generate_embeddings_model_load_error(chunked_dataframe, mocker):
    """Test handling of errors during model loading."""
    # Mock SentenceTransformer constructor to raise an exception
    mocker.patch('parallm.rag.embedding.SentenceTransformer', side_effect=RuntimeError("Model load failed"))
    
    with pytest.raises(RuntimeError, match="Model load failed"):
        generate_embeddings(chunked_dataframe, embedding_model_name="bad-model")

def test_generate_embeddings_encode_error(chunked_dataframe, mock_sentence_transformer):
    """Test handling of errors during the encode call."""
    mock_st_class, mock_st_instance = mock_sentence_transformer
    # Make the mock encode method raise an exception
    mock_st_instance.encode.side_effect = ValueError("Encode failed")
    
    with pytest.raises(ValueError, match="Encode failed"):
        generate_embeddings(chunked_dataframe.copy(), embedding_model_name="mock-model") 