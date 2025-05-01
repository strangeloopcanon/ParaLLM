# tests/test_rag/test_indexing.py
import pytest
import pandas as pd
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

# Assume rank_bm25 installed for type hints, but mock its usage
from rank_bm25 import BM25Okapi

from parallm.rag.indexing import create_bm25_index, load_bm25_index, _tokenize_text

# --- Fixtures ---

@pytest.fixture
def chunked_dataframe_for_bm25() -> pd.DataFrame:
    data = [
        {
            'chunk_id': 'c1',
            'chunk_text': 'Simple test text for bm25.',
            'metadata': {'file': 'f1.txt'}
        },
        {
            'chunk_id': 'c2',
            'chunk_text': 'Another test with different words.',
            'metadata': {'file': 'f2.txt'}
        },
         {
            'chunk_id': 'c3',
            'chunk_text': '', # Empty chunk
            'metadata': {'file': 'f3.txt'}
        }
    ]
    return pd.DataFrame(data)

@pytest.fixture
def mock_bm25_okapi(mocker) -> tuple[MagicMock, MagicMock]: # Return tuple of mocks
    """Mocks the BM25Okapi class."""
    mock_instance = MagicMock(spec=BM25Okapi)
    mock_class = mocker.patch('parallm.rag.indexing.BM25Okapi', return_value=mock_instance)
    return mock_class, mock_instance

# --- Tests for _tokenize_text (simple example) ---

def test_tokenize_text_basic():
    assert _tokenize_text("Hello World! Test.") == ["hello", "world!", "test."]

def test_tokenize_text_empty():
    assert _tokenize_text("") == []

# --- Tests for create_bm25_index ---

def test_create_bm25_index_success(chunked_dataframe_for_bm25, mock_bm25_okapi, tmp_path):
    """Test successful creation and saving of BM25 index."""
    mock_class, mock_instance = mock_bm25_okapi
    output_path = tmp_path / "test_index.pkl"
    
    # Mock pickle.dump to check what's being saved
    # Also mock builtins.open for the writing part
    with patch('pickle.dump') as mock_pickle_dump, \
         patch('builtins.open', MagicMock()) as mock_open:
             
        create_bm25_index(chunked_dataframe_for_bm25, str(output_path))
        
        # Check BM25Okapi was called with tokenized corpus
        expected_corpus = [
            ['simple', 'test', 'text', 'for', 'bm25.'], 
            ['another', 'test', 'with', 'different', 'words.'],
            [] # Empty string results in empty list
        ]
        mock_class.assert_called_once_with(expected_corpus)
        
        # Check pickle.dump was called
        # mock_open.assert_called_once_with(str(output_path), "wb") # Verify file open args
        assert mock_pickle_dump.call_count == 1
        args, kwargs = mock_pickle_dump.call_args # Use call_args to get args and kwargs
        saved_data = args[0]
        # Check saved data structure
        assert 'bm25_index' in saved_data
        assert 'chunk_mapping' in saved_data
        assert saved_data['bm25_index'] is mock_instance # Check it saved the mock index
        # Check chunk mapping content
        assert len(saved_data['chunk_mapping']) == 3
        assert saved_data['chunk_mapping'][0]['chunk_id'] == 'c1'
        assert saved_data['chunk_mapping'][0]['chunk_text'] == 'Simple test text for bm25.'
        assert saved_data['chunk_mapping'][0]['metadata'] == {'file': 'f1.txt'}
        assert saved_data['chunk_mapping'][1]['chunk_id'] == 'c2'
        assert saved_data['chunk_mapping'][2]['chunk_id'] == 'c3'
        assert saved_data['chunk_mapping'][2]['chunk_text'] == ''

def test_create_bm25_index_empty_df(tmp_path):
    """Test create with empty dataframe."""
    output_path = tmp_path / "empty_index.pkl"
    empty_df = pd.DataFrame(columns=['chunk_id', 'chunk_text'])
    with patch('pickle.dump') as mock_pickle_dump:
        create_bm25_index(empty_df, str(output_path))
        mock_pickle_dump.assert_not_called()

def test_create_bm25_index_missing_cols():
    """Test create with missing columns."""
    bad_df = pd.DataFrame([{'chunk_id': 'c1'}])
    with pytest.raises(ValueError, match="missing required columns for BM25"):
        create_bm25_index(bad_df, "some_path.pkl")

# --- Tests for load_bm25_index ---

def test_load_bm25_index_success(mock_bm25_okapi, tmp_path):
    """Test successful loading of a pickled index file."""
    _, mock_instance = mock_bm25_okapi
    mock_instance.get_scores = MagicMock() 
    
    index_path = tmp_path / "load_test.pkl"
    valid_data = {
        'bm25_index': mock_instance,
        'chunk_mapping': {0: {'chunk_id': 'c1'}, 1: {'chunk_id': 'c2'}}
    }
    with open(index_path, "wb") as f: 
        pickle.dump(valid_data, f)
        
    bm25, mapping = load_bm25_index(str(index_path))
    assert bm25 is mock_instance
    assert mapping == valid_data['chunk_mapping']

def test_load_bm25_index_file_not_found():
    """Test loading a non-existent file."""
    bm25, mapping = load_bm25_index("non_existent_index.pkl")
    assert bm25 is None
    assert mapping is None

def test_load_bm25_index_bad_pickle(tmp_path):
    """Test loading a file that isn't valid pickle data."""
    bad_file = tmp_path / "bad.pkl"
    bad_file.write_text("this is not pickle data")
    bm25, mapping = load_bm25_index(str(bad_file))
    assert bm25 is None
    assert mapping is None

def test_load_bm25_index_missing_keys(tmp_path):
    """Test loading pickle data missing required keys."""
    index_path = tmp_path / "missing_keys.pkl"
    invalid_data = {'index': MagicMock(), 'map': {}}
    with open(index_path, "wb") as f:
        pickle.dump(invalid_data, f)
        
    bm25, mapping = load_bm25_index(str(index_path))
    assert bm25 is None
    assert mapping is None

def test_load_bm25_index_invalid_object(tmp_path):
    """Test loading pickle data where bm25_index is not a valid object."""
    index_path = tmp_path / "invalid_object.pkl"
    invalid_data = {'bm25_index': "not an object", 'chunk_mapping': {}}
    with open(index_path, "wb") as f:
        pickle.dump(invalid_data, f)
        
    bm25, mapping = load_bm25_index(str(index_path))
    assert bm25 is None
    assert mapping is None 