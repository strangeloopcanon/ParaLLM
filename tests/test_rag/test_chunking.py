# tests/test_rag/test_chunking.py

import pytest
import pandas as pd
from parallm.rag.chunking import chunk_dataframe

# Sample DataFrame for testing
@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    data = [
        {
            'doc_id': 'doc1',
            'text': 'This is the first sentence. This is the second sentence. This is the third.',
            'metadata': {'filename': 'doc1.txt'}
        },
        {
            'doc_id': 'doc2',
            'text': 'Short.',
            'metadata': {'filename': 'doc2.txt'}
        },
         {
            'doc_id': 'doc3',
            'text': '', # Empty text
            'metadata': {'filename': 'doc3.txt'}
        }
    ]
    return pd.DataFrame(data)

def test_chunk_fixed_size_basic(sample_dataframe):
    """Test fixed size chunking with reasonable overlap."""
    chunked_df = chunk_dataframe(sample_dataframe, strategy='fixed_size', chunk_size=30, overlap=10)
    
    assert isinstance(chunked_df, pd.DataFrame)
    assert all(col in chunked_df.columns for col in ['doc_id', 'chunk_id', 'chunk_text', 'metadata'])
    
    doc1_chunks = chunked_df[chunked_df['doc_id'] == 'doc1']
    assert len(doc1_chunks) > 2 # Expect multiple chunks
    assert doc1_chunks.iloc[0]['chunk_text'] == 'This is the first sentence. Th' # First 30 chars
    assert doc1_chunks.iloc[1]['chunk_text'] == 'ntence. This is the second sen' # Starts at position 20 (30-10 overlap)
    assert len(doc1_chunks.iloc[1]['chunk_text']) == 30 # Should be exactly 30 chars
    assert doc1_chunks.iloc[0]['chunk_id'] == 'doc1_chunk_0'
    assert doc1_chunks.iloc[1]['chunk_id'] == 'doc1_chunk_1'
    assert doc1_chunks.iloc[0]['metadata']['filename'] == 'doc1.txt'
    
    doc2_chunks = chunked_df[chunked_df['doc_id'] == 'doc2']
    assert len(doc2_chunks) == 1 # Short text should yield one chunk
    assert doc2_chunks.iloc[0]['chunk_text'] == 'Short.'
    assert doc2_chunks.iloc[0]['chunk_id'] == 'doc2_chunk_0'
    
    doc3_chunks = chunked_df[chunked_df['doc_id'] == 'doc3']
    assert len(doc3_chunks) == 0 # Empty text should yield zero chunks

def test_chunk_fixed_size_no_overlap(sample_dataframe):
    """Test fixed size chunking with zero overlap."""
    chunked_df = chunk_dataframe(sample_dataframe, strategy='fixed_size', chunk_size=20, overlap=0)
    
    doc1_chunks = chunked_df[chunked_df['doc_id'] == 'doc1']
    assert len(doc1_chunks) > 3
    assert doc1_chunks.iloc[0]['chunk_text'] == 'This is the first se' # First 20 chars
    assert doc1_chunks.iloc[1]['chunk_text'] == 'ntence. This is the ' # Next 20 chars, no overlap
    assert doc1_chunks.iloc[2]['chunk_text'] == 'second sentence. Thi'

def test_chunk_fixed_size_large_chunk(sample_dataframe):
    """Test when chunk size is larger than document."""
    chunked_df = chunk_dataframe(sample_dataframe, strategy='fixed_size', chunk_size=1000, overlap=100)
    
    doc1_chunks = chunked_df[chunked_df['doc_id'] == 'doc1']
    assert len(doc1_chunks) == 1
    assert doc1_chunks.iloc[0]['chunk_text'] == sample_dataframe.iloc[0]['text']
    
    doc2_chunks = chunked_df[chunked_df['doc_id'] == 'doc2']
    assert len(doc2_chunks) == 1
    assert doc2_chunks.iloc[0]['chunk_text'] == 'Short.'

def test_chunk_invalid_strategy(sample_dataframe):
    """Test using an unknown chunking strategy."""
    with pytest.raises(ValueError, match="Unknown chunking strategy: invalid_strategy"):
        chunk_dataframe(sample_dataframe, strategy='invalid_strategy')

def test_chunk_missing_params(sample_dataframe):
    """Test fixed_size strategy without chunk_size."""
    with pytest.raises(ValueError, match="'chunk_size' parameter must be a positive integer"):
        chunk_dataframe(sample_dataframe, strategy='fixed_size', overlap=10) # Missing chunk_size

def test_chunk_invalid_overlap(sample_dataframe):
    """Test fixed_size strategy with overlap >= chunk_size."""
    with pytest.raises(ValueError, match="Chunk size must be greater than overlap"):
        chunk_dataframe(sample_dataframe, strategy='fixed_size', chunk_size=10, overlap=10)
    with pytest.raises(ValueError, match="Chunk size must be greater than overlap"):
        chunk_dataframe(sample_dataframe, strategy='fixed_size', chunk_size=10, overlap=20)

def test_chunk_empty_dataframe():
     """Test chunking an empty DataFrame."""
     empty_df = pd.DataFrame(columns=['doc_id', 'text', 'metadata'])
     chunked_df = chunk_dataframe(empty_df, strategy='fixed_size', chunk_size=100, overlap=10)
     assert chunked_df.empty
     assert list(chunked_df.columns) == ['doc_id', 'chunk_id', 'chunk_text', 'metadata']

def test_chunk_missing_columns():
    """Test chunking DataFrame without required columns."""
    bad_df = pd.DataFrame([{'doc_id': 'd1'}])
    with pytest.raises(ValueError, match="Input DataFrame missing required columns"):
         chunk_dataframe(bad_df, strategy='fixed_size', chunk_size=100, overlap=10) 