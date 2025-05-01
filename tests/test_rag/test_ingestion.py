# tests/test_rag/test_ingestion.py
import pytest
import pandas as pd
from pathlib import Path

# Attempt to import optional dependencies
try:
    import pypdf
    pypdf_installed = True
except ImportError:
    pypdf_installed = False

try:
    import docx
    # Create a dummy docx file content (requires python-docx)
    from docx import Document
    docx_installed = True
except ImportError:
    docx_installed = False

from parallm.rag.ingestion import load_documents

# --- Fixtures --- 

@pytest.fixture
def sample_docs(tmp_path: Path) -> Path:
    """Create a directory with sample .txt, .pdf, .docx files."""
    docs_dir = tmp_path / "sample_docs"
    docs_dir.mkdir()
    
    # TXT file
    txt_file = docs_dir / "doc1.txt"
    txt_file.write_text("This is the first text document.")
    
    # Second TXT file in subdirectory
    sub_dir = docs_dir / "subdir"
    sub_dir.mkdir()
    sub_txt_file = sub_dir / "doc2.txt"
    sub_txt_file.write_text("Second document in subdirectory.")
    
    # PDF file (Requires pypdf and reportlab for creation)
    if pypdf_installed:
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            pdf_file = docs_dir / "doc3.pdf"
            c = canvas.Canvas(str(pdf_file), pagesize=letter)
            c.drawString(72, 720, "This is a test PDF document.")
            c.save()
        except ImportError:
             print("\nWarning: reportlab not installed. Cannot create test PDF.")
             # Create an empty file as placeholder if reportlab is missing
             pdf_file = docs_dir / "doc3.pdf"
             pdf_file.touch()
    
    # DOCX file (Requires python-docx)
    if docx_installed:
        docx_file = docs_dir / "doc4.docx"
        document = Document()
        document.add_paragraph("This is a test DOCX document.")
        document.save(str(docx_file))
        
    # Unsupported file
    unsupported_file = docs_dir / "image.jpg"
    unsupported_file.touch()
        
    return docs_dir

# --- Tests --- 

def test_load_documents_non_recursive(sample_docs: Path):
    """Test loading only from the top directory."""
    df = load_documents(str(sample_docs), recursive=False)
    
    assert isinstance(df, pd.DataFrame)
    # Should find .txt, potentially .pdf and .docx if libs are installed
    expected_min_docs = 1 # doc1.txt
    if pypdf_installed: expected_min_docs += 1
    if docx_installed: expected_min_docs += 1
    assert len(df) >= expected_min_docs 
    assert len(df) <= 3 # Shouldn't find subdir doc or jpg
    assert "doc1.txt" in df['doc_id'].tolist()
    assert "subdir/doc2.txt" not in df['doc_id'].tolist()
    assert all(col in df.columns for col in ['doc_id', 'text', 'metadata'])
    # Check text extraction for txt
    assert df[df['doc_id'] == 'doc1.txt']['text'].iloc[0] == "This is the first text document."

def test_load_documents_recursive(sample_docs: Path):
    """Test loading recursively."""
    df = load_documents(str(sample_docs), recursive=True)
    
    assert isinstance(df, pd.DataFrame)
    expected_min_docs = 2 # doc1.txt, subdir/doc2.txt
    if pypdf_installed: expected_min_docs += 1
    if docx_installed: expected_min_docs += 1
    assert len(df) >= expected_min_docs
    assert len(df) <= 4 # Shouldn't find jpg
    assert "doc1.txt" in df['doc_id'].tolist()
    assert "subdir/doc2.txt" in df['doc_id'].tolist()
    assert all(col in df.columns for col in ['doc_id', 'text', 'metadata'])
    assert df[df['doc_id'] == 'subdir/doc2.txt']['text'].iloc[0] == "Second document in subdirectory."

@pytest.mark.skipif(not pypdf_installed, reason="pypdf is not installed")
def test_load_documents_pdf(sample_docs: Path):
    """Test PDF loading specifically (if pypdf is installed)."""
    df = load_documents(str(sample_docs), recursive=True)
    pdf_row = df[df['doc_id'] == 'doc3.pdf']
    assert len(pdf_row) == 1
    # Text extraction might vary slightly based on library versions
    assert "This is a test PDF document." in pdf_row['text'].iloc[0]
    assert pdf_row['metadata'].iloc[0]['extension'] == ".pdf"

@pytest.mark.skipif(not docx_installed, reason="python-docx is not installed")
def test_load_documents_docx(sample_docs: Path):
    """Test DOCX loading specifically (if python-docx is installed)."""
    df = load_documents(str(sample_docs), recursive=True)
    docx_row = df[df['doc_id'] == 'doc4.docx']
    assert len(docx_row) == 1
    assert "This is a test DOCX document." in docx_row['text'].iloc[0]
    assert docx_row['metadata'].iloc[0]['extension'] == ".docx"

def test_load_documents_empty_dir(tmp_path: Path):
    """Test loading from an empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    df = load_documents(str(empty_dir))
    assert df.empty
    assert list(df.columns) == ['doc_id', 'text', 'metadata']

def test_load_documents_dir_not_found():
    """Test loading from a non-existent directory."""
    with pytest.raises(FileNotFoundError):
        load_documents("non_existent_dir_for_testing")

# Note: Need reportlab to *create* the test PDF in the fixture.
# If reportlab is not installed, the PDF test might not run correctly 
# depending on how the placeholder empty file is handled by pypdf reader. 