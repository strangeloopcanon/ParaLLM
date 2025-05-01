import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

# PDF handling dependency
try:
    import pypdf
except ImportError:
    print("Warning: pypdf not installed. PDF ingestion will not be available.")
    pypdf = None 

# DOCX handling dependency
try:
    import docx
except ImportError:
    print("Warning: python-docx not installed. DOCX ingestion will not be available.")
    docx = None

# HTML handling dependency
try:
    from bs4 import BeautifulSoup
    bs4_installed = True
except ImportError:
    print("Warning: beautifulsoup4 not installed. HTML ingestion will not be available.")
    bs4_installed = False

def _extract_text_from_html(content: str) -> str:
    """Extracts text content from HTML string, removing scripts and styles."""
    soup = BeautifulSoup(content, 'lxml')
    
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()
        
    body = soup.body
    if body:
        text = body.get_text(separator='\n', strip=True)
    else:
        text = soup.get_text(separator='\n', strip=True)
        
    return text

def load_documents(source_path: str, recursive: bool = True) -> pd.DataFrame:
    """Loads text documents from a specified directory.

    Currently supports .txt, .pdf (if pypdf installed), 
    .docx (if python-docx installed), and .html/.htm (if beautifulsoup4 installed).

    Args:
        source_path: The path to the directory containing documents.
        recursive: Whether to search for files recursively in subdirectories.

    Returns:
        A Pandas DataFrame with columns: 'doc_id', 'text', 'metadata'.
        'doc_id' is the relative path to the file from the source_path.
        'metadata' is a dictionary containing {'filename', 'full_path', 'extension'}.

    Raises:
        FileNotFoundError: If the source_path does not exist or is not a directory.
    """
    source_dir = Path(source_path)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source path '{source_path}' is not a valid directory.")

    supported_extensions = [".txt"]
    if pypdf:
        supported_extensions.append(".pdf")
    if docx:
        supported_extensions.append(".docx")
    if bs4_installed:
        supported_extensions.extend([".html", ".htm"])
        
    print(f"Scanning for {', '.join(supported_extensions)} files in '{source_dir}'{' recursively' if recursive else ''}...")

    all_files: List[Path] = []
    for ext in supported_extensions:
        file_pattern = f"**/*{ext}" if recursive else f"*{ext}"
        all_files.extend(list(source_dir.glob(file_pattern)))
        
    if not all_files:
        print(f"Warning: No supported files ({supported_extensions}) found in '{source_dir}'.")
        return pd.DataFrame(columns=['doc_id', 'text', 'metadata'])

    print(f"Found {len(all_files)} supported files.")

    data: List[Dict[str, Any]] = []
    for file_path in all_files:
        text = None
        encoding_to_try = 'utf-8'
        try:
            file_ext = file_path.suffix.lower()
            
            if file_ext == ".txt":
                with open(file_path, 'r', encoding=encoding_to_try) as f:
                    text = f.read()
            elif file_ext == ".pdf" and pypdf:
                 reader = pypdf.PdfReader(file_path)
                 text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            elif file_ext == ".docx" and docx:
                 document = docx.Document(file_path)
                 text = "\n".join([paragraph.text for paragraph in document.paragraphs if paragraph.text])
            elif file_ext in [".html", ".htm"] and bs4_installed:
                with open(file_path, 'r', encoding=encoding_to_try) as f:
                     content = f.read()
                text = _extract_text_from_html(content)
            
            if text is not None and text.strip():
                relative_path = str(file_path.relative_to(source_dir))
                metadata = {
                    'filename': file_path.name,
                    'full_path': str(file_path.resolve()),
                    'extension': file_path.suffix
                }
                
                data.append({
                    'doc_id': relative_path,
                    'text': text,
                    'metadata': metadata
                })
            elif text is None:
                if file_path.suffix.lower() in supported_extensions:
                     print(f"Warning: Could not extract text from supported file '{file_path}'. Library might be missing or file corrupted.")
                
        except UnicodeDecodeError:
             print(f"Warning: Encoding error reading '{file_path}' with utf-8. Trying latin-1...")
             try:
                  encoding_to_try = 'latin-1'
                  if file_ext == ".txt" or (file_ext in [".html", ".htm"] and bs4_installed):
                      with open(file_path, 'r', encoding=encoding_to_try) as f:
                           raw_content = f.read()
                      if file_ext == ".txt":
                           text = raw_content
                      else:
                           text = _extract_text_from_html(raw_content)
                           
                      if text is not None and text.strip():
                          relative_path = str(file_path.relative_to(source_dir))
                          metadata = {'filename': file_path.name, 'full_path': str(file_path.resolve()), 'extension': file_path.suffix}
                          data.append({'doc_id': relative_path, 'text': text, 'metadata': metadata})
                  else:
                       print(f"Warning: Encoding error on non-text/html file '{file_path}', skipping.")
             except Exception as e_retry:
                  print(f"Error processing file '{file_path}' after encoding retry: {e_retry}")
                  
        except Exception as e:
            print(f"Error processing file '{file_path}': {e}")
            continue

    df = pd.DataFrame(data)
    return df
