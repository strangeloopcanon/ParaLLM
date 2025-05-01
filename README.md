# ParaLLM

ParaLLM is a command-line tool and Python package for efficiently querying language models. It supports batch processing with multiple prompts and models, and includes structured JSON output via schemas.

## Features

- **Multi-Model Querying:** Query multiple LLMs simultaneously, comparing their outputs
- **CSV Input/Output:** Use CSV files for batch processing of prompts
- **Structured JSON Output:** Get responses formatted to JSON schemas or Pydantic models
- **High Performance:** Leverages Bodo for parallel execution of queries
- **Multiple Providers:** Support for OpenAI, AWS Bedrock, and Google Gemini

## Installation

```bash
pip install parallm
```

Or install from source:

```bash
git clone https://github.com/strangeloopcanon/parallm.git
cd parallm
pip install -e .
```

You'll need to install Simon Willison's `llm` package and set up your API keys. For AWS Bedrock, ensure you have AWS credentials configured. For Gemini, set the `GEMINI_API_KEY` environment variable.

## Command-Line Usage

### Batch Processing (CSV Files)

Process multiple prompts from a CSV file with one or more models:

```bash
# Default mode (OpenAI/llm)
parallm --prompts data/prompts.csv --models gpt-4 claude-3-sonnet-20240229

# AWS Bedrock mode
parallm aws --prompts data/prompts.csv --models anthropic.claude-3-sonnet-20240229 amazon.titan-text-express-v1

# Gemini mode
parallm gemini --prompts data/prompts.csv --models gemini-2.0-flash
```

### Single Prompt Processing

Process a single prompt with optional repeat functionality:

```bash
# Default mode (OpenAI/llm)
parallm --prompts "What is the capital of France?" --models gpt-4 --repeat 5

# AWS Bedrock mode
parallm aws --prompts "What is the capital of France?" --models anthropic.claude-3-sonnet-20240229 --repeat 5

# Gemini mode
parallm gemini --prompts "What is the capital of France?" --models gemini-2.0-flash --repeat 5
```

### Structured Output

Get responses formatted according to a JSON schema or Pydantic model:

```bash
# Using a JSON schema
parallm --prompts data/prompts.csv --models gpt-4 --schema '{
  "type": "object",
  "properties": {
    "answer": {"type": "string"},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
  },
  "required": ["answer", "confidence"]
}'

# Using a schema from file
parallm --prompts data/prompts.csv --models gpt-4 --schema schema.json

# Using a Pydantic model
parallm --prompts data/prompts.csv --models gpt-4 --pydantic models.py:ResponseModel
```

## Python API Usage

### Batch Processing

```python
from parallm import query_model_all, bedrock_query_model_all, gemini_query_model_all

# Default mode (OpenAI/llm)
df = query_model_all("data/prompts.csv", ["gpt-4", "claude-3-sonnet-20240229"])
print(df)

# AWS Bedrock
df = bedrock_query_model_all("data/prompts.csv", ["anthropic.claude-3-sonnet-20240229"])
print(df)

# Gemini
df = gemini_query_model_all("data/prompts.csv", ["gemini-2.0-flash"])
print(df)
```

### Single Prompt Processing

```python
from parallm import query_model_repeat, bedrock_query_model_repeat, gemini_query_model_repeat

# Default mode (OpenAI/llm)
df = query_model_repeat("What is the capital of France?", "gpt-4", repeat=5)
print(df)

# AWS Bedrock
df = bedrock_query_model_repeat("What is the capital of France?", "anthropic.claude-3-sonnet-20240229", repeat=5)
print(df)

# Gemini
df = gemini_query_model_repeat("What is the capital of France?", "gemini-2.0-flash", repeat=5)
print(df)
```

### Structured Output

```python
from parallm import query_model_json
from pydantic import BaseModel

# Using a Pydantic model
class Response(BaseModel):
    answer: str
    confidence: float

result = query_model_json("What is the capital of France?", "gpt-4", schema=Response)
print(result)
```

## Retrieval-Augmented Generation (RAG)

ParaLLM now includes a modular RAG pipeline to allow querying language models with context retrieved from your own documents.

### Overview

The RAG system processes your documents through a configurable pipeline:

1.  **Ingestion:** Loads documents from a specified directory. Supports `.txt`, `.pdf`, `.docx`, and `.html`/`.htm` files.
2.  **Chunking:** Splits documents into smaller chunks using different strategies:
    *   `fixed_size`: Overlapping chunks of a defined character size.
    *   `semantic`: Groups sentences together (using NLTK).
3.  **Embedding:** Generates vector embeddings for each chunk using a specified Sentence Transformer model (e.g., `all-MiniLM-L6-v2`).
4.  **Indexing:** Stores the chunks, embeddings, and metadata in:
    *   A vector store (currently ChromaDB) for semantic search.
    *   A keyword index (using BM25) for lexical search.

When querying, the system retrieves relevant chunks using vector search, keyword search, or a hybrid combination, augments the prompt with this context, and then sends it to the specified language model.

### Configuration (`rag_config.yaml`)

The entire RAG pipeline is configured using a YAML file (e.g., `rag_config.yaml`). This file defines the sequence of steps, parameters for each step (like source paths, chunking strategy, embedding model, index paths), and the retrieval strategy.

See `examples/rag_config.yaml` for a detailed example.

### RAG CLI Usage

Use the `rag` subcommand for building indexes and querying.

**1. Build the RAG Index:**

This command runs the ingestion, chunking, embedding, and indexing pipeline defined in your config file.

```bash
python -m parallm rag build --config path/to/your_rag_config.yaml
```

*   Replace `path/to/your_rag_config.yaml` with the actual path to your configuration file.
*   This needs to be run once initially and then again whenever your source documents or pipeline configuration change.
*   Indexes (ChromaDB, BM25 pickle file) will be created/updated based on paths specified in the config.

**2. Query the RAG System:**

This command uses a previously built index to retrieve context, augment a prompt, and query an LLM.

```bash
python -m parallm rag query --config path/to/your_rag_config.yaml --query "Your question here?" --llm-model gpt-4o-mini
```

*   `--config`: Specifies the RAG configuration file (used to load the retriever and embedding models).
*   `--query` / `-q`: The question you want to ask.
*   `--llm-model`: (Optional) The language model to use for generating the final answer (defaults to the model specified in the script, e.g., `gpt-4o-mini`).

### RAG Dependencies

Using the RAG features requires additional dependencies:

```
PyYAML          # For parsing rag_config.yaml
sentence-transformers # For embedding generation
chromadb        # Vector store
rank_bm25       # Keyword indexing
pypdf           # PDF ingestion
python-docx     # DOCX ingestion
beautifulsoup4  # HTML ingestion
lxml            # HTML parsing backend for beautifulsoup4
nltk            # Semantic chunking (sentence tokenization)
reportlab       # Required by test suite to generate test PDFs
```

Ensure NLTK's `punkt` tokenizer data is downloaded:
`python -m nltk.downloader punkt`

## CSV Format

Your `prompts.csv` file should have a header row with "prompt" as the column name:

```csv
prompt
What is machine learning?
Explain quantum computing
How does blockchain work?
```

## Dependencies

- **pandas:** Data processing and CSV handling
- **bodo:** Parallel execution for performance
- **llm:** Simon Willison's LLM interface library
- **python-dotenv:** Environment variable management
- **pydantic:** Data validation for structured output
- **boto3:** AWS SDK for Python (required for AWS Bedrock)
- **google-generativeai:** Google's Gemini API client (required for Gemini)

## Author

Rohit Krishnan