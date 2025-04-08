# ParaLLM

ParaLLM is a command-line tool and Python package for efficiently querying language models. It supports batch processing with multiple prompts and models, and now includes structured JSON output via schemas.

## Features

- **Multi-Model Querying:** Query multiple LLMs simultaneously, comparing their outputs
- **CSV Input/Output:** Use CSV files for batch processing of prompts
- **Structured JSON Output:** Get responses formatted to JSON schemas or Pydantic models
- **Single-Query Mode:** Use `llm-query` for quick one-off queries with schema support
- **High Performance:** Leverages Bodo for parallel execution of queries

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

You'll need to install Simon Willison's `llm` package and set up your API keys.

## Command-Line Usage

### Batch Processing Multiple Prompts

```bash
# Process prompts.csv with two different models
parallm --prompts data/prompts.csv --models gpt-4o-mini claude-3-sonnet-20240229

# Using a JSON schema for structured output
parallm --prompts data/prompts.csv --models gpt-4o-mini --schema '{
  "properties": {
    "answer": {"type": "string"},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
  },
  "required": ["answer", "confidence"]
}'

# Using a schema from file
parallm --prompts data/prompts.csv --models gpt-4o-mini --schema schema.json

# Using a Pydantic model
parallm --prompts data/prompts.csv --models gpt-4o-mini --pydantic models.py:ResponseModel
```

### Single Query Tool

```bash
# Basic query
llm-query "What is the capital of France?"

# Specify a different model
llm-query "What is the capital of France?" --model claude-3-opus-20240229

# Get structured JSON response using inline schema
llm-query "Describe a nice dog" --schema '{
  "properties": {
    "name": {"type": "string"},
    "age": {"type": "integer"},
    "breed": {"type": "string"},
    "personality_traits": {"type": "array", "items": {"type": "string"}}
  },
  "required": ["name", "age", "breed", "personality_traits"]
}'

# Using schema from a file
llm-query "List the top 5 programming languages" --schema schemas/languages.json

# Using a Pydantic model
llm-query "Describe a car" --pydantic models.py:Car
```

## Python API Usage

### Basic Queries

```python
from parallm import query_model

# Simple text query
response = query_model("What is the capital of France?")
print(response)  # Paris

# Different model
response = query_model("What is the capital of France?", model_given="claude-3-sonnet-20240229")
print(response)
```

### Structured JSON Output

```python
from parallm import query_model_json

# Using JSON schema
schema = {
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "is_friendly": {"type": "boolean"}
    },
    "required": ["name", "age", "is_friendly"]
}

result = query_model_json("Describe a nice dog", schema=schema)
print(result)
# Output: {'name': 'Buddy', 'age': 3, 'is_friendly': True}

# Access fields directly
print(f"The dog's name is {result['name']} and it is {result['age']} years old")
```

### Using Pydantic Models

```python
from pydantic import BaseModel
from typing import List
from parallm import query_model_json

class Dog(BaseModel):
    name: str
    age: int
    breed: str
    personality_traits: List[str]

# Get response structured according to the Pydantic model
dog = query_model_json("Describe a nice dog", schema=Dog)

print(f"Name: {dog['name']}")
print(f"Age: {dog['age']}")
print(f"Breed: {dog['breed']}")
print("Personality traits:")
for trait in dog['personality_traits']:
    print(f"- {trait}")
```

### Batch Processing

```python
from parallm import query_model_all
import pandas as pd

# Process multiple prompts with multiple models
df = query_model_all("data/prompts.csv", ["gpt-4o-mini", "claude-3-sonnet-20240229"])
print(df)

# With JSON schema
schema = {
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"}
    },
    "required": ["answer", "confidence"]
}

df = query_model_all("data/prompts.csv", ["gpt-4o-mini"], schema=schema)

# Process the results
for _, row in df.iterrows():
    result = json.loads(row['response'])
    print(f"Prompt: {row['prompt']}")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']}")
    print("---")

# Save results
df.to_csv("structured_results.csv", index=False)
```

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

## Author

Rohit Krishnan