# ParaLLM

ParaLLM is a command-line tool and Python package for efficiently querying language models. It supports batch processing with multiple prompts and models, and now includes structured JSON output via schemas.

## Features

- **Multi-Model Querying:** Query multiple LLMs simultaneously, comparing their outputs
- **CSV Input/Output:** Use CSV files for batch processing of prompts
- **Structured JSON Output:** Get responses formatted to JSON schemas or Pydantic models
- **Single-Query Mode:** Use `parallm single` for quick one-off queries with schema support
- **High Performance:** Leverages Bodo for parallel execution of queries
- **AWS Bedrock Support:** Query AWS Bedrock models with the same interface
- **Google Gemini Support:** Query Google Gemini models with native schema support

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

### Regular LLM Queries

```bash
# Process prompts.csv with two different models
parallm --prompts data/prompts.csv --models gpt-4o-mini claude-3-sonnet-20240229

# Using a JSON schema for structured output
parallm --prompts data/prompts.csv --models gpt-4o-mini --schema '{
  "type": "object",
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

### Single Query Mode

```bash
# Basic query
parallm single "What is the capital of France?"

# Specify a different model
parallm single "What is the capital of France?" --model claude-3-opus-20240229

# Repeat the same query multiple times
parallm single "What is the capital of France?" --repeat 100

# Get structured JSON response using inline schema
parallm single "Describe a nice dog" --schema '{
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "age": {"type": "integer"},
    "breed": {"type": "string"},
    "personality_traits": {"type": "array", "items": {"type": "string"}}
  },
  "required": ["name", "age", "breed", "personality_traits"]
}'
```

### AWS Bedrock Queries

```bash
# Single query with AWS Bedrock
parallm aws single "What is the capital of France?" --model anthropic.claude-3-sonnet-20240229

# Repeat a query multiple times with AWS Bedrock
parallm aws single "What is the capital of France?" --model anthropic.claude-3-sonnet-20240229 --repeat 100

# Batch process with AWS Bedrock models
parallm aws batch --prompts data/prompts.csv --models anthropic.claude-3-sonnet-20240229 amazon.titan-text-express-v1
```

### Google Gemini Queries

```bash
# Single query with Gemini
parallm gemini "What is the capital of France?" --model gemini-2.0-flash

# Repeat a query multiple times with Gemini
parallm gemini "What is the capital of France?" --model gemini-2.0-flash --repeat 100

# Batch process with Gemini
parallm gemini --prompts data/prompts.csv --models gemini-2.0-flash

# Using a Pydantic model for structured output
parallm gemini "List a few popular cookie recipes" --pydantic models.py:Recipe

# Using a JSON schema for structured output
parallm gemini "Describe a nice dog" --schema '{
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "age": {"type": "integer"},
    "breed": {"type": "string"},
    "personality_traits": {"type": "array", "items": {"type": "string"}}
  },
  "required": ["name", "age", "breed", "personality_traits"]
}'
```

### Batch Repeating Queries (CSV, Multiple Models)

#### Command-Line Usage

```bash
# Default (OpenAI/llm): Repeat each (prompt, model) pair 10 times
parallm --prompts data/prompts.csv --models gpt-4o-mini claude-3-sonnet-20240229 --repeat 10

# AWS Bedrock: Repeat each (prompt, model) pair 5 times
parallm aws batch --prompts data/prompts.csv --models anthropic.claude-3-sonnet-20240229 --repeat 5

# Gemini: Repeat each (prompt, model) pair 7 times
parallm gemini --prompts data/prompts.csv --models gemini-2.0-flash --repeat 7
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

### AWS Bedrock Queries

```python
from parallm import bedrock_query_model

# Simple text query with AWS Bedrock
response = bedrock_query_model("What is the capital of France?", model_id="anthropic.claude-3-sonnet-20240229")
print(response)

# Batch processing with AWS Bedrock
from parallm import bedrock_query_model_all
import pandas as pd

df = bedrock_query_model_all("data/prompts.csv", ["anthropic.claude-3-sonnet-20240229", "amazon.titan-text-express-v1"])
print(df)
```

### Google Gemini Queries

```python
from parallm import gemini_query
from pydantic import BaseModel

# Simple text query
response = gemini_query.query_model("What is the capital of France?")
print(response)

# Using a Pydantic model
class Recipe(BaseModel):
    recipe_name: str
    ingredients: list[str]

# Get parsed Pydantic model directly
recipes = gemini_query.query_model(
    prompt="List a few popular cookie recipes. Be sure to include the amounts of ingredients.",
    model_id="gemini-2.0-flash",
    schema=Recipe
)
print(recipes.recipe_name)
print(recipes.ingredients)

# Using JSON schema
json_schema = {
    "type": "object",
    "properties": {
        "recipe_name": {"type": "string"},
        "ingredients": {"type": "array", "items": {"type": "string"}}
    }
}

# Get JSON string
json_response = gemini_query.query_model(
    prompt="List a few popular cookie recipes. Be sure to include the amounts of ingredients.",
    model_id="gemini-2.0-flash",
    schema=json_schema
)

# Batch processing
df = gemini_query.query_model_all(
    file_path="data/prompts.csv",
    models=["gemini-2.0-flash"]
)
print(df)
```

### Structured JSON Output

```python
from parallm import query_model_json

# Using JSON schema
schema = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"}
    },
    "required": ["answer", "confidence"]
}

result = query_model_json("Describe a nice dog", schema=schema)
print(result)
# Output: {'name': 'Buddy', 'age': 3, 'is_friendly': True}
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
    "type": "object",
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

### Repeating a Query N Times (Easy Top-Level Import)

```python
from parallm import query_model_repeat, bedrock_query_model_repeat, gemini_query_model_repeat

# Default (OpenAI/llm)
df = query_model_repeat("What is the capital of France?", repeat=10)
print(df)

# AWS Bedrock
df = bedrock_query_model_repeat("What is the capital of France?", model_id="anthropic.claude-3-sonnet-20240229", repeat=5)
print(df)

# Gemini
df = gemini_query_model_repeat("What is the capital of France?", model_id="gemini-2.0-flash", repeat=7)
print(df)
```

### Batch Repeating Queries (CSV, Multiple Models)

#### Python API Usage

```python
from parallm import query_model_all_repeat, bedrock_query_model_all_repeat, gemini_query_model_all_repeat

# Default (OpenAI/llm)
df = query_model_all_repeat("data/prompts.csv", ["gpt-4o-mini", "claude-3-sonnet-20240229"], repeat=10)
print(df)

# AWS Bedrock
df = bedrock_query_model_all_repeat("data/prompts.csv", ["anthropic.claude-3-sonnet-20240229"], repeat=5)
print(df)

# Gemini
df = gemini_query_model_all_repeat("data/prompts.csv", ["gemini-2.0-flash"], repeat=7)
print(df)
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
- **boto3:** AWS SDK for Python (required for AWS Bedrock)
- **google-generativeai:** Google's Gemini API client (required for Gemini)

## Author

Rohit Krishnan