# ParaLLM
ParaLLM is a command-line tool that queries multiple language models using prompts read from a CSV file. It efficiently creates a Cartesian product of your prompts and model names, sends them to the models, and outputs the responses as a CSV file.

## Features

- **Multi-Model Querying:** Easily query several models in one go.
- **CSV Input/Output:** Use CSV files to manage your prompts and capture responses.
- **Simple CLI Interface:** Run your queries directly from the terminal.

## Installation

```bash
pip install parallm
```

Or clone the repository and install the package locally:

```bash
git clone https://github.com/strangeloopcanon/parallm.git
cd parallm
pip install e .
```
You would also need to install SimonW's llm package, and set up the keys.

## Usage
Once installed, you can use the `parallm` command from your terminal. For example:

```bash
parallm --prompts path/to/prompts.csv --models gpt-4o gemini-2.5-pro-exp-03-25 claude-3.7-sonnet
```
**Python API**: After you install, you can, for instance, run this in a python script.
```bash
from parallm import query_model_all

df = query_model_all("data/prompts.csv",
["gpt-4", "gemini-2.0-flash"])
print(df)
```

This command:
- Reads the prompts from the specified CSV file (the file should have a header named `prompt`).
- Creates a query for each combination of prompt and provided model names.
- Outputs a CSV with columns for `prompt`, `model`, and `response`.
- It will save the output into 'output.csv'

## CSV Format
Your `prompts.csv` file should have a header as "prompt". Example:

```csv
prompt
Hello, world!
Tell me a joke.
```

## Dependencies
ParaLLM relies on the following Python packages:
- **pandas**
- **python-dotenv**
- **bodo**
- **llm**

These dependencies are automatically installed when you install ParaLLM.

## Author
Rohit Krishnan