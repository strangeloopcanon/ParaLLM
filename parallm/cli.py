# cli.py
import pandas as pd
import typer
import json
import sys
import time
import os
from typing import Any, List, Optional
from pathlib import Path

# Import core query modules
from parallm import model_query
from parallm import bedrock_query
from parallm import gemini_query

# Import RAG modules
from parallm.rag.pipeline import run_pipeline as run_rag_build_pipeline
from parallm.rag.pipeline import setup_retriever as setup_rag_retriever

# Main Typer application
cli = typer.Typer(help="Parallel LLM Query Tool with RAG support.")

# --- Helper Functions (mostly unchanged from original) ---

def load_schema(schema_arg: Optional[str], pydantic_arg: Optional[str]) -> Optional[Any]:
    """Helper function to load schema from either JSON or Pydantic"""
    if schema_arg:
        try:
            return json.loads(schema_arg)
        except json.JSONDecodeError:
            try:
                with open(schema_arg, 'r') as f:
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                typer.echo(f"Error: Could not parse schema '{schema_arg}' as JSON or read from file.", err=True)
                raise typer.Exit(code=1)
    elif pydantic_arg:
        try:
            # Attempt to handle relative/absolute paths better
            pydantic_path = Path(pydantic_arg)
            if not pydantic_path.is_file():
                 raise FileNotFoundError(f"Pydantic file not found: {pydantic_arg}")
                 
            module_path_str = str(pydantic_path.resolve())
            if not module_path_str.endswith('.py'):
                 raise ValueError("Pydantic file must end with .py")
                 
            # Split path and class name
            # Assuming format like path/to/model.py:ClassName
            if ':' not in module_path_str:
                raise ValueError("Pydantic argument format must be 'path/to/model.py:ClassName'")
                
            file_part, class_name = module_path_str.rsplit(':', 1)
            module_dir = str(Path(file_part).parent)
            module_name = Path(file_part).stem
            
            # Temporarily add directory to sys.path
            sys.path.insert(0, module_dir)
            try:
                module = __import__(module_name)
                model_class = getattr(module, class_name)
            finally:
                # Remove directory from sys.path
                if sys.path[0] == module_dir:
                    sys.path.pop(0)
                    
            return model_class
        except (ValueError, ImportError, AttributeError, FileNotFoundError) as e:
            typer.echo(f"Error loading Pydantic model '{pydantic_arg}': {e}", err=True)
            typer.echo("Format should be 'path/to/model.py:ClassName'", err=True)
            raise typer.Exit(code=1)
    return None

def _process_query_results(result_df: pd.DataFrame, schema: Optional[Any], repeat: int):
    """Helper to print results uniformly."""
    if result_df is None or result_df.empty:
        print("No results generated.")
        return
        
    print(f"\n--- Results ---")
    # Simple print for single prompt, single model, no repeat
    if len(result_df) == 1 and repeat == 1:
         response = result_df.iloc[0]['response']
         if schema:
              try:
                  # Try parsing if it's a string representation of JSON
                  if isinstance(response, str):
                      response_data = json.loads(response)
                  else:
                      response_data = response # Assume already parsed dict/object
                  print(json.dumps(response_data, indent=2))
              except Exception:
                  print(response) # Print raw if parsing fails
         else:
              print(response)
    else:
        # More detailed print for batch/repeat/multiple models
        print(f"\nReadable table:\n{result_df}\n")
        print(f"\nCSV format output:\n")
        try:
            csv_output = result_df.to_csv(index=False)
            print(csv_output)
            output_file = "output.csv"
            result_df.to_csv(output_file, index=False)
            print(f"\nOutput also saved to {output_file}")
        except AttributeError:
            typer.echo("Error: Could not generate CSV output. Result was not a DataFrame.", err=True)
            print("Result dump:", result_df)
            
# --- Core Query Command Logic (Adapted for Typer) ---

def run_core_query(
    prompts_or_path: str, 
    models: List[str], 
    schema_path: Optional[str],
    pydantic_path: Optional[str],
    repeat: int,
    query_module: Any,
    provider_name: str
):
    """Runs the core LLM query logic for default, aws, gemini modes."""
    schema = load_schema(schema_path, pydantic_path)
    is_file = os.path.isfile(prompts_or_path)
    
    try:
        result_df = None
        if is_file:
             typer.echo(f"Processing prompts from file: {prompts_or_path}...")
             # Use repeat logic for batch mode if applicable
             if repeat > 1:
                  result_df = query_module.query_model_all_repeat(prompts_or_path, models, repeat, schema)
             else:
                  result_df = query_module.query_model_all(prompts_or_path, models, schema)
        else:
             typer.echo(f"Processing single prompt...")
             # Handle single prompt with repeat
             if repeat > 1:
                  if len(models) > 1:
                       typer.echo("Warning: Repeat > 1 only supports a single model for single prompts. Using the first model.", err=True)
                  result_df = query_module.query_model_repeat(prompts_or_path, models[0], repeat, schema)
             else:
                 # Simulate batch processing for a single prompt to get DataFrame output
                 # Create a temporary DataFrame/CSV? Or adapt query_model_all? 
                 # Easiest might be to call query_model_all with a fake CSV path or directly create the df.
                 # Let's try creating the DataFrame directly for simplicity.
                 
                 # Prepare input DataFrame structure expected by query_model_all 
                 # (This assumes query_model_all handles cross product internally)
                 prompts_df = pd.DataFrame({"prompt": [prompts_or_path]})
                 models_df = pd.DataFrame({"model": models})
                 # We need to know how query_model_all forms the final df to replicate input
                 # Let's assume query_model_all can take a DataFrame input directly if refactored,
                 # or we pass the prompt string and models list.
                 
                 # Reverting to original logic: call query_model_all which handles single prompt case
                 result_df = query_module.query_model_all(prompts_or_path, models, schema)
                 
        _process_query_results(result_df, schema, repeat)
        
    except FileNotFoundError as e:
         typer.echo(f"Error: Prompts file not found - {e}", err=True)
         raise typer.Exit(code=1)
    except Exception as e:
         typer.echo(f"Error during query execution ({provider_name}): {e}", err=True)
         raise typer.Exit(code=1)

# --- Typer Commands ---

@cli.command(name="default", help="Run queries using default provider (e.g., OpenAI via llm). Add commands like aws or gemini for specific providers.")
def run_default(
    prompts: str = typer.Argument(..., help="Single prompt text or path to CSV file with prompts."),
    models: List[str] = typer.Option(["gpt-4o-mini"], "--models", "-m", help="List of model IDs to use."),
    schema: Optional[str] = typer.Option(None, help="JSON schema file or string for structured output."),
    pydantic: Optional[str] = typer.Option(None, help="Pydantic model file for structured output (e.g., 'path/to/model.py:ClassName')."),
    repeat: int = typer.Option(1, help="Number of times to repeat each query.")
):
    run_core_query(prompts, models, schema, pydantic, repeat, model_query, "default")

@cli.command()
def aws(
    prompts: str = typer.Argument(..., help="Single prompt text or path to CSV file with prompts."),
    models: List[str] = typer.Option(["anthropic.claude-3-sonnet-20240229-v1:0"], "--models", "-m", help="List of AWS Bedrock model IDs to use."),
    schema: Optional[str] = typer.Option(None, help="JSON schema file or string for structured output."),
    pydantic: Optional[str] = typer.Option(None, help="Pydantic model file for structured output (e.g., 'path/to/model.py:ClassName')."),
    repeat: int = typer.Option(1, help="Number of times to repeat each query.")
):
    """Run queries using AWS Bedrock provider."""
    run_core_query(prompts, models, schema, pydantic, repeat, bedrock_query, "AWS")

@cli.command()
def gemini(
    prompts: str = typer.Argument(..., help="Single prompt text or path to CSV file with prompts."),
    models: List[str] = typer.Option(["gemini-pro"], "--models", "-m", help="List of Google Gemini model IDs to use."),
    schema: Optional[str] = typer.Option(None, help="JSON schema file or string for structured output."),
    pydantic: Optional[str] = typer.Option(None, help="Pydantic model file for structured output (e.g., 'path/to/model.py:ClassName')."),
    repeat: int = typer.Option(1, help="Number of times to repeat each query.")
):
    """Run queries using Google Gemini provider."""
    run_core_query(prompts, models, schema, pydantic, repeat, gemini_query, "Gemini")

# --- RAG Subcommand --- 
rag_app = typer.Typer(help="Build and query RAG indexes.")
cli.add_typer(rag_app, name="rag")

@rag_app.command()
def build(
    config: Path = typer.Option(..., "--config", "-c", help="Path to the RAG YAML configuration file.", exists=True, file_okay=True, dir_okay=False, readable=True)
):
    """Builds the RAG index based on the provided configuration file."""
    typer.echo(f"Starting RAG index build using config: {config}")
    try:
        run_rag_build_pipeline(str(config))
        typer.echo("RAG index build finished successfully.")
    except FileNotFoundError as e:
        typer.echo(f"Error: Config file not found at {config}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error during RAG build: {e}", err=True)
        raise typer.Exit(code=1)

@rag_app.command()
def query(
    query: str = typer.Option(..., "--query", "-q", help="The query to ask the RAG system."),
    config: Path = typer.Option(..., "--config", "-c", help="Path to the RAG YAML configuration file.", exists=True, file_okay=True, dir_okay=False, readable=True),
    llm_model: str = typer.Option("gpt-4o-mini", "--llm-model", help="LLM model to use for final answer generation.")
):
    """Queries the RAG system using the specified configuration and query."""
    typer.echo(f"Querying RAG system using config: {config}")
    typer.echo(f"Query: '{query}'")
    
    try:
        # 1. Setup Retriever
        typer.echo("Setting up retriever...")
        retriever = setup_rag_retriever(str(config))
        typer.echo("Retriever setup complete.")
        
        # 2. Retrieve Context
        typer.echo("Retrieving context...")
        retrieved_docs = retriever(query)
        typer.echo(f"Retrieved {len(retrieved_docs)} context chunks.")
        
        if not retrieved_docs:
             typer.echo("Warning: No relevant context found.")
             context_text = "No relevant context was found."
        else:
            typer.echo("--- Top Retrieved Context Chunks (for info) ---")
            # Show top 3 snippets for context
            for i, doc in enumerate(retrieved_docs[:3]):
                 typer.echo(f"  Chunk {i+1} (Score: {doc.get('score', 'N/A'):.4f}): {doc.get('chunk_text', '')[:100]}...")
            typer.echo("--------------------------------------------")
            context_text = "\n\n---\n\n".join([doc['chunk_text'] for doc in retrieved_docs if doc.get('chunk_text')])

        # 3. Augment Prompt
        prompt_template = f"""Context information is below.
---------------------
{context_text}
---------------------

Given the context information and not prior knowledge, answer the query.
Query: {query}
Answer:"""
        typer.echo("Augmented prompt prepared.")

        # 4. Call LLM
        typer.echo(f"Sending augmented prompt to LLM ({llm_model})...")
        # Assuming model_query uses dotenv or environment vars for keys
        response = model_query.query_model(prompt_template, model_given=llm_model)
        
        typer.echo("\n--- RAG Response ---")
        typer.echo(response)
        typer.echo("--------------------")
        
    except FileNotFoundError as e:
        typer.echo(f"Error: Config file not found at {config}", err=True)
        raise typer.Exit(code=1)
    except ImportError as e:
         typer.echo(f"\nError: Missing dependency for model querying: {e}", err=True)
         typer.echo("Please install required LLM client libraries (e.g., 'pip install llm-gpt')", err=True)
         raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error during RAG query execution: {e}", err=True)
        raise typer.Exit(code=1)

# Entry point for the CLI
if __name__ == "__main__":
    cli()