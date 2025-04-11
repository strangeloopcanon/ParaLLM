# cli.py
import pandas as pd
import argparse
import json
import sys
import time
from parallm import model_query
from parallm import bedrock_query

def cli(mode=None):
    # If mode is not specified, try to detect it
    if mode is None:
        # Check if we have any non-flag arguments (potential prompt)
        args = [arg for arg in sys.argv[1:] if not arg.startswith('--')]
        if args and not any(arg.startswith('--') for arg in sys.argv[1:]):
            mode = "single"
        else:
            mode = "batch"

    if mode == "single":
        parser = argparse.ArgumentParser(
            description="Query a model with a single prompt."
        )
        parser.add_argument(
            "prompt", type=str,
            help="The prompt to send to the model."
        )
        parser.add_argument(
            "--model", type=str, default="gpt-4o-mini",
            help="The model to query (default: gpt-4o-mini)."
        )
        parser.add_argument(
            "--schema", type=str,
            help="JSON schema file path or JSON string for structured output."
        )
        parser.add_argument(
            "--pydantic", type=str,
            help="Python file:class specification for Pydantic model (e.g., 'models.py:Dog')."
        )
        args = parser.parse_args()
        handle_single_query(args, aws=(mode == "aws"))
    elif mode == "aws":
        parser = argparse.ArgumentParser(
            description="Query an AWS Bedrock model with a single prompt or in batch mode."
        )
        subparsers = parser.add_subparsers(dest="command", help="Command to run")
        
        # AWS Single mode
        single_parser = subparsers.add_parser("single", help="Send a single prompt to AWS Bedrock")
        single_parser.add_argument(
            "prompt", type=str,
            help="The prompt to send to the AWS Bedrock model."
        )
        single_parser.add_argument(
            "--model", type=str, default="anthropic.claude-3-sonnet-20240229",
            help="The AWS Bedrock model ID to query (default: anthropic.claude-3-sonnet-20240229)."
        )
        single_parser.add_argument(
            "--schema", type=str,
            help="JSON schema file path or JSON string for structured output."
        )
        single_parser.add_argument(
            "--pydantic", type=str,
            help="Python file:class specification for Pydantic model (e.g., 'models.py:Dog')."
        )
        
        # AWS Batch mode
        batch_parser = subparsers.add_parser("batch", help="Process batch prompts with AWS Bedrock")
        batch_parser.add_argument(
            "--prompts", type=str, required=True,
            help="Path to the prompts CSV file."
        )
        batch_parser.add_argument(
            "--models", type=str, nargs='+', required=True,
            help="List of AWS Bedrock model IDs to query (space separated)."
        )
        batch_parser.add_argument(
            "--schema", type=str,
            help="JSON schema file path or JSON string for structured output."
        )
        batch_parser.add_argument(
            "--pydantic", type=str,
            help="Python file:class specification for Pydantic model (e.g., 'models.py:Dog')."
        )
        
        args = parser.parse_args()
        
        if args.command == "single":
            handle_single_query(args, aws=True)
        elif args.command == "batch":
            handle_batch_query(args, aws=True)
        else:
            parser.print_help()
            sys.exit(1)
    else:  # batch mode
        parser = argparse.ArgumentParser(
            description="Query multiple models with prompts from a CSV file."
        )
        parser.add_argument(
            "--prompts", type=str, required=True,
            help="Path to the prompts CSV file."
        )
        parser.add_argument(
            "--models", type=str, nargs='+', required=True,
            help="List of model names to query (space separated)."
        )
        parser.add_argument(
            "--schema", type=str,
            help="JSON schema file path or JSON string for structured output."
        )
        parser.add_argument(
            "--pydantic", type=str,
            help="Python file:class specification for Pydantic model (e.g., 'models.py:Dog')."
        )
        args = parser.parse_args()
        handle_batch_query(args)

def load_schema(schema_arg, pydantic_arg):
    """Helper function to load schema from either JSON or Pydantic"""
    if schema_arg:
        try:
            return json.loads(schema_arg)
        except json.JSONDecodeError:
            try:
                with open(schema_arg, 'r') as f:
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                print(f"Error: Could not parse schema '{schema_arg}' as JSON or read from file.")
                sys.exit(1)
    elif pydantic_arg:
        try:
            file_path, class_name = pydantic_arg.split(':')
            if file_path.endswith('.py'):
                file_path = file_path[:-3]
            module_path = file_path.replace('/', '.')
            module = __import__(module_path, fromlist=[class_name])
            return getattr(module, class_name)
        except (ValueError, ImportError, AttributeError) as e:
            print(f"Error loading Pydantic model: {e}")
            print("Format should be 'file_path:ClassName' (e.g., 'models.py:Dog')")
            sys.exit(1)
    return None

def handle_single_query(args, aws=False):
    schema = load_schema(args.schema, args.pydantic)
    try:
        if aws:
            query_module = bedrock_query
        else:
            query_module = model_query
            
        if schema is None:
            result = query_module.query_model(args.prompt, args.model)
            print(result)
        else:
            result = query_module.query_model_json(args.prompt, args.model, schema)
            print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def handle_batch_query(args, aws=False):
    schema = load_schema(args.schema, args.pydantic)
    
    if aws:
        query_module = bedrock_query
    else:
        query_module = model_query
    
    if schema is not None:
        print("Schema provided. Running queries sequentially (non-parallel)...")
        t_start = time.time()
        try:
            prompts_df = pd.read_csv(args.prompts)
            prompts_df["prompt"] = prompts_df["prompt"].str.strip().str.lower()

            models = pd.Series(args.models).str.strip().str.lower().tolist()
            models_df = pd.DataFrame({"model": models})
            combined_df = prompts_df.merge(models_df, how='cross')

            combined_df["response"] = combined_df.apply(
                lambda row: query_module.query_model(row["prompt"], row["model"], schema),
                axis=1
            )
            result_df = combined_df
            print(f"Sequential processing time: {time.time() - t_start:.2f} seconds")

        except Exception as e:
            print(f"\nError during sequential processing: {e}")
            sys.exit(1)
    else:
        print("No schema provided. Running queries in parallel using Bodo...")
        try:
            result_df = query_module.query_model_all(args.prompts, args.models, schema)
        except Exception as e:
            print(f"\nError during parallel processing: {e}")
            sys.exit(1)

    print(f"\nReadable table for humans: \n{result_df}\n")
    print(f"\nReadable table for machines (CSV to stdout): \n")
    try:
        csv_output = result_df.to_csv(index=False)
        print(csv_output)
        result_df.to_csv("output.csv", index=False)
        print("\nOutput also saved to output.csv")
    except AttributeError:
        print("Error: Could not generate CSV output. Result was not a DataFrame.")
        print("Result dump:", result_df)

if __name__ == "__main__":
    cli()