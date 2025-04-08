# cli!
import pandas as pd
import argparse
import json
import sys
from parallm import model_query

def main():
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

    schema = None
    if args.schema:
        # Try to parse as direct JSON string
        try:
            schema = json.loads(args.schema)
        except json.JSONDecodeError:
            # Try to read from file
            try:
                with open(args.schema, 'r') as f:
                    schema = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                print(f"Error: Could not parse schema '{args.schema}' as JSON or read from file.")
                sys.exit(1)
    
    elif args.pydantic:
        try:
            file_path, class_name = args.pydantic.split(':')
            
            # Remove .py extension if present
            if file_path.endswith('.py'):
                file_path = file_path[:-3]
                
            # Import the module and get the class
            module_path = file_path.replace('/', '.')
            module = __import__(module_path, fromlist=[class_name])
            schema = getattr(module, class_name)
        except (ValueError, ImportError, AttributeError) as e:
            print(f"Error loading Pydantic model: {e}")
            print("Format should be 'file_path:ClassName' (e.g., 'models.py:Dog')")
            sys.exit(1)

    result_df = model_query.query_model_all(args.prompts, args.models, schema)
    print(f"\n Readable table for humans: {result_df}\n")
    # Output the result as CSV to stdout.
    print(f"\n Readable table for machines: \n")
    print(pd.DataFrame(result_df).to_csv(index=False))
    pd.DataFrame(result_df).to_csv("output.csv", index=False)
    
def query():
    """Entry point for single query CLI tool"""
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

    schema = None
    if args.schema:
        # Try to parse as direct JSON string
        try:
            schema = json.loads(args.schema)
        except json.JSONDecodeError:
            # Try to read from file
            try:
                with open(args.schema, 'r') as f:
                    schema = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                print(f"Error: Could not parse schema '{args.schema}' as JSON or read from file.")
                sys.exit(1)
    
    elif args.pydantic:
        try:
            file_path, class_name = args.pydantic.split(':')
            
            # Remove .py extension if present
            if file_path.endswith('.py'):
                file_path = file_path[:-3]
                
            # Import the module and get the class
            module_path = file_path.replace('/', '.')
            module = __import__(module_path, fromlist=[class_name])
            schema = getattr(module, class_name)
        except (ValueError, ImportError, AttributeError) as e:
            print(f"Error loading Pydantic model: {e}")
            print("Format should be 'file_path:ClassName' (e.g., 'models.py:Dog')")
            sys.exit(1)

    try:
        if schema is None:
            result = model_query.query_model(args.prompt, args.model)
            print(result)
        else:
            result = model_query.query_model_json(args.prompt, args.model, schema)
            print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
