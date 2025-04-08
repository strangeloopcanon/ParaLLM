# cli.py
import pandas as pd
import argparse
import json
import sys
import time  # Import time for timing the sequential version
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
    # --- Start: Schema Loading Logic (same as before) ---
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
    # --- End: Schema Loading Logic ---

    # --- Start: Conditional Execution Logic ---
    if schema is not None:
        # --- Non-parallel execution path when schema is provided ---
        print("Schema provided. Running queries sequentially (non-parallel)...")
        t_start = time.time()
        try:
            # Replicate the core logic of query_model_all without Bodo
            prompts_df = pd.read_csv(args.prompts)
            prompts_df["prompt"] = prompts_df["prompt"].str.strip().str.lower()

            models = pd.Series(args.models).str.strip().str.lower().tolist()
            models_df = pd.DataFrame({"model": models})
            combined_df = prompts_df.merge(models_df, how='cross')

            # Use standard pandas apply (which runs sequentially)
            # Call the basic query_model function directly
            combined_df["response"] = combined_df.apply(
                lambda row: model_query.query_model(row["prompt"], row["model"], schema),
                axis=1
            )
            result_df = combined_df
            print(f"Sequential processing time: {time.time() - t_start:.2f} seconds")

        except Exception as e:
            # Catch errors during the sequential processing
            print(f"\nError during sequential processing: {e}")
            # Optionally add traceback here for more detail
            # import traceback
            # traceback.print_exc()
            sys.exit(1)

    else:
        # --- Parallel execution path when no schema is provided ---
        print("No schema provided. Running queries in parallel using Bodo...")
        try:
            # Call the original Bodo-jitted function
            # It will print its own timing message
            result_df = model_query.query_model_all(args.prompts, args.models, schema) # schema is None here
        except Exception as e:
            # Catch errors during the parallel processing (like the MPI error)
            print(f"\nError during parallel processing: {e}")
            # Optionally add traceback here
            # import traceback
            # traceback.print_exc()
            sys.exit(1)
    # --- End: Conditional Execution Logic ---


    # --- Output Section (runs after either path) ---
    print(f"\n Readable table for humans: \n{result_df}\n")
    print(f"\n Readable table for machines (CSV to stdout): \n")
    # Use try-except for safety in case result_df isn't a DataFrame (though it should be)
    try:
        csv_output = result_df.to_csv(index=False)
        print(csv_output)
        # Save to file
        result_df.to_csv("output.csv", index=False)
        print("\nOutput also saved to output.csv")
    except AttributeError:
         print("Error: Could not generate CSV output. Result was not a DataFrame.")
         print("Result dump:", result_df) # Print whatever result_df is


# Keep the query() function as is
def query():
    # ... (existing query function code remains unchanged) ...
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
            # Call the basic query function
            result = model_query.query_model(args.prompt, args.model)
            print(result)
        else:
            # Call the JSON version when schema is present for the single query tool
            result = model_query.query_model_json(args.prompt, args.model, schema)
            print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # This part remains the same, determining whether to run main() or query()
    if len(sys.argv) > 1 and sys.argv[1] == "query":
        sys.argv.pop(1)
        query()
    else:
        main()