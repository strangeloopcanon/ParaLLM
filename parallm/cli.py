# cli.py
import pandas as pd
import argparse
import json
import sys
import time
import os
from parallm import model_query
from parallm import bedrock_query
from parallm import gemini_query

def cli(mode=None):
    parser = argparse.ArgumentParser(description='Parallel LLM Query Tool')
    
    if mode == "aws":
        # AWS Bedrock mode
        parser.add_argument('--prompts', nargs='+', required=True, help='List of prompts to query')
        parser.add_argument('--models', nargs='+', default=['anthropic.claude-3-sonnet-20240229-v1:0'],
                          help='List of model IDs to use')
        parser.add_argument('--schema', help='JSON schema file for structured output')
        parser.add_argument('--pydantic', help='Pydantic model file for structured output')
        parser.add_argument('--repeat', type=int, default=1, help='Number of times to repeat each query')
        
    elif mode == "gemini":
        # Gemini mode
        parser.add_argument('--prompts', nargs='+', required=True, help='List of prompts to query')
        parser.add_argument('--models', nargs='+', default=['gemini-pro'],
                          help='List of model IDs to use')
        parser.add_argument('--schema', help='JSON schema file for structured output')
        parser.add_argument('--pydantic', help='Pydantic model file for structured output')
        parser.add_argument('--repeat', type=int, default=1, help='Number of times to repeat each query')
        
    else:
        # Default mode (single/batch)
        parser.add_argument('--prompts', nargs='+', required=True, help='List of prompts to query')
        parser.add_argument('--models', nargs='+', default=['gpt-4'],
                          help='List of model IDs to use')
        parser.add_argument('--schema', help='JSON schema file for structured output')
        parser.add_argument('--pydantic', help='Pydantic model file for structured output')
        parser.add_argument('--repeat', type=int, default=1, help='Number of times to repeat each query')
    
    # Remove the mode argument from sys.argv if it exists
    if mode and len(sys.argv) > 1 and sys.argv[1] == mode:
        sys.argv.pop(1)
    
    args = parser.parse_args()
    
    if mode == "aws":
        from parallm.aws_query import query_aws_bedrock
        query_aws_bedrock(args.prompts, args.models, args.schema, args.pydantic, args.repeat)
    elif mode == "gemini":
        from parallm.gemini_query import query_model_repeat, query_model_all_repeat
        # For single prompt, use query_model_repeat
        if len(args.prompts) == 1:
            result = query_model_repeat(args.prompts[0], args.models[0], args.repeat, args.schema)
        else:
            # For multiple prompts, create a temporary CSV file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                f.write("prompt\n")
                for prompt in args.prompts:
                    f.write(f"{prompt}\n")
                temp_file = f.name
            try:
                result = query_model_all_repeat(temp_file, args.models, args.repeat, args.schema)
            finally:
                os.unlink(temp_file)
        print(result)
    else:
        from parallm.query import query_llm
        query_llm(args.prompts, args.models, args.schema, args.pydantic, args.repeat)

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

def handle_single_query(args, mode="default"):
    schema = load_schema(args.schema, args.pydantic)
    try:
        if mode == "aws":
            query_module = bedrock_query
        elif mode == "gemini":
            query_module = gemini_query
        else:
            query_module = model_query
        
        # Parallel repeat for all providers
        if args.repeat > 1:
            if mode == "gemini":
                df = query_module.query_model_repeat(args.prompt, args.model, args.repeat, schema)
            elif mode == "aws":
                df = query_module.query_model_repeat(args.prompt, args.model, args.repeat, schema)
            else:
                df = query_module.query_model_repeat(args.prompt, args.model, args.repeat, schema)
            for _, row in df.iterrows():
                print(f"Response {row['repeat_index']}/{args.repeat}:")
                if schema is None:
                    print(row['response'])
                else:
                    try:
                        if isinstance(row['response'], dict):
                            print(json.dumps(row['response'], indent=2))
                        else:
                            print(json.dumps(json.loads(row['response']), indent=2))
                    except Exception:
                        print(row['response'])
                print("---")
        else:
            if schema is None:
                result = query_module.query_model(args.prompt, args.model)
                print(result)
            else:
                result = query_module.query_model_json(args.prompt, args.model, schema)
                print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def handle_batch_query(args, mode="default"):
    schema = load_schema(args.schema, args.pydantic)
    repeat = getattr(args, 'repeat', 1)
    
    if mode == "aws":
        query_module = bedrock_query
    elif mode == "gemini":
        query_module = gemini_query
    else:
        query_module = model_query
    
    # Use the new repeat logic for batch mode
    if repeat > 1:
        if mode == "aws":
            result_df = query_module.query_model_all_repeat(args.prompts, args.models, repeat, schema)
        elif mode == "gemini":
            result_df = query_module.query_model_all_repeat(args.prompts, args.models, repeat, schema)
        else:
            result_df = query_module.query_model_all_repeat(args.prompts, args.models, repeat, schema)
    else:
        if schema is not None:
            print("Schema provided. Running queries sequentially (non-parallel)...")
            t_start = time.time()
            try:
                prompts_df = pd.read_csv(args.prompts)
                prompts_df["prompt"] = prompts_df["prompt"].str.strip().str.lower()

                models = pd.Series(args.models).str.strip().str.lower().tolist()
                models_df = pd.DataFrame({"model": models})
                combined_df = prompts_df.merge(models_df, how='cross')

                # Repeat each (prompt, model) pair 'repeat' times
                repeated_rows = []
                for _, row in combined_df.iterrows():
                    try:
                        response = query_module.query_model(row["prompt"], row["model"], schema)
                    except Exception as e:
                        response = f"Error: {e}"
                    repeated_rows.append({
                        "prompt": row["prompt"],
                        "model": row["model"],
                        "repeat_index": 1,
                        "response": response
                    })
                result_df = pd.DataFrame(repeated_rows)
                print(f"Sequential processing time: {time.time() - t_start:.2f} seconds")

            except Exception as e:
                print(f"\nError during sequential processing: {e}")
                sys.exit(1)
        else:
            print("No schema provided. Running queries in parallel using Bodo...")
            try:
                prompts_df = pd.read_csv(args.prompts)
                prompts_df["prompt"] = prompts_df["prompt"].str.strip().str.lower()
                models = pd.Series(args.models).str.strip().str.lower().tolist()
                models_df = pd.DataFrame({"model": models})
                combined_df = prompts_df.merge(models_df, how='cross')
                repeated_df = pd.DataFrame(
                    [
                        {"prompt": row["prompt"], "model": row["model"], "repeat_index": 1}
                        for _, row in combined_df.iterrows()
                    ]
                )
                def run_query(row):
                    try:
                        return query_module.query_model(row["prompt"], row["model"])
                    except Exception as e:
                        return f"Error: {e}"
                repeated_df["response"] = repeated_df.apply(run_query, axis=1)
                result_df = repeated_df
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