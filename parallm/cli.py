# cli!
import sys
import argparse
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
    args = parser.parse_args()

    result_df = model_query.query_model_all(args.prompts, args.models)
    print(f"\n Readable table for humans: {result_df}\n")
    # Output the result as CSV to stdout.
    print(f"\n Readable table for machines: \n")
    print(result_df.to_csv(index=False))

if __name__ == "__main__":
    main()
