import pandas as pd
import time
from dotenv import load_dotenv
import llm
import bodo

load_dotenv()

@bodo.wrap_python(bodo.string_type)
def query_model(prompt, model_given="gpt-4o-mini"):
    """
    Sends a prompt to the AI Suite and returns the response.
    """
    model = llm.get_model(model_given)
    response = model.prompt(prompt)
    return response.text()

@bodo.jit
def query_model_all(file_path, models):
    """
    Reads the prompts from `file_path` and a list of model names,
    creates a Cartesian product of prompts and models, and queries each pair.
    """
    t0 = time.time()
    prompts_df = pd.read_csv(file_path)
    prompts_df["prompt"] = prompts_df["prompt"].str.strip().str.lower()

    # Normalize models list
    models = pd.Series(models).str.strip().str.lower()
    models_df = pd.DataFrame({"model": models})
    combined_df = prompts_df.merge(models_df, how='cross')

    combined_df["response"] = combined_df.apply(
        lambda row: query_model(row["prompt"], row["model"]), axis=1
    )

    print("Processing time:", time.time() - t0)
    return combined_df
