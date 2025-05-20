import pandas as pd
import time
import bodo

@bodo.jit
def query_all(file_path, models, query_func, schema=None):
    t0 = time.time()
    prompts_df = pd.read_csv(file_path, usecols=["prompt"])
    prompts_df["prompt"] = prompts_df["prompt"].str.strip().str.lower()
    models = pd.Series(models).str.strip().str.lower().tolist()
    models_df = pd.DataFrame({"model": models})
    combined_df = prompts_df.merge(models_df, how="cross")
    combined_df["response"] = combined_df.apply(
        lambda row: query_func(row["prompt"], row["model"], schema), axis=1
    )
    print("Processing time:", time.time() - t0)
    return combined_df

@bodo.jit
def query_repeat(prompt, model_id, query_func, repeat=1, schema=None):
    df = pd.DataFrame({
        "repeat_index": range(1, repeat + 1),
        "prompt": [prompt] * repeat,
        "model": [model_id] * repeat,
    })
    df["response"] = df.apply(
        lambda row: query_func(row["prompt"], row["model"], schema), axis=1
    )
    return df[["repeat_index", "response"]]

@bodo.jit
def query_all_repeat(file_path, models, query_func, repeat=1, schema=None):
    prompts_df = pd.read_csv(file_path, usecols=["prompt"])
    prompts_df["prompt"] = prompts_df["prompt"].str.strip().str.lower()
    models = pd.Series(models).str.strip().str.lower().tolist()
    models_df = pd.DataFrame({"model": models})
    combined_df = prompts_df.merge(models_df, how="cross")
    repeated_df = pd.DataFrame([
        {"prompt": row["prompt"], "model": row["model"], "repeat_index": r + 1}
        for _, row in combined_df.iterrows()
        for r in range(repeat)
    ])
    repeated_df["response"] = repeated_df.apply(
        lambda row: query_func(row["prompt"], row["model"], schema), axis=1
    )
    return repeated_df[["prompt", "model", "repeat_index", "response"]]

