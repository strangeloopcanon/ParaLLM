# model_query.py
import pandas as pd
import time
import json
from dotenv import load_dotenv
import llm
import bodo
from typing import Dict, Any, Optional, Union, Type
from pydantic import BaseModel

load_dotenv()

@bodo.wrap_python(bodo.string_type)
def query_model(prompt, model_given="gemini-2.0-flash", schema=None):
    """
    Sends a prompt to the AI Suite and returns the response.
    
    Args:
        prompt: The text prompt to send to the model
        model_given: The name of the model to use
        schema: Optional schema to format response as JSON
               Can be a Pydantic model class or a dict representing JSON schema
    """
    model = llm.get_model(model_given)
    
    response = model.prompt(prompt, schema=schema)
    return response.text()

def query_model_json(prompt, model_given="gemini-2.0-flash", schema=None):
    """
    Sends a prompt to the AI Suite and returns parsed JSON response.
    
    Args:
        prompt: The text prompt to send to the model
        model_given: The name of the model to use
        schema: Optional schema to format response as JSON
               Can be a Pydantic model class or a dict representing JSON schema
    
    Returns:
        Parsed JSON object
    """
    result = query_model(prompt, model_given, schema)
    return json.loads(result)

@bodo.jit
def query_model_all(file_path, models, schema=None):
    """
    Reads the prompts from `file_path` and a list of model names,
    creates a Cartesian product of prompts and models, and queries each pair.
    
    Args:
        file_path: Path to CSV file with prompts
        models: List of model names to query
        schema: Optional schema to format responses as JSON
    """
    t0 = time.time()
    prompts_df = pd.read_csv(file_path, usecols=["prompt"])
    prompts_df["prompt"] = prompts_df["prompt"].str.strip().str.lower()

    models = pd.Series(models).str.strip().str.lower().tolist()
    
    models_df = pd.DataFrame({"model": models})
    combined_df = prompts_df.merge(models_df, how='cross')

    combined_df["response"] = combined_df.apply(
            lambda row: query_model(row["prompt"], row["model"], schema), axis=1
        )

    print("Processing time:", time.time() - t0)
    return combined_df

@bodo.jit
def query_model_repeat(prompt, model_given="gemini-2.0-flash", repeat=1, schema=None):
    """
    Runs the same prompt against the same model N times in parallel using Bodo.
    Returns a DataFrame with columns: repeat_index, response
    """
    df = pd.DataFrame({
        "repeat_index": range(1, repeat + 1),
        "prompt": [prompt] * repeat,
        "model": [model_given] * repeat
    })
    def run_query(row):
        return query_model(row["prompt"], row["model"], schema)
    df["response"] = df.apply(run_query, axis=1)
    return df[["repeat_index", "response"]]

@bodo.jit
def query_model_all_repeat(file_path, models, repeat=1, schema=None):
    """
    Reads the prompts from `file_path` and a list of model names,
    creates a Cartesian product of prompts, models, and repeat_index, and queries each combination.
    Returns a DataFrame with columns: prompt, model, repeat_index, response
    """
    prompts_df = pd.read_csv(file_path, usecols=["prompt"])
    prompts_df["prompt"] = prompts_df["prompt"].str.strip().str.lower()
    models = pd.Series(models).str.strip().str.lower().tolist()
    models_df = pd.DataFrame({"model": models})
    combined_df = prompts_df.merge(models_df, how='cross')
    repeated_df = pd.DataFrame([
        {"prompt": row["prompt"], "model": row["model"], "repeat_index": r+1}
        for _, row in combined_df.iterrows() for r in range(repeat)
    ])
    def run_query(row):
        return query_model(row["prompt"], row["model"], schema)
    repeated_df["response"] = repeated_df.apply(run_query, axis=1)
    return repeated_df[["prompt", "model", "repeat_index", "response"]]
