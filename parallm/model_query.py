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
def query_model(prompt, model_given="gpt-4o-mini", schema=None):
    """
    Sends a prompt to the AI Suite and returns the response.
    
    Args:
        prompt: The text prompt to send to the model
        model_given: The name of the model to use
        schema: Optional schema to format response as JSON
               Can be a Pydantic model class or a dict representing JSON schema
    """
    model = llm.get_model(model_given)
    
    if schema is None:
        response = model.prompt(prompt)
        return response.text()
    else:
        response = model.prompt(prompt, schema=schema)
        return response.text()

def query_model_json(prompt, model_given="gpt-4o-mini", schema=None):
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

def query_model_all(file_path, models, schema=None):
    """
    Reads the prompts from `file_path` and a list of model names,
    creates a Cartesian product of prompts and models, and queries each pair.
    
    Args:
        file_path: Path to CSV file with prompts
        models: List of model names to query
        schema: Optional schema to format responses as JSON
    """
    # Note: We've temporarily removed the @bodo.jit decorator for simplicity
    # while testing with schemas
    t0 = time.time()
    prompts_df = pd.read_csv(file_path)
    prompts_df["prompt"] = prompts_df["prompt"].str.strip().str.lower()

    # Normalize models list
    if isinstance(models, list):
        models = [model.strip().lower() for model in models]
    else:
        models = pd.Series(models).str.strip().str.lower().tolist()
    
    models_df = pd.DataFrame({"model": models})
    combined_df = prompts_df.merge(models_df, how='cross')

    # Process with or without schema
    if schema is None:
        combined_df["response"] = combined_df.apply(
            lambda row: query_model(row["prompt"], row["model"]), axis=1
        )
    else:
        combined_df["response"] = combined_df.apply(
            lambda row: query_model(row["prompt"], row["model"], schema), axis=1
        )

    print("Processing time:", time.time() - t0)
    return combined_df
