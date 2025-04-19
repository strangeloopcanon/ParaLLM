# gemini_query.py
import pandas as pd
import time
import json
import bodo
from dotenv import load_dotenv
from typing import Dict, Any, Optional, Union, Type, List
from pydantic import BaseModel
from google import genai
import os

load_dotenv()

@bodo.wrap_python(bodo.string_type)
def query_model(prompt, model_id="gemini-2.0-flash", schema=None):
    """
    Sends a prompt to Google Gemini and returns the response.
    
    Args:
        prompt: The text prompt to send to the model
        model_id: The ID of the Gemini model to use
        schema: Optional schema to format response as JSON
               Can be a Pydantic model class or a dict representing JSON schema
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    
    client = genai.Client(api_key=api_key)
    
    try:
        if schema:
            # If schema is a Pydantic model class
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                response = client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config={
                        'response_mime_type': 'application/json',
                        'response_schema': schema,
                    }
                )
                # Return the parsed Pydantic model
                return response.parsed
            # If schema is a dict (JSON schema)
            elif isinstance(schema, dict):
                response = client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config={
                        'response_mime_type': 'application/json',
                        'response_schema': schema,
                    }
                )
                # Return the JSON string
                return response.text
        else:
            # Regular text response
            response = client.models.generate_content(
                model=model_id,
                contents=[prompt]
            )
            return response.text
    except Exception as e:
        print(f"Error invoking Gemini model: {e}")
        raise

def query_model_json(prompt, model_id="gemini-2.0-flash", schema=None):
    """
    Sends a prompt to Google Gemini and returns parsed JSON response.
    
    Args:
        prompt: The text prompt to send to the model
        model_id: The ID of the Gemini model to use
        schema: Optional schema to format response as JSON
               Can be a Pydantic model class or a dict representing JSON schema
    
    Returns:
        Parsed JSON object or Pydantic model instance
    """
    result = query_model(prompt, model_id, schema)
    
    # If schema was a Pydantic model, result is already parsed
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return result
    
    # Otherwise, parse the JSON string
    try:
        return json.loads(result)
    except json.JSONDecodeError:
        raise ValueError("Response could not be parsed as JSON")

@bodo.jit
def query_model_all(file_path, models, schema=None):
    """
    Reads the prompts from `file_path` and a list of model IDs,
    creates a Cartesian product of prompts and models, and queries each pair.
    
    Args:
        file_path: Path to CSV file with prompts
        models: List of Gemini model IDs to query
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
def query_model_repeat(prompt, model_id="gemini-2.0-flash", repeat=1, schema=None):
    """
    Runs the same prompt against the same model N times in parallel using Bodo.
    Returns a DataFrame with columns: repeat_index, response
    """
    df = pd.DataFrame({
        "repeat_index": range(1, repeat + 1),
        "prompt": [prompt] * repeat,
        "model": [model_id] * repeat
    })
    def run_query(row):
        return query_model(row["prompt"], row["model"], schema)
    df["response"] = df.apply(run_query, axis=1)
    return df[["repeat_index", "response"]] 

@bodo.jit
def query_model_all_repeat(file_path, models, repeat=1, schema=None):
    """
    Reads the prompts from `file_path` and a list of model IDs,
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