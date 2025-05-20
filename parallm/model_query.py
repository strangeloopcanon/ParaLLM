# model_query.py
import json
from dotenv import load_dotenv
import llm
import bodo
from typing import Dict, Any, Optional, Union, Type
from pydantic import BaseModel
from .query_utils import query_all, query_repeat, query_all_repeat

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

def query_model_all(file_path, models, schema=None):
    """Batch process prompts across multiple models."""
    return query_all(file_path, models, query_model, schema)

def query_model_repeat(prompt, model_given="gemini-2.0-flash", repeat=1, schema=None):
    """Repeat a single prompt multiple times."""
    return query_repeat(prompt, model_given, query_model, repeat, schema)

def query_model_all_repeat(file_path, models, repeat=1, schema=None):
    """Batch process prompts across models with repeats."""
    return query_all_repeat(file_path, models, query_model, repeat, schema)

