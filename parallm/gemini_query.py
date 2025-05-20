# gemini_query.py
import json
import bodo
from dotenv import load_dotenv
from typing import Dict, Any, Optional, Union, Type, List
from pydantic import BaseModel
from google import genai
import os
from .query_utils import query_all, query_repeat, query_all_repeat

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

def query_model_all(file_path, models, schema=None):
    """Batch process prompts across multiple Gemini models."""
    return query_all(file_path, models, query_model, schema)

def query_model_repeat(prompt, model_id="gemini-2.0-flash", repeat=1, schema=None):
    """Repeat a prompt multiple times using a Gemini model."""
    return query_repeat(prompt, model_id, query_model, repeat, schema)

def query_model_all_repeat(file_path, models, repeat=1, schema=None):
    """Batch process prompts across models with repeats."""
    return query_all_repeat(file_path, models, query_model, repeat, schema)
