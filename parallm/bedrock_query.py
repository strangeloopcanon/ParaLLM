# bedrock_query.py
import json
import boto3
import bodo
from dotenv import load_dotenv
from typing import Dict, Any, Optional, Union, Type
from pydantic import BaseModel
from botocore.exceptions import ClientError
from .query_utils import query_all, query_repeat, query_all_repeat

load_dotenv()

@bodo.wrap_python(bodo.string_type)
def query_model(prompt, model_id="anthropic.claude-3-sonnet-20240229", schema=None):
    """
    Sends a prompt to AWS Bedrock and returns the response.
    
    Args:
        prompt: The text prompt to send to the model
        model_id: The ID of the Bedrock model to use
        schema: Optional schema to format response as JSON
               Can be a Pydantic model class or a dict representing JSON schema
    """
    bedrock_client = boto3.client(service_name="bedrock-runtime")
    
    # Format the prompt based on model provider
    provider = model_id.split('.')[0].lower()
    formatted_prompt = format_prompt_for_provider(prompt, provider)
    
    try:
        # Invoke the model
        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps(formatted_prompt)
        )
        
        # Parse the response
        response_body = json.loads(response['body'].read().decode('utf-8'))
        result = extract_response_for_provider(response_body, provider)
        
        # Apply schema if provided
        if schema:
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                # Handle Pydantic model
                result_json = json.loads(result)
                return json.dumps(schema(**result_json).dict())
            elif isinstance(schema, dict):
                # Handle JSON schema
                try:
                    result_json = json.loads(result)
                    # Here we would validate against schema, but for now we just return the parsed JSON
                    return json.dumps(result_json)
                except json.JSONDecodeError:
                    # If the result is not valid JSON, return as is
                    return result
        
        return result
    except ClientError as e:
        print(f"Error invoking Bedrock model: {e}")
        raise

def format_prompt_for_provider(prompt, provider):
    """
    Format the prompt according to the provider's required format.
    
    Args:
        prompt: The text prompt
        provider: The model provider (e.g., 'anthropic', 'amazon', 'ai21', etc.)
    
    Returns:
        Formatted prompt ready for the specific model
    """
    if provider == 'anthropic':
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
    elif provider == 'amazon':
        return {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 1000,
                "temperature": 0.7
            }
        }
    elif provider == 'ai21':
        return {
            "prompt": prompt,
            "maxTokens": 1000,
            "temperature": 0.7
        }
    elif provider == 'cohere':
        return {
            "prompt": prompt,
            "max_tokens": 1000,
            "temperature": 0.7
        }
    else:
        # Default format
        return {"prompt": prompt}

def extract_response_for_provider(response_body, provider):
    """
    Extract the text from the model response based on provider.
    
    Args:
        response_body: The parsed JSON response body
        provider: The model provider
    
    Returns:
        Extracted text response
    """
    if provider == 'anthropic':
        return response_body.get('content', [{}])[0].get('text', '')
    elif provider == 'amazon':
        return response_body.get('results', [{}])[0].get('outputText', '')
    elif provider == 'ai21':
        return response_body.get('completions', [{}])[0].get('data', {}).get('text', '')
    elif provider == 'cohere':
        return response_body.get('generations', [{}])[0].get('text', '')
    else:
        # Default extraction - return the whole response
        return json.dumps(response_body)

def query_model_json(prompt, model_id="anthropic.claude-3-sonnet-20240229", schema=None):
    """
    Sends a prompt to AWS Bedrock and returns parsed JSON response.
    
    Args:
        prompt: The text prompt to send to the model
        model_id: The ID of the Bedrock model to use
        schema: Optional schema to format response as JSON
               Can be a Pydantic model class or a dict representing JSON schema
    
    Returns:
        Parsed JSON object
    """
    result = query_model(prompt, model_id, schema)
    return json.loads(result)

def query_model_all(file_path, models, schema=None):
    """Batch process prompts across multiple Bedrock models."""
    return query_all(file_path, models, query_model, schema)

def query_model_repeat(prompt, model_id="anthropic.claude-3-sonnet-20240229", repeat=1, schema=None):
    """Repeat a prompt multiple times using a Bedrock model."""
    return query_repeat(prompt, model_id, query_model, repeat, schema)

def query_model_all_repeat(file_path, models, repeat=1, schema=None):
    """Batch process prompts across models with repeats."""
    return query_all_repeat(file_path, models, query_model, repeat, schema)
