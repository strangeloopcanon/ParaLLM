"""
Lists the available Amazon Bedrock models and stores them in a CSV file.
"""
import logging
import json
import boto3
import csv
import os
from botocore.exceptions import ClientError
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def list_foundation_models(bedrock_client):
    """
    Gets a list of available Amazon Bedrock foundation models.

    :param bedrock_client: The Bedrock client to use
    :return: The list of available bedrock foundation models.
    """
    try:
        response = bedrock_client.list_foundation_models()
        models = response["modelSummaries"]
        logger.info("Got %s foundation models.", len(models))
        return models
    except ClientError:
        logger.error("Couldn't list foundation models.")
        raise

def get_foundation_model(bedrock_client, model_identifier):
    """
    Get details about an Amazon Bedrock foundation model.

    :param bedrock_client: The Bedrock client to use
    :param model_identifier: The identifier of the model to get details for
    :return: The foundation model's details.
    """
    try:
        return bedrock_client.get_foundation_model(
            modelIdentifier=model_identifier
        )["modelDetails"]
    except ClientError:
        logger.error(
            f"Couldn't get foundation models details for {model_identifier}"
        )
        raise

def save_models_to_csv(models, filename="scripts/bedrock_models.csv"):
    """
    Save model information to a CSV file.
    
    :param models: List of model dictionaries
    :param filename: Name of the CSV file to save
    """
    # Define the fields we want to save
    fields = [
        'modelName',
        'modelId',
        'providerName',
        'inputModalities',
        'outputModalities',
        'responseStreamingSupported',
        'customizationsSupported',
        'inferenceTypesSupported',
        'modelLifecycle.status'
    ]
    
    # Create the CSV file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        
        for model in models:
            # Create a row with only the fields we want
            row = {}
            for field in fields:
                if field == 'modelLifecycle.status':
                    # Handle nested field
                    row[field] = model.get('modelLifecycle', {}).get('status', '')
                else:
                    # Convert lists to strings for CSV
                    value = model.get(field, '')
                    if isinstance(value, list):
                        value = '|'.join(value)
                    row[field] = value
            writer.writerow(row)
    
    logger.info(f"Saved {len(models)} models to {filename}")

def main():
    """Entry point for the example. Uses the AWS SDK for Python (Boto3)
    to create an Amazon Bedrock client. Then lists the available Bedrock models
    in the region set in the callers profile and credentials.
    """
    bedrock_client = boto3.client(service_name="bedrock")
    
    # List all available models and save to CSV
    try:
        fm_models = list_foundation_models(bedrock_client)
        save_models_to_csv(fm_models)
        
        # Print a sample of the models
        print("\nSample of available models:")
        for model in fm_models[:3]:  # Show first 3 models
            print(f"\nModel: {model['modelName']}")
            print(f"ID: {model['modelId']}")
            print(f"Provider: {model['providerName']}")
            print("---------------------------")
        
        print(f"\nTotal models found: {len(fm_models)}")
        print(f"Model information saved to bedrock_models.csv")
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")

    logger.info("Done.")

if __name__ == "__main__":
    main()

