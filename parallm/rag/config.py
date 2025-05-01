import yaml
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Loads the RAG pipeline configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A dictionary representing the loaded configuration.

    Raises:
        FileNotFoundError: If the config file is not found.
        yaml.YAMLError: If the config file is not valid YAML.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            raise ValueError("Configuration file is not a valid dictionary.")
        # Basic validation (can be expanded later)
        if 'pipeline' not in config or not isinstance(config['pipeline'], list):
            raise ValueError("Config missing required 'pipeline' list.")
        if 'retrieval' not in config or not isinstance(config['retrieval'], dict):
            raise ValueError("Config missing required 'retrieval' dictionary.")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {config_path}: {e}")
        raise
    except ValueError as e:
        print(f"Error in configuration structure: {e}")
        raise
