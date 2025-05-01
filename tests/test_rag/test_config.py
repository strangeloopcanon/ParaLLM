# tests/test_rag/test_config.py

import pytest
import yaml
from pathlib import Path

from parallm.rag.config import load_config

# Minimal valid config structure
VALID_CONFIG_DICT = {
    'pipeline': [
        {'name': 'ingest', 'params': {'source_path': './docs'}}
    ],
    'retrieval': {
        'strategy': 'vector'
    }
}

VALID_CONFIG_YAML = yaml.dump(VALID_CONFIG_DICT)

INVALID_YAML_SYNTAX = "pipeline: [\n name: ingest"

MISSING_PIPELINE_YAML = yaml.dump({'retrieval': {'strategy': 'vector'}})
MISSING_RETRIEVAL_YAML = yaml.dump({'pipeline': []})
NOT_A_DICT_YAML = yaml.dump([1, 2, 3])


def test_load_config_valid(tmp_path: Path):
    """Test loading a structurally valid config file."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(VALID_CONFIG_YAML)
    
    loaded_config = load_config(str(config_file))
    assert loaded_config == VALID_CONFIG_DICT

def test_load_config_file_not_found():
    """Test loading a non-existent config file."""
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_config.yaml")

def test_load_config_invalid_yaml_syntax(tmp_path: Path):
    """Test loading a file with invalid YAML syntax."""
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text(INVALID_YAML_SYNTAX)
    
    with pytest.raises(yaml.YAMLError):
        load_config(str(config_file))

def test_load_config_missing_pipeline(tmp_path: Path):
    """Test loading config missing the 'pipeline' key."""
    config_file = tmp_path / "missing_pipeline.yaml"
    config_file.write_text(MISSING_PIPELINE_YAML)
    
    with pytest.raises(ValueError, match="missing required 'pipeline' list"):
        load_config(str(config_file))

def test_load_config_missing_retrieval(tmp_path: Path):
    """Test loading config missing the 'retrieval' key."""
    config_file = tmp_path / "missing_retrieval.yaml"
    config_file.write_text(MISSING_RETRIEVAL_YAML)
    
    with pytest.raises(ValueError, match="missing required 'retrieval' dictionary"):
        load_config(str(config_file))

def test_load_config_not_a_dict(tmp_path: Path):
    """Test loading config that is not a dictionary at the top level."""
    config_file = tmp_path / "not_dict.yaml"
    config_file.write_text(NOT_A_DICT_YAML)
    
    with pytest.raises(ValueError, match="not a valid dictionary"):
        load_config(str(config_file)) 