"""Unit tests for the utility functions in binaural.utils module."""

import pytest

from binaural.exceptions import (
    ConfigFileNotFoundError,
    ConfigurationError,
    YAMLParsingError,
)
from binaural.utils import load_yaml_config


def test_non_existent_file(tmp_path):
    "Attempt to load a non-existent file should raise ConfigFileNotFoundError"
    non_existent = tmp_path / "nonexistent.yaml"
    with pytest.raises(ConfigFileNotFoundError):
        load_yaml_config(str(non_existent))


def test_invalid_yaml(tmp_path):
    "Write an invalid YAML content and ensure YAMLParsingError is raised"
    file = tmp_path / "invalid.yaml"
    file.write_text(":::: invalid yaml ::")
    with pytest.raises(YAMLParsingError):
        load_yaml_config(str(file))


def test_missing_steps(tmp_path):
    "YAML file without a 'steps' key should raise ConfigurationError"
    file = tmp_path / "missing_steps.yaml"
    file.write_text("key: value\n")
    with pytest.raises(ConfigurationError):
        load_yaml_config(str(file))


def test_valid_yaml(tmp_path):
    "Valid YAML should load and contain the 'steps' key"
    file = tmp_path / "valid.yaml"
    content = "steps:\n  - type: stable\n    frequency: 10\n    duration: 5\n"
    file.write_text(content)
    config = load_yaml_config(str(file))
    assert "steps" in config
