"""Utility functions for loading and validating YAML configuration files."""

import yaml
from binaural.exceptions import (
    ConfigFileNotFoundError,
    YAMLParsingError,
    ConfigurationError,
)


def load_yaml_config(path: str) -> dict:
    """Loads and validates YAML configuration."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        if not isinstance(config, dict) or "steps" not in config:
            raise ConfigurationError("YAML must have 'steps' key.")
        return config
    except FileNotFoundError as e:
        raise ConfigFileNotFoundError(f"Config file '{path}' not found.") from e
    except yaml.YAMLError as e:
        raise YAMLParsingError(f"Error parsing YAML file '{path}': {e}") from e
    except ValueError as e:
        raise ConfigurationError(f"Configuration error: {e}") from e
