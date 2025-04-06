"""Configuration utilities for the Binaural Beat Generator."""

from typing import Any

import streamlit as st
import yaml

from binaural.utils import get_all_script_configs


def load_config_file(file_path: str) -> dict[str, Any]:
    """Load a YAML configuration file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

            if not isinstance(config, dict):
                st.error("Invalid configuration format: Root must be a dictionary.")
                return {}

            if "background_noise" in config:
                bg_noise = config["background_noise"]
                if not isinstance(bg_noise, dict):
                    config["background_noise"] = {"type": "none", "amplitude": 0.0}
            else:
                config["background_noise"] = {"type": "none", "amplitude": 0.0}

            return config
    except FileNotFoundError:
        st.error(f"Error: Configuration file not found at {file_path}")
        return {}
    except PermissionError:
        st.error(f"Error: Permission denied reading file {file_path}")
        return {}
    except yaml.YAMLError as e:
        st.error(f"Error parsing YAML configuration: {e}")
        return {}
    except IOError as e:
        st.error(f"Error reading configuration file: {e}")
        return {}


def get_config_steps(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract steps from configuration dictionary."""
    if "steps" in config and isinstance(config["steps"], list):
        return config["steps"]
    return []


def format_time(seconds: int) -> str:
    """Format seconds as mm:ss."""
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes):02d}:{int(seconds):02d}"


def get_available_configs(scripts_dir: str = "scripts") -> dict[str, str]:
    """Get all available configuration files with their titles.

    Args:
        scripts_dir: Directory containing YAML configuration files

    Returns:
        Dictionary mapping configuration titles to file paths
    """
    # Use the utility function from binaural.utils
    return get_all_script_configs(scripts_dir)
