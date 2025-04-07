"""Sidebar components for the Binaural Beat Generator."""

import os
from typing import Any

import streamlit as st

from binaural_generator.core.constants import (
    AUTHOR_EMAIL,
    AUTHOR_NAME,
    DEFAULT_BASE_FREQUENCY,
    GITHUB_URL,
    LICENSE,
)
from binaural_generator.webui.components.config_utils import (
    get_available_configs,
    load_config_file,
)
from binaural_generator.webui.constants import BRAINWAVE_PRESETS


def _handle_example_loading(config: dict[str, Any]) -> None:
    """Handle the loading of example configurations in the sidebar."""
    # Get configurations dynamically from YAML files
    available_configs = get_available_configs()

    example_name = st.selectbox(
        "Load Example Configuration",
        ["Custom"] + sorted(list(available_configs.keys())),
    )

    if example_name != "Custom" and st.button("Load Example"):
        loaded_config = load_config_file(available_configs[example_name])
        if loaded_config:
            config.clear()
            config.update(loaded_config)
            st.success(f"Loaded configuration: {example_name}")
            st.rerun()


def _render_global_settings(config: dict[str, Any]) -> None:
    """Render global settings controls in the sidebar."""
    st.subheader("Global Settings")
    # Add title input field
    config["title"] = st.text_input(
        "Title",
        value=config.get("title", "Binaural Beat Session"),
        help="Title of your binaural beat session",
    )

    config["base_frequency"] = st.number_input(
        "Base Carrier Frequency (Hz)",
        min_value=50,
        max_value=500,
        value=config.get("base_frequency", DEFAULT_BASE_FREQUENCY),
        help="The base frequency used for both channels.",
    )


def _render_noise_settings(config: dict[str, Any], noise_types: list[str]) -> None:
    """Render background noise settings controls in the sidebar."""
    st.subheader("Background Noise")
    current_noise = config.get("background_noise", {}).get("type", "none")
    index = noise_types.index(current_noise) if current_noise in noise_types else 0

    noise_type = st.selectbox("Noise Type", noise_types, index=index)
    noise_amplitude = 0.0

    if noise_type != "none":
        default_amplitude = config.get("background_noise", {}).get("amplitude", 0.0)
        noise_amplitude = st.slider(
            "Noise Amplitude",
            min_value=0.0,
            max_value=1.0,
            value=float(default_amplitude),
            step=0.01,
            help="Relative volume of the background noise.",
        )

    config["background_noise"] = {"type": noise_type, "amplitude": noise_amplitude}


def _render_output_settings(config: dict[str, Any]) -> None:
    """Render output file settings controls in the sidebar."""
    st.subheader("Output Settings")
    current_filename = config.get("output_filename", "")
    current_format_index = 0 if current_filename.lower().endswith(".flac") else 1
    output_format = st.radio(
        "Audio Format", ["FLAC", "WAV"], index=current_format_index
    )

    default_basename = "my_session"
    if current_filename:
        default_basename = os.path.splitext(os.path.basename(current_filename))[0]

    output_filename_base = st.text_input("Output Filename", value=default_basename)
    extension = ".flac" if output_format == "FLAC" else ".wav"
    config["output_filename"] = f"audio/{output_filename_base}{extension}"


def _render_brainwave_info():
    """Render the brainwave information expander in the sidebar."""
    with st.expander("Brainwave States Information"):
        for wave, description in BRAINWAVE_PRESETS.items():
            st.markdown(f"**{wave}**: {description}")


def _render_repo_info():
    """Render the repository information in the sidebar."""
    st.markdown("## Repository Information")
    st.markdown(
        f"Binaural Beat Generator project is licensed under the {LICENSE}. "
        f"Find the source code and contribute on [GitHub]({GITHUB_URL})."
    )
    st.markdown(f"Copyright © 2025 [{AUTHOR_NAME}](mailto:{AUTHOR_EMAIL}) ")


def render_sidebar(config: dict[str, Any], noise_types: list[str]) -> None:
    """Render the entire sidebar content."""
    with st.sidebar:
        st.header("Templates & Settings")
        _handle_example_loading(config)
        st.divider()
        _render_global_settings(config)
        _render_noise_settings(config, noise_types)
        _render_output_settings(config)
        st.divider()
        _render_brainwave_info()
        _render_repo_info()
