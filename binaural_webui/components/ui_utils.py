"""UI utility functions for the Binaural Beat Generator."""

from typing import Any

import streamlit as st
import yaml

from binaural.constants import DEFAULT_BASE_FREQUENCY, DEFAULT_SAMPLE_RATE
from binaural.exceptions import BinauralError
from binaural.parallel import prepare_audio_steps
from binaural_webui.constants import DEFAULT_STEP_DURATION, FREQUENCY_PRESETS


def initialize_session_state() -> dict[str, Any]:
    """Initialize Streamlit session state variables if they don't exist."""
    if "config" not in st.session_state:
        st.session_state.config = {
            "title": "Binaural Beat Generator",
            "base_frequency": DEFAULT_BASE_FREQUENCY,
            "sample_rate": DEFAULT_SAMPLE_RATE,
            "output_filename": "audio/my_session.flac",
            "background_noise": {"type": "none", "amplitude": 0.0},
            "steps": [],
        }
    if "audio_preview" not in st.session_state:
        st.session_state.audio_preview = None
    if "generated_audio" not in st.session_state:
        st.session_state.generated_audio = None
    return st.session_state.config


def render_frequency_preset_selector(config: dict[str, Any]) -> None:
    """Render the controls for adding a step from frequency presets."""
    preset_col1, preset_col2 = st.columns(2)
    with preset_col1:
        selected_preset = st.selectbox(
            "Add from Frequency Preset",
            list(FREQUENCY_PRESETS.keys()),
            index=2,
            help="Quickly add a stable frequency step from common brainwave ranges.",
        )

    with preset_col2:
        selected_freq = st.selectbox(
            f"Select {selected_preset} Frequency (Hz)",
            FREQUENCY_PRESETS[selected_preset],
        )

        if st.button("Add Frequency"):
            new_step = {
                "type": "stable",
                "frequency": selected_freq,
                "duration": DEFAULT_STEP_DURATION,
            }
            config["steps"].append(new_step)
            st.session_state.audio_preview = None
            st.session_state.generated_audio = None
            st.rerun()


def display_total_duration(config: dict[str, Any]) -> None:
    """Calculate and display the total duration of the sequence."""
    total_duration = sum(step.get("duration", 0) for step in config["steps"])
    minutes, seconds = divmod(total_duration, 60)
    st.info(f"Total Duration: {int(minutes)} minutes, {int(seconds)} seconds")


def render_add_step_buttons(config: dict[str, Any]) -> None:
    """Render buttons for adding new empty steps or transition steps."""
    if st.button("Add Empty Step"):
        new_step = {
            "type": "stable",
            "frequency": 10.0,
            "duration": DEFAULT_STEP_DURATION,
        }
        config["steps"].append(new_step)
        st.session_state.audio_preview = None
        st.session_state.generated_audio = None
        st.rerun()

    if len(config["steps"]) >= 1:
        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            transition_duration = st.number_input(
                "Transition Duration (seconds)", min_value=1, value=120, step=10
            )

        with col2:
            if st.button("Add Transition to Next Frequency"):
                current_steps = config["steps"]
                last_freq = 10.0
                if current_steps:
                    try:
                        processed_steps = prepare_audio_steps(current_steps)
                        last_freq = processed_steps[-1].freq.end
                    except BinauralError:
                        st.warning(
                            "Could not determine last frequency, defaulting to 10Hz."
                        )
                        last_freq = 10.0

                next_freq = 4.0 if last_freq > 7.0 else 10.0

                new_step = {
                    "type": "transition",
                    "end_frequency": next_freq,
                    "duration": transition_duration,
                }

                config["steps"].append(new_step)
                st.session_state.audio_preview = None
                st.session_state.generated_audio = None
                st.rerun()


def display_yaml_config(config: dict[str, Any]) -> None:
    """Display the current configuration as YAML and provide a download button."""
    with st.expander("View/Download YAML Configuration"):
        display_config = config.copy()

        if "background_noise" not in display_config:
            display_config["background_noise"] = {"type": "none", "amplitude": 0.0}

        key_order = [
            "title",
            "base_frequency",
            "sample_rate",
            "output_filename",
            "background_noise",
            "steps",
        ]

        ordered_config = {}
        for key in key_order:
            if key in display_config:
                if key == "background_noise":
                    bg_noise = display_config[key]
                    if (
                        bg_noise.get("type", "none") != "none"
                        or bg_noise.get("amplitude", 0.0) > 0
                    ):
                        ordered_config[key] = bg_noise
                else:
                    ordered_config[key] = display_config[key]

        for key, value in display_config.items():
            if key not in ordered_config and key != "noise_config":
                ordered_config[key] = value

        try:
            # Generate the YAML without header comments first
            raw_yaml_text = yaml.dump(
                ordered_config, default_flow_style=False, sort_keys=False
            )
            # Add explanatory header comments
            config_title = ordered_config.get("title", "Binaural Beat Configuration")
            yaml_text = f"# {config_title}\n"
            yaml_text += "# Generated by Binaural Beat Generator\n\n"
            yaml_text += raw_yaml_text
        except yaml.YAMLError as e:
            st.error(f"Error generating YAML: {e}")
            yaml_text = "# Error generating YAML configuration"

        st.code(yaml_text, language="yaml")

        st.download_button(
            label="Download YAML Configuration",
            data=yaml_text,
            file_name="binaural_config.yaml",
            mime="application/x-yaml",
        )
