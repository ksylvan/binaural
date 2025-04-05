"""Streamlit web UI for the Binaural Beat Generator."""

import io
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import streamlit as st
import yaml

from binaural.constants import (
    AUTHOR_EMAIL,
    AUTHOR_NAME,
    DEFAULT_BASE_FREQUENCY,
    DEFAULT_SAMPLE_RATE,
    GITHUB_URL,
    LICENSE,
)
from binaural.data_types import NoiseConfig
from binaural.exceptions import BinauralError
from binaural.noise import NoiseFactory
from binaural.parallel import prepare_audio_steps
from binaural.tone_generator import generate_audio_sequence, save_audio_file

# Get available noise types from the NoiseFactory
# Ensure 'none' is first and list is sorted otherwise
all_noise_types = NoiseFactory.strategies()
if "none" in all_noise_types:
    all_noise_types.remove("none")
    NOISE_TYPES = ["none"] + sorted(all_noise_types)
else:
    NOISE_TYPES = sorted(all_noise_types)

# Constants for the UI
BRAINWAVE_PRESETS = {
    "Delta (0.5-4 Hz)": "Deep sleep, healing",
    "Theta (4-7 Hz)": "Meditation, creativity",
    "Alpha (8-12 Hz)": "Relaxation, calmness",
    "Beta (13-30 Hz)": "Focus, alertness",
    "Gamma (30-100 Hz)": "Peak concentration",
}

FREQUENCY_PRESETS = {
    "Delta": [0.5, 1, 2, 3, 4],
    "Theta": [4, 5, 6, 7],
    "Alpha": [8, 9, 10, 11, 12],
    "Beta": [13, 15, 18, 20, 25, 30],
    "Gamma": [35, 40, 50, 60],
}

STEP_TYPES = ["stable", "transition"]
DEFAULT_STEP_DURATION = 300  # 5 minutes in seconds
EXAMPLE_CONFIGS = {
    "Relaxation (Alpha)": "scripts/relaxation_alpha.yaml",
    "Focus (Beta)": "scripts/focus_beta.yaml",
    "Focus (Gamma)": "scripts/focus_gamma.yaml",
    "Meditation (Theta)": "scripts/meditation_theta.yaml",
    "Sleep (Delta)": "scripts/sleep_delta.yaml",
    "Creativity (Theta)": "scripts/creativity_theta.yaml",
    "Creativity (Blue Noise)": "scripts/creativity_blue.yaml",
    "Focus (Violet Noise)": "scripts/focus_violet.yaml",
    "Relaxation (Grey Noise)": "scripts/relaxation_grey.yaml",
    "Relaxation (Rain)": "scripts/relaxation_rain.yaml",
    "Relaxation (Ocean)": "scripts/relaxation_ocean.yaml",  # Added Ocean example
    "Lucid Dreaming": "scripts/lucid_dreaming.yaml",
    "Lucid Dreaming (Pink Noise)": "scripts/lucid_dream_pink_noise.yaml",
    "Migraine Relief": "scripts/migraine_relief.yaml",
}


def load_config_file(file_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    try:
        # Open and read the YAML file
        with open(file_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

            # Ensure the loaded config is a dictionary
            if not isinstance(config, dict):
                st.error("Invalid configuration format: Root must be a dictionary.")
                return {}

            # Convert noise settings to the expected format
            # This ensures compatibility between the core library and the web UI
            if "background_noise" in config:
                bg_noise = config["background_noise"]
                if not isinstance(bg_noise, dict):
                    # If background_noise is not a dict, replace with default
                    config["background_noise"] = {"type": "none", "amplitude": 0.0}
            else:
                # If no background_noise in the loaded config, add default
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


def get_config_steps(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract steps from configuration dictionary."""
    # Check if 'steps' key exists and is a list
    if "steps" in config and isinstance(config["steps"], list):
        return config["steps"]
    # Return an empty list if 'steps' is missing or not a list
    return []


def format_time(seconds: int) -> str:
    """Format seconds as mm:ss."""
    # Calculate minutes and remaining seconds
    minutes, seconds = divmod(seconds, 60)
    # Return formatted string with leading zeros
    return f"{int(minutes):02d}:{int(seconds):02d}"


def _extract_noise_config(config: Dict[str, Any]) -> NoiseConfig:
    """Extract noise configuration from the given config."""
    # Get the background_noise dictionary, defaulting if missing
    noise_dict = config.get("background_noise", {"type": "none", "amplitude": 0.0})
    # Create and return a NoiseConfig object
    return NoiseConfig(
        type=noise_dict.get("type", "none"),
        amplitude=noise_dict.get("amplitude", 0.0),
    )


def generate_preview_audio(
    config: Dict[str, Any], duration: int = 30
) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    """Generate a preview of the audio based on the current configuration."""
    try:
        # Extract parameters from the config, using defaults if missing
        sample_rate = config.get("sample_rate", DEFAULT_SAMPLE_RATE)
        base_freq = config.get("base_frequency", DEFAULT_BASE_FREQUENCY)
        steps = config.get("steps", [])

        # Extract noise configuration
        noise_config = _extract_noise_config(config)

        # Create shortened steps for preview
        preview_steps = []
        total_original_duration = sum(step.get("duration", 0) for step in steps)

        # If total duration is zero or negative, cannot generate preview
        if total_original_duration <= 0:
            st.warning("Cannot generate preview: Total duration is zero or negative.")
            return None

        # Calculate the ratio to scale down the duration
        ratio = min(1.0, duration / total_original_duration)

        # Create preview steps by scaling duration and fades
        for step in steps:
            preview_step = step.copy()
            # Scale duration, ensuring it's at least 1 second
            preview_step["duration"] = max(1, int(step.get("duration", 0) * ratio))

            # Scale fade durations proportionally
            # ensuring they don't exceed half the new duration
            if "fade_in_duration" in preview_step:
                preview_step["fade_in_duration"] = min(
                    preview_step["duration"] / 2,
                    preview_step["fade_in_duration"] * ratio,
                )

            if "fade_out_duration" in preview_step:
                preview_step["fade_out_duration"] = min(
                    preview_step["duration"] / 2,
                    preview_step["fade_out_duration"] * ratio,
                )

            preview_steps.append(preview_step)

        # Generate the audio sequence using the preview steps
        left_channel, right_channel, total_duration = generate_audio_sequence(
            sample_rate=sample_rate,
            base_freq=base_freq,
            steps=preview_steps,
            noise_config=noise_config,
        )

        return left_channel, right_channel, total_duration

    except BinauralError as e:
        # Handle specific binaural errors
        st.error(f"Error generating preview: {e}")
        return None
    except (ValueError, KeyError, TypeError) as e:
        # Handle configuration related errors
        st.error(f"Configuration error during preview generation: {e}")
        return None
    except OSError as e:
        # Handle file system errors (less likely here but possible)
        st.error(f"I/O error during preview generation: {e}")
        return None


def _get_implied_start_frequency(
    step_index: int, step: Dict[str, Any], all_steps: List[Dict[str, Any]]
) -> Optional[float]:
    """Get the implied start frequency for transition steps."""
    # Check if it's a transition step, doesn't have explicit start_frequency,
    # and is not the first step
    if (
        step.get("type") == "transition"
        and "start_frequency" not in step
        and step_index > 0
    ):
        try:
            # Prepare the steps up to the current one to find the end frequency
            # of the previous step
            current_steps = all_steps[: step_index + 1]
            processed_steps = prepare_audio_steps(current_steps)
            # Return the start frequency of the current step
            # (which is the end freq of the previous)
            return processed_steps[-1].freq.start
        except (IndexError, AttributeError, ValueError, BinauralError):
            # If any error occurs during processing, cannot determine implied frequency
            return None
    # Not applicable or cannot determine implied frequency
    return None


def _handle_stable_step(
    step_index: int, step: Dict[str, Any], step_type: str, duration: int
) -> Dict[str, Any]:
    """Handle editing for stable frequency step type."""
    # Get current frequency or default to 10.0 Hz (Alpha)
    freq_value = float(step.get("frequency", 10.0))
    # Create number input for frequency
    frequency = st.number_input(
        "Frequency (Hz)",
        min_value=0.1,
        max_value=100.0,
        value=freq_value,
        step=0.1,
        format="%.1f",
        key=f"frequency_{step_index}",
        help="The constant binaural beat frequency for this step.",
    )

    # Return the updated step dictionary
    return {
        "type": step_type,
        "frequency": frequency,
        "duration": duration,
    }


def _handle_transition_step(
    step_index: int,
    step: Dict[str, Any],
    step_type: str,
    duration: int,
    implied_start_freq: Optional[float],
) -> Dict[str, Any]:
    """Handle editing for transition step type."""
    implied_label = ""

    # Determine the start frequency value and label
    if implied_start_freq is not None:
        start_freq_value = float(implied_start_freq)
        implied_label = " (implied)"  # Indicate that this value is derived
    else:
        # Use explicit start frequency or default to 10.0 Hz
        start_freq_value = float(step.get("start_frequency", 10.0))

    # Create number input for start frequency
    start_freq = st.number_input(
        f"Start Frequency (Hz){implied_label}",
        min_value=0.1,
        max_value=100.0,
        value=start_freq_value,
        step=0.1,
        format="%.1f",
        key=f"start_freq_{step_index}",
        help="The binaural beat frequency at the beginning of the transition.",
    )

    # Get current end frequency or default to 4.0 Hz (Theta)
    end_freq_value = float(step.get("end_frequency", 4.0))
    # Create number input for end frequency
    end_freq = st.number_input(
        "End Frequency (Hz)",
        min_value=0.1,
        max_value=100.0,
        value=end_freq_value,
        step=0.1,
        format="%.1f",
        key=f"end_freq_{step_index}",
        help="The binaural beat frequency at the end of the transition.",
    )

    # Base structure for the updated step
    updated_step = {
        "type": step_type,
        "end_frequency": end_freq,
        "duration": duration,
    }

    # Only include start_frequency in the output dictionary if it was explicitly
    # set in the original step or if it couldn't be implied (i.e., it's the first step).
    # This preserves the behavior where omitted start_frequency implies continuation.
    if "start_frequency" in step or implied_label == "":
        updated_step["start_frequency"] = start_freq

    return updated_step


def _add_fade_controls(
    step_index: int, step: Dict[str, Any], updated_step: Dict[str, Any], duration: int
) -> Dict[str, Any]:
    """Add fade in/out controls to the step."""
    # Create two columns for fade controls
    col1, col2 = st.columns(2)

    # Fade In control
    with col1:
        fade_in = st.number_input(
            "Fade In (seconds)",
            min_value=0.0,
            # Max fade-in can be half the duration to allow for fade-out
            max_value=float(duration / 2),
            value=float(step.get("fade_in_duration", 0.0)),
            step=0.5,
            format="%.1f",
            key=f"fade_in_{step_index}",
            help="Duration of volume fade-in at the start of the step.",
        )
        # Add to step dictionary only if greater than zero
        if fade_in > 0:
            updated_step["fade_in_duration"] = fade_in

    # Fade Out control
    with col2:
        fade_out = st.number_input(
            "Fade Out (seconds)",
            min_value=0.0,
            # Max fade-out can be half the duration
            max_value=float(duration / 2),
            value=float(step.get("fade_out_duration", 0.0)),
            step=0.5,
            format="%.1f",
            key=f"fade_out_{step_index}",
            help="Duration of volume fade-out at the end of the step.",
        )
        # Add to step dictionary only if greater than zero
        if fade_out > 0:
            updated_step["fade_out_duration"] = fade_out

    # Validate that sum of fades does not exceed duration
    current_fade_in = updated_step.get("fade_in_duration", 0.0)
    current_fade_out = updated_step.get("fade_out_duration", 0.0)
    if current_fade_in + current_fade_out > duration:
        st.warning(
            f"Sum of fade-in ({current_fade_in:.1f}s)"
            f" and fade-out ({current_fade_out:.1f}s)"
            f" exceeds step duration ({duration}s)."
            " Fades might overlap or be truncated."
        )

    return updated_step


def ui_step_editor(
    step_index: int,
    step: Dict[str, Any],
    all_steps: List[Dict[str, Any]],
    on_delete=None,
) -> Dict[str, Any]:
    """UI component for editing a single audio step."""
    # Determine if the start frequency is implied from the previous step
    implied_start_freq = _get_implied_start_frequency(step_index, step, all_steps)

    # Use an expander to contain the controls for each step
    with st.expander(
        f"Step {step_index + 1}: {step.get('type', 'stable')} -"
        f" {format_time(step.get('duration', 0))}",
        expanded=True,
    ):
        # Layout columns for step type/duration and frequency controls
        col1, col2 = st.columns(2)

        # Left column: Step type and duration
        with col1:
            # Select box for step type (stable or transition)
            step_type = st.selectbox(
                "Step Type",
                STEP_TYPES,
                index=STEP_TYPES.index(step.get("type", "stable")),
                key=f"step_type_{step_index}",
                help="'stable' holds a frequency, "
                "'transition' changes between frequencies.",
            )

            # Number input for step duration
            duration_value = int(step.get("duration", DEFAULT_STEP_DURATION))
            duration = st.number_input(
                "Duration (seconds)",
                min_value=1,
                value=duration_value,
                key=f"duration_{step_index}",
                help="Duration of this audio segment in seconds.",
            )

        # Right column: Frequency controls (depend on step type)
        with col2:
            if step_type == "stable":
                # Handle controls for stable frequency step
                updated_step = _handle_stable_step(
                    step_index, step, step_type, duration
                )
            else:  # transition
                # Handle controls for transition frequency step
                updated_step = _handle_transition_step(
                    step_index, step, step_type, duration, implied_start_freq
                )

        # Add fade controls below frequency controls
        updated_step = _add_fade_controls(step_index, step, updated_step, duration)

        # Add delete button if callback provided
        if on_delete:
            if st.button("Delete Step", key=f"delete_step_{step_index}"):
                # Call the provided delete function with the step index
                on_delete(step_index)
                # Important: Rerun to reflect the deletion immediately
                st.rerun()

    # Return the potentially modified step dictionary
    return updated_step


def _initialize_session_state():
    """Initialize Streamlit session state variables if they don't exist."""
    # Initialize main configuration dictionary
    if "config" not in st.session_state:
        st.session_state.config = {
            "base_frequency": DEFAULT_BASE_FREQUENCY,
            "sample_rate": DEFAULT_SAMPLE_RATE,
            "output_filename": "audio/my_session.flac",
            "background_noise": {"type": "none", "amplitude": 0.0},
            "steps": [],
        }
    # Initialize placeholder for audio preview data
    if "audio_preview" not in st.session_state:
        st.session_state.audio_preview = None
    # Initialize placeholder for generated full audio data
    if "generated_audio" not in st.session_state:
        st.session_state.generated_audio = None


def _handle_example_loading():
    """Handle the loading of example configurations in the sidebar."""
    # Select box for choosing an example or 'Custom'
    example_name = st.selectbox(
        "Load Example Configuration", ["Custom"] + list(EXAMPLE_CONFIGS.keys())
    )

    # If an example is selected (not 'Custom')
    if example_name != "Custom":
        # Button to trigger loading the selected example
        if st.button("Load Example"):
            config_file = EXAMPLE_CONFIGS[example_name]
            loaded_config = load_config_file(config_file)
            # If loading was successful
            if loaded_config:
                # Update session state with the loaded config
                st.session_state.config = loaded_config
                st.success(f"Loaded configuration: {example_name}")
                # Reset audio preview and generated audio states
                st.session_state.audio_preview = None
                st.session_state.generated_audio = None
                # Rerun the app to reflect the loaded configuration
                st.rerun()


def _render_global_settings():
    """Render global settings controls in the sidebar."""
    st.subheader("Global Settings")
    # Input for base carrier frequency
    st.session_state.config["base_frequency"] = st.number_input(
        "Base Carrier Frequency (Hz)",
        min_value=50,
        max_value=500,
        value=st.session_state.config.get("base_frequency", DEFAULT_BASE_FREQUENCY),
        help="The base frequency used for both channels. "
        "The binaural beat is created by the difference between the two channels.",
    )


def _render_noise_settings():
    """Render background noise settings controls in the sidebar."""
    st.subheader("Background Noise")
    # Use the globally defined NOISE_TYPES
    noise_options = NOISE_TYPES

    # Determine the current noise type and its index safely
    current_noise = st.session_state.config.get("background_noise", {}).get(
        "type", "none"
    )
    try:
        index = noise_options.index(current_noise)
    except ValueError:
        index = 0  # Default to 'none' if current type is invalid or not found

    # Select box for noise type
    noise_type = st.selectbox("Noise Type", noise_options, index=index)

    # Slider for noise amplitude (only shown if noise type is not 'none')
    noise_amplitude = 0.0
    if noise_type != "none":
        default_amplitude = st.session_state.config.get("background_noise", {}).get(
            "amplitude", 0.0
        )
        noise_amplitude = st.slider(
            "Noise Amplitude",
            min_value=0.0,
            max_value=1.0,
            value=float(default_amplitude),  # Ensure value is float
            step=0.01,
            help="Relative volume of the background noise (0.0=silent, 1.0=max).",
        )

    # Update noise settings in session state config
    st.session_state.config["background_noise"] = {
        "type": noise_type,
        "amplitude": noise_amplitude,
    }


def _render_output_settings():
    """Render output file settings controls in the sidebar."""
    st.subheader("Output Settings")
    # Radio buttons for choosing audio format (FLAC or WAV)
    current_filename = st.session_state.config.get("output_filename", "")
    current_format_index = 0 if current_filename.lower().endswith(".flac") else 1
    output_format = st.radio(
        "Audio Format", ["FLAC", "WAV"], index=current_format_index
    )

    # Text input for the base filename (without extension)
    default_basename = "my_session"
    if current_filename:
        default_basename = os.path.splitext(os.path.basename(current_filename))[0]

    output_filename_base = st.text_input("Output Filename", value=default_basename)

    # Update the full output filename in config based on format and base name
    extension = ".flac" if output_format == "FLAC" else ".wav"
    # Prepend 'audio/' directory
    st.session_state.config["output_filename"] = (
        f"audio/{output_filename_base}{extension}"
    )


def _render_brainwave_info():
    """Render the brainwave information expander in the sidebar."""
    with st.expander("Brainwave States Information"):
        for wave, description in BRAINWAVE_PRESETS.items():
            st.markdown(f"**{wave}**: {description}")


def _render_repo_info():
    """Render the repository information in the sidebar."""
    st.markdown("## Repository Information")
    st.markdown(
        f"Binaural Beat Generator project is licensed under {LICENSE}. "
        f"Find the source code and contribute on [GitHub]({GITHUB_URL})."
    )
    st.markdown(f"Copyright © 2025 [{AUTHOR_NAME}](mailto:{AUTHOR_EMAIL}) ")


def _render_sidebar():
    """Render the entire sidebar content."""
    with st.sidebar:
        st.header("Templates & Settings")
        _handle_example_loading()
        st.divider()
        _render_global_settings()
        _render_noise_settings()
        _render_output_settings()
        st.divider()
        _render_brainwave_info()
        _render_repo_info()


def _render_frequency_preset_selector():
    """Render the controls for adding a step from frequency presets."""
    preset_col1, preset_col2 = st.columns(2)
    with preset_col1:
        # Select box for choosing a brainwave preset category
        selected_preset = st.selectbox(
            "Add from Frequency Preset",
            list(FREQUENCY_PRESETS.keys()),
            index=2,  # Default to Alpha
            help="Quickly add a stable frequency step from common brainwave ranges.",
        )

    with preset_col2:
        # Select box for choosing a specific frequency within the selected preset
        selected_freq = st.selectbox(
            f"Select {selected_preset} Frequency (Hz)",
            FREQUENCY_PRESETS[selected_preset],
        )

        # Button to add the selected frequency as a new stable step
        if st.button("Add Frequency"):
            new_step = {
                "type": "stable",
                "frequency": selected_freq,
                "duration": DEFAULT_STEP_DURATION,
            }
            st.session_state.config["steps"].append(new_step)
            # Reset audio previews since the configuration changed
            st.session_state.audio_preview = None
            st.session_state.generated_audio = None
            st.rerun()  # Rerun to show the new step


def _display_total_duration():
    """Calculate and display the total duration of the sequence."""
    # Sum durations of all steps in the config
    total_duration = sum(
        step.get("duration", 0) for step in st.session_state.config["steps"]
    )
    # Format and display the total duration
    minutes, seconds = divmod(total_duration, 60)
    st.info(f"Total Duration: {int(minutes)} minutes, {int(seconds)} seconds")


def _render_step_editors():
    """Render the editor UI for each step in the sequence."""

    # Function to handle step deletion
    def delete_step(index):
        # Remove step at the given index
        st.session_state.config["steps"].pop(index)
        # Reset audio previews
        st.session_state.audio_preview = None
        st.session_state.generated_audio = None
        # No need to rerun here, ui_step_editor handles it

    updated_steps = []
    steps_list = st.session_state.config["steps"]
    # Iterate through each step and render its editor
    for i, step in enumerate(steps_list):
        # Pass the current step, its index, all steps (for context),
        # and the delete callback
        updated_step = ui_step_editor(
            i, step, all_steps=steps_list, on_delete=delete_step
        )
        updated_steps.append(updated_step)

    # Update the session state with potentially modified steps
    st.session_state.config["steps"] = updated_steps


def _render_add_step_buttons():
    """Render buttons for adding new empty steps or transition steps."""
    # Button to add a new, default stable step
    if st.button("Add Empty Step"):
        new_step = {
            "type": "stable",
            "frequency": 10.0,  # Default to alpha
            "duration": DEFAULT_STEP_DURATION,
        }
        st.session_state.config["steps"].append(new_step)
        # Reset previews and rerun
        st.session_state.audio_preview = None
        st.session_state.generated_audio = None
        st.rerun()

    # Add transition step functionality (only if there's at least one step already)
    if len(st.session_state.config["steps"]) >= 1:
        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            # Input for the duration of the transition
            transition_duration = st.number_input(
                "Transition Duration (seconds)", min_value=1, value=120, step=10
            )

        with col2:
            # Button to add a transition step
            if st.button("Add Transition to Next Frequency"):
                current_steps = st.session_state.config["steps"]
                last_freq = (
                    10.0  # Default start if no steps exist (though button is hidden)
                )
                if current_steps:
                    try:
                        # Determine the end frequency of the last step correctly
                        processed_steps = prepare_audio_steps(current_steps)
                        last_freq = processed_steps[-1].freq.end
                    except BinauralError:
                        st.warning(
                            "Could not determine last frequency, defaulting to 10Hz."
                        )
                        last_freq = 10.0

                # Simple logic for default next frequency
                next_freq = 4.0 if last_freq > 7.0 else 10.0

                # Create the new transition step
                new_step = {
                    "type": "transition",
                    # No start_frequency needed here, it will be implied
                    "end_frequency": next_freq,
                    "duration": transition_duration,
                }

                st.session_state.config["steps"].append(new_step)
                # Reset previews and rerun
                st.session_state.audio_preview = None
                st.session_state.generated_audio = None
                st.rerun()


def _render_main_content():
    """Render the main content area (step configuration)."""
    st.header("Audio Sequence")
    _render_frequency_preset_selector()
    _display_total_duration()
    _render_step_editors()
    _render_add_step_buttons()


def _handle_preview_generation():
    """Handle the 'Generate Preview' button click and audio generation."""
    if st.button("Generate Preview (30s)"):
        # Show spinner while generating
        with st.spinner("Generating audio preview..."):
            # Call the preview generation function
            preview_result = generate_preview_audio(st.session_state.config)

            if preview_result:
                left, right, _ = preview_result
                # Combine channels for stereo playback
                stereo_data = np.column_stack((left, right))
                sample_rate = st.session_state.config.get(
                    "sample_rate", DEFAULT_SAMPLE_RATE
                )

                # Save audio to an in-memory buffer
                buffer = io.BytesIO()
                sf.write(buffer, stereo_data, sample_rate, format="WAV")
                buffer.seek(0)
                # Store buffer in session state for the audio player
                st.session_state.audio_preview = buffer
            else:
                # Clear preview if generation failed
                st.session_state.audio_preview = None


def _handle_full_audio_generation():
    """Handle the 'Generate Full Audio' button click and audio generation/saving."""
    if st.button("Generate Full Audio"):
        # Check if there are any steps defined
        if not st.session_state.config["steps"]:
            st.error("Please add at least one step before generating.")
            return

        # Show spinner during generation
        with st.spinner("Generating full audio..."):
            try:
                # Extract parameters from config
                sample_rate = st.session_state.config.get(
                    "sample_rate", DEFAULT_SAMPLE_RATE
                )
                base_freq = st.session_state.config.get(
                    "base_frequency", DEFAULT_BASE_FREQUENCY
                )
                steps = st.session_state.config.get("steps", [])
                noise_config = _extract_noise_config(st.session_state.config)

                # Generate the full audio sequence
                left_channel, right_channel, total_duration = generate_audio_sequence(
                    sample_rate=sample_rate,
                    base_freq=base_freq,
                    steps=steps,
                    noise_config=noise_config,
                )

                # Determine output format and temporary file suffix
                output_filename = st.session_state.config["output_filename"]
                suffix = (
                    ".flac" if output_filename.lower().endswith(".flac") else ".wav"
                )

                # Use a temporary file to save the audio
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp_name = tmp.name
                    # Save audio to the temp file
                    save_audio_file(
                        filename=tmp_name,
                        sample_rate=sample_rate,
                        left=left_channel,
                        right=right_channel,
                        total_duration_sec=total_duration,
                    )

                # Read the saved audio data back from the temp file
                with open(tmp_name, "rb") as f:
                    audio_data = f.read()

                # Store data and metadata in session state for download
                st.session_state.generated_audio = {
                    "data": audio_data,
                    "filename": os.path.basename(output_filename),
                    "duration": total_duration,
                }

                # Clean up the temporary file
                try:
                    os.unlink(tmp_name)
                except OSError as unlink_err:
                    st.warning(
                        f"Could not delete temporary file {tmp_name}: {unlink_err}"
                    )

                st.success(
                    "Audio generated successfully! Duration:"
                    f" {format_time(total_duration)}"
                )

            # Handle potential errors during generation/saving
            except BinauralError as e:
                st.error(f"Error generating audio: {e}")
            except (ValueError, TypeError) as e:
                st.error(f"Configuration error: {e}")
            except (IOError, OSError) as e:
                st.error(f"File operation error: {e}")
            finally:
                # Ensure generated_audio is cleared if an error occurred
                if "generated_audio" not in st.session_state:
                    st.session_state.generated_audio = None


def _render_preview_generate():
    """Render the Preview & Generate section with buttons."""
    st.header("Preview & Generate")
    col1, col2 = st.columns(2)
    with col1:
        _handle_preview_generation()
    with col2:
        _handle_full_audio_generation()


def _display_audio_players():
    """Display audio players for preview and full generated audio if available."""
    # Display preview audio player
    if st.session_state.audio_preview:
        st.subheader("Audio Preview")
        st.audio(st.session_state.audio_preview, format="audio/wav")

    # Display download button and info for full generated audio
    if st.session_state.generated_audio:
        st.subheader("Download Generated Audio")
        generated_info = st.session_state.generated_audio
        mime_type = (
            "audio/flac"
            if generated_info["filename"].lower().endswith(".flac")
            else "audio/wav"
        )
        # Download button
        st.download_button(
            label=f"Download {generated_info['filename']}",
            data=generated_info["data"],
            file_name=generated_info["filename"],
            mime=mime_type,
        )
        # Display duration info
        duration_min, duration_sec = divmod(generated_info["duration"], 60)
        st.info(
            f"Audio duration: {int(duration_min)} minutes, {duration_sec:.1f} seconds"
        )


def _display_yaml_config():
    """Display the current configuration as YAML and provide a download button."""
    with st.expander("View/Download YAML Configuration"):
        # Create a clean copy of the config for display
        display_config = st.session_state.config.copy()

        # Ensure background_noise is present for ordering, even if default
        if "background_noise" not in display_config:
            display_config["background_noise"] = {"type": "none", "amplitude": 0.0}

        # Define the desired order of keys
        key_order = [
            "base_frequency",
            "sample_rate",
            "output_filename",
            "background_noise",
            "steps",
        ]

        # Create an ordered dictionary for YAML output
        ordered_config = {}
        for key in key_order:
            if key in display_config:
                # Special handling for background noise: only include if not default
                if key == "background_noise":
                    bg_noise = display_config[key]
                    if (
                        bg_noise.get("type", "none") != "none"
                        or bg_noise.get("amplitude", 0.0) > 0
                    ):
                        ordered_config[key] = bg_noise
                else:
                    ordered_config[key] = display_config[key]

        # Add any other keys not in the defined order (e.g., custom keys if any)
        for key, value in display_config.items():
            if (
                key not in ordered_config and key != "noise_config"
            ):  # Exclude internal object
                ordered_config[key] = value

        # Convert the ordered config to YAML text
        try:
            yaml_text = yaml.dump(
                ordered_config, default_flow_style=False, sort_keys=False
            )
        except yaml.YAMLError as e:
            st.error(f"Error generating YAML: {e}")
            yaml_text = "# Error generating YAML configuration"

        # Display the YAML in a code block
        st.code(yaml_text, language="yaml")

        # Add download button for the YAML configuration
        st.download_button(
            label="Download YAML Configuration",
            data=yaml_text,
            file_name="binaural_config.yaml",
            mime="application/x-yaml",
        )


def main():
    """Main Streamlit application entry point."""
    # Configure Streamlit page settings
    st.set_page_config(
        page_title="Binaural Beat Generator", page_icon="🔊", layout="wide"
    )

    sine_svg = r"""
<svg width="100%" height="50" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <pattern id="sinePattern" patternUnits="userSpaceOnUse" width="200" height="50">
      <!-- One cycle of a sine wave -->
      <path d="M0,25 Q50,0 100,25 T200,25" fill="none" stroke="#000" stroke-width="2"/>
    </pattern>
  </defs>
  <!-- Fill a rectangle with the repeating pattern -->
  <rect width="100%" height="50" fill="url(#sinePattern)"/>
</svg>
"""
    st.markdown(sine_svg, unsafe_allow_html=True)

    # Set the main title and introductory markdown
    st.title("Binaural Beat Generator")
    st.markdown(
        """
    Create custom binaural beat audio for meditation, focus, relaxation, and more.
    Configure your audio sequence below and download the result.
    """
    )

    # Initialize session state variables
    _initialize_session_state()

    # Render the sidebar with settings and examples
    _render_sidebar()

    # Render the main content area for step configuration
    _render_main_content()

    # Render the section for previewing and generating audio
    _render_preview_generate()

    # Display audio players for preview/generated audio if available
    _display_audio_players()

    # Display the current configuration in YAML format
    _display_yaml_config()


if __name__ == "__main__":
    main()
