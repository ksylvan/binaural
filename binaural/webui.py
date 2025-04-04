"""Streamlit web UI for the Binaural Beat Generator."""

import io
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import streamlit as st
import yaml

from binaural.constants import DEFAULT_BASE_FREQUENCY, DEFAULT_SAMPLE_RATE
from binaural.data_types import NoiseConfig
from binaural.exceptions import BinauralError
from binaural.noise import NoiseFactory
from binaural.parallel import prepare_audio_steps
from binaural.tone_generator import generate_audio_sequence, save_audio_file

# Get available noise types from the NoiseFactory
NOISE_TYPES = NoiseFactory.strategies()

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
    "Lucid Dreaming": "scripts/lucid_dreaming.yaml",
    "Lucid Dreaming (Pink Noise)": "scripts/lucid_dream_pink_noise.yaml",
    "Migraine Relief": "scripts/migraine_relief.yaml",
}


def load_config_file(file_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

            # Convert noise settings to the expected format
            # This ensures compatibility between the core library and the web UI
            if "background_noise" in config:
                # Keep background_noise for the UI display
                bg_noise = config["background_noise"]
                if not isinstance(bg_noise, dict):
                    bg_noise = {"type": "none", "amplitude": 0.0}
            else:
                # If no background_noise in the loaded config, add default
                config["background_noise"] = {"type": "none", "amplitude": 0.0}

            return config
    except (FileNotFoundError, PermissionError, yaml.YAMLError, IOError) as e:
        st.error(f"Error loading configuration: {e}")
        return {}


def get_config_steps(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract steps from configuration dictionary."""
    if "steps" in config and isinstance(config["steps"], list):
        return config["steps"]
    return []


def format_time(seconds: int) -> str:
    """Format seconds as mm:ss."""
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes):02d}:{int(seconds):02d}"


def _extract_noise_config(config: Dict[str, Any]) -> NoiseConfig:
    """Extract noise configuration from the given config."""
    noise_dict = config.get("background_noise", {"type": "none", "amplitude": 0.0})
    return NoiseConfig(
        type=noise_dict.get("type", "none"),
        amplitude=noise_dict.get("amplitude", 0.0),
    )


def generate_preview_audio(
    config: Dict[str, Any], duration: int = 30
) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    """Generate a preview of the audio based on the current configuration."""
    try:
        # Extract parameters from the config
        sample_rate = config.get("sample_rate", DEFAULT_SAMPLE_RATE)
        base_freq = config.get("base_frequency", DEFAULT_BASE_FREQUENCY)
        steps = config.get("steps", [])

        # Extract noise configuration
        noise_config = _extract_noise_config(config)

        # Create shortened steps for preview
        preview_steps = []
        total_original_duration = sum(step.get("duration", 0) for step in steps)

        if total_original_duration <= 0:
            return None

        ratio = min(1.0, duration / total_original_duration)

        for step in steps:
            preview_step = step.copy()
            preview_step["duration"] = max(1, int(step.get("duration", 0) * ratio))

            # Scale fade durations proportionally
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

        # Generate the audio
        left_channel, right_channel, total_duration = generate_audio_sequence(
            sample_rate=sample_rate,
            base_freq=base_freq,
            steps=preview_steps,
            noise_config=noise_config,
        )

        return left_channel, right_channel, total_duration

    except BinauralError as e:
        st.error(f"Error generating preview: {e}")
        return None
    except (ValueError, KeyError, TypeError) as e:
        st.error(f"Configuration error: {e}")
        return None
    except OSError as e:
        st.error(f"I/O error: {e}")
        return None


def _get_implied_start_frequency(
    step_index: int, step: Dict[str, Any], all_steps: List[Dict[str, Any]]
) -> Optional[float]:
    """Get the implied start frequency for transition steps."""
    if (
        step.get("type") == "transition"
        and "start_frequency" not in step
        and step_index > 0
    ):
        try:
            current_steps = all_steps[: step_index + 1]
            processed_steps = prepare_audio_steps(current_steps)
            return processed_steps[-1].freq.start
        except (IndexError, AttributeError, ValueError, BinauralError):
            return None
    return None


def _handle_stable_step(
    step_index: int, step: Dict[str, Any], step_type: str, duration: int
) -> Dict[str, Any]:
    """Handle editing for stable frequency step type."""
    freq_value = float(step.get("frequency", 10.0))
    frequency = st.number_input(
        "Frequency (Hz)",
        min_value=0.1,
        max_value=100.0,
        value=freq_value,
        step=0.1,
        format="%.1f",
        key=f"frequency_{step_index}",
    )

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

    if implied_start_freq is not None:
        start_freq_value = float(implied_start_freq)
        implied_label = " (implied)"
    else:
        start_freq_value = float(step.get("start_frequency", 10.0))

    start_freq = st.number_input(
        f"Start Frequency (Hz){implied_label}",
        min_value=0.1,
        max_value=100.0,
        value=start_freq_value,
        step=0.1,
        format="%.1f",
        key=f"start_freq_{step_index}",
    )

    end_freq_value = float(step.get("end_frequency", 4.0))
    end_freq = st.number_input(
        "End Frequency (Hz)",
        min_value=0.1,
        max_value=100.0,
        value=end_freq_value,
        step=0.1,
        format="%.1f",
        key=f"end_freq_{step_index}",
    )

    updated_step = {
        "type": step_type,
        "end_frequency": end_freq,
        "duration": duration,
    }

    # Only include start_frequency if it was in the original step
    # This preserves the implied frequency behavior in YAML
    if "start_frequency" in step or implied_label == "":
        updated_step["start_frequency"] = start_freq

    return updated_step


def _add_fade_controls(
    step_index: int, step: Dict[str, Any], updated_step: Dict[str, Any], duration: int
) -> Dict[str, Any]:
    """Add fade in/out controls to the step."""
    col1, col2 = st.columns(2)

    with col1:
        fade_in = st.number_input(
            "Fade In (seconds)",
            min_value=0.0,
            max_value=float(duration / 2),
            value=float(step.get("fade_in_duration", 0.0)),
            step=0.5,
            format="%.1f",
            key=f"fade_in_{step_index}",
        )

        if fade_in > 0:
            updated_step["fade_in_duration"] = fade_in

    with col2:
        fade_out = st.number_input(
            "Fade Out (seconds)",
            min_value=0.0,
            max_value=float(duration / 2),
            value=float(step.get("fade_out_duration", 0.0)),
            step=0.5,
            format="%.1f",
            key=f"fade_out_{step_index}",
        )

        if fade_out > 0:
            updated_step["fade_out_duration"] = fade_out

    return updated_step


def ui_step_editor(
    step_index: int,
    step: Dict[str, Any],
    all_steps: List[Dict[str, Any]],
    on_delete=None,
) -> Dict[str, Any]:
    """UI component for editing a single audio step."""
    implied_start_freq = _get_implied_start_frequency(step_index, step, all_steps)

    with st.expander(
        f"Step {step_index + 1}: {step.get('type', 'stable')} -"
        f" {format_time(step.get('duration', 0))}",
        expanded=True,
    ):
        col1, col2 = st.columns(2)

        with col1:
            step_type = st.selectbox(
                "Step Type",
                STEP_TYPES,
                index=STEP_TYPES.index(step.get("type", "stable")),
                key=f"step_type_{step_index}",
            )

            duration_value = int(step.get("duration", DEFAULT_STEP_DURATION))
            duration = st.number_input(
                "Duration (seconds)",
                min_value=1,
                value=duration_value,
                key=f"duration_{step_index}",
            )

        with col2:
            if step_type == "stable":
                updated_step = _handle_stable_step(
                    step_index, step, step_type, duration
                )
            else:  # transition
                updated_step = _handle_transition_step(
                    step_index, step, step_type, duration, implied_start_freq
                )

        # Add fade controls
        updated_step = _add_fade_controls(step_index, step, updated_step, duration)

        # Delete button
        if on_delete:
            if st.button("Delete Step", key=f"delete_step_{step_index}"):
                on_delete(step_index)

    return updated_step


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Binaural Beat Generator", page_icon="🔊", layout="wide"
    )

    st.title("Binaural Beat Generator")
    st.markdown(
        """
    Create custom binaural beat audio for meditation, focus, relaxation, and more.
    Configure your audio sequence below and download the result.
    """
    )

    # Initialize session state
    if "config" not in st.session_state:
        st.session_state.config = {
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

    # Sidebar for loading examples and global settings
    with st.sidebar:
        st.header("Templates & Settings")

        example_name = st.selectbox(
            "Load Example Configuration", ["Custom"] + list(EXAMPLE_CONFIGS.keys())
        )

        if example_name != "Custom":
            if st.button("Load Example"):
                config_file = EXAMPLE_CONFIGS[example_name]
                loaded_config = load_config_file(config_file)
                if loaded_config:
                    st.session_state.config = loaded_config
                    st.success(f"Loaded configuration: {example_name}")
                    # Reset the audio preview and generated audio
                    st.session_state.audio_preview = None
                    st.session_state.generated_audio = None
                    st.rerun()

        st.divider()

        # Global settings
        st.subheader("Global Settings")
        st.session_state.config["base_frequency"] = st.number_input(
            "Base Carrier Frequency (Hz)",
            min_value=50,
            max_value=500,
            value=st.session_state.config.get("base_frequency", DEFAULT_BASE_FREQUENCY),
            help="The base frequency used for both channels. "
            "The binaural beat is created by the difference between the two channels.",
        )

        # Background noise settings
        st.subheader("Background Noise")
        # Get all noise types and make sure 'none' is first in the list
        # if not already included
        noise_options = NOISE_TYPES if "none" in NOISE_TYPES else ["none"] + NOISE_TYPES

        # Find the index of the current noise type, with safe default
        try:
            # Handle different possible structures of the config
            if "background_noise" in st.session_state.config:
                if isinstance(st.session_state.config["background_noise"], dict):
                    current_noise = st.session_state.config["background_noise"].get(
                        "type", "none"
                    )
                else:
                    current_noise = "none"
            else:
                current_noise = "none"

            # Add background_noise to config if it doesn't exist
            if "background_noise" not in st.session_state.config:
                st.session_state.config["background_noise"] = {
                    "type": "none",
                    "amplitude": 0.0,
                }

            # Get the index
            index = (
                noise_options.index(current_noise)
                if current_noise in noise_options
                else 0
            )
        except (ValueError, KeyError):
            current_noise = "none"
            index = 0

        noise_type = st.selectbox("Noise Type", noise_options, index=index)

        noise_amplitude = 0.0
        if noise_type != "none":
            # Get amplitude with safe default
            try:
                default_amplitude = 0.0
                if "background_noise" in st.session_state.config and isinstance(
                    st.session_state.config["background_noise"], dict
                ):
                    default_amplitude = st.session_state.config["background_noise"].get(
                        "amplitude", 0.0
                    )

                noise_amplitude = st.slider(
                    "Noise Amplitude",
                    min_value=0.0,
                    max_value=1.0,
                    value=default_amplitude,
                    step=0.01,
                    help="The relative volume of the background noise. "
                    "0.0 is silent, 1.0 is maximum.",
                )
            except (KeyError, TypeError):
                noise_amplitude = st.slider(
                    "Noise Amplitude",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.01,
                    help="The relative volume of the background noise. "
                    "0.0 is silent, 1.0 is maximum.",
                )

        st.session_state.config["background_noise"] = {
            "type": noise_type,
            "amplitude": noise_amplitude,
        }

        # Output settings
        st.subheader("Output Settings")
        output_format = st.radio(
            "Audio Format",
            ["FLAC", "WAV"],
            index=(
                0
                if st.session_state.config.get("output_filename", "")
                .lower()
                .endswith(".flac")
                else 1
            ),
        )

        output_filename = st.text_input(
            "Output Filename",
            value=(
                "my_session"
                if "output_filename" not in st.session_state.config
                else os.path.splitext(
                    os.path.basename(st.session_state.config["output_filename"])
                )[0]
            ),
        )

        # Update output filename with the correct extension
        extension = ".flac" if output_format == "FLAC" else ".wav"
        st.session_state.config["output_filename"] = (
            f"audio/{output_filename}{extension}"
        )

        st.divider()

        # Information about brainwave states
        with st.expander("Brainwave States Information"):
            for wave, description in BRAINWAVE_PRESETS.items():
                st.markdown(f"**{wave}**: {description}")

    # Main content - Step configuration
    st.header("Audio Sequence")

    # Quick frequency preset selector
    preset_col1, preset_col2 = st.columns(2)
    with preset_col1:
        selected_preset = st.selectbox(
            "Add from Frequency Preset",
            list(FREQUENCY_PRESETS.keys()),
            index=2,  # Alpha by default
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
            st.session_state.config["steps"].append(new_step)
            # Reset audio preview
            st.session_state.audio_preview = None
            st.session_state.generated_audio = None

    # Display total duration
    total_duration = sum(
        step.get("duration", 0) for step in st.session_state.config["steps"]
    )
    minutes, seconds = divmod(total_duration, 60)
    st.info(f"Total Duration: {int(minutes)} minutes, {seconds} seconds")

    # Function to delete a step
    def delete_step(index):
        st.session_state.config["steps"].pop(index)
        # Reset audio preview
        st.session_state.audio_preview = None
        st.session_state.generated_audio = None
        st.rerun()

    # Display all steps with editors
    updated_steps = []
    # Pass the complete steps list to each editor
    # so it can calculate implied frequencies
    steps_list = st.session_state.config["steps"]
    for i, step in enumerate(steps_list):
        updated_step = ui_step_editor(
            i, step, all_steps=steps_list, on_delete=delete_step
        )
        updated_steps.append(updated_step)

    # Update the config with the edited steps
    st.session_state.config["steps"] = updated_steps

    # Add new step button
    if st.button("Add Empty Step"):
        new_step = {
            "type": "stable",
            "frequency": 10.0,  # Default to alpha
            "duration": DEFAULT_STEP_DURATION,
        }
        st.session_state.config["steps"].append(new_step)
        # Reset audio preview
        st.session_state.audio_preview = None
        st.session_state.generated_audio = None
        st.rerun()

    # Add transition step between frequencies
    if len(st.session_state.config["steps"]) >= 1:
        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            transition_duration = st.number_input(
                "Transition Duration (seconds)", min_value=1, value=120, step=10
            )

        with col2:
            if st.button("Add Transition to Next Frequency"):
                # Process the current step sequence to get the proper ending frequency
                # of the last step, using the same function used in audio generation
                current_steps = st.session_state.config["steps"]

                if current_steps:
                    # Use prepare_audio_steps to correctly determine ending frequency
                    processed_steps = prepare_audio_steps(current_steps)
                    last_freq = (
                        processed_steps[-1].freq.end if processed_steps else 10.0
                    )
                else:
                    # Default if there are no steps yet
                    last_freq = 10.0

                # Create a transition to the next default frequency
                # Use different target frequencies depending on current frequency
                next_freq = 4.0 if last_freq > 7.0 else 10.0

                new_step = {
                    "type": "transition",
                    # Explicitly include start_frequency to match
                    # the ending frequency of the last step
                    "start_frequency": last_freq,
                    "end_frequency": next_freq,
                    "duration": transition_duration,
                }

                st.session_state.config["steps"].append(new_step)
                # Reset audio preview
                st.session_state.audio_preview = None
                st.session_state.generated_audio = None
                st.rerun()

    # Preview and Generate section
    st.header("Preview & Generate")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Generate Preview (30s)"):
            with st.spinner("Generating audio preview..."):
                preview_result = generate_preview_audio(st.session_state.config)

                if preview_result:
                    left, right, _ = preview_result

                    # Convert to stereo audio for preview
                    stereo_data = np.column_stack((left, right))
                    sample_rate = st.session_state.config.get(
                        "sample_rate", DEFAULT_SAMPLE_RATE
                    )

                    # Save to a byte buffer
                    buffer = io.BytesIO()
                    sf.write(buffer, stereo_data, sample_rate, format="WAV")
                    buffer.seek(0)

                    st.session_state.audio_preview = buffer

    with col2:
        if st.button("Generate Full Audio"):
            if not st.session_state.config["steps"]:
                st.error("Please add at least one step before generating")
            else:
                with st.spinner("Generating full audio..."):
                    try:
                        # Extract parameters from the config
                        sample_rate = st.session_state.config.get(
                            "sample_rate", DEFAULT_SAMPLE_RATE
                        )
                        base_freq = st.session_state.config.get(
                            "base_frequency", DEFAULT_BASE_FREQUENCY
                        )
                        steps = st.session_state.config.get("steps", [])

                        # Extract noise configuration
                        noise_dict = st.session_state.config.get(
                            "background_noise", {"type": "none", "amplitude": 0.0}
                        )
                        noise_config = NoiseConfig(
                            type=noise_dict.get("type", "none"),
                            amplitude=noise_dict.get("amplitude", 0.0),
                        )

                        # Generate the audio
                        left_channel, right_channel, total_duration = (
                            generate_audio_sequence(
                                sample_rate=sample_rate,
                                base_freq=base_freq,
                                steps=steps,
                                noise_config=noise_config,
                            )
                        )

                        # Convert to stereo audio
                        stereo_data = np.column_stack((left_channel, right_channel))

                        # Create temporary file
                        suffix = (
                            ".flac"
                            if st.session_state.config["output_filename"].endswith(
                                ".flac"
                            )
                            else ".wav"
                        )
                        with tempfile.NamedTemporaryFile(
                            suffix=suffix, delete=False
                        ) as tmp:
                            # Save to the temporary file
                            save_audio_file(
                                filename=tmp.name,
                                sample_rate=sample_rate,
                                left=left_channel,
                                right=right_channel,
                                total_duration_sec=total_duration,
                            )

                            # Read the data back for downloading
                            with open(tmp.name, "rb") as f:
                                audio_data = f.read()

                            # Store for download
                            st.session_state.generated_audio = {
                                "data": audio_data,
                                "filename": os.path.basename(
                                    st.session_state.config["output_filename"]
                                ),
                                "duration": total_duration,
                            }

                        # Clean up the temporary file
                        try:
                            os.unlink(tmp.name)
                        except OSError:
                            pass  # Ignore errors when deleting temporary file

                        st.success(
                            "Audio generated successfully! "
                            f"Duration: {format_time(total_duration)}"
                        )

                    except BinauralError as e:
                        st.error(f"Error generating audio: {e}")
                    except (ValueError, TypeError) as e:
                        st.error(f"Configuration error: {e}")
                    except (IOError, OSError) as e:
                        st.error(f"File operation error: {e}")

    # Display the preview audio player if available
    if st.session_state.audio_preview:
        st.subheader("Audio Preview")
        st.audio(st.session_state.audio_preview, format="audio/wav")

    # Display the generated audio for download if available
    if st.session_state.generated_audio:
        st.subheader("Download Generated Audio")

        st.download_button(
            label=f"Download {st.session_state.generated_audio['filename']}",
            data=st.session_state.generated_audio["data"],
            file_name=st.session_state.generated_audio["filename"],
            mime=(
                "audio/flac"
                if st.session_state.generated_audio["filename"].endswith(".flac")
                else "audio/wav"
            ),
        )

        duration_min, duration_sec = divmod(
            st.session_state.generated_audio["duration"], 60
        )
        st.info(
            f"Audio duration: {int(duration_min)} minutes, {duration_sec:.1f} seconds"
        )

    # Display the current YAML configuration
    with st.expander("View YAML Configuration"):
        # Create a copy of the config to modify for display
        display_config = st.session_state.config.copy()

        # Create an ordered version of the config with fields in the desired order
        ordered_config = {}

        # First, add the standard header fields in order
        for key in ["base_frequency", "sample_rate", "output_filename"]:
            if key in display_config:
                ordered_config[key] = display_config[key]

        # Then add background_noise (if present and not default)
        bg_noise = display_config.get(
            "background_noise", {"type": "none", "amplitude": 0.0}
        )
        if bg_noise["type"] != "none" or bg_noise["amplitude"] > 0:
            ordered_config["background_noise"] = bg_noise

        # Finally add the steps
        if "steps" in display_config:
            ordered_config["steps"] = display_config["steps"]

        # Add any remaining keys that weren't explicitly ordered
        for key, value in display_config.items():
            # Skip noise_config (internal object) and background_noise (already handled)
            if (
                key not in ordered_config
                and key != "noise_config"
                and key != "background_noise"
            ):
                ordered_config[key] = value

        # Skip the noise_config object (shouldn't appear in YAML)

        yaml_text = yaml.dump(ordered_config, default_flow_style=False, sort_keys=False)
        st.code(yaml_text, language="yaml")

        if st.download_button(
            label="Download YAML Configuration",
            data=yaml_text,
            file_name="binaural_config.yaml",
            mime="application/x-yaml",
        ):
            st.success("Configuration downloaded successfully!")


if __name__ == "__main__":
    main()
