"""Audio generation and handling components for the Binaural Beat Generator."""

import io
import os
import tempfile
from typing import Any, Optional, Tuple

import numpy as np
import soundfile as sf
import streamlit as st

from binaural_generator.core.constants import (
    DEFAULT_BASE_FREQUENCY,
    DEFAULT_SAMPLE_RATE,
)
from binaural_generator.core.data_types import NoiseConfig
from binaural_generator.core.exceptions import BinauralError
from binaural_generator.core.tone_generator import (
    generate_audio_sequence,
    save_audio_file,
)


def _extract_noise_config(config: dict[str, Any]) -> NoiseConfig:
    """Extract noise configuration from the given config."""
    noise_dict = config.get("background_noise", {"type": "none", "amplitude": 0.0})
    return NoiseConfig(
        type=noise_dict.get("type", "none"),
        amplitude=noise_dict.get("amplitude", 0.0),
    )


def generate_preview_audio(
    config: dict[str, Any], duration: int = 30
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


def handle_preview_generation(config: dict[str, Any]) -> Optional[io.BytesIO]:
    """Handle the 'Generate Preview' button click and audio generation."""
    if st.button("Generate Preview (30s)"):
        with st.spinner("Generating audio preview..."):
            preview_result = generate_preview_audio(config)
            if preview_result:
                left, right, _ = preview_result
                stereo_data = np.column_stack((left, right))
                sample_rate = config.get("sample_rate", DEFAULT_SAMPLE_RATE)

                buffer = io.BytesIO()
                sf.write(buffer, stereo_data, sample_rate, format="WAV")
                buffer.seek(0)
                return buffer
    return None


def _create_audio_file(
    sample_rate: int,
    base_freq: float,
    steps: list[dict[str, Any]],
    noise_config: NoiseConfig,
    output_filename: str,
) -> Tuple[bytes, float]:
    """Create and save audio file, return the file data and duration."""
    left_channel, right_channel, total_duration = generate_audio_sequence(
        sample_rate=sample_rate,
        base_freq=base_freq,
        steps=steps,
        noise_config=noise_config,
    )

    suffix = ".flac" if output_filename.lower().endswith(".flac") else ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_name = tmp.name
        save_audio_file(
            filename=tmp_name,
            sample_rate=sample_rate,
            left=left_channel,
            right=right_channel,
            total_duration_sec=total_duration,
        )

    with open(tmp_name, "rb") as f:
        audio_data = f.read()

    try:
        os.unlink(tmp_name)
    except OSError as unlink_err:
        st.warning(f"Could not delete temporary file {tmp_name}: {unlink_err}")

    return audio_data, total_duration


def handle_full_audio_generation(config: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Handle the 'Generate Full Audio' button click and audio generation/saving."""
    if not st.button("Generate Full Audio"):
        return None

    if not config["steps"]:
        st.error("Please add at least one step before generating.")
        return None

    with st.spinner("Generating full audio..."):
        try:
            sample_rate = config.get("sample_rate", DEFAULT_SAMPLE_RATE)
            base_freq = config.get("base_frequency", DEFAULT_BASE_FREQUENCY)
            steps = config["steps"]
            noise_config = _extract_noise_config(config)
            output_filename = config["output_filename"]

            audio_data, total_duration = _create_audio_file(
                sample_rate, base_freq, steps, noise_config, output_filename
            )

            return {
                "data": audio_data,
                "filename": os.path.basename(output_filename),
                "duration": total_duration,
            }

        except BinauralError as e:
            st.error(f"Error generating audio: {e}")
        except (ValueError, TypeError) as e:
            st.error(f"Configuration error: {e}")
        except (IOError, OSError) as e:
            st.error(f"File operation error: {e}")

    return None


def display_audio_players(
    audio_preview: Optional[io.BytesIO], generated_audio: Optional[dict[str, Any]]
) -> None:
    """Display audio players for preview and full generated audio if available."""
    if audio_preview:
        st.subheader("Audio Preview")
        st.audio(audio_preview, format="audio/wav")

    if generated_audio:
        st.subheader("Download Generated Audio")
        mime_type = (
            "audio/flac"
            if generated_audio["filename"].lower().endswith(".flac")
            else "audio/wav"
        )
        st.download_button(
            label=f"Download {generated_audio['filename']}",
            data=generated_audio["data"],
            file_name=generated_audio["filename"],
            mime=mime_type,
        )
        duration_min, duration_sec = divmod(generated_audio["duration"], 60)
        st.info(
            f"Audio duration: {int(duration_min)} minutes, {duration_sec:.1f} seconds"
        )
