"""Step editor components for the Binaural Beat Generator."""

from typing import Any, Callable, Optional

import streamlit as st

from binaural_generator.core.exceptions import BinauralError
from binaural_generator.core.parallel import prepare_audio_steps
from binaural_generator.webui.components.config_utils import format_time
from binaural_generator.webui.constants import DEFAULT_STEP_DURATION, STEP_TYPES


def _get_implied_start_frequency(
    step_index: int, step: dict[str, Any], all_steps: list[dict[str, Any]]
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
    step_index: int, step: dict[str, Any], step_type: str, duration: int
) -> dict[str, Any]:
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
        help="The constant binaural beat frequency for this step.",
    )
    return {"type": step_type, "frequency": frequency, "duration": duration}


def _handle_transition_step(
    step_index: int,
    step: dict[str, Any],
    step_type: str,
    duration: int,
    implied_start_freq: Optional[float],
) -> dict[str, Any]:
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
        help="The binaural beat frequency at the beginning of the transition.",
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
        help="The binaural beat frequency at the end of the transition.",
    )

    updated_step = {
        "type": step_type,
        "end_frequency": end_freq,
        "duration": duration,
    }

    if "start_frequency" in step or implied_label == "":
        updated_step["start_frequency"] = start_freq

    return updated_step


def _add_fade_controls(
    step_index: int, step: dict[str, Any], updated_step: dict[str, Any], duration: int
) -> dict[str, Any]:
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
            help="Duration of volume fade-in at the start of the step.",
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
            help="Duration of volume fade-out at the end of the step.",
        )
        if fade_out > 0:
            updated_step["fade_out_duration"] = fade_out

    current_fade_in = updated_step.get("fade_in_duration", 0.0)
    current_fade_out = updated_step.get("fade_out_duration", 0.0)
    if current_fade_in + current_fade_out > duration:
        st.warning(
            f"Sum of fade-in ({current_fade_in:.1f}s) "
            f"and fade-out ({current_fade_out:.1f}s) "
            f"exceeds step duration ({duration}s)."
        )

    return updated_step


def ui_step_editor(
    step_index: int,
    step: dict[str, Any],
    all_steps: list[dict[str, Any]],
    on_delete: Optional[Callable[[int], None]] = None,
) -> dict[str, Any]:
    """UI component for editing a single audio step."""
    implied_start_freq = _get_implied_start_frequency(step_index, step, all_steps)

    with st.expander(
        f"Step {step_index + 1}: {step.get('type', 'stable')} - "
        f"{format_time(step.get('duration', 0))}",
        expanded=True,
    ):
        col1, col2 = st.columns(2)

        with col1:
            step_type = st.selectbox(
                "Step Type",
                STEP_TYPES,
                index=STEP_TYPES.index(step.get("type", "stable")),
                key=f"step_type_{step_index}",
                help="'stable' holds a frequency, "
                "'transition' moves smoothly between two frequencies.",
            )
            duration_value = int(step.get("duration", DEFAULT_STEP_DURATION))
            duration = st.number_input(
                "Duration (seconds)",
                min_value=1,
                value=duration_value,
                key=f"duration_{step_index}",
                help="Duration of this audio segment in seconds.",
            )

        with col2:
            if step_type == "stable":
                updated_step = _handle_stable_step(
                    step_index, step, step_type, duration
                )
            else:
                updated_step = _handle_transition_step(
                    step_index, step, step_type, duration, implied_start_freq
                )

        updated_step = _add_fade_controls(step_index, step, updated_step, duration)

        if on_delete and st.button("Delete Step", key=f"delete_step_{step_index}"):
            on_delete(step_index)
            st.rerun()

    return updated_step
