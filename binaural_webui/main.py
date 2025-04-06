"""Main entry point for the Binaural Beat Generator web UI."""

from typing import Any

import streamlit as st

from binaural.noise import NoiseFactory
from binaural_webui.components.audio_handlers import (
    display_audio_players,
    handle_full_audio_generation,
    handle_preview_generation,
)
from binaural_webui.components.sidebar import render_sidebar
from binaural_webui.components.step_editor import ui_step_editor
from binaural_webui.components.ui_utils import (
    display_total_duration,
    display_yaml_config,
    initialize_session_state,
    render_add_step_buttons,
    render_frequency_preset_selector,
)


def render_main_content(config: dict[str, Any]) -> None:
    """Render the main content area (step configuration)."""
    st.header("Audio Sequence")
    render_frequency_preset_selector(config)
    display_total_duration(config)

    # Render step editors
    def delete_step(index):
        config["steps"].pop(index)
        st.session_state.audio_preview = None
        st.session_state.generated_audio = None

    updated_steps = []
    for i, step in enumerate(config["steps"]):
        updated_step = ui_step_editor(
            i, step, all_steps=config["steps"], on_delete=delete_step
        )
        updated_steps.append(updated_step)
    config["steps"] = updated_steps

    render_add_step_buttons(config)


def render_preview_generate(config: dict[str, Any]) -> None:
    """Render the Preview & Generate section with buttons."""
    st.header("Preview & Generate")
    col1, col2 = st.columns(2)
    with col1:
        audio_preview = handle_preview_generation(config)
        if audio_preview:
            st.session_state.audio_preview = audio_preview
    with col2:
        generated_audio = handle_full_audio_generation(config)
        if generated_audio:
            st.session_state.generated_audio = generated_audio


def main():
    """Main Streamlit application entry point."""
    # Initialize session state and get config
    config = initialize_session_state()
    # Use the title from config if available for the page title
    st.set_page_config(
        page_title=f"Binaural - {config.get('title', 'Beat Generator')}",
        page_icon="🔊",
        layout="wide",
    )

    sine_svg = r"""
    <svg width="100%" height="50" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <pattern id="sinePattern" patternUnits="userSpaceOnUse" width="200" height="50">
          <path d="M0,25 Q50,0 100,25 T200,25" fill="none" stroke="#000" stroke-width="2"/>
        </pattern>
      </defs>
      <rect width="100%" height="50" fill="url(#sinePattern)"/>
    </svg>
    """
    st.markdown(sine_svg, unsafe_allow_html=True)

    # Display custom title based on configuration
    st.title(f"{config.get('title', 'Beat Generator')}")
    st.markdown(
        """
        Create custom binaural beat audio for meditation, focus, relaxation, and more.
        Configure your audio sequence below and download the result.
        """
    )

    # Get available noise types
    all_noise_types = NoiseFactory.strategies()
    if "none" in all_noise_types:
        all_noise_types.remove("none")
        noise_types = ["none"] + sorted(all_noise_types)
    else:
        noise_types = sorted(all_noise_types)

    # Render the UI components
    render_sidebar(config, noise_types)
    render_main_content(config)
    render_preview_generate(config)

    # Display audio players and YAML config
    display_audio_players(
        st.session_state.get("audio_preview"), st.session_state.get("generated_audio")
    )
    display_yaml_config(config)


if __name__ == "__main__":
    main()
