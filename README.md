# Binaural

Binaural is a Python tool that generates binaural beat audio (WAV or FLAC) designed to induce different brain wave states, configured via a simple YAML script. It can optionally mix background noise (white, pink, or brown) with the binaural beats.

## Table of Contents

- [Binaural](#binaural)
  - [Table of Contents](#table-of-contents)
  - [Description](#description)
  - [Background](#background)
    - [What Are Binaural Beats?](#what-are-binaural-beats)
    - [Background Noise Types](#background-noise-types)
      - [Standard Noise Types](#standard-noise-types)
      - [Advanced Noise Types](#advanced-noise-types)
      - [Nature Sounds](#nature-sounds)
    - [Brainwave Entrainment](#brainwave-entrainment)
    - [Brainwave States](#brainwave-states)
  - [Scientific Research](#scientific-research)
  - [Installation](#installation)
    - [Requirements](#requirements)
    - [Setup](#setup)
  - [Contributing](#contributing)
  - [Usage](#usage)
    - [Web Interface](#web-interface)
    - [Command Line Interface](#command-line-interface)
  - [YAML Script Format](#yaml-script-format)
  - [Script Library](#script-library)
    - [Standard Scripts](#standard-scripts)
    - [Advanced Scripts with Specialized Noise](#advanced-scripts-with-specialized-noise)
  - [File Structure](#file-structure)
  - [Resources](#resources)
    - [Further Reading](#further-reading)
    - [References](#references)
  - [License](#license)

## Description

This tool reads a YAML script defining a sequence of binaural beat frequencies, durations, optional volume fades, and optional background noise settings. It then creates an audio file based on that sequence. It supports output in both WAV and FLAC formats. It allows for stable frequency segments, smooth transitions between frequencies, and gradual fade-in/fade-out for each segment.

The program uses a configurable base carrier frequency (defaulting to 100 Hz) and creates stereo audio. The frequency difference between the left and right channels creates the binaural beat effect, which is intended to influence brainwave activity. Background noise, if configured, is added equally to both channels.

**Note:** All duration values (i.e., duration, fade_in_duration, and fade_out_duration) in the YAML configuration are specified in seconds.

## Background

### What Are Binaural Beats?

Binaural beats are an auditory illusion perceived when two slightly different frequencies are presented separately to each ear. The brain detects the phase difference between these frequencies and attempts to reconcile this difference, which creates the sensation of a third "beat" frequency equal to the difference between the two tones.

For example, if a 100 Hz tone is presented to the left ear and a 110 Hz tone to the right ear, the brain perceives a 10 Hz binaural beat. This perceived frequency corresponds to specific brainwave patterns.

### Background Noise Types

#### Standard Noise Types

- **White Noise**: Contains equal energy across all audible frequencies. Sounds like a hiss (e.g., static, fan).
- **Pink Noise**: Energy decreases with increasing frequency (specifically, 3dB per octave). Sounds deeper than white noise (e.g., steady rainfall, wind).
- **Brown Noise (Brownian/Red Noise)**: Energy decreases more steeply than pink noise (6dB per octave). Sounds even deeper (e.g., strong waterfall, thunder rumble).

#### Advanced Noise Types

- **Blue Noise (Azure Noise)**: Energy increases with frequency (specifically, 3dB per octave). Has more high-frequency content than white noise, creating a "brighter" sound.
- **Violet Noise (Purple Noise)**: Energy increases steeply with frequency (6dB per octave). Strong emphasis on high frequencies, creating a "sharp" or "hissing" sound.
- **Grey Noise**: White noise filtered to match the ear's frequency response. Emphasizes frequencies where human hearing is most sensitive (2-5 kHz), creating a perceptually balanced sound.

#### Nature Sounds

- **Rain**: Natural rain sound simulation that provides a calming and consistent audio backdrop. Helps mask external distractions while creating a soothing ambience.
- **Ocean**: Simulates the rhythmic sound of ocean waves, combining a low-frequency rumble with periodic wave crests. Creates a dynamic yet calming natural soundscape.

Adding background noise can help mask distracting environmental sounds or provide a constant auditory backdrop. Different noise types may be beneficial for different use cases based on their frequency characteristics.

### Brainwave Entrainment

Brainwave entrainment refers to the brain's electrical response to rhythmic sensory stimulation, such as pulses of sound or light. When the brain is presented with a stimulus with a frequency corresponding to a specific brainwave state, it tends to synchronize its electrical activity with that frequency—a process called "frequency following response."

Binaural beats are one method of achieving brainwave entrainment, potentially helping to induce specific mental states associated with different brainwave frequencies.

### Brainwave States

- **Gamma Waves (30-100 Hz)**: The fastest brainwaves, linked to high-level cognitive functions such as sensory integration, focused attention, and advanced mental processing.
Gamma activity plays a key role in binding together information from different brain regions and is often enhanced during peak concentration and certain meditative states.
- **Beta Waves (13-30 Hz)**: Alertness, concentration, active thinking, problem-solving.
  *Note*: Higher Beta (e.g., 18-30 Hz) may correlate with stress or anxiety, while lower Beta (12-15 Hz) is linked to relaxed focus.
- **Alpha Waves (8-12 Hz)**: Relaxation, calmness, light meditation, daydreaming, and passive attention (e.g., closing your eyes or mindfulness practices).
  Acts as a bridge between conscious (Beta) and subconscious (Theta) states.
- **Theta Waves (4-7 Hz)**: Deep meditation, creativity, intuition, drowsiness (stage 1 NREM sleep), and light sleep (stage 2 NREM).
- **Delta Waves (0.5-4 Hz)**: Deep, dreamless sleep (NREM stages 3-4, "slow-wave sleep"), physical healing, and regeneration. Dominant in restorative sleep, critical for immune function and memory consolidation.

*Note*: While Theta waves are present in REM sleep, they are not the dominant pattern. REM is characterized by mixed-frequency activity
(including Beta-like waves) due to heightened brain activity during dreaming. Theta is more prominent during pre-sleep relaxation and early sleep stages.

## Scientific Research

Research on binaural beats has shown mixed results, but several studies suggest potential benefits:

- **Stress Reduction**: Some studies indicate that binaural beats in the alpha frequency range may help reduce anxiety and stress ([Wahbeh et al., 2007](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5370608/))
- **Cognitive Enhancement**: Research suggests potential improvements in attention, working memory, and other cognitive functions ([Kraus & Porubanová, 2015](https://www.sciencedirect.com/science/article/abs/pii/S1053810015300593))
- **Sleep Quality**: Delta frequency binaural beats may improve sleep quality in some individuals ([Jirakittayakorn & Wongsawat, 2018](https://www.frontiersin.org/articles/10.3389/fnhum.2018.00387/full))

## Installation

### Requirements

- Python 3.12+
- Dependencies listed in `pyproject.toml` (numpy, PyYAML, soundfile, scipy).

### Setup

1. **Automatic setup** with the provided script:

    ```bash
    ./bin/setup.sh
    source .venv/bin/activate
    ```

    This installs `uv` if it's not already installed, and uses it to create the `.venv/` virtual
    environment and installs the required packages.

    > Note: If using VS Code, the workspace is configured to run the setup script automatically when opening
      the folder.

## Contributing

- Fork the repository.
- Create a feature branch (`git checkout -b feature/awesome-feature`).
- Write clear, concise code with type hints and docstrings.
- Ensure new features are tested and add appropriate unit tests.
- Ensure code passes linters (`pylint binaural tests`) and tests (`pytest`).
- Run tests:

  ```bash
  pytest
  ```

  You can also use the `--run-performance` flag to run normally skipped tests marked as `performance` in the tests. These
  are usually skipped.

- Run linters:

  ```bash
  pylint .
  ```

- Submit a pull request with a clear description of your changes.

## Usage

### Web Interface

For a more interactive experience, run the web-based user interface:

```bash
python run_webapp.py
```

This launches a Streamlit-based web interface that allows you to:

- Create and edit audio sequences through a visual interface
- Load example configurations
- Preview audio before generating the full file
- Customize background noise settings
- Download generated audio and configuration files

Once launched, open your web browser and navigate to `http://localhost:8501` to access the interface.

### Command Line Interface

Run the script with the path to a YAML configuration file:

```bash
python generate.py <path_to_script.yaml> [options]
```

**Arguments:**

- `<path_to_script.yaml>`: YAML file defining the binaural beat sequence and settings.
- `-o <output_file>`, `--output <output_file>` (Optional): Specify the output audio file path. The file extension determines the format (`.wav` or `.flac`) and overrides the `output_filename` in the YAML.
- `--verbose` (Optional): Enable verbose logging output.

**Example:**

To use the example script provided (which defaults to FLAC output):

```bash
python generate.py example_script.yaml
```

This will generate `audio/example_fade_noise.flac` (or the filename specified in `example_script.yaml`) in the `audio/` directory.

To use one of the pre-defined scripts from the library and output as WAV:

```bash
python generate.py scripts/relaxation_alpha.yaml -o audio/relaxation_alpha.wav
```

This will generate `relaxation_alpha.wav` in the `audio/` directory, overriding the default name in the script.

To generate a FLAC file with a custom name:

```bash
python generate.py scripts/focus_beta.yaml -o my_focus_session.flac
```

## YAML Script Format

The YAML script defines the parameters and sequence for audio generation.

**Global Settings (Optional):**

- `base_frequency`: The carrier frequency in Hz (e.g., 100). Default: `100`.
- `sample_rate`: The audio sample rate in Hz (e.g., 44100). Default: `44100`.
- `output_filename`: The default name for the output audio file (e.g., `"audio/my_session.flac"` or `"audio/my_session.wav"`). The extension (`.wav` or `.flac`) determines the output format. Default: `"output.flac"`.
- `background_noise` (Optional): Settings for adding background noise.
  - `type`: The type of noise. Options:
    - Standard: `"white"`, `"pink"`, `"brown"`
    - Advanced: `"blue"`, `"violet"`, `"grey"`
    - Nature: `"rain"`, `"ocean"`
    - No noise: `"none"`
    - Default: `"none"`.
  - `amplitude`: The relative amplitude (volume) of the noise, from `0.0` (silent) to `1.0` (maximum relative level). Default: `0.0`. The binaural beat signal is scaled down by `(1 - amplitude)` before mixing to prevent clipping.

**Steps (Required):**

A list under the `steps:` key, where each item defines an audio segment. Each step can be one of the following types:

- **`type: stable`**: Holds a constant binaural beat frequency.
- `frequency`: The binaural beat frequency in Hz.
- `duration`: The duration of this segment in seconds.
- `fade_in_duration` (Optional): Duration of a linear volume fade-in at the beginning of the step, in seconds. Default: `0.0`.
- `fade_out_duration` (Optional): Duration of a linear volume fade-out at the end of the step, in seconds. Default: `0.0`.

For the `transition` step type, we have the following:

- **`type: transition`**: Linearly changes the binaural beat frequency over time.
- `start_frequency`: The starting binaural beat frequency in Hz. If omitted, it uses the end frequency of the previous step for a smooth transition (cannot be omitted for the first step).
- `end_frequency`: The ending binaural beat frequency in Hz.
- `duration`: The duration of this transition in seconds.
- `fade_in_duration` (Optional): Duration of a linear volume fade-in at the beginning of the step, in seconds. Default: `0.0`.
- `fade_out_duration` (Optional): Duration of a linear volume fade-out at the end of the step, in seconds. Default: `0.0`.

**Important Notes on Fades:**

- Fades are applied *within* the specified `duration` of the step.
- The sum of `fade_in_duration` and `fade_out_duration` for a single step cannot exceed the step's `duration`.

**Example YAML (`example_script.yaml`):**

```yaml
# Example Binaural Beat Generation Script with Fades and Background Noise

# Global settings
title: Example Binaural Beat Script
base_frequency: 100 # Hz (carrier frequency)
sample_rate: 44100 # Hz (audio sample rate)
output_filename: "audio/example_fade_noise.flac" # Default output file name

# Background noise settings (optional)
background_noise:
  type: "pink" # Type of noise: "white", "pink", "brown", "blue", "violet", "grey", "rain", "ocean", or "none"
  amplitude: 0.15 # Relative amplitude (0.0 to 1.0)

# Sequence of audio generation steps (Total Duration: 1500 seconds = 25 minutes)
steps:
  # 1. Beta phase (stable 18 Hz beat) with fade-in
  - type: stable
    frequency: 18 # Hz (binaural beat frequency)
    duration: 180 # seconds (3 minutes)
    fade_in_duration: 6 # seconds

  # 2. Transition from Beta (18 Hz) to Alpha (10 Hz)
  - type: transition
    start_frequency: 18 # Hz (explicit, could be implied)
    end_frequency: 10 # Hz
    duration: 300 # seconds (5 minutes)

  # 3. Transition from Alpha (10 Hz) to Theta (6 Hz) with fades
  - type: transition
    # start_frequency: 10 (implied from previous step)
    end_frequency: 6 # Hz
    duration: 300 # seconds (5 minutes)
    fade_in_duration: 3 # seconds
    fade_out_duration: 3 # seconds

  # 4. Transition from Theta (6 Hz) to Delta (2 Hz) with fade-out
  - type: transition
    # start_frequency: 6 (implied)
    end_frequency: 2 # Hz
    duration: 420 # seconds (7 minutes)
    fade_out_duration: 12 # seconds

  # 5. Transition from Delta (2 Hz) to Gamma (40 Hz) with fades
  - type: transition
    # start_frequency: 2 (implied)
    end_frequency: 40 # Hz
    duration: 300 # seconds (5 minutes)
    fade_in_duration: 6 # seconds
    fade_out_duration: 15 # seconds
```

## Script Library

A collection of pre-defined YAML scripts for common use-cases is available in the `scripts/` directory.
Most scripts default to `.flac` output. Some include background noise as noted below.

### Standard Scripts

- **`scripts/focus_beta.yaml`**: Designed to enhance concentration and alertness using Beta waves (14-18 Hz).
- **`scripts/focus_gamma.yaml`**: Targets peak concentration and problem-solving with Gamma waves (40 Hz).
- **`scripts/relaxation_alpha.yaml`**: Aims to reduce stress and promote calmness using Alpha waves (8-10 Hz).
- **`scripts/meditation_theta.yaml`**: Facilitates deep meditation and introspection using Theta waves (6 Hz).
- **`scripts/sleep_delta.yaml`**: Guides the brain towards deep sleep states using Delta waves (2 Hz).

### Advanced Scripts with Specialized Noise

- **`scripts/focus_violet.yaml`**: Concentration enhancement with Gamma waves (40 Hz) and violet noise for heightened alertness.
- **`scripts/relaxation_grey.yaml`**: Stress reduction with Alpha waves (8-10 Hz) and perceptually balanced grey noise.
- **`scripts/relaxation_rain.yaml`**: Calming experience with Alpha waves (8-10 Hz) and natural rain sounds.
- **`scripts/relaxation_ocean.yaml`**: Deep relaxation with Alpha waves (8-10 Hz) and simulated ocean sounds.
- **`scripts/creativity_blue.yaml`**: Creative flow enhancement with Theta waves (6-7.83 Hz) and blue noise for clarity.
- **`scripts/creativity_theta.yaml`**: Intended to foster an intuitive and creative mental state using Theta waves (7 Hz).
- **`scripts/lucid_dream_pink.yaml`**: Aims to facilitate REM sleep states potentially conducive to lucid
  dreaming (90 minutes, with pink noise).
- **`scripts/lucid_dreaming.yaml`**: Aims to facilitate REM sleep states potentially conducive to lucid dreaming.
- **`scripts/migraine_relief.yaml`**: Uses specific frequencies and transitions aimed at reducing migraine symptoms.

You can use these scripts directly, modify them (e.g., add `background_noise`), or use the `-o` command-line option to change the output format/name.

Example usage for WAV output with added noise (assuming you modify the script):

```bash
# (First, edit scripts/sleep_delta.yaml to add background_noise section)
python generate.py scripts/sleep_delta.yaml -o audio/sleep_delta_with_noise.wav
```

## File Structure

- `generate.py`: Main script entry point.
- `example_script.yaml`: Example YAML script with fades and noise.
- `scripts/`: Directory containing pre-defined YAML scripts.
- `binaural/`: Source code package.
  - `__init__.py`
  - `cli.py`: Command-line interface logic.
  - `constants.py`: Default values and constants.
  - `data_types.py`: Dataclasses for configuration objects (AudioStep, NoiseConfig, etc.).
  - `exceptions.py`: Custom exception classes.
  - `fade.py`: Audio fade logic.
  - `noise.py`: Background noise generation functions.
  - `parallel.py`: Parallel processing utilities.
  - `tone_generator.py`: Core audio generation logic for beats and mixing.
  - `utils.py`: YAML loading and validation utilities.
- `binaural_webui/`: Modular web UI implementation for the Binaural Beat Generator.
  - `__init__.py`
  - `main.py`: Main Streamlit application entry point.
  - `constants.py`: Constants for UI components (brainwave presets, step types, etc.).
  - `components/`: Directory containing modular UI code.
    - `audio_handlers.py`: Audio generation & handling logic (preview and full audio clips).
    - `config_utils.py`: Configuration loading/parsing utilities.
    - `sidebar.py`: Sidebar layout and controls.
    - `step_editor.py`: Editing components for individual steps.
    - `ui_utils.py`: Common Streamlit UI helpers.
- `tests/`: Directory of unit tests.
  - `test_common.py`: Common test utilities.
  - `test_data_types.py`: data type/validation tests.
  - `test_fade.py`: fade data type tests.
  - `test_noise.py`: Tests for standard noise types.
  - `test_new_noise_types.py`: Tests for advanced/nature noise types.
  - `test_ocean_noise.py`: Specific tests for ocean noise.
  - `test_parallel.py`: Tests for parallel generation.
  - `test_property_based.py`: Hypothesis property-based tests.
  - `test_rain_noise.py`: Specific tests for rain noise.
  - `test_tone_generator.py`
  - `test_utils.py`
- `bin/setup.sh`: Setup script for development environment.
- `pyproject.toml`: Project configuration and dependencies.
- `.python-version`: Specifies Python version.
- `README.md`: This file.
- `LICENSE`: Project license information.
- `conftest.py`: Pytest configuration.
- `cspell.json`: Spell checking configuration.
- `run_webapp.py`: Script to launch the web UI.

## Resources

### Further Reading

- [The Discovery of Binaural Beats][discovery-binaural-beats]
- [Healthline - Binaural Beats: Do They Really Affect Your Brain?][healthline] - Discusses the potential cognitive and mood benefits of binaural beats
- [Sleep Foundation - Binaural Beats and Sleep][sleep-foundation] - Examines the impact of binaural beats on sleep quality
- [Binaural beats to entrain the brain? A systematic review of the effects of binaural beat stimulation][plos-one-ruth-research] - Published in 2023.

### References

- Oster, G. (1973). Auditory beats in the brain. Scientific American, 229(4), 94-102.
- Huang, T. L., & Charyton, C. (2008). A comprehensive review of the psychological effects of brainwave entrainment. Alternative Therapies in Health and Medicine, 14(5), 38-50.
- Le Scouarnec, R. P., Poirier, R. M., Owens, J. E., Gauthier, J., Taylor, A. G., & Foresman, P. A. (2001). Use of binaural beat tapes for treatment of anxiety: A pilot study. Alternative Therapies in Health and Medicine, 7(1), 58-63.
- Chaieb, L., Wilpert, E. C., Reber, T. P., & Fell, J. (2015). Auditory beat stimulation and its effects on cognition and mood states. Frontiers in Psychiatry, 6, 70.
- Wahbeh, H., Calabrese, C., & Zwickey, H. (2007). Binaural beat technology in humans: a pilot study to assess psychologic and physiologic effects. Journal of Alternative and Complementary Medicine, 13(1), 25-32.
- Kraus, J., & Porubanová, M. (2015). The effect of binaural beats on working memory capacity. Studia Psychologica, 57(2), 135-145.
- Jirakittayakorn, N., & Wongsawat, Y. (2018). A novel insight of effects of a 3-Hz binaural beat on sleep stages during sleep. Frontiers in Human Neuroscience, 12, 387.
- Stumbrys, T., Erlacher, D., & Schredl, M. (2014). Testing the potential of binaural beats to induce lucid dreams. Dreaming, 24(3), 208–217.
- Prinsloo, S., Lyle, R., & Sewell, D. (2018). Alpha-Theta Neurofeedback for Chronic Pain: A Pilot Study. Journal of Neurotherapy, 22(3), 193-211.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

Copyright (c) 2025 Kayvan Sylvan

[discovery-binaural-beats]: https://www.binauralbeatsmeditation.com/dr-gerald-oster-auditory-beats-in-the-brain/
[healthline]: https://www.healthline.com/health/binaural-beats
[sleep-foundation]: https://www.sleepfoundation.org/bedroom-environment/binaural-beats
[plos-one-ruth-research]: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0286023
