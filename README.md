# Binaural

Binaural is a Python tool that generates binaural beat audio (WAV or FLAC) designed to induce different brain wave states, configured via a simple YAML script.

## Description

This tool reads a YAML script defining a sequence of binaural beat frequencies, durations, and optional volume fades, then creates an audio file based on that sequence. It supports output in both WAV and FLAC formats. It allows for stable frequency segments, smooth transitions between frequencies, and gradual fade-in/fade-out for each segment.

The program uses a configurable base carrier frequency (defaulting to 100 Hz) and creates stereo audio. The frequency difference between the left and right channels creates the binaural beat effect, which is intended to influence brainwave activity.

## Background

### What Are Binaural Beats?

Binaural beats are an auditory illusion perceived when two slightly different frequencies are presented separately to each ear. The brain detects the phase difference between these frequencies and attempts to reconcile this difference, which creates the sensation of a third "beat" frequency equal to the difference between the two tones.

For example, if a 100 Hz tone is presented to the left ear and a 110 Hz tone to the right ear, the brain perceives a 10 Hz binaural beat. This perceived frequency corresponds to specific brainwave patterns.

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

- Python 3.x
- Dependencies listed in `requirements.txt`:
  - `numpy`: For numerical operations and array handling.
  - `PyYAML`: For parsing the configuration script.
  - `soundfile`: For writing audio files (WAV, FLAC).

### Setup

1. **Automatic setup** with the provided script:

    ```bash
    ./bin/setup.sh
    ```

    This script will:
    - Install `uv` (a fast Python package installer/resolver) if not already installed.
    - Create a virtual environment named `.venv` in the current directory.
    - Install all required dependencies from `requirements.txt` and `requirements-bootstrap.txt` into the virtual environment.

2. **Activate the virtual environment**:

    ```bash
    source .venv/bin/activate
    ```

    > Note: If using VS Code, the workspace is configured to run the setup script automatically when opening the folder.

## Usage

Run the script with the path to a YAML configuration file:

```bash
python binaural.py <path_to_script.yaml> [options]
```

**Arguments:**

- `<path_to_script.yaml>` (Required): Path to the YAML file defining the binaural beat sequence.
- `-o <output_file>`, `--output <output_file>` (Optional): Specify the output audio file path. The file extension determines the format (e.g., `.wav` for WAV, `.flac` for FLAC). This overrides the `output_filename` setting in the YAML script.

**Example:**

To use the example script provided (which defaults to FLAC output):

```bash
python binaural.py example_script.yaml
```

This will generate `example_fade.flac` (or the filename specified in `example_script.yaml`) in the `audio/` directory.

To use one of the pre-defined scripts from the library and output as WAV:

```bash
python binaural.py scripts/relaxation_alpha.yaml -o audio/relaxation_alpha.wav
```

This will generate `relaxation_alpha.wav` in the `audio/` directory, overriding the default name in the script.

To generate a FLAC file with a custom name:

```bash
python binaural.py scripts/focus_beta.yaml -o my_focus_session.flac
```

## YAML Script Format

The YAML script defines the parameters and sequence for audio generation.

**Global Settings (Optional):**

- `base_frequency`: The carrier frequency in Hz (e.g., 100). Default: `100`.
- `sample_rate`: The audio sample rate in Hz (e.g., 44100). Default: `44100`.
- `output_filename`: The default name for the output audio file (e.g., `"audio/my_session.flac"` or `"audio/my_session.wav"`).
  The extension (`.wav` or `.flac`) determines the output format. Default: `"output.flac"`.

**Steps (Required):**

A list under the `steps:` key, where each item defines an audio segment. Each step can be one of the following types:

- **`type: stable`**: Holds a constant binaural beat frequency.
  - `frequency`: The binaural beat frequency in Hz.
  - `duration`: The duration of this segment in minutes.
  - `fade_in_duration` (Optional): Duration of a linear volume fade-in at the beginning of the step, in minutes. Default: `0`.
  - `fade_out_duration` (Optional): Duration of a linear volume fade-out at the end of the step, in minutes. Default: `0`.

- **`type: transition`**: Linearly changes the binaural beat frequency over time.
  - `start_frequency`: The starting binaural beat frequency in Hz. If omitted, it uses the end frequency of the previous step for a smooth transition.
  - `end_frequency`: The ending binaural beat frequency in Hz.
  - `duration`: The duration of this transition in minutes.
  - `fade_in_duration` (Optional): Duration of a linear volume fade-in at the beginning of the step, in minutes. Default: `0`.
  - `fade_out_duration` (Optional): Duration of a linear volume fade-out at the end of the step, in minutes. Default: `0`.

**Important Notes on Fades:**

- Fades are applied *within* the specified `duration` of the step.
- The sum of `fade_in_duration` and `fade_out_duration` for a single step cannot exceed the step's `duration`.

**Example YAML (`example_script.yaml`):**

```yaml
# Example Binaural Beat Generation Script with Fades

# Global settings (optional)
base_frequency: 100 # Hz (carrier frequency)
sample_rate: 44100 # Hz (audio sample rate)
output_filename: "audio/example_fade.flac" # Default output file name (FLAC format)

# Sequence of audio generation steps (Total Duration: 20 min)
steps:
  # 1. Beta phase with fade-in
  - type: stable
    frequency: 18 # Hz (binaural beat frequency)
    duration: 3 # minutes
    fade_in_duration: 0.1 # 6 seconds fade-in

  # 2. Transition from Beta (18 Hz) to Alpha (10 Hz)
  - type: transition
    start_frequency: 18 # Hz
    end_frequency: 10 # Hz
    duration: 5 # minutes
    # No fade in/out specified, uses defaults (0)

  # 3. Transition from Alpha (10 Hz) to Theta (6 Hz)
  - type: transition
    start_frequency: 10 # Hz
    end_frequency: 6 # Hz
    duration: 5 # minutes

  # 4. Transition from Theta (6 Hz) to Delta (2 Hz) with fade-out
  - type: transition
    start_frequency: 6 # Hz
    end_frequency: 2 # Hz
    duration: 7 # minutes
    fade_out_duration: 0.2 # 12 seconds fade-out at the very end
```

## Script Library

A collection of pre-defined YAML scripts for common use-cases is available in the `scripts/` directory.
These currently default to `.flac` output but can be easily changed by modifying the `output_filename` field or
using the `-o` command-line option with a `.wav` extension.

- **`scripts/focus_beta.yaml`**: Designed to enhance concentration and alertness using Beta waves (14-18 Hz).
- **`scripts/focus_gamma.yaml`**: Targets peak concentration and problem-solving with Gamma waves (40 Hz).
- **`scripts/relaxation_alpha.yaml`**: Aims to reduce stress and promote calmness using Alpha waves (8-10 Hz).
- **`scripts/meditation_theta.yaml`**: Facilitates deep meditation and introspection using Theta waves (6 Hz).
- **`scripts/sleep_delta.yaml`**: Guides the brain towards deep sleep states using Delta waves (2 Hz).
- **`scripts/creativity_theta.yaml`**: Intended to foster an intuitive and creative mental state using Theta waves (7 Hz).
- **`scripts/lucid_dreaming.yaml`**: Aims to facilitate REM sleep states potentially conducive to lucid dreaming.
- **`scripts/migraine_relief.yaml`**: Uses specific frequencies and transitions aimed at reducing migraine symptoms.

You can use these scripts directly or modify them to suit your needs, including adding fades.

Example usage for WAV output:

```bash
python binaural.py scripts/sleep_delta.yaml -o audio/sleep_delta.wav
```

## File Structure

- `binaural.py`: Main script that generates the binaural beats audio.
- `example_script.yaml`: Example YAML script with fades.
- `scripts/`: Directory containing pre-defined YAML scripts for various use-cases.
  - `focus_beta.yaml`
  - `focus_gamma.yaml`
  - `relaxation_alpha.yaml`
  - `meditation_theta.yaml`
  - `sleep_delta.yaml`
  - `creativity_theta.yaml`
  - `lucid_dreaming.yaml`
  - `migraine_relief.yaml`
- `bin/setup.sh`: Setup script to prepare the development environment.
- `requirements.txt`: Python dependencies (numpy, PyYAML, soundfile).
- `requirements-bootstrap.txt`: Bootstrap dependencies for setup (uv).
- `README.md`: This file.
- `LICENSE`: Project license information.

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
