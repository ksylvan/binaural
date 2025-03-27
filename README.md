# Binaural

Binaural is a Python tool that generates binaural beat audio designed to induce different brain wave states, configured via a simple YAML script.

## Description

This tool reads a YAML script defining a sequence of binaural beat frequencies and durations, then creates a WAV audio file based on that sequence. It allows for both stable frequency segments and smooth transitions between frequencies.

The program uses a configurable base carrier frequency (defaulting to 100 Hz) and creates stereo audio. The frequency difference between the left and right channels creates the binaural beat effect, which is intended to influence brainwave activity.

## Background

### What Are Binaural Beats?

Binaural beats are an auditory illusion perceived when two slightly different frequencies are presented separately to each ear. The brain detects the phase difference between these frequencies and attempts to reconcile this difference, which creates the sensation of a third "beat" frequency equal to the difference between the two tones.

For example, if a 100 Hz tone is presented to the left ear and a 110 Hz tone to the right ear, the brain perceives a 10 Hz binaural beat. This perceived frequency corresponds to specific brainwave patterns.

### Brainwave Entrainment

Brainwave entrainment refers to the brain's electrical response to rhythmic sensory stimulation, such as pulses of sound or light. When the brain is presented with a stimulus with a frequency corresponding to a specific brainwave state, it tends to synchronize its electrical activity with that frequency—a process called "frequency following response."

Binaural beats are one method of achieving brainwave entrainment, potentially helping to induce specific mental states associated with different brainwave frequencies.

### Brainwave States

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
- `-o <output_file.wav>`, `--output <output_file.wav>` (Optional): Specify the output WAV file path. This overrides the `output_filename` setting in the YAML script.

**Example:**

To use the default script provided:

```bash
python binaural.py default_script.yaml
```

This will generate `binaural_beats.wav` (or the filename specified in `default_script.yaml`) in the current directory.

To specify a different output file:

```bash
python binaural.py my_custom_script.yaml -o custom_audio.wav
```

## YAML Script Format

The YAML script defines the parameters and sequence for audio generation.

**Global Settings (Optional):**

- `base_frequency`: The carrier frequency in Hz (e.g., 100). Default: `100`.
- `sample_rate`: The audio sample rate in Hz (e.g., 44100). Default: `44100`.
- `output_filename`: The default name for the output WAV file. Default: `"binaural_beats.wav"`.

**Steps (Required):**

A list under the `steps:` key, where each item defines an audio segment.

- **`type: stable`**: Holds a constant binaural beat frequency.
  - `frequency`: The binaural beat frequency in Hz.
  - `duration`: The duration of this segment in minutes.

- **`type: transition`**: Linearly changes the binaural beat frequency over time.
  - `start_frequency`: The starting binaural beat frequency in Hz. If omitted, it uses the end frequency of the previous step for a smooth transition.
  - `end_frequency`: The ending binaural beat frequency in Hz.
  - `duration`: The duration of this transition in minutes.

**Example YAML (`default_script.yaml`):**

```yaml
# Global settings
base_frequency: 100
sample_rate: 44100
output_filename: "binaural_beats.wav"

# Sequence of audio generation steps
steps:
  # 1. Beta phase (stable 18 Hz beat for 3 minutes)
  - type: stable
    frequency: 18
    duration: 3

  # 2. Transition from Beta (18 Hz) to Alpha (10 Hz) over 5 minutes
  - type: transition
    start_frequency: 18 # Can be omitted if previous step ended at 18 Hz
    end_frequency: 10
    duration: 5

  # 3. Transition from Alpha (10 Hz) to Theta (6 Hz) over 5 minutes
  - type: transition
    start_frequency: 10 # Can be omitted
    end_frequency: 6
    duration: 5

  # 4. Transition from Theta (6 Hz) to Delta (2 Hz) over 7 minutes
  - type: transition
    start_frequency: 6 # Can be omitted
    end_frequency: 2
    duration: 7
```

## File Structure

- `binaural.py`: Main script that generates the binaural beats audio based on a YAML script.
- `default_script.yaml`: Example YAML script defining the default sequence.
- `bin/setup.sh`: Setup script to prepare the development environment.
- `requirements.txt`: Python dependencies (numpy, PyYAML).
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

## License

This project is licensed under the MIT License - see the LICENSE file for details.

Copyright (c) 2025 Kayvan Sylvan

[discovery-binaural-beats]: https://www.binauralbeatsmeditation.com/dr-gerald-oster-auditory-beats-in-the-brain/
[healthline]: https://www.healthline.com/health/binaural-beats
[sleep-foundation]: https://www.sleepfoundation.org/bedroom-environment/binaural-beats
[plos-one-ruth-research]: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0286023
