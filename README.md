# Binaural

Binaural is a Python tool that generates binaural beat audio designed to induce different brain wave states.

## Description

This tool creates a WAV file that transitions through different brain wave frequency bands using binaural beats:

- Beta (18 Hz) - 3 minutes - Associated with alertness and concentration
- Alpha (transition from 18 Hz to 10 Hz) - 5 minutes - Associated with relaxation and calmness
- Theta (transition from 10 Hz to 6 Hz) - 5 minutes - Associated with deep meditation and light sleep
- Delta (transition from 6 Hz to 2 Hz) - 7 minutes - Associated with deep sleep and healing

The program uses a 100 Hz base carrier frequency and creates stereo audio where the frequency difference between left and right channels creates the binaural beat effect.

## Background

### What Are Binaural Beats?

Binaural beats are an auditory illusion perceived when two slightly different frequencies are presented separately to each ear. The brain detects the phase difference between these frequencies and attempts to reconcile this difference, which creates the sensation of a third "beat" frequency.

For example, if a 100 Hz tone is presented to the left ear and a 110 Hz tone to the right ear, the brain perceives a 10 Hz binaural beat. This perceived frequency corresponds to specific brainwave patterns.

### Brainwave Entrainment

Brainwave entrainment refers to the brain's electrical response to rhythmic sensory stimulation, such as pulses of sound or light. When the brain is presented with a stimulus with a frequency corresponding to a specific brainwave state, it tends to synchronize its electrical activity with that frequency—a process called "frequency following response."

Binaural beats are one method of achieving brainwave entrainment, potentially helping to induce specific mental states associated with different brainwave frequencies.

### Brainwave States

#### Beta Waves (13-30 Hz)

- **Mental State**: Alertness, concentration, active thinking
- **Activities**: Problem-solving, active conversation, analytical tasks
- **In This Program**: 18 Hz for 3 minutes to begin in an alert state

#### Alpha Waves (8-12 Hz)

- **Mental State**: Relaxation, calmness, passive attention
- **Activities**: Light meditation, relaxation practices, creative visualization
- **In This Program**: Transition from 18 Hz to 10 Hz over 5 minutes to induce relaxation

#### Theta Waves (4-7 Hz)

- **Mental State**: Deep meditation, drowsiness, REM sleep
- **Activities**: Deep meditation, creative insight, dream states
- **In This Program**: Transition from 10 Hz to 6 Hz over 5 minutes to deepen relaxation

#### Delta Waves (0.5-4 Hz)

- **Mental State**: Deep, dreamless sleep, healing, regeneration
- **Activities**: Unconscious bodily functions, deep restorative sleep
- **In This Program**: Transition from 6 Hz to 2 Hz over 7 minutes for deep relaxation

## Scientific Research

Research on binaural beats has shown mixed results, but several studies suggest potential benefits:

- **Stress Reduction**: Some studies indicate that binaural beats in the alpha frequency range may help reduce anxiety and stress ([Wahbeh et al., 2007](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5370608/))

- **Cognitive Enhancement**: Research suggests potential improvements in attention, working memory, and other cognitive functions ([Kraus & Porubanová, 2015](https://www.sciencedirect.com/science/article/abs/pii/S1053810015300593))

- **Sleep Quality**: Delta frequency binaural beats may improve sleep quality in some individuals ([Jirakittayakorn & Wongsawat, 2018](https://www.frontiersin.org/articles/10.3389/fnhum.2018.00387/full))

## Installation

### Requirements

- Python 3.x
- Dependencies listed in requirements.txt:
  - numpy
  - wave

### Setup

1. Automatic setup with the provided script:

    ```bash
    ./bin/setup.sh
    ```

    This script will:

   - Install uv (if not already installed)
   - Create a virtual environment
   - Install all required dependencies

2. Activate the virtual environment:

    ```bash
    source .venv/bin/activate
    ```

    > Note: If using VS Code, the workspace is configured to run the setup script automatically when opening the folder.

## Usage

Run the script to generate a binaural beat WAV file:

```bash
python binaural.py
```

This will create `binaural_beats.wav` in the current directory.

## File Structure

- binaural.py - Main script that generates the binaural beats audio
- bin/setup.sh - Setup script to prepare the development environment
- requirements.txt - Python dependencies
- requirements-bootstrap.txt - Bootstrap dependencies for setup

## Resources

### Further Reading

- [The Discovery of Binaural Beats][discovery-binaural-beats]
- [Healthline - Binaural Beats: Do They Really Affect Your Brain?][healthline] - Discusses the potential cognitive and mood benefits of binaural beats
- [Sleep Foundation - Binaural Beats and Sleep][sleep-foundation] - Examines the impact of binaural beats on sleep quality
- [Binaural beats to entrain the brain? A systematic review of the effects of binaural beat stimulation][plos-one-ruth-research] - Published in 2023.

### References

- Oster, G. (1973). Auditory beats in the brain. Scientific American, 229(4), 94-102.
- Huang, T. L., & Charyton, C. (2008). A comprehensive review of the psychological effects of
  brainwave entrainment. Alternative Therapies in Health and Medicine, 14(5), 38-50.
- Le Scouarnec, R. P., Poirier, R. M., Owens, J. E., Gauthier, J., Taylor, A. G., & Foresman,
  P. A. (2001). Use of binaural beat tapes for treatment of anxiety: A pilot study. Alternative
  Therapies in Health and Medicine, 7(1), 58-63.
- Chaieb, L., Wilpert, E. C., Reber, T. P., & Fell, J. (2015). Auditory beat stimulation and its
  effects on cognition and mood states. Frontiers in Psychiatry, 6, 70.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

Copyright (c) 2025 Kayvan Sylvan

[discovery-binaural-beats]: https://www.binauralbeatsmeditation.com/dr-gerald-oster-auditory-beats-in-the-brain/
[healthline]: https://www.healthline.com/health/binaural-beats
[sleep-foundation]: https://www.sleepfoundation.org/bedroom-environment/binaural-beats
[plos-one-ruth-research]: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0286023
