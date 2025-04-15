# Development Plan: Ambient Music Integration for Binaural Generator

## Objective

Enable dynamic generation and seamless blending of ambient music loops with binaural beat audio, using the `music` Python package, to enhance the end-user experience without relying on pre-recorded music files.

---

## Step-by-Step Implementation Plan

### 1. Research and Evaluate the `music` Package

- Review [music PyPI documentation](https://pypi.org/project/music/) and test core generative features: loop generation, instruments, tempo, etc.
- Prototype basic ambient textures suitable for background blending (e.g., pads, drones, simple arpeggios).

### 2. Design Integration API

- Extend project data models (YAML script format and `AudioStep`/`NoiseConfig`) to optionally include ambient music layer configuration:
  - instrument type(s)
  - style/preset (if supported)
  - volume/amplitude
  - seed (for reproducibility)
  - loop duration / fade / crossfade parameters

### 3. Build Ambient Music Generator

- Implement a new module (e.g., `core/music.py`) that:
  - Generates a loop or sequence based on YAML/config parameters using `music`.
  - Exposes API for: create track, render loop, customize instrument/style, repeat/generate longer sequences.
- Unit test core music generation features for deterministic output given a seed/config.

### 4. Integrate With Audio Pipeline

- Update core audio pipeline (in `tone_generator.py` and/or `noise.py`):
  - Add function to layer ambient music track(s) on top of generated binaural/noise output.
  - Implement seamless looping, crossfades, and level mixing (matching gain scaling logic for noise).
  - Support fade-in/out and duration-to-match generated output.

### 5. Extend YAML Script Schema (Backwards Compatible)

- Add new ambient music config section in YAML (optional):

```yaml
ambient_music:
  enabled: true
  instrument: "pad_swell"
  style: "ambient"
  amplitude: 0.2
  seed: 42
  loop_seconds: 60
```

- Update YAML loader/validator in `utils.py` and corresponding data classes in `data_types.py`.

### 6. User Interface/CLI Enhancements

- Add CLI & WebUI options for ambient music: enable/disable, select instrument/style, intensity/volume, etc.
- Update help/usage docs.

### 7. Testing & Validation

- Add comprehensive tests (unit, property-based, and integration) to:
  - Validate correct blending, seamless looping, and absence of artifacts.
  - Confirm YAML and CLI options work as intended.
  - Ensure backward compatibility with scripts missing ambient sections.

### 8. Documentation & Demos

- Document usage in README and provide example YAML scripts.
- Briefly explain algorithmic approaches and user options for customizing ambient textures.
- Provide a sample `ambient_music.yaml` with several preset configs.

---

## Risks/Challenges

- Ensuring the generated music blends musically and does not distract from core binaural/brainwave effects.
- Avoiding sudden starts/stops via proper crossfading/looping strategies.
- Performance: Keep generation responsive on typical hardware.

## Milestones

- [ ] Prototype generative loops with `music`
- [ ] YAML and core API extended
- [ ] Generator module integrated and tested
- [ ] CLI/WebUI support ready
- [ ] Docs and demo scripts added
