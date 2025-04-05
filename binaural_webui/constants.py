"""Constants for the Binaural Beat Generator web UI."""

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
