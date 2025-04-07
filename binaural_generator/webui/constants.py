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
