"""Binaural Beat Generator

This module generates a binaural beat WAV file that transitions through different brain wave
frequency bands:

- Beta (18 Hz) for 3 minutes
- Alpha (transition from 18 Hz to 10 Hz) for 5 minutes
- Theta (transition from 10 Hz to 6 Hz) for 5 minutes
- Delta (transition from 6 Hz to 2 Hz) for 7 minutes

The program uses a base carrier frequency and creates stereo audio where the frequency difference
between left and right channels creates the desired binaural beat effect.

    str: Message indicating successful file generation

"""

import wave

import numpy as np

# Parameters
SAMPLE_RATE = 44100  # samples per second
BASE_FREQ = 100  # base frequency for both channels in Hz


def generate_tone(duration, base_freq, start_diff, end_diff):
    """
    Generate stereo tones for the given duration.
    Left channel is a sine wave at base_freq.
    Right channel is a sine wave at base_freq plus a varying difference (binaural beat).
    """
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    # Linear ramp for the binaural beat difference
    diff = np.linspace(start_diff, end_diff, t.size)
    left = np.sin(2 * np.pi * base_freq * t)
    right = np.sin(2 * np.pi * (base_freq + diff) * t)
    return left, right


# Define durations (in seconds) for each segment
DURATION_BETA = 3 * 60  # 3 minutes
DURATION_ALPHA = 5 * 60  # 5 minutes
DURATION_THETA = 5 * 60  # 5 minutes
DURATION_DELTA = 7 * 60  # 7 minutes

# Segment 1: Beta phase (constant 18 Hz beat)
left_beta, right_beta = generate_tone(DURATION_BETA, BASE_FREQ, 18, 18)

# Segment 2: Transition from Beta (18 Hz) to Alpha (10 Hz)
left_alpha, right_alpha = generate_tone(DURATION_ALPHA, BASE_FREQ, 18, 10)

# Segment 3: Transition from Alpha (10 Hz) to Theta (6 Hz)
left_theta, right_theta = generate_tone(DURATION_THETA, BASE_FREQ, 10, 6)

# Segment 4: Transition from Theta (6 Hz) to Delta (2 Hz)
left_delta, right_delta = generate_tone(DURATION_DELTA, BASE_FREQ, 6, 2)

# Concatenate segments for left and right channels
left_channel = np.concatenate([left_beta, left_alpha, left_theta, left_delta])
right_channel = np.concatenate([right_beta, right_alpha, right_theta, right_delta])

# Combine into stereo signal (interleaved channels)
stereo = np.vstack((left_channel, right_channel)).T

# Normalize the audio signal to 16-bit range
max_val = np.max(np.abs(stereo))
stereo_int16 = np.int16(stereo / max_val * 32767)

# Write to a stereo WAV file
OUTPUT_FILENAME = "binaural_beats.wav"
with wave.Wave_write(OUTPUT_FILENAME) as wf:
    wf.setnchannels(2)  # Stereo
    wf.setsampwidth(2)  # 16 bits per sample (2 bytes)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(stereo_int16.tobytes())

print(f"Audio file '{OUTPUT_FILENAME}' generated successfully.")
