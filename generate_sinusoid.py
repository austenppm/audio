import sys
import math
import numpy as np
import scipy.io.wavfile

def generate_sinusoid(sampling_rate, frequency, duration):
    sampling_interval = 1.0 / sampling_rate
    t = np.arange(sampling_rate * duration) * sampling_interval
    waveform = np.sin(2.0 * math.pi * frequency * t)
    return waveform

def combine_waveforms(waveforms):
    return sum(waveforms) / len(waveforms)

# Sampling rate and duration
sampling_rate = 16000.0
duration = 2.0  # seconds

# Collect frequencies from the user
frequencies = []
print("Enter frequencies in Hz, and enter 0 to finish:")
while True:
    try:
        freq = float(input("Enter frequency (Hz): "))
        if freq == 0:
            break
        frequencies.append(freq)
    except ValueError:
        print("Please enter a valid number.")

# Generate sinusoids for each frequency
waveforms = [generate_sinusoid(sampling_rate, freq, duration) for freq in frequencies]

# Combine waveforms
combined_waveform = combine_waveforms(waveforms)

# Normalize to maximum value of 0.9
combined_waveform = combined_waveform * 0.9

# Convert value range from [-1.0, +1.0] to [-32768, +32767]
combined_waveform = (combined_waveform * 32768.0).astype('int16')

# Output to a WAV file
filename = 'sinusoid_test.wav'
scipy.io.wavfile.write(filename, int(sampling_rate), combined_waveform)

print(f"Sinusoid test file '{filename}' created successfully.")
