import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load the audio file
y, sr = librosa.load('nekocut.wav', sr=None)

# Function to apply tremolo effect
def apply_tremolo(audio, sr, rate, depth):
    # Create tremolo waveform (sine wave)
    t = np.linspace(0, len(audio) / sr, num=len(audio))
    tremolo_wave = (1 - depth) + depth * np.sin(2 * np.pi * rate * t)
    
    # Apply tremolo effect
    return audio * tremolo_wave

# Function to plot waveform
def plot_waveform(audio, sr, title):
    plt.figure(figsize=(12, 4))
    times = np.arange(len(audio)) / sr
    plt.plot(times, audio)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.savefig(f'25{title}.png', dpi=300)
    plt.show()    

# Original audio
plot_waveform(y, sr, 'Original Audio')

# Parameters for tremolo
rate = 2  # Rate of tremolo in Hz
depth = 0.5  # Depth of tremolo (0 to 1)

# Apply tremolo
y_tremolo = apply_tremolo(y, sr, rate, depth)

# Tremolo-processed audio
plot_waveform(y_tremolo, sr, 'Tremolo-Processed Audio')
