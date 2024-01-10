import numpy as np
import librosa
import matplotlib.pyplot as plt

# Load the audio file
y, sr = librosa.load('aiueo.wav', sr=None)

# Parameters for spectrogram
SR = 16000
size_frame = 512
size_shift = SR / 100
hamming_window = np.hamming(size_frame)

# Function to create a spectrogram
def create_spectrogram(signal, sr):
    spectrogram = []
    for i in np.arange(0, len(signal) - size_frame, size_shift):
        idx = int(i)
        frame = signal[idx:idx + size_frame]
        fft_spec = np.fft.rfft(frame * hamming_window)
        fft_log_abs_spec = np.log(np.abs(fft_spec))
        spectrogram.append(fft_log_abs_spec)
    return np.flipud(np.array(spectrogram).T)

# Create a sine wave for voice change
def create_sine_wave(freq, duration, sr):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t)

# Apply sine wave to the audio
sine_freq = 4400  # Frequency of the sine wave in Hz
sine_wave = create_sine_wave(sine_freq, 7.0, sr)  # 1 second duration
y_modified = np.copy(y)
y_modified[:len(sine_wave)] *= sine_wave

# Generate spectrograms
original_spectrogram = create_spectrogram(y, sr)
modified_spectrogram = create_spectrogram(y_modified, sr)

# Plotting
plt.figure(figsize=(12, 6))

# Original Spectrogram
plt.subplot(2, 1, 1)
plt.imshow(original_spectrogram, extent=[0, len(y), 0, SR/2], aspect='auto', interpolation='nearest')
plt.title('Original Spectrogram')
plt.xlabel('Sample')
plt.ylabel('Frequency [Hz]')

# Modified Spectrogram
plt.subplot(2, 1, 2)
plt.imshow(modified_spectrogram, extent=[0, len(y_modified), 0, SR/2], aspect='auto', interpolation='nearest')
plt.title('Modified Spectrogram')
plt.xlabel('Sample')
plt.ylabel('Frequency [Hz]')

plt.tight_layout()
plt.savefig('24.png', dpi=300)
plt.show()
