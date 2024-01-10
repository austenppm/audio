########### USE OTHER FILE ###########
import numpy as np
import librosa
import matplotlib.pyplot as plt

def is_peak(a, index, threshold=0.5):
    """Check if the element at index is a peak, considering a threshold."""
    if index <= 0 or index >= len(a) - 1:
        return False
    return a[index] > a[index - 1] and a[index] > a[index + 1] and a[index] > threshold

def estimate_fundamental_frequency(signal, frame_size, sr):
    frames = [signal[i:i + frame_size] for i in range(0, len(signal), frame_size)]
    fundamental_frequencies = []

    for frame in frames:
        # Calculate autocorrelation
        autocorr = np.correlate(frame, frame, 'full')
        autocorr = autocorr[len(autocorr) // 2:]

        # Normalize and apply threshold
        autocorr /= np.max(autocorr)
        threshold = 0.5

        # Find peaks
        peak_indices = [i for i in range(1, len(autocorr)) if is_peak(autocorr, i, threshold)]
        
        if peak_indices:
            # Find the index of the maximum peak
            max_peak_index = max(peak_indices, key=lambda index: autocorr[index])

            # Convert index to frequency
            frequency = sr / max_peak_index
            fundamental_frequencies.append(frequency)
        else:
            fundamental_frequencies.append(0)

    return fundamental_frequencies

# Constants
FRAME_SIZE = 512  # Increased frame size
SR = 16000  # Sampling rate

# Load the audio file
x, _ = librosa.load('aiueo_.wav', sr=SR)

# Estimate fundamental frequencies for each frame
fundamental_frequencies = estimate_fundamental_frequency(x, FRAME_SIZE, SR)

# Plot the fundamental frequencies
plt.figure(figsize=(10, 4))
plt.plot(fundamental_frequencies, label='Fundamental Frequency')
plt.xlabel('Frame')
plt.ylabel('Frequency (Hz)')
plt.title('Fundamental Frequency Estimation')
plt.legend()
plt.show()
