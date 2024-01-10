import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.signal import find_peaks

# Load the audio file
y, sr = librosa.load('singing.wav', sr=None)

# Parameters for frame-based analysis
size_frame = 512  # Frame size
size_shift = int(sr / 100)  # Shift size (10 msec)

# Generate a Hamming window
hamming_window = np.hamming(size_frame)

# Compute spectrogram
spectrogram = []
for i in np.arange(0, len(y) - size_frame, size_shift):
    idx = int(i)
    x_frame = y[idx:idx + size_frame]
    fft_spec = np.fft.rfft(x_frame * hamming_window)
    fft_log_abs_spec = np.log(np.abs(fft_spec))
    spectrogram.append(fft_log_abs_spec)

# Plot the spectrogram
plt.figure(figsize=(12, 9))
plt.subplot(3, 1, 1)
plt.xlabel('Sample')
plt.ylabel('Frequency [Hz]')
plt.imshow(np.flipud(np.array(spectrogram).T), extent=[0, len(y), 0, sr/2], aspect='auto', interpolation='nearest')
plt.title('Spectrogram')

# Display the waveform
plt.subplot(3, 1, 2)
librosa.display.waveshow(y, sr=sr)
plt.title('Waveform of the Audio')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Pitch Estimation using Autocorrelation
def estimate_pitch_autocorrelation(frame, sr, fmin=50, fmax=2000):
    # Compute autocorrelation using statsmodels
    auto = sm.tsa.acf(frame, nlags=2000)

    # Find peaks in the autocorrelation function
    peaks = find_peaks(auto)[0]
    if len(peaks) == 0:
        return 0  # No peaks found

    # Use the first peak as our pitch component lag
    lag = peaks[0]

    # Transform lag into frequency
    pitch = sr / lag
    return pitch if fmin <= pitch <= fmax else 0

# Calculate pitch for each frame
pitch_values = []
for i in np.arange(0, len(y) - size_frame, size_shift):
    idx = int(i)
    frame = y[idx:idx+size_frame]
    pitch = estimate_pitch_autocorrelation(frame, sr)
    pitch_values.append(pitch)

# Plot the pitch curve
plt.subplot(3, 1, 3)
plt.plot(pitch_values)
plt.title('Pitch Estimation using Autocorrelation')
plt.xlabel('Time')
plt.ylabel('Pitch (Hz)')

plt.tight_layout()
plt.savefig('20.2.png', dpi=300)
plt.show()
