import matplotlib.pyplot as plt
import numpy as np
import librosa

def is_peak(a, index):
    if index == 0 or index == len(a) - 1:
        return False
    return a[index] > a[index-1] and a[index] > a[index+1]

def find_fundamental_frequency(autocorr, sr):
    peak_indices = [i for i in range(1, len(autocorr) - 1) if is_peak(autocorr, i)]

    if not peak_indices:
        return 0  # No peaks found, return 0 as a default value

    max_peak_index = max(peak_indices, key=lambda index: autocorr[index])
    frequency = sr / max_peak_index
    return frequency

# Sampling Rate and Frame Size
SR = 16000
FRAME_SIZE = 1024
HOP_SIZE = SR // 100

# Load audio file
x, _ = librosa.load('aiueo.wav', sr=SR)
y, _ = librosa.load('aiueo_.wav', sr=SR)
# Process each frame
fundamental_frequencies = []
fundamental_frequencies_ = []


for i in range(0, len(x) - FRAME_SIZE, HOP_SIZE):
    frame = x[i:i + FRAME_SIZE]
    autocorr = np.correlate(frame, frame, 'full')
    autocorr = autocorr[len(autocorr) // 2:]

    frequency = find_fundamental_frequency(autocorr, SR)
    fundamental_frequencies.append(frequency)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(fundamental_frequencies, label='Fundamental Frequency')
plt.xlabel('Frame')
plt.ylabel('Frequency (Hz)')
plt.title('Fundamental Frequency Estimation for Each Frame in aiueo.wav')
plt.legend()
plt.show()

for i in range(0, len(y) - FRAME_SIZE, HOP_SIZE):
    frame_ = y[i:i + FRAME_SIZE]
    autocorr_ = np.correlate(frame_, frame_, 'full')
    autocorr_ = autocorr_[len(autocorr_) // 2:]

    frequency_ = find_fundamental_frequency(autocorr_, SR)
    fundamental_frequencies_.append(frequency_)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(fundamental_frequencies_, label='Fundamental Frequency')
plt.xlabel('Frame')
plt.ylabel('Frequency (Hz)')
plt.title('Fundamental Frequency Estimation for Each Frame in aiueo_.wav')
plt.legend()
plt.show()