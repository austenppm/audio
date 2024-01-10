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

# Load audio files
x, _ = librosa.load('aiueo.wav', sr=SR)
y, _ = librosa.load('aiueo_.wav', sr=SR)

# Utterance periods in seconds
utterances_aiueo = [(1.12, 1.32), (1.99, 2.18), (2.79, 2.99), (3.61, 3.80), (4.40, 4.60)]
utterances_aiueo_ = [(0.49, 1.24), (1.33, 2.12), (2.19, 2.95), (3.03, 3.60), (3.85, 4.21)]

def process_utterances(utterances, audio_data):
    fundamental_frequencies = []
    for start, end in utterances:
        start_sample = int(start * SR)
        end_sample = int(end * SR)
        frame = audio_data[start_sample:end_sample]

        if len(frame) >= FRAME_SIZE:
            autocorr = np.correlate(frame, frame, 'full')
            autocorr = autocorr[len(autocorr) // 2:]
            frequency = find_fundamental_frequency(autocorr, SR)
            fundamental_frequencies.append(frequency)
        else:
            fundamental_frequencies.append(0)

    return fundamental_frequencies

# Process each utterance period
fundamental_frequencies_aiueo = process_utterances(utterances_aiueo, x)
fundamental_frequencies_aiueo_ = process_utterances(utterances_aiueo_, y)

# Plotting for aiueo.wav
plt.figure(figsize=(10, 4))
plt.plot(fundamental_frequencies_aiueo, marker='o', label='Fundamental Frequency')
plt.xlabel('Utterance')
plt.ylabel('Frequency (Hz)')
plt.title('Fundamental Frequency Estimation in aiueo.wav')
plt.xticks(range(len(utterances_aiueo)), ['A', 'I', 'U', 'E', 'O'])
plt.ylim(80, 120)
plt.legend()
plt.show()

# Plotting for aiueo_.wav
plt.figure(figsize=(10, 4))
plt.plot(fundamental_frequencies_aiueo_, marker='o', label='Fundamental Frequency')
plt.xlabel('Utterance')
plt.ylabel('Frequency (Hz)')
plt.title('Fundamental Frequency Estimation in aiueo_.wav')
plt.xticks(range(len(utterances_aiueo_)), ['A_', 'I_', 'U_', 'E_', 'O_'])
plt.ylim(80, 120)
plt.legend()
plt.show()
