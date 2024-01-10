import numpy as np
import librosa
import matplotlib.pyplot as plt

SR = 16000
SIZE_FRAME = 512
SHIFT_SIZE = int(SR / 100)  # 10 msec

def zero_cross(waveform):
    d = np.array(waveform)
    return sum([1 if x < 0.0 else 0 for x in d[1:] * d[:-1]])

def zero_cross_rate(waveform, sample_rate):
    return zero_cross(waveform) / (len(waveform) / sample_rate)

def zero_cross_rates(waveform, sample_rate, size_frame):
    zcrs = []
    for i in np.arange(0, len(waveform) - size_frame, size_frame):
        x_frame = waveform[int(i): int(i) + size_frame]
        zcrs.append(zero_cross_rate(x_frame, sample_rate))
    return zcrs

def is_peak(arr, index):
    if index == 0 or index == len(arr) - 1:
        return False
    return arr[index - 1] < arr[index] and arr[index] > arr[index + 1]

def get_f0(waveform, sampling_rate):
    autocorr = np.correlate(waveform, waveform, 'full')
    autocorr = autocorr[len(autocorr) // 2:]  # Discard the first half

    peak_indices = [i for i in range(len(autocorr)) if is_peak(autocorr, i) and i != 0]
    if len(peak_indices) == 0:
        return 0

    max_peak_index = max(peak_indices, key=lambda index: autocorr[index])
    f0 = sampling_rate / max_peak_index
    return f0

def get_f0s_voiced(waveform, sampling_rate, size_frame):
    f0s = []
    for i in np.arange(0, len(waveform) - size_frame, size_frame):
        x_frame = waveform[int(i): int(i) + size_frame]
        if zero_cross_rate(x_frame, sampling_rate) < 0.01:  # Threshold for voiced segments
            f0 = get_f0(x_frame, sampling_rate)
        else:
            f0 = 0
        f0s.append(f0)
    return f0s

def spectrogram(waveform, size_frame, size_shift):
    spectrogram = []
    hamming_window = np.hamming(size_frame)
    for i in np.arange(0, len(waveform) - size_frame, size_shift):
        x_frame = waveform[int(i): int(i) + size_frame]
        fft_spec = np.fft.rfft(x_frame * hamming_window)
        fft_log_abs_spec = np.log(np.abs(fft_spec) + 1e-10)
        spectrogram.append(fft_log_abs_spec)
    return spectrogram

# Load the audio file
x, _ = librosa.load('output.wav', sr=SR)

# Calculate spectrogram
spec = spectrogram(x, SIZE_FRAME, SHIFT_SIZE)

# Estimate fundamental frequencies for voiced segments
f0s_voiced = get_f0s_voiced(x, SR, SIZE_FRAME)

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))

# Spectrogram
img = ax.imshow(np.flipud(np.array(spec).T), extent=[0, len(x)/SR, 0, SR/2], aspect='auto', interpolation='nearest')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Frequency (Hz)')
ax.set_ylim(0, 2000)
fig.colorbar(img, ax=ax, format='%+2.0f dB', label='Log Amplitude')

# Overlaying fundamental frequencies
times = np.linspace(0, len(x)/SR, len(f0s_voiced))
ax.plot(times, f0s_voiced, label='Voiced F0', color='yellow')
ax.legend()

plt.title('Spectrogram with Fundamental Frequency of Voiced Segments')
plt.show()

# Save the plot
fig.savefig('exercise15_2.png')
