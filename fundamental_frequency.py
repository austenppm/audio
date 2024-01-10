import numpy as np
import librosa
import matplotlib.pyplot as plt

def extract_utterances(filename, SR, size_frame, size_shift, threshold, time_start=None, time_end=None):
    x, _ = librosa.load(filename, sr=SR)

    # If time_start and time_end are specified, slice the audio data
    if time_start is not None and time_end is not None:
        start_sample = int(time_start * SR)
        end_sample = int(time_end * SR)
        x = x[start_sample:end_sample]

    energy = []
    utterances = []
    in_utterance = False

    for i in range(0, len(x) - size_frame, int(size_shift)):
        frame = x[i:i+size_frame]
        frame_energy = 10 * np.log10(np.sum(frame**2))
        energy.append(frame_energy)

        if frame_energy > threshold and not in_utterance:
            start = i / SR + (time_start if time_start else 0)
            in_utterance = True
        elif frame_energy <= threshold and in_utterance:
            end = i / SR + (time_start if time_start else 0)
            utterances.append((start, end))
            in_utterance = False

    if in_utterance:
        end_time = (len(x) / SR) + (time_start if time_start else 0)
        utterances.append((start, end_time))

    return utterances

def generate_spectrogram(filename, SR, size_frame, time_start, time_end):
    x, _ = librosa.load(filename, sr=SR)
    start_sample = int(time_start * SR)
    end_sample = int(time_end * SR)
    x_segment = x[start_sample:end_sample]

    hamming_window = np.hamming(size_frame)
    spectrogram = []

    for i in range(0, len(x_segment) - size_frame, int(size_frame / 2)):  # 50% overlap
        x_frame = x_segment[i:i + size_frame]
        fft_spec = np.fft.rfft(x_frame * hamming_window)
        fft_log_abs_spec = np.log(np.abs(fft_spec) + 1e-10)
        spectrogram.append(fft_log_abs_spec)

    return np.array(spectrogram)

def find_fundamental_frequency(spectrogram, SR, size_frame):
    avg_spectrum = np.mean(spectrogram, axis=0)
    fundamental_freq_idx = np.argmax(avg_spectrum)
    fundamental_freq = fundamental_freq_idx * SR / (2 * size_frame)
    return fundamental_freq

# Main
SR = 16000
size_frame = 1024
size_shift = 16000 / 100
threshold = -5

# Process 'aiueo.wav'
utterances_aiueo = extract_utterances('aiueo.wav', SR, size_frame, size_shift, threshold)
for i, (start, end) in enumerate(utterances_aiueo):
    spectrogram = generate_spectrogram('aiueo.wav', SR, size_frame, start, end)
    fundamental_freq = find_fundamental_frequency(spectrogram, SR, size_frame)
    print(f"Utterance {i+1} in 'aiueo.wav' ({start}-{end} seconds): Fundamental Frequency = {fundamental_freq:.2f} Hz")

# Process 'aiueo_.wav'
time_start_ = 0
time_end_ = 5
threshold_ = -3
utterances_aiueo_ = extract_utterances('aiueo_.wav', SR, size_frame, size_shift, threshold_, time_start_, time_end_)
for i, (start, end) in enumerate(utterances_aiueo_):
    spectrogram_ = generate_spectrogram('aiueo_.wav', SR, size_frame, start, end)
    fundamental_freq = find_fundamental_frequency(spectrogram_, SR, size_frame)
    print(f"Utterance {i+1} in 'aiueo_.wav' ({start}-{end} seconds): Fundamental Frequency = {fundamental_freq:.2f} Hz")

print(spectrogram.shape)

# # # Plotting the specific segment's spectrogram
# plt.figure(figsize=(10, 4))
# plt.imshow(
#     np.flipud(spectrogram.T),
#     extent=[0, 7, 0, SR/2],
#     aspect='auto',
#     interpolation='nearest'
# )

# plt.title('Spectrogram of a specific segment in aiueo.wav')
# plt.xlabel('Time [s]')
# plt.ylabel('Frequency [Hz]')
# plt.ylim([0, 1000])
# plt.colorbar(label='Log Amplitude')
# plt.show()

# # Plotting the specific segment's spectrogram
# plt.figure(figsize=(10, 4))
# plt.imshow(
#     np.flipud(spectrogram_.T),
#     extent=[0, 7, 0, SR/2],
#     aspect='auto',
#     interpolation='nearest'
# )
# plt.title('Spectrogram of a specific segment in aiueo_.wav')
# plt.xlabel('Time [s]')
# plt.ylabel('Frequency [Hz]')
# plt.ylim([0, 1000])
# plt.colorbar(label='Log Amplitude')
# plt.show()

# Plotting spectrograms for all segments in 'aiueo.wav'
for i, (start, end) in enumerate(utterances_aiueo):
    spectrogram = generate_spectrogram('aiueo.wav', SR, size_frame, start, end)
    plt.figure(figsize=(10, 4))
    plt.imshow(
        np.flipud(spectrogram.T),
        extent=[0, end-start, 0, SR/2],
        aspect='auto',
        interpolation='nearest'
    )
    plt.title(f'Spectrogram of segment {i+1} in aiueo.wav ({start}-{end} seconds)')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.ylim([0, 1000])
    plt.colorbar(label='Log Amplitude')
    plt.show()

# Plotting spectrograms for all segments in 'aiueo.wav'
for i, (start, end) in enumerate(utterances_aiueo_):
    spectrogram_ = generate_spectrogram('aiueo_.wav', SR, size_frame, start, end)
    plt.figure(figsize=(10, 4))
    plt.imshow(
        np.flipud(spectrogram_.T),
        extent=[0, end-start, 0, SR/2],
        aspect='auto',
        interpolation='nearest'
    )
    plt.title(f'Spectrogram of segment {i+1} in aiueo_.wav ({start}-{end} seconds)')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.ylim([0, 1000])
    plt.colorbar(label='Log Amplitude')
    plt.show()

