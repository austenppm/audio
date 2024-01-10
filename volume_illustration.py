import matplotlib.pyplot as plt
import numpy as np
import librosa

# Function to generate spectrogram
def generate_spectrogram(filename, SR, size_frame, size_shift):
    x, _ = librosa.load(filename, sr=SR)
    hamming_window = np.hamming(size_frame)
    spectrogram = []

    for i in np.arange(0, len(x) - size_frame, size_shift):
        idx = int(i)
        x_frame = x[idx:idx+size_frame]
        fft_spec = np.fft.rfft(x_frame * hamming_window)
        fft_log_abs_spec = np.log(np.abs(fft_spec))
        spectrogram.append(fft_log_abs_spec)

    return np.array(spectrogram), len(x)

# Settings
SR = 16000
size_frame = 512
size_shift = 16000 / 100

# Generate spectrograms
spectrogram1, len_x1 = generate_spectrogram('aiueo.wav', SR, size_frame, size_shift)
spectrogram2, len_x2 = generate_spectrogram('aiueo_.wav', SR, size_frame, size_shift)

# Plot settings
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Plot first spectrogram
axs[0].set_title('Spectrogram of aiueo.wav')
axs[0].imshow(
    np.flipud(spectrogram1.T),
    extent=[0, len_x1, 0, SR/2],
    aspect='auto',
    interpolation='nearest'
)
axs[0].set_ylabel('Frequency [Hz]')

# Plot second spectrogram
axs[1].set_title('Spectrogram of aiueo_.wav')
axs[1].imshow(
    np.flipud(spectrogram2.T),
    extent=[0, len_x2, 0, SR/2],
    aspect='auto',
    interpolation='nearest'
)
axs[1].set_xlabel('Time (samples)')
axs[1].set_ylabel('Frequency [Hz]')

plt.tight_layout()
plt.show()
