import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF

# Custom Spectrogram Function
def compute_magnitude_spectrogram(y, sr, size_frame, size_shift):
    hamming_window = np.hamming(size_frame)
    spectrogram = []
    for i in np.arange(0, len(y) - size_frame, size_shift):
        idx = int(i)
        x_frame = y[idx:idx + size_frame]
        fft_spec = np.fft.rfft(x_frame * hamming_window)
        mag_spec = np.abs(fft_spec)
        spectrogram.append(mag_spec)
    return np.array(spectrogram).T

# Load Audio File
y, sr = librosa.load('nekocut.wav', sr=16000)

# Compute Magnitude Spectrogram
size_frame = 512
size_shift = sr // 100
spectrogram = compute_magnitude_spectrogram(y, sr, size_frame, size_shift)

# Apply NMF for different numbers of bases
for k in [5, 10, 15, 20]:  # Different values for k (number of bases)
    model = NMF(n_components=k, init='random', random_state=0)
    W = model.fit_transform(spectrogram)
    H = model.components_

    # Determine rows and cols for subplot
    nrows, ncols = k // 2 + k % 2, 2

    # Plotting U (W) matrix
    plt.figure(figsize=(15, 6))
    for i in range(k):
        plt.subplot(nrows, ncols, i+1)
        plt.plot(W[:, i])
        plt.title(f'Base {i+1}')
        plt.xlabel('Time')
        plt.ylabel('Component')

    plt.tight_layout()
    plt.suptitle(f'NMF Bases (k={k})', fontsize=16)
    plt.savefig(f'22_bases_k={k}.png', dpi=300)
    # plt.show()

    # Plotting H matrix
    plt.figure(figsize=(15, 6))
    for i in range(k):
        plt.subplot(nrows, ncols, i+1)
        plt.plot(H[i, :])
        plt.title(f'Coefficient {i+1}')
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.suptitle(f'NMF Coefficients (k={k})', fontsize=16)
    plt.savefig(f'22_coefficients_k={k}.png', dpi=300)
    # plt.show()
