import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def plot_and_save_spectrum(file_name, intervals, sr=16000, n_mfcc=13):
    y, sr = librosa.load(file_name, sr=sr)

    for label, (start, end) in intervals.items():
        # Extract the segment for the vowel
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        y_segment = y[start_sample:end_sample]

        # Extract MFCCs and invert to Mel spectrogram
        mfccs = librosa.feature.mfcc(y=y_segment, sr=sr, n_mfcc=n_mfcc)
        mel_spectrogram = librosa.feature.inverse.mfcc_to_mel(mfccs)

        # Convert Mel spectrogram to power spectrogram
        power_spectrogram = librosa.feature.melspectrogram(S=mel_spectrogram)

        # Convert power spectrogram to amplitude spectrogram
        amplitude_spectrogram = librosa.power_to_db(power_spectrogram, ref=np.max)

        # Original logarithmic amplitude spectrum
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y_segment)), ref=np.max)

        # Plot
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Logarithmic Amplitude Spectrum of {label} in {file_name}')

        plt.subplot(2, 1, 2)
        librosa.display.specshow(amplitude_spectrogram, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectral Envelope from MFCCs of {label} in {file_name}')

        plt.tight_layout()

        # Save the plot
        plot_file_name = f"{file_name.split('.')[0]}_{label}_plot.png"
        plt.savefig(plot_file_name)
        plt.close()
        print(f"Saved plot as {plot_file_name}")

# Vowel intervals for aiueo.wav and aiueo_.wav
intervals_aiueo = {
    'A': (1.12, 1.32), 'I': (1.99, 2.18), 'U': (2.79, 2.99),
    'E': (3.61, 3.80), 'O': (4.40, 4.60)
}
intervals_aiueo_ = {
    'A_': (0.49, 1.24), 'I_': (1.33, 2.12), 'U_': (2.19, 2.95),
    'E_': (3.03, 3.60), 'O_': (3.85, 4.21)
}

# Plot and Save MFCCs and Spectrum for aiueo.wav
plot_and_save_spectrum('aiueo.wav', intervals_aiueo)

# Plot and Save MFCCs and Spectrum for aiueo_.wav
plot_and_save_spectrum('aiueo_.wav', intervals_aiueo_)
