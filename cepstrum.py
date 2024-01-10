import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def plot_mfccs_and_spectrum(file_name, sr=16000, n_mfcc=13):
    # Load audio file
    y, sr = librosa.load(file_name, sr=sr)

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=int(n_mfcc))

    # Invert MFCCs to Mel spectrogram
    mel_spectrogram = librosa.feature.inverse.mfcc_to_mel(mfccs)

    # Convert Mel spectrogram to power spectrogram
    power_spectrogram = librosa.feature.melspectrogram(S=mel_spectrogram, sr=sr)

    # Convert power spectrogram to amplitude spectrogram
    amplitude_spectrogram = librosa.power_to_db(power_spectrogram, ref=np.max)

    # Original logarithmic amplitude spectrum
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    # Plot
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Logarithmic Amplitude Spectrum of ' + file_name)

    plt.subplot(2, 1, 2)
    librosa.display.specshow(amplitude_spectrogram, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectral Envelope from MFCCs of ' + file_name)

    plt.tight_layout()
    
    # Save the plot with a name derived from the input file name
    plot_file_name = file_name.split('.')[0] + '_plot.png'
    plt.savefig(plot_file_name)
    plt.close()
    print(f"Saved plot as {plot_file_name}")

# Plot and Save MFCCs and Spectrum for aiueo.wav
plot_mfccs_and_spectrum('aiueo.wav')

# Plot and Save MFCCs and Spectrum for aiueo_.wav
plot_mfccs_and_spectrum('aiueo_.wav')
