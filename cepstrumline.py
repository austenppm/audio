import matplotlib.pyplot as plt
import numpy as np
import librosa

SR = 16000
LEVEL = 13

def get_log_spec(x_frame):
    spec = np.fft.rfft(x_frame)
    log_spec = np.log(np.abs(spec))
    return log_spec

def get_cepstrum(x_frame):
    spec = np.fft.rfft(x_frame)
    log_spec = np.log(np.abs(spec))
    cepstrum = np.fft.rfft(log_spec)
    return cepstrum[:LEVEL]

def plot_spectrum_and_cepstrum(file_name, sr=SR, order=LEVEL):
    # Load the audio file
    x, _ = librosa.load(file_name, sr=sr)

    # Calculate the logarithmic amplitude spectrum
    log_spec = get_log_spec(x)

    # Compute the cepstrum
    cepstrum = get_cepstrum(x)

    # Prepare frequency axis for the spectrum
    freq_axis = np.linspace(0, sr/2, len(log_spec))

    # Reconstruct the spectral envelope from the low-level cepstrum
    r = np.fft.irfft(cepstrum, n=len(log_spec))

    # Plot
    fig = plt.figure(figsize=(12, 6))
    plt.plot(freq_axis, log_spec, label='Logarithmic Amplitude Spectrum')
    plt.plot(freq_axis, r, label='Reconstructed Spectral Envelope')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.title('Spectrum and Cepstrum Analysis of ' + file_name)
    plt.legend()
    plt.show()

    # Save the plot
    plot_file_name = f'{file_name.split(".")[0]}_spectrum_cepstrum.png'
    fig.savefig(plot_file_name)
    plt.close(fig)

# Plot Spectrum and Cepstrum for aiueo.wav and aiueo_.wav
plot_spectrum_and_cepstrum('aiueo.wav')
plot_spectrum_and_cepstrum('aiueo_.wav')
