import matplotlib.pyplot as plt
import numpy as np
import librosa

SR = 16000
LEVEL = 13

def get_log_spec(x_frame):
    spec = np.fft.rfft(x_frame)
    log_spec = np.log(np.abs(spec) + 1e-10)  # Adding epsilon to avoid log(0)
    return log_spec

def get_cepstrum(x_frame):
    spec = np.fft.rfft(x_frame)
    log_spec = np.log(np.abs(spec) + 1e-10)  # Adding epsilon to avoid log(0)
    cepstrum_full = np.fft.rfft(log_spec)
    return cepstrum_full

def plot_spectrum_and_cepstrum(file_name, vowel, start_time, end_time, sr=SR, min_quef=0.005):
    # Load the audio file
    x, _ = librosa.load(file_name, sr=sr)

    # Extract the segment
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    x_segment = x[start_sample:end_sample]

    # Calculate the logarithmic amplitude spectrum
    log_spec = get_log_spec(x_segment)

    # Compute the full cepstrum
    cepstrum_full = get_cepstrum(x_segment)

    # Prepare frequency axis for the spectrum
    freq_axis = np.linspace(0, sr/2, len(log_spec))

    # Prepare quefrency axis for the full cepstrum
    quef_axis = np.linspace(0, len(x_segment) / sr / 2, num=len(cepstrum_full))

    # Ignore initial part of the cepstrum
    valid_indices = quef_axis >= min_quef
    quef_axis = quef_axis[valid_indices]
    cepstrum_full_adjusted = cepstrum_full[valid_indices]

    # Plot
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    # Plotting the logarithmic amplitude spectrum and low-order cepstrum
    axs[0].plot(freq_axis, log_spec, label='Logarithmic Amplitude Spectrum')
    axs[0].plot(freq_axis, np.fft.irfft(cepstrum_full.real[:LEVEL], n=len(log_spec)), label='Reconstructed Spectral Envelope', color='orange')
    axs[0].set_xlabel('Frequency [Hz]')
    axs[0].set_ylabel('Amplitude')
    axs[0].set_title(f'Logarithmic Amplitude Spectrum and Low-Order Cepstrum of {vowel} in {file_name}')
    axs[0].legend()

    # Plotting the full cepstrum
    axs[1].plot(quef_axis, cepstrum_full_adjusted.real, label='Full Cepstrum', color='green')
    axs[1].set_xlabel('Quefrency [s]')
    axs[1].set_ylabel('Cepstral Coefficients')
    # axs[1].set_xlim([0.01, 0.4])
    axs[1].set_title(f'Full Cepstrum of {vowel} in {file_name}')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

    # Save the plot
    plot_file_name = f'{file_name.split(".")[0]}_{vowel}_ignored_spectrum_cepstrum.png'
    fig.savefig(plot_file_name)
    plt.close(fig)

# Define intervals for aiueo.wav and aiueo_.wav
intervals_aiueo = {
    'A': (1.12, 1.32), 'I': (1.99, 2.18), 'U': (2.79, 2.99),
    'E': (3.61, 3.80), 'O': (4.40, 4.60)
}
intervals_aiueo_ = {
    'A_': (0.49, 1.24), 'I_': (1.33, 2.12), 'U_': (2.19, 2.95),
    'E_': (3.03, 3.60), 'O_': (3.85, 4.21)
}

# Plot and save spectrum and cepstrum for each interval
for vowel, (start, end) in intervals_aiueo.items():
    plot_spectrum_and_cepstrum('aiueo.wav', vowel, start, end)
for vowel, (start, end) in intervals_aiueo_.items():
    plot_spectrum_and_cepstrum('aiueo_.wav', vowel, start, end)
