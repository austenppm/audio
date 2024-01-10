import numpy as np
import matplotlib.pyplot as plt

def compute_autocorrelation_via_fft(signal):
    # Compute the Fourier Transform
    F_signal = np.fft.fft(signal)

    # Compute the power spectrum
    power_spectrum = np.abs(F_signal) ** 2

    # Compute the inverse Fourier Transform of the power spectrum
    autocorrelation = np.fft.ifft(power_spectrum)

    return autocorrelation.real

# Example usage
signal = np.array([1, 2, 3, 4, 3, 2, 1])
autocorrelation = compute_autocorrelation_via_fft(signal)

# Plotting
plt.stem(autocorrelation)
plt.title('Autocorrelation of the Signal')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.show()
