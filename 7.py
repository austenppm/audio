import numpy as np
import librosa
import matplotlib.pyplot as plt
import time

# Custom FFT Implementation
def custom_fft(x):
    N = len(x)
    if N <= 1:
        return x

    # Pad the array if its length is not a power of two
    if np.log2(N) % 1 > 0:
        next_pow2 = int(2 ** np.ceil(np.log2(N)))
        x = np.append(x, [0] * (next_pow2 - N))
        N = next_pow2

    even = custom_fft(x[0::2])
    odd = custom_fft(x[1::2])

    T = [np.exp(-2j * np.pi * k / N) * odd[k % (N//2)] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]

# Load the audio file
y, sr = librosa.load('aiueo.wav', sr=None)

# Apply custom FFT
start_time_custom = time.time()
fft_custom = custom_fft(y)
end_time_custom = time.time()

# Apply NumPy's rFFT
start_time_np = time.time()
fft_np = np.fft.rfft(y)
end_time_np = time.time()

# Compute execution times
execution_time_custom = end_time_custom - start_time_custom
execution_time_np = end_time_np - start_time_np

# Plotting the results for comparison
plt.figure(figsize=(12, 6))

# Custom FFT Result
plt.subplot(2, 1, 1)
plt.plot(np.abs(fft_custom)[:len(fft_custom)//2])
plt.title('Custom FFT Magnitude Spectrum')
plt.xlabel('Frequency Bin')
plt.ylabel('Magnitude')

# NumPy rFFT Result
plt.subplot(2, 1, 2)
plt.plot(np.abs(fft_np))
plt.title("NumPy's rFFT Magnitude Spectrum")
plt.xlabel('Frequency Bin')
plt.ylabel('Magnitude')

plt.tight_layout()
plt.show()

# Print execution times
print(f"Execution Time - Custom FFT: {execution_time_custom} seconds")
print(f"Execution Time - NumPy rFFT: {execution_time_np} seconds")
