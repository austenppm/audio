import numpy as np
import time

def custom_fft(x):
    N = len(x)
    if N <= 1: 
        return x

    even = custom_fft(x[0::2])
    odd = custom_fft(x[1::2])

    T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]

# Test the custom FFT implementation
x = np.array([1, 0, 3, 2, 4, 0, 2, 0])
fft_custom = custom_fft(x)



# Using NumPy's rFFT
start_time_np = time.time()
fft_np = np.fft.rfft(x)
end_time_np = time.time()
np_time = end_time_np - start_time_np

# Timing custom FFT
start_time_custom = time.time()
fft_custom = custom_fft(x)
end_time_custom = time.time()
custom_time = end_time_custom - start_time_custom

print("NumPy FFT result:", fft_np)
print("Custom FFT result:", fft_custom)
print("\nExecution time (NumPy's rFFT):", np_time)
print("Execution time (Custom FFT):", custom_time)
