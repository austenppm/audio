import matplotlib.pyplot as plt
import numpy as np
import librosa

# Function to calculate short-time energy
def calculate_energy(filename, SR, size_frame, size_shift):
    x, _ = librosa.load(filename, sr=SR)
    energy = []

    for i in range(0, len(x) - size_frame, int(size_shift)):
        frame = x[i:i+size_frame]
        energy.append(10 * np.log10(np.sum(frame**2)))

    return energy, np.linspace(0, len(x) / SR, len(energy))

# Settings
SR = 16000
size_frame = 512
size_shift = 16000 / 100  # 0.001 sec (10 msec)

# Calculate energy
energy1, time1 = calculate_energy('aiueo.wav', SR, size_frame, size_shift)
energy2, time2 = calculate_energy('aiueo_.wav', SR, size_frame, size_shift)

# Plot
fig, axs = plt.subplots(2, 1, figsize=(10, 6))

# Plot for aiueo.wav
axs[0].plot(time1, energy1)
axs[0].set_title('Volume vs Time for aiueo.wav')
axs[0].set_xlabel('Time (seconds)')
axs[0].set_ylabel('Volume (dB)')

# Plot for aiueo_.wav
axs[1].plot(time2, energy2)
axs[1].set_title('Volume vs Time for aiueo_.wav')
axs[1].set_xlabel('Time (seconds)')
axs[1].set_ylabel('Volume (dB)')

plt.tight_layout()
plt.show()
