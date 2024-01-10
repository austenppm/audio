import matplotlib.pyplot as plt
import numpy as np
import librosa

# Function to calculate short-time energy and extract utterances
def extract_utterances(filename, SR, size_frame, size_shift, threshold):
    x, _ = librosa.load(filename, sr=SR)
    energy = []
    utterances = []
    in_utterance = False

    for i in range(0, len(x) - size_frame, int(size_shift)):
        frame = x[i:i+size_frame]
        frame_energy = 10 * np.log10(np.sum(frame**2))
        energy.append(frame_energy)

        # Check if current frame is above the threshold
        if frame_energy > threshold and not in_utterance:
            start = i / SR  # Convert frame index to time
            in_utterance = True
        elif frame_energy <= threshold and in_utterance:
            end = i / SR  # Convert frame index to time
            utterances.append((start, end))
            in_utterance = False

    # Check for last utterance not ending
    if in_utterance:
        utterances.append((start, len(x) / SR))

    return energy, utterances

# Settings
SR = 16000
size_frame = 512
size_shift = 16000 / 100  # 0.001 sec (10 msec)
threshold = -10  # Threshold for volume (dB)

# Process files and extract utterances
energy, utterances = extract_utterances('aiueo.wav', SR, size_frame, size_shift, threshold)

# Label utterances
vowels = ['a', 'i', 'u', 'e', 'o']
print("Utterance periods in 'aiueo.wav':")
for i, (start, end) in enumerate(utterances):
    if i < len(vowels):
        print(f"{vowels[i].upper()} : Start: {start:.2f}s, End: {end:.2f}s")
    else:
        break  # In case there are more than 5 utterances detected

# Plotting is optional
plt.plot(np.linspace(0, len(energy) * size_shift / SR, len(energy)), energy)
plt.axhline(y=threshold, color='r', linestyle='-')
plt.xlabel('Time (seconds)')
plt.ylabel('Volume (dB)')
plt.show()

def extract_utterances_(filename, SR, size_frame, size_shift, threshold, time_start, time_end):
    x, _ = librosa.load(filename, sr=SR)
    energy = []
    utterances = []
    in_utterance = False

    # Convert time range to sample range
    start_sample = int(time_start * SR)
    end_sample = int(time_end * SR)

    # Slicing the audio data to the specified time range
    x = x[start_sample:end_sample]

    for i in range(0, len(x) - size_frame, int(size_shift)):
        frame = x[i:i+size_frame]
        frame_energy = 10 * np.log10(np.sum(frame**2))
        energy.append(frame_energy)

        # Check if current frame is above the threshold
        if frame_energy > threshold and not in_utterance:
            start = i / SR + time_start  # Adjust start time by adding time_start
            in_utterance = True
        elif frame_energy <= threshold and in_utterance:
            end = i / SR + time_start  # Adjust end time by adding time_start
            utterances.append((start, end))
            in_utterance = False

    # Check for last utterance not ending
    if in_utterance:
        utterances.append((start, (len(x) + start_sample) / SR))

    return energy, utterances

# Settings
SR_ = 16000
size_frame_ = 512
size_shift_ = 16000 / 100  # 0.001 sec (10 msec)
threshold_ = -4.5  # Threshold for volume (dB)
time_start = 0  # Start time in seconds
time_end = 5    # End time in seconds

# Process files and extract utterances within the specified time range
energy, utterances = extract_utterances_('aiueo_.wav', SR_, size_frame_, size_shift_, threshold_, time_start, time_end)
print("Utterance periods in 'aiueo_.wav':")
for i, (start, end) in enumerate(utterances):
    if i < len(vowels):
        print(f"{vowels[i].upper()}_ : Start: {start:.2f}s, End: {end:.2f}s")
    else:
        break  # In case there are more than 5 utterances detected

# Plotting is optional, if you want to visualize
plt.plot(np.linspace(time_start, time_end, len(energy)), energy)
plt.axhline(y=threshold_, color='r', linestyle='-')
plt.xlabel('Time (seconds)')
plt.ylabel('Volume (dB)')
plt.xlim(time_start, time_end)
plt.show()