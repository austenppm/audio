import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from sklearn.svm import SVC  # Support Vector Classifier

def train_model(features, labels):
    # Train a simple SVM classifier
    clf = SVC()
    clf.fit(features, labels)
    return clf

def recognizeB(waveform, model, size_frame):
    recognized = np.array([])
    for i in np.arange(0, len(waveform) - size_frame, size_frame):
        idx = int(i)
        x_frame = waveform[idx: idx + size_frame]
        cepstrum = get_cepstrum(x_frame)
        cepstrum = cepstrum.reshape(1, -1)  # Reshape for single prediction
        pred = model.predict(cepstrum)[0]
        recognized = np.append(recognized, pred)
    return recognized

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

        if frame_energy > threshold and not in_utterance:
            start = i / SR
            in_utterance = True
        elif frame_energy <= threshold and in_utterance:
            end = i / SR
            utterances.append((start, end))
            in_utterance = False

    if in_utterance:
        utterances.append((start, len(x) / SR))

    return energy, utterances

def extract_utterances_(filename, SR, size_frame, size_shift, threshold, time_start, time_end):
    x, _ = librosa.load(filename, sr=SR)
    energy = []
    utterances = []
    in_utterance = False

    start_sample = int(time_start * SR)
    end_sample = int(time_end * SR)
    x = x[start_sample:end_sample]

    for i in range(0, len(x) - size_frame, int(size_shift)):
        frame = x[i:i+size_frame]
        frame_energy = 10 * np.log10(np.sum(frame**2))
        energy.append(frame_energy)

        if frame_energy > threshold and not in_utterance:
            start = i / SR + time_start
            in_utterance = True
        elif frame_energy <= threshold and in_utterance:
            end = i / SR + time_start
            utterances.append((start, end))
            in_utterance = False

    if in_utterance:
        utterances.append((start, (len(x) + start_sample) / SR))

    return energy, utterances

# Functions from exercise16.py
def get_spec(x_frame):
    spec = np.fft.rfft(x_frame)
    log_spec = np.log(np.abs(spec) + 1e-10 )
    return log_spec

def get_cepstrum(x_frame):
    spec = get_spec(x_frame)
    cepstrum = np.fft.rfft(spec)
    log_cepstrum = np.log(np.abs(cepstrum) + 1e-10)
    return log_cepstrum

def get_cepstrums(waveform, size_frame):
    cepstrums = np.array([])
    length1 = 0
    for i in np.arange(0, len(waveform) - size_frame, size_frame):
        idx = int(i)
        x_frame = waveform[idx: idx + size_frame]
        cepstrum = get_cepstrum(x_frame)
        cepstrums = np.append(cepstrums, cepstrum)
        length1 += 1
        length2 = len(cepstrum)
    cepstrums = cepstrums.reshape(length1, length2)
    return cepstrums

def spectrogram(waveform, size_frame, size_shift):
    spectrogram = []
    hamming_window = np.hamming(size_frame)

    for i in np.arange(0, len(waveform) - size_frame, size_shift):
        idx = int(i)
        x_frame = waveform[idx: idx + size_frame]
        fft_spec = np.fft.rfft(x_frame * hamming_window)
        fft_log_abs_spec = np.log(np.abs(fft_spec) + 1e-10)
        spectrogram.append(fft_log_abs_spec)
    return spectrogram

def learn_avg(arr):
    return np.average(arr, axis=0)

def learn_var(arr, avg):
    return np.average((arr - avg) ** 2, axis=0)

def likelihood(x, avg, var):
    return - np.sum((x - avg) ** 2 / var / 2 + np.log(var))

def predict(x, avgs, vars):
    ans = None
    max_likelihood = - np.inf
    for i, (avg, var) in enumerate(zip(avgs, vars)):
        cepstrums = get_cepstrum(x)
        like = likelihood(cepstrums, avg, var)
        if like > max_likelihood:
            max_likelihood = like
            ans = i
    return ans

def recognizeA(waveform, avgs, vars, size_frame):
    recognized = np.array([])
    for i in np.arange(0, len(waveform) - size_frame, size_frame):
        idx = int(i)
        x_frame = waveform[idx: idx + size_frame]
        pred = predict(x_frame, avgs, vars)
        if pred is None:
            pred = -1  # Assign a default value when no matching vowel is found
        recognized = np.append(recognized, pred)
    return recognized


# Settings
SR = 16000
size_frame = 512
size_shift = 16000 / 100  # 0.001 sec (10 msec)
threshold = -10
threshold_ = -4.5  # Threshold for volume (dB)
time_start = 0  # Start time in seconds
time_end = 5 


# Load audio files
x_short, _ = librosa.load('aiueo.wav', sr=SR)
x_long, _ = librosa.load('aiueo_.wav', sr=SR)

# Extract intervals using utterances from the short "aiueo" file
_, intervals_shorta = extract_utterances('aiueo.wav', SR, size_frame, size_shift, threshold)
_, intervals_longa = extract_utterances_('aiueo_.wav', SR, size_frame, size_shift, threshold_, time_start, time_end)


def adjust_intervals(intervals, delta=0.02):
    # Adjust each interval by decreasing the start by delta and increasing the end by delta
    adjusted_intervals = [(start - delta, end + delta) for start, end in intervals]
    return adjusted_intervals

# Adjusted intervals
intervals_short = adjust_intervals(intervals_shorta)
intervals_long = adjust_intervals(intervals_longa)

# Ensure that there are at least 5 utterances (for vowels a, i, u, e, o)
if len(intervals_short) < 5:
    raise ValueError("Less than 5 utterances detected in the short file.")

if len(intervals_long) < 5:
    raise ValueError("Less than 5 utterances detected in the long file.")

vowels = ['a', 'i', 'u', 'e', 'o']

print(intervals_long)
print(intervals_short)
# print("Utterance periods in 'aiueo_.wav':")
# for i, (start, end) in enumerate(intervals_long):
#     if i < len(vowels):
#         print(f"{vowels[i].upper()}_ : Start: {start:.2f}s, End: {end:.2f}s")
#     else:
#         break  # In case there are more than 5 utterances detected

# Learning phase using intervals from short "aiueo"
avgs = []
vars = []
for i, (start, end) in enumerate(intervals_short[:5]):  # Use first 5 utterances
    start_sample = int(start * SR)
    end_sample = int(end * SR)
    segment = x_short[start_sample:end_sample]
    cepstrums = get_cepstrums(segment, size_frame)
    avg = learn_avg(cepstrums)
    var = learn_var(cepstrums, avg)
    avgs.append(avg)
    vars.append(var)

x_recognized = x_short
# Recognition phase using the same "aiueo.wav" file
recognized = recognizeA(x_recognized, avgs, vars, size_frame)
print(recognized)

# Plotting
fig, ax = plt.subplots(figsize=(10, 4))

x_spec = x_long

# Spectrogram of the same file used for training and recognition
spec = spectrogram(x_spec, size_frame, size_shift)
img = np.flipud(np.array(spec).T)
extent = [0, len(x_spec) / SR, 0, SR / 2]
ax.imshow(img, extent=extent, aspect='auto', interpolation='nearest')

# Recognized phonemes
times = np.linspace(0, len(x_spec) / SR, len(recognized))
# Filter out -1 values or handle them as needed
filtered_recognized = np.where(recognized == -1, np.nan, recognized)  # Replace -1 with NaN
ax.plot(times, filtered_recognized * 500, color='red', linewidth=1)

# Prepare data for training
# Extract features and labels for each vowel segment
features = []
labels = []
vowels = ['a', 'i', 'u', 'e', 'o']
for i, (start, end) in enumerate(intervals_short[:5]):
    start_sample = int(start * SR)
    end_sample = int(end * SR)
    segment = x_short[start_sample:end_sample]
    cepstrums = get_cepstrums(segment, size_frame)
    features.extend(cepstrums)
    labels.extend([i] * len(cepstrums))  # Label each vowel with an integer

# Convert lists to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Train model
model = train_model(features, labels)

# Recognize vowels using the trained model
recognized_labels = recognizeB(x_short, model, size_frame)
# print(recognized_labels)

# Apply post-processing smoothing
smoothed_recognized_labels = median_filter(recognized_labels, size=5)

# Apply median filter for smoothing
smoothed_recognized = median_filter(filtered_recognized, size=5)  # Adjust size as needed

# Plotting with smoothed labels
ax.plot(times, smoothed_recognized * 500, color='blue', linewidth=1, label='Smoothed Recognized Vowels')

ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Frequency (Hz) (x500 of the recognized phonemes)))')
ax.set_ylim(0, 2000)
ax.set_title('Vowel Recognition on Spectrogram of aiueo.wav')

plt.show()
fig.savefig('exercise16_same_fileAB.png')

# # Plotting
# fig, ax = plt.subplots(figsize=(10, 4))
# spec = spectrogram(x_short, size_frame, size_shift)
# img = np.flipud(np.array(spec).T)
# extent = [0, len(x_short) / SR, 0, SR / 2]
# ax.imshow(img, extent=extent, aspect='auto', interpolation='nearest')

# # Plot recognized phonemes
# times = np.linspace(0, len(x_short) / SR, len(smoothed_recognized_labels))
# ax.plot(times, smoothed_recognized_labels * 500, color='blue', linewidth=1, label='Smoothed Recognized Vowels')

# ax.set_xlabel('Time (seconds)')
# ax.set_ylabel('Frequency (Hz) (x500 of the recognized phonemes)')
# ax.set_ylim(0, 2000)
# ax.set_title('Vowel Recognition on Spectrogram of aiueo.wav')
# ax.legend()

# plt.show()
# fig.savefig('exercise16_same_fileB.png')

