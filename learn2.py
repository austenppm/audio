import numpy as np
import librosa
import matplotlib.pyplot as plt

def get_spec(x_frame):
    spec = np.fft.rfft(x_frame)
    log_spec = np.log(np.abs(spec))
    return log_spec

def get_cepstrum(x_frame):
    spec = get_spec(x_frame)
    cepstrum = np.fft.rfft(spec)
    log_cepstrum = np.log(np.abs(cepstrum))
    return log_cepstrum

def get_cepstrums(waveform, size_frame):
    cepstrums = []
    for i in np.arange(0, len(waveform) - size_frame, size_frame):
        idx = int(i)
        x_frame = waveform[idx: idx + size_frame]
        cepstrum = get_cepstrum(x_frame)
        if cepstrum.size > 0:  # Check if cepstrum is not empty
            cepstrums.append(cepstrum)
    return np.array(cepstrums)  # Convert list to numpy array

def spectrogram(waveform, size_frame, size_shift):
    spectrogram = []
    hamming_window = np.hamming(size_frame)
    for i in np.arange(0, len(waveform) - size_frame, size_shift):
        x_frame = waveform[int(i): int(i) + size_frame]
        fft_spec = np.fft.rfft(x_frame * hamming_window)
        fft_log_abs_spec = np.log(np.abs(fft_spec))
        spectrogram.append(fft_log_abs_spec)
    return spectrogram


def learn_avg(arr):
    if arr.size == 0:
        return None  # Return None if array is empty
    return np.average(arr, axis=0)

def learn_var(arr, avg):
    if arr.size == 0 or avg is None:
        return None  # Return None if array is empty or avg is None
    return np.average((arr - avg) ** 2, axis=0)

def likelihood(x, avg, var):
    return - np.sum((x - avg) ** 2 / var / 2 + np.log(var))

def predict(x, avgs, vars):
    max_likelihood = - np.inf
    ans = None
    for i, (avg, var) in enumerate(zip(avgs, vars)):
        like = likelihood(x, avg, var)
        if like > max_likelihood:
            max_likelihood = like
            ans = i
    return ans

def recognize(waveform, avgs, vars, size_frame):
    recognized = np.array([])
    for i in np.arange(0, len(waveform) - size_frame, size_frame):
        x_frame = waveform[int(i): int(i) + size_frame]
        pred = predict(x_frame, avgs, vars)
        recognized = np.append(recognized, pred if pred is not None else -1)
    return recognized

def extract_intervals(filename, sr, size_frame, size_shift, threshold):
    x, _ = librosa.load(filename, sr=sr)
    utterances = []
    in_utterance = False
    for i in range(0, len(x) - size_frame, int(size_shift)):
        frame = x[i:i+size_frame]
        frame_energy = 10 * np.log10(np.sum(frame**2))
        if frame_energy > threshold and not in_utterance:
            start = i / sr
            in_utterance = True
        elif frame_energy <= threshold and in_utterance:
            end = i / sr
            utterances.append((start, end))
            in_utterance = False
    if in_utterance:
        utterances.append((start, len(x) / sr))
    return utterances

# Main execution
SR = 16000
SIZE_FRAME = 512
SHIFT_SIZE = 16000 / 100  # 10 msec
THRESHOLD = -10

x_long, _ = librosa.load('aiueo_.wav', sr=SR)
x_short, _ = librosa.load('aiueo.wav', sr=SR)

intervals_short = extract_intervals('aiueo.wav', SR, SIZE_FRAME, SHIFT_SIZE, THRESHOLD)
intervals_long = extract_intervals('aiueo_.wav', SR, SIZE_FRAME, SHIFT_SIZE, THRESHOLD)

learn_data = [x_long[int(start*SR):int(end*SR)] for start, end in intervals_long[:5]]


# Update the code where you calculate averages and variances
avgs = []
vars = []
for data in learn_data:
    cepstrums = get_cepstrums(data, SIZE_FRAME)
    avg = learn_avg(cepstrums)
    var = learn_var(cepstrums, avg)
    if avg is not None and var is not None:
        avgs.append(avg)
        vars.append(var)


x_recognized = x_short
# Now, check if avgs and vars are not empty before calling recognize function
if avgs and vars:
    recognized = recognize(x_recognized, avgs, vars, SIZE_FRAME)
else:
    recognized = np.array([])  # Return an empty array if avgs or vars are empty
    

fig, ax = plt.subplots(figsize=(10, 4))
spec = spectrogram(x_recognized, SIZE_FRAME, SHIFT_SIZE)
img = np.flipud(np.array(spec).T)
ax.imshow(img, extent=[0, len(x_recognized) / SR, 0, SR / 2], aspect='auto', interpolation='nearest')

times = np.linspace(0, len(x_recognized) / SR, len(recognized))
ax.plot(times, recognized * 500, color='red', linewidth=1)

ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Frequency (Hz)')
ax.set_ylim(0, 2000)
ax.set_title('Vowel Recognition on Spectrogram')

plt.show()
fig.savefig('vowel_recognition.png')
