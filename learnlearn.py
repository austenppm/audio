import numpy as np
import librosa
import matplotlib.pyplot as plt
import pickle


def get_spec(x_frame):
  spec = np.fft.rfft(x_frame)
  log_spec = np.log(np.abs(spec))
  return log_spec


def get_cepstrum(x_frame):
  spec = get_spec(x_frame)
  cepstrum = np.fft.rfft(spec)
  log_cepstrum = np.log(np.abs(cepstrum))
  return log_cepstrum  # 横軸をfreqとする一次元配列


def get_cepstrums(waveform, size_frame):
  # waveform: 波を表す一次元配列
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
  return cepstrums  # 二次元配列


def spectrogram(waveform, size_frame, size_shift):
  spectrogram = []
  hamming_window = np.hamming(size_frame)

  for i in np.arange(0, len(waveform) - size_frame, size_shift):
    idx = int(i)
    x_frame = waveform[idx: idx + size_frame]

    # 窓掛けしたデータをFFT
    fft_spec = np.fft.rfft(x_frame * hamming_window)

    # 振幅スペクトルを対数化
    fft_log_abs_spec = np.log(np.abs(fft_spec))

    # 配列に保存
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


def recognize(waveform, avgs, vars, size_frame):
    recognized = np.array([])
    if waveform is None or len(waveform) == 0:
        return recognized  # Return empty array if waveform is None or empty

    for i in np.arange(0, len(waveform) - size_frame, size_frame):
        idx = int(i)
        x_frame = waveform[idx: idx + size_frame]
        pred = predict(x_frame, avgs, vars)
        if pred is None:
            pred = -1  # Assign a default value when no matching vowel is found
        recognized = np.append(recognized, pred)

    return recognized


def intervals_to_samples(intervals, sr):
    sample_intervals = {}
    vowels = ['a', 'i', 'u', 'e', 'o']

    for vowel, (start, end) in zip(vowels, intervals):
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        sample_intervals[vowel] = (start_sample, end_sample)

    return sample_intervals

SR = 16000

x_long, _ = librosa.load('aiueo_.wav', sr=SR)
x_short, _ = librosa.load('aiueo.wav', sr=SR)

# Usage
intervals_short = [(1.1, 1.34), (1.97, 2.2), (2.77, 3.01), (3.59, 3.82), (4.38, 4.62)]
intervals_long = [(0.47, 1.26), (1.31, 2.14), (2.17, 2.97), (3.01, 3.62), (3.83, 4.23)]
sample_intervals_short = intervals_to_samples(intervals_short, SR)
sample_intervals_long  = intervals_to_samples(intervals_long, SR)
print(sample_intervals_short)

# Assign segments to variables
a = x_short[sample_intervals_short['a'][0]:sample_intervals_short['a'][1]]
i = x_short[sample_intervals_short['i'][0]:sample_intervals_short['i'][1]]
u = x_short[sample_intervals_short['u'][0]:sample_intervals_short['u'][1]]
e = x_short[sample_intervals_short['e'][0]:sample_intervals_short['e'][1]]
o = x_short[sample_intervals_short['o'][0]:sample_intervals_short['o'][1]]

a_ = x_long[sample_intervals_long['a'][0]:sample_intervals_long['a'][1]]
i_ = x_long[sample_intervals_long['i'][0]:sample_intervals_long['i'][1]]    
u_ = x_long[sample_intervals_long['u'][0]:sample_intervals_long['u'][1]]
e_ = x_long[sample_intervals_long['e'][0]:sample_intervals_long['e'][1]]
o_ = x_long[sample_intervals_long['o'][0]:sample_intervals_long['o'][1]]

# # For the second set of data
# a = x_short[17600:21440]
# i = x_short[31520:35200]
# u = x_short[44320:48160]
# e = x_short[57440:61120]
# o = x_short[70080:73920]
learn_data = [a, i, u, e, o]
learn_data_ = [a_, i_, u_, e_, o_]

SIZE_FRAME = 512

avgs = []
vars = []
for data in learn_data:
  cepstrums = get_cepstrums(data, SIZE_FRAME)
  # print(cepstrums)
  avg = learn_avg(cepstrums)
  var = learn_var(cepstrums, avg)
  avgs.append(avg)
  vars.append(var)

# シフトサイズ
SHIFT_SIZE = 16000 / 100  # 10 msec

x_short, _ = librosa.load('aiueo_.wav', sr=SR)
x_implement = x_short

spec = spectrogram(x_implement, SIZE_FRAME, SHIFT_SIZE)
rec = recognize(x_implement, avgs, vars, SIZE_FRAME)

# Save the model
model = {'avgs': avgs, 'vars': vars}
with open('aiueo_model.pkl', 'wb') as file:
    pickle.dump(model, file)


fig = plt.figure()

plt.xlabel('sample')
plt.ylabel('frequency [Hz]')

plt.plot(np.linspace(0, len(x_implement), len(rec)), rec * 500, color = 'red')

plt.imshow(
    np.flipud(np.array(spec).T),
    extent=[0, len(x_implement), 0, SR / 2],
    aspect='auto',
    interpolation='nearest'
)
plt.ylim(0, 2000)

plt.show()
fig.savefig('exercise16learnlearn.png')