import numpy as np
import librosa
import matplotlib.pyplot as plt


def zero_cross(waveform):
  d = np.array(waveform)
  return sum([1 if x < 0.0 else 0 for x in d[1:] * d[:-1]])


def zero_cross_rate(waveform, sample_rate):
  # 単位時間あたりのゼロクロス数
  return zero_cross(waveform) / (len(waveform) / sample_rate)


def zero_cross_rates(waveform, sample_rate, size_frame):
  zcrs = []
  for i in np.arange(0, len(waveform) - size_frame, size_frame):
    idx = int(i)
    x_frame = waveform[idx: idx + size_frame]
    zcr = zero_cross_rate(x_frame, sample_rate)
    zcrs.append(zcr)
  return zcrs


def is_peak(arr, index):
  if index == 0 or index == len(arr) - 1:
    return False
  return arr[index - 1] < arr[index] and arr[index] > arr[index + 1]


def get_f0(waveform, sampling_rate):
  autocorr = np.correlate(waveform, waveform, 'full')
  autocorr = autocorr[len(autocorr) // 2:]  # 不要な前半を捨てる

  # ピークを検出
  peak_indices = [i for i in range(len(autocorr)) if is_peak(autocorr, i)]
  peak_indices = [i for i in peak_indices if i != 0]  # 最初のピークは除く

  if len(peak_indices) == 0:
    return 0

  max_peak_index = max(peak_indices, key=lambda index: autocorr[index])

  # 基本周波数を推定
  f0 = sampling_rate / max_peak_index
  return f0


def get_f0s(waveform, sampling_rate, size_frame):
  f0s = []
  for i in np.arange(0, len(waveform) - size_frame, size_frame):
    idx = int(i)
    x_frame = waveform[idx: idx + size_frame]
    f0 = get_f0(x_frame, sampling_rate)
    f0s.append(f0)
  return f0s


def is_voiced_sound(waveform, sampling_rate):
  f0 = get_f0(waveform, sampling_rate)
  zcr = zero_cross_rate(waveform, sampling_rate)
  if f0 == 0:
    return False
  rate = zcr / f0
  return 1.95 < rate < 6.0


def get_f0s_voiced(waveform, sampling_rate, size_frame):
  f0s = []
  for i in np.arange(0, len(waveform) - size_frame, size_frame):
    idx = int(i)
    x_frame = waveform[idx: idx + size_frame]
    if is_voiced_sound(x_frame, sampling_rate):
      f0 = get_f0(x_frame, sampling_rate)
    else:
      f0 = 0
    f0s.append(f0)
  return f0s


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


SR = 16000
SIZE_FRAME = 512
SHIFT_SIZE = int(SR / 100)  # 10 msec

# Load the audio file
x, _ = librosa.load('output.wav', sr=SR)

# Calculate spectrogram
spec = spectrogram(x, SIZE_FRAME, SHIFT_SIZE)

# Estimate fundamental frequencies
f0s = get_f0s(x, SR, SIZE_FRAME)

# Estimate fundamental frequencies for voiced segments
f0s_voiced = get_f0s_voiced(x, SR, SIZE_FRAME)

# Calculate zero-crossing rates
zcrs = zero_cross_rates(x, SR, SIZE_FRAME)

# Plotting
fig, ax1 = plt.subplots(figsize=(12, 6))

# Spectrogram
img = ax1.imshow(np.flipud(np.array(spec).T), extent=[0, len(x)/SR, 0, SR/2], aspect='auto', interpolation='nearest')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Frequency (Hz)')
ax1.set_ylim(0, 2000)
fig.colorbar(img, ax=ax1, format='%+2.0f dB', label='Log Amplitude')

# Overlaying fundamental frequencies and zero-crossing rates
times = np.linspace(0, len(x)/SR, len(f0s_voiced))
ax2 = ax1.twinx()  # Create another y-axis
ax2.plot(times, f0s_voiced, label='Voiced F0', color='white')
ax2.plot(times, f0s, label='All F0', color='cyan', alpha=0.8)
ax2.plot(times, zcrs, label='Zero-Crossing Rates', color='red', alpha=0.8)
ax2.set_ylabel('Frequency (Hz) / ZCR')
ax2.legend(loc='upper right')

plt.title('Spectrogram with Fundamental Frequency and Zero-Crossing Rates')
plt.show()

# Save the plot
fig.savefig('zerocross.png')