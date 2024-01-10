import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import math

# Your provided functions
def hz2nn(frequency):
    return int(round(12.0 * (math.log(frequency / 440.0) / math.log(2.0)))) + 69

# Function to estimate chords based on chroma vectors
def estimate_chords(chromagram):
    chords = []
    # Define the chord templates for major and minor in all 12 keys
    CHORDS = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
    MAJOR = [0, 4, 7]  # Template for major chord
    MINOR = [0, 3, 7]  # Template for minor chord

    for chroma in chromagram.T:
        major_chords = [sum(chroma[(np.array(MAJOR) + n) % 12]) for n in range(12)]
        minor_chords = [sum(chroma[(np.array(MINOR) + n) % 12]) for n in range(12)]
        major_idx, minor_idx = np.argmax(major_chords), np.argmax(minor_chords)
        if major_chords[major_idx] > minor_chords[minor_idx]:
            chords.append(CHORDS[major_idx] + " major")
        else:
            chords.append(CHORDS[minor_idx] + " minor")
    return chords

# Load the audio file
y, sr = librosa.load('nekocut.wav')

# サ ン プ リ ン グ レ ー ト
SR = 16000

# 短 時 間 フ ー リ エ 変 換

# フ レ ー ム サ イ ズ
size_frame = 512 # 2の べ き 乗

# フ レ ー ム サ イ ズ に 合 わ せ て ハ ミ ン グ 窓 を 作 成
hamming_window = np.hamming(size_frame)

# シ フ ィ フ ト サ イ ズ
size_shift = 16000 / 100 # 0.001 秒 (10 msec)

# ス ペ ク ト ロ グ ラ ム を 保 存 す る list
spectrogram = []

# size_shift 分ずらしながら size_frame 分のデータを取得
for i in np.arange(0, len(y) - size_frame, size_shift):
    # 該 当 フ レ ー ム の デ ー タ を 取 得
    idx = int(i)
    x_frame = y[idx:idx+size_frame]
    # 窓 掛 け し た デ ー タ を FFT
    fft_spec = np.fft.rfft(x_frame * hamming_window)
    # 複 素 ス ペ ク ト ロ グ ラ ム を 対 数 振 幅 ス ペ ク ト ロ グ ラ ム に
    fft_log_abs_spec = np.log(np.abs(fft_spec))
    # 計 算 し た 対 数 振 幅 ス ペ ク ト ロ グ ラ ム を 配 列 に 保 存
    spectrogram.append(fft_log_abs_spec)

# Create a chromagram
chromagram = librosa.feature.chroma_stft(y=y, sr=sr)

# Estimate chords
estimated_chords = estimate_chords(chromagram)

# Improved figure size and layout
plt.figure(figsize=(12, 9))

# Plot the spectrogram
plt.subplot(3, 1, 1)

# ス ペ ク ト ロ グ ラ ム を 描 画
plt.xlabel('sample') # x 軸 の ラ ベ ル を 設 定
plt.ylabel('frequency [Hz]') # y 軸 の ラ ベ ル を 設 定
plt.imshow(
    np.flipud(np.array(spectrogram).T), # 画 像 と み な す た め に ， デ ー タ を 転 地 し て 上 下 反 転
    extent=[0, len(y), 0, SR/2], # (横 軸 の 原 点 の 値 ， 横 軸 の 最 大 値 ， 縦 軸 の 原 点 の 値 ， 縦 軸 の 最 大 値)
    aspect='auto',
    interpolation='nearest'
)
plt.title('Spectrogram')

# Plot the chromagram
plt.subplot(3, 1, 2)
librosa.display.specshow(chromagram, y_axis='chroma', x_axis='time', cmap='viridis')
plt.colorbar()
plt.title('Chromagram')

# Plot the estimated chords
plt.subplot(3, 1, 3)
times = np.linspace(0, len(y) / sr, len(estimated_chords))
plt.plot(times, estimated_chords, label='Estimated Chords', color='darkred')
plt.xlabel('Time (s)')
plt.ylabel('Chord')
plt.xticks(rotation=45)  # Rotate x labels for better readability
plt.legend(loc='upper right')
plt.tight_layout()

plt.savefig('19(2).png', dpi=300)
plt.show()
