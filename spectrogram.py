# Load the audio file
y, sr = librosa.load('nekocut.wav', sr=None)

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