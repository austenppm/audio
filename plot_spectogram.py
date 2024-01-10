# 計 算 機 科 学 実 験 及 演 習 4「 音 響 信 号 処 理 」
# サ ン プ ル ソ ー ス コ ー ド
#
# 音 声 フ ァ イ ル を 読 み 込 み ， ス ペ ク ト ロ グ ラ ム を 計 算 し て 図 示 す る ．

# ラ イ ブ ラ リ の 読 み 込 み
import matplotlib.pyplot as plt
import numpy as np
import librosa

# サ ン プ リ ン グ レ ー ト
SR = 16000

# 音 声 フ ァ イ ル の 読 み 込 み
x, _ = librosa.load('aiueo.wav', sr=SR)

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
for i in np.arange(0, len(x) - size_frame, size_shift):
    # 該 当 フ レ ー ム の デ ー タ を 取 得
    idx = int(i)
    x_frame = x[idx:idx+size_frame]
    # 窓 掛 け し た デ ー タ を FFT
    fft_spec = np.fft.rfft(x_frame * hamming_window)
    # 複 素 ス ペ ク ト ロ グ ラ ム を 対 数 振 幅 ス ペ ク ト ロ グ ラ ム に
    fft_log_abs_spec = np.log(np.abs(fft_spec))
    # 計 算 し た 対 数 振 幅 ス ペ ク ト ロ グ ラ ム を 配 列 に 保 存
    spectrogram.append(fft_log_abs_spec)

# ス ペ ク ト ロ グ ラ ム を 画 像 に 表 示 ・ 保 存

# 画 像 と し て 保 存 す る た め の 設 定
fig = plt.figure()

# ス ペ ク ト ロ グ ラ ム を 描 画
plt.xlabel('sample') # x 軸 の ラ ベ ル を 設 定
plt.ylabel('frequency [Hz]') # y 軸 の ラ ベ ル を 設 定
plt.imshow(
    np.flipud(np.array(spectrogram).T), # 画 像 と み な す た め に ， デ ー タ を 転 地 し て 上 下 反 転
    extent=[0, len(x), 0, SR/2], # (横 軸 の 原 点 の 値 ， 横 軸 の 最 大 値 ， 縦 軸 の 原 点 の 値 ， 縦 軸 の 最 大 値)
    aspect='auto',
    interpolation='nearest'
)
plt.show()

# 縦 軸 の 最 大 値 は サ ン プ リ ン グ 周 波 数 の 半 分 = 16000 / 2 = 8000 Hz と な る

# 画 像 フ ァ イ ル に 保 存
fig.savefig('plot-spectrogram.png')
