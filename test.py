# 計 算 機 科 学 実 験 及 演 習 4「 音 響 信 号 処 理 」
# サ ン プ ル ソ ー ス コ ー ド
#
# 音 声 フ ァ イ ル を 読 み 込 み ， フ ー リ エ 変 換 を 行 う ．

# ラ イ ブ ラ リ の 読 み 込 み
import matplotlib.pyplot as plt
import numpy as np
import librosa

# サ ン プ リ ン グ レ ー ト
SR = 16000

# 音 声 フ ァ イ ル の 読 み 込 み
x, _ = librosa.load('aiueo_.wav', sr=SR)

# # 高 速 フ ー リ エ 変 換
# # np.fft. r f f t を 使 用 す る と F F T の 前 半 部 分 の み が 得 ら れ る
# fft_spec = np.fft.rfft(x)

# # 複 素 ス ペ ク ト ル を 対 数 振 幅 ス ペ ク ト ル に
# fft_log_abs_spec = np.log(np.abs(fft_spec))

# # ス ペ ク ト ル を 画 像 に 表 示 ・ 保 存

# 画 像 と し て 保 存 す る た め の 設 定
fig = plt.figure()

# ス ペ ク ト ロ グ ラ ム を 描 画
plt.xlabel('frequency [Hz]') # x 軸 の ラ ベ ル を 設 定
plt.ylabel('amplitude') # y 軸 の ラ ベ ル を 設 定
# plt.xlim([0, SR/2]) # x 軸 の 範 囲 を 設 定

# x 軸 の デ ー タ を 生 成 （ 元 々 の デ ー タ が 0 ˜8000 H z に 対 応 す る よ う に す る ）
x_data = np.linspace((SR/2)/len(x), SR/2, len(x))
print(len(x))
print((SR/2)/len(x))
print(SR/2)
plt.plot(x_data, x) # 描 画

# 縦 軸 の 最 大 値 は サ ン プ リ ン グ 周 波 数 の 半 分 = 16000 / 2 = 8000 Hz と な る
plt.show()

# 画 像 フ ァ イ ル に 保 存
fig.savefig('test.png')