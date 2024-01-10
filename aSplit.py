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

# Separate the audio into different segments
x_a = x[15000:25000] 
x_i = x[25000:40000]
x_u = x[40000:55000]
x_e = x[55000:70000]
x_o = x[70000:85000]

# 高 速 フ ー リ エ 変 換
fft_spec = np.fft.rfft(x)
fft_spec_a = np.fft.rfft(x_a)
fft_spec_i = np.fft.rfft(x_i)   
fft_spec_u = np.fft.rfft(x_u)   
fft_spec_e = np.fft.rfft(x_e)   
fft_spec_o = np.fft.rfft(x_o)  

# Convert the complex spectrum to logarithmic amplitude spectrum
fft_log_abs_spec = np.log(np.abs(fft_spec) + 1e-10)
fft_log_abs_spec_a = np.log(np.abs(fft_spec_a) + 1e-10)
fft_log_abs_spec_i = np.log(np.abs(fft_spec_i) + 1e-10)
fft_log_abs_spec_u = np.log(np.abs(fft_spec_u) + 1e-10)
fft_log_abs_spec_e = np.log(np.abs(fft_spec_e) + 1e-10)
fft_log_abs_spec_o = np.log(np.abs(fft_spec_o) + 1e-10)

# Prepare x-axis data for plotting
x_data = np.linspace(0, SR/2, len(fft_log_abs_spec))
x_data_a = np.linspace(0, SR/2, len(fft_log_abs_spec_a))
x_data_i = np.linspace(0, SR/2, len(fft_log_abs_spec_i))
x_data_u = np.linspace(0, SR/2, len(fft_log_abs_spec_u))   
x_data_e = np.linspace(0, SR/2, len(fft_log_abs_spec_e))
x_data_o = np.linspace(0, SR/2, len(fft_log_abs_spec_o))

# Plot and save figure for whole spectrum
fig = plt.figure()
plt.xlabel('frequency [Hz]')
plt.ylabel('amplitude')
plt.xlim([0, SR/2])
plt.plot(x_data, fft_log_abs_spec)
fig.savefig('plot-spectrum-whole.png')

# Plot and save figures for each vowel
def plot_and_save_vowel(data, fft_data, filename):
    fig = plt.figure()
    plt.xlabel('frequency [Hz]')
    plt.ylabel('amplitude')
    plt.xlim([0, SR/2])
    plt.plot(data, fft_data)
    fig.savefig(filename)

plot_and_save_vowel(x_data_a, fft_log_abs_spec_a, 'plot-spectrum-a.png')
plot_and_save_vowel(x_data_i, fft_log_abs_spec_i, 'plot-spectrum-i.png')
plot_and_save_vowel(x_data_u, fft_log_abs_spec_u, 'plot-spectrum-u.png')
plot_and_save_vowel(x_data_e, fft_log_abs_spec_e, 'plot-spectrum-e.png')
plot_and_save_vowel(x_data_o, fft_log_abs_spec_o, 'plot-spectrum-o.png')

# Plot and save figure with enlarged x-axis limit
fig = plt.figure()
plt.xlabel('frequency [Hz]')
plt.ylabel('amplitude')
plt.xlim([0, 2000])
plt.plot(x_data, fft_log_abs_spec)
fig.savefig('plot-spectrum-2000.png')
