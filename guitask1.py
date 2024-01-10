import librosa
import numpy as np
import matplotlib.pyplot as plt
import tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Constants
size_frame = 4096
SR = 16000
size_shift = 16000 / 100

# Load the audio file
x, _ = librosa.load('aiueo.wav', sr=SR)
duration = len(x) / SR
fundamental_frequencies = []
FRAME_SIZE = 1024
HOP_SIZE = SR // 100

# Compute spectrogram and volume
hamming_window = np.hamming(size_frame)
spectrogram = []
volume = []
for i in np.arange(0, len(x) - size_frame, size_shift):
    idx = int(i)
    x_frame = x[idx:idx+size_frame]
    fft_spec = np.fft.rfft(x_frame * hamming_window)
    fft_log_abs_spec = np.log(np.abs(fft_spec))
    spectrogram.append(fft_log_abs_spec)
    vol = 20 * np.log10(np.mean(x_frame ** 2))
    volume.append(vol)

# Fundamental frequency functions
def is_peak(a, index):
    if index == 0 or index == len(a) - 1:
        return False
    return a[index] > a[index - 1] and a[index] > a[index + 1]

def find_fundamental_frequency(autocorr, sr):
    peak_indices = [i for i in range(1, len(autocorr) - 1) if is_peak(autocorr, i)]
    if not peak_indices:
        return 0
    max_peak_index = max(peak_indices, key=lambda index: autocorr[index])
    frequency = sr / max_peak_index
    return frequency

def calculate_overall_fundamental_freq():
    autocorr = np.correlate(x, x, 'full')
    autocorr = autocorr[len(autocorr)//2:]
    return find_fundamental_frequency(autocorr, SR)

overall_fundamental_freq = calculate_overall_fundamental_freq()

for i in range(0, len(x) - FRAME_SIZE, HOP_SIZE):
    frame = x[i:i + FRAME_SIZE]
    autocorr = np.correlate(frame, frame, 'full')
    autocorr = autocorr[len(autocorr) // 2:]

    frequency = find_fundamental_frequency(autocorr, SR)
    fundamental_frequencies.append(frequency)

# Initialize Tkinter
root = tkinter.Tk()
root.wm_title("Audio Signal Analysis")

# Frame setup
frame1 = tkinter.Frame(root)
frame2 = tkinter.Frame(root)
frame1.pack(side="left")
frame2.pack(side="left")

# Spectrogram plot
fig, ax1 = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=frame1)
spectrogram_image = ax1.imshow(np.flipud(np.array(spectrogram).T), extent=[0, duration, 0, 8000], aspect='auto', interpolation='nearest')
ax1.set_xlabel('sec')
ax1.set_ylabel('frequency [Hz]')
# Line for the overall fundamental frequency
ax1.plot(np.arange(0, duration, duration/len(fundamental_frequencies)), fundamental_frequencies, color='r', linestyle='-', label=f'Overall Fundamental Freq: {overall_fundamental_freq:.2f} Hz')
ax1.legend(loc='upper right')
canvas.draw()
canvas.get_tk_widget().pack(side="left")

# Volume plot
ax2 = ax1.twinx()
ax2.set_ylabel('volume [dB]')
x_data = np.linspace(0, duration, len(volume))
ax2.plot(x_data, volume, c='y')

# Callback function for slider
def _draw_spectrum(v):
    index = int((len(spectrogram)-1) * (float(v) / duration))
    frame = x[int(float(v) * SR):int(float(v) * SR) + size_frame]
    if len(frame) >= size_frame:
        autocorr = np.correlate(frame, frame, 'full')
        autocorr = autocorr[len(autocorr)//2:]
        fundamental_freq = find_fundamental_frequency(autocorr, SR)
        fundamental_freq_label.config(text=f"Fundamental Frequency: {fundamental_freq:.2f} Hz")
    spectrum = spectrogram[index]
    plt.cla()
    x_data = np.fft.rfftfreq(size_frame, d=1/SR)
    ax2.plot(x_data, spectrum)
    ax2.set_ylim(-10, 5)
    ax2.set_xlim(0, SR/2)
    ax2.set_ylabel('amplitude')
    ax2.set_xlabel('frequency [Hz]')
    canvas2.draw()

# Label to display the fundamental frequency
fundamental_freq_label = tkinter.Label(frame2, text="Fundamental Frequency: 0 Hz", font=("", 20))
fundamental_freq_label.pack()

# Spectrum plot
fig2, ax2 = plt.subplots()
canvas2 = FigureCanvasTkAgg(fig2, master=frame2)
canvas2.get_tk_widget().pack(side="top")

# Slider for time selection
scale = tkinter.Scale(command=_draw_spectrum, master=frame2, from_=0, to=duration, resolution=size_shift/SR, label='Select Time (sec)', orient=tkinter.HORIZONTAL, length=600, width=50, font=("", 20))
scale.pack(side="top")

# Start the GUI
root.mainloop()
