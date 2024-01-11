import librosa
import numpy as np
import matplotlib.pyplot as plt
import tkinter
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pickle

# Load the model
with open('aiueo_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Constants
size_frame = 4096
SR = 16000
size_shift = 16000 / 100
FRAME_SIZE = 1024
HOP_SIZE = SR // 100

# Initialize variables
duration = 0  # Default value for duration
x = np.array([])  # Empty array for audio data
volume = np.array([])  # Empty array for volume

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

def process_audio_file(filename):
    global x, duration, spectrogram, volume, fundamental_frequencies, overall_fundamental_freq, vowel_predictions
    
    x, _ = librosa.load(filename, sr=SR)
    duration = len(x) / SR
    spectrogram = []
    volume = []
    fundamental_frequencies = []
    vowel_predictions = []

    # Compute spectrogram and volume
    hamming_window = np.hamming(size_frame)
    for i in np.arange(0, len(x) - size_frame, size_shift):
        idx = int(i)
        x_frame = x[idx:idx+size_frame]
        fft_spec = np.fft.rfft(x_frame * hamming_window)
        fft_log_abs_spec = np.log(np.abs(fft_spec))
        spectrogram.append(fft_log_abs_spec)
        vol = 20 * np.log10(np.mean(x_frame ** 2))
        volume.append(vol)

    autocorr = np.correlate(x, x, 'full')
    autocorr = autocorr[len(autocorr)//2:]
    overall_fundamental_freq = find_fundamental_frequency(autocorr, SR)
    
    for i in range(0, len(x) - FRAME_SIZE, HOP_SIZE):
        frame = x[i:i + FRAME_SIZE]
        mfcc = librosa.feature.mfcc(y=frame, sr=SR, n_mfcc=20, n_fft=1024)  # Reduce n_fft to match frame size
        mfcc_mean = np.mean(mfcc, axis=1).reshape(1, -1)  # Reshape for single sample prediction
        prediction = model.predict(mfcc_mean)[0]
        vowel_predictions.append(prediction * 1500)  # Multiply by 1500 for graphing


    for i in range(0, len(x) - FRAME_SIZE, HOP_SIZE):
        frame = x[i:i + FRAME_SIZE]
        autocorr = np.correlate(frame, frame, 'full')
        autocorr = autocorr[len(autocorr) // 2:]
        frequency = find_fundamental_frequency(autocorr, SR)
        fundamental_frequencies.append(frequency)
        
    update_plots()
    
def update_plots():
    global ax1, ax2, ax3, canvas, scale
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax1.imshow(np.flipud(np.array(spectrogram).T), extent=[0, duration, 0, 8000], aspect='auto', interpolation='nearest')
    ax1.plot(np.arange(0, duration, duration/len(fundamental_frequencies)), fundamental_frequencies, color='r', linestyle='-', label=f'Overall Fundamental Freq: {overall_fundamental_freq:.2f} Hz')
    ax1.plot(np.arange(0, duration, duration/len(vowel_predictions)), vowel_predictions, color='b', linestyle='-', label='Vowel Predictions')
    ax1.legend(loc=(0.5, 1))  # Adjust the numbers as needed
    ax1.set_xlabel('sec')
    ax1.set_ylabel('frequency [Hz]')

    ax2.set_ylabel('volume [dB]')
    x_data = np.linspace(0, duration, len(volume))
    ax2.plot(x_data, volume, c='y')

    canvas.draw()

    scale.config(from_=0, to=duration)
    
def load_audio_file():
    filename = filedialog.askopenfilename(title="Select Audio File", filetypes=[("WAV Files", "*.wav")])
    if filename:
        process_audio_file(filename)
        
# Callback function for slider
def _draw_spectrum(v):
    # ax3 = ax1.twinx()
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
    ax3.plot(x_data, spectrum)
    ax3.set_ylim(-10, 5)
    ax3.set_xlim(0, SR/2)
    ax3.set_ylabel('amplitude')
    ax3.set_xlabel('frequency [Hz]')
    canvas2.draw()
        
# Initialize Tkinter
root = tkinter.Tk()
root.wm_title("Audio Signal Analysis")

# Frame setup for the left graph (Spectrogram) and its control (Load button)
frame_left = tkinter.Frame(root)
frame_left.pack(side="left", fill='both', expand=True)

frame_left_top = tkinter.Frame(frame_left)
frame_left_top.pack(side="top", fill='both', expand=True)

frame_left_bottom = tkinter.Frame(frame_left)
frame_left_bottom.pack(side="bottom", fill='x')

# Spectrogram plot setup 
fig, ax1 = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=frame_left_top)
canvas.get_tk_widget().pack(side="top", fill='both', expand=True)
ax1.set_xlabel('sec')
ax1.set_ylabel('frequency [Hz]')

# Volume plot setup
ax2 = ax1.twinx()

# Button for loading audio file
load_button = tkinter.Button(frame_left_bottom, text="Load Audio File", command=load_audio_file)
load_button.pack()

# Frame setup for the right graph (Spectrum) and its control (Slider)
frame_right = tkinter.Frame(root)
frame_right.pack(side="right", fill='both', expand=True)

frame_right_top = tkinter.Frame(frame_right)
frame_right_top.pack(side="top", fill='both', expand=True)

frame_right_bottom = tkinter.Frame(frame_right)
frame_right_bottom.pack(side="bottom", fill='x')

# Label to display the fundamental frequency
fundamental_freq_label = tkinter.Label(frame_right_top, text="Fundamental Frequency: 0 Hz", font=("", 20))
fundamental_freq_label.pack()

# Spectrum plot
fig3, ax3 = plt.subplots()
canvas2 = FigureCanvasTkAgg(fig3, master=frame_right_top)
canvas2.get_tk_widget().pack(side="top", fill='both', expand=True)

# Slider for time selection
scale = tkinter.Scale(frame_right_bottom, command=_draw_spectrum, from_=0, to=duration, resolution=size_shift/SR, label='Select Time (sec)', orient=tkinter.HORIZONTAL)
scale.pack(fill='x')

# Start the GUI
root.mainloop()