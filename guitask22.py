import librosa
import numpy as np
import matplotlib.pyplot as plt
import tkinter
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pickle
import time
import threading
import vlc  # Import VLC module
from pydub import AudioSegment

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
duration, x, volume = 0, np.array([]), np.array([])
vlc_instance, vlc_player = vlc.Instance(), vlc.Instance().media_player_new()

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
    ax1.legend(loc='lower left', bbox_to_anchor=(0,1))  # Adjust the numbers as needed
    ax1.set_xlabel('sec')
    ax1.set_ylabel('frequency [Hz]')

    ax2.set_ylabel('volume [dB]')
    x_data = np.linspace(0, duration, len(volume))
    ax2.plot(x_data, volume, c='y', label='Volume')
    ax2.legend(loc='lower right', bbox_to_anchor=(1,1))  # Adjust the numbers as needed

    canvas.draw()

    scale.config(from_=0, to=duration)
    
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

def on_position_changed(_):
    # Update the slider based on the current playback position
    current_position = vlc_player.get_time() / 1000.0  # Convert milliseconds to seconds
    audio_seek_slider.set(current_position)

def load_audio_file():
    global duration, x, spectrogram, volume, fundamental_frequencies, overall_fundamental_freq, vowel_predictions
    filename = filedialog.askopenfilename(title="Select Audio File", filetypes=[("Audio Files", "*.wav *.mp3")])
    if filename:
        process_audio_file(filename)
        audio_segment = AudioSegment.from_file(filename)
        duration = len(audio_segment) / 1000  # Duration in seconds

        # Load audio into VLC player
        media = vlc_instance.media_new(filename)
        vlc_player.set_media(media)

        # Attach event to VLC player for position change
        event_manager = vlc_player.event_manager()
        event_manager.event_attach(vlc.EventType.MediaPlayerPositionChanged, on_position_changed)

        # Update duration in slider
        update_audio_duration(duration)


def play_audio():
    if vlc_player.get_media():
        vlc_player.play()
        # Update the slider in a background thread
        threading.Thread(target=update_slider, args=(duration,)).start()

def update_slider(audio_length):
    while vlc_player.is_playing():
        time.sleep(0.01)  # Reduce the delay to improve responsiveness
        current_position = vlc_player.get_time() / 1000.0  # Convert to seconds
        if current_position >= audio_length:
            audio_seek_slider.set(audio_length)
            break
        else:
            audio_seek_slider.set(current_position)  # Update the audio seek slider with the current position

def stop_audio():
    vlc_player.stop()
    audio_seek_slider.set(0)

def pause_audio():
    vlc_player.pause()

def unpause_audio():
    vlc_player.play()

def seek_audio(position):
    if vlc_player.get_media():
        vlc_player.set_time(int(position * 1000))  # VLC uses milliseconds

def update_slider(audio_length):
    while vlc_player.is_playing():
        time.sleep(0.1)
        current_position = vlc_player.get_time() / 1000.0  # Convert to seconds
        if current_position >= audio_length:
            audio_seek_slider.set(audio_length)
            break
        else:
            audio_seek_slider.set(current_position)

def update_audio_duration(new_duration):
    scale.config(to=new_duration)
    audio_seek_slider.config(to=new_duration)

def on_slider_change(val):
    seek_audio(float(val))

def start_slider_thread():
    slider_thread = threading.Thread(target=update_slider)
    slider_thread.start()

def set_playback_speed(val):
    speed = float(val)
    vlc_player.set_rate(speed)

def seek_forward():
    # Seek forward by a fixed amount, e.g., 5 seconds
    current_time = vlc_player.get_time()
    vlc_player.set_time(current_time + 5000)

def seek_backward():
    # Seek backward by a fixed amount, e.g., 5 seconds
    current_time = vlc_player.get_time()
    vlc_player.set_time(max(current_time - 5000, 0))

# Initialize Tkinter
root = tkinter.Tk()
root.wm_title("Audio Signal Analysis")

# Top Frame for Graphs
top_frame = tkinter.Frame(root)
top_frame.pack(side="top", fill='both', expand=True)

# Frame for Left Graph
frame_left = tkinter.Frame(top_frame)
frame_left.pack(side="left", fill='both', expand=True)

# Frame for Right Graph and Controls
frame_right = tkinter.Frame(top_frame)
frame_right.pack(side="right", fill='both', expand=True)

# Bottom Frame for Media Player
bottom_frame = tkinter.Frame(root)
bottom_frame.pack(side="bottom", fill='x')

bottom_frame_top = tkinter.Frame(bottom_frame)
bottom_frame_top.pack(side="top", fill='x')

# Spectrogram plot setup 
fig, ax1 = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=frame_left)
canvas.get_tk_widget().pack(side="top", fill='both', expand=True)
ax1.set_xlabel('sec')
ax1.set_ylabel('frequency [Hz]')

# Volume plot setup
ax2 = ax1.twinx()

frame_right_top = tkinter.Frame(frame_right)
frame_right_top.pack(side="top", fill='both', expand=True)

frame_right_bottom = tkinter.Frame(frame_right)
frame_right_bottom.pack(side="bottom", fill='x')

# Label to display the fundamental frequency
fundamental_freq_label = tkinter.Label(frame_right, text="Fundamental Frequency: 0 Hz", font=("", 20))
fundamental_freq_label.pack()

# Spectrum plot
fig3, ax3 = plt.subplots()
canvas2 = FigureCanvasTkAgg(fig3, master=frame_right_top)
canvas2.get_tk_widget().pack(side="top", fill='both', expand=True)

# Slider for time selection
scale = tkinter.Scale(frame_right_bottom, command=_draw_spectrum, from_=0, to=duration, resolution=size_shift/SR, label='Select Time (sec)', orient=tkinter.HORIZONTAL)
scale.pack(fill='x')
 
# Media Player Controls in Bottom Frame
load_button = tkinter.Button(bottom_frame_top, text="Load Audio File", command=load_audio_file)
load_button.pack(side="left", fill='x')

play_button = tkinter.Button(bottom_frame_top, text="Play", command=play_audio)
play_button.pack(side="left", fill='x')

# Update pause_button to use toggle_pause
pause_button = tkinter.Button(bottom_frame_top, text="Pause/Unpause", command=pause_audio)
pause_button.pack(side="left", fill='x')

stop_button = tkinter.Button(bottom_frame_top, text="Stop", command=stop_audio)
stop_button.pack(side="left", fill='x')

# Speed Adjustment
speed_scale = tkinter.Scale(bottom_frame_top, from_=0.5, to=2.0, resolution=0.1, orient=tkinter.HORIZONTAL, label='Speed', command=set_playback_speed)
speed_scale.set(1.0)  # Default speed
speed_scale.pack(side='right')

seek_backward_button = tkinter.Button(bottom_frame_top, text="<<", command=seek_backward)
seek_backward_button.pack(side="left", fill='x')

# Add buttons to the GUI for seeking
seek_forward_button = tkinter.Button(bottom_frame_top, text=">>", command=seek_forward)
seek_forward_button.pack(side="left", fill='x')

# Seekbar for Audio
audio_seek_slider = tkinter.Scale(bottom_frame, from_=0, to=duration, resolution=size_shift/SR, orient=tkinter.HORIZONTAL, command=on_slider_change)
audio_seek_slider.pack(side='bottom', fill='x', expand=True)

# Update function calls
load_button.config(command=load_audio_file)
play_button.config(command=play_audio)
pause_button.config(command=pause_audio)
stop_button.config(command=stop_audio)
audio_seek_slider.config(command=on_slider_change)

root.mainloop()  # Start the Tkinter GUI in the main thread
