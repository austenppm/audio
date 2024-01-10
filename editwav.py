import librosa
import soundfile as sf

def cut_audio(file_path, start_sec, end_sec=None, duration_sec=None, output_path='nekocut.wav'):
    """
    Cuts a portion of an audio file.

    :param file_path: Path to the input WAV file.
    :param start_sec: Start time in seconds.
    :param end_sec: End time in seconds (optional if duration is provided).
    :param duration_sec: Duration in seconds (optional if end time is provided).
    :param output_path: Path to save the cut audio.
    """
    # Load the full audio file
    audio, sr = librosa.load(file_path, sr=None)

    # Convert times to sample indices
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr) if end_sec is not None else int((start_sec + duration_sec) * sr)

    # Ensure end_sample does not exceed audio length
    end_sample = min(end_sample, len(audio))

    # Cut the audio
    cut_audio = audio[start_sample:end_sample]

    # Save the cut audio to a new file
    sf.write(output_path, cut_audio, sr)
    print(f"Cut audio saved to {output_path}")

# Example usage
file_path = 'neko.wav'  # Replace with your file path
start_sec = 55  # Start at x seconds
end_sec = 65  # End at y seconds
# Alternatively, you can specify the duration
# duration_sec = 10  # Duration of 10 seconds from the start point

cut_audio(file_path, start_sec, end_sec=end_sec)  # Specify either end_sec or duration_sec
