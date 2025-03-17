import sounddevice as sd
import scipy.io.wavfile as wav

# Define sample rate and duration
sample_rate = 44100  # 44.1 kHz
duration = 5  # Maximum recording duration in seconds

# Record audio
print("Recording...")
audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
sd.wait()  # Wait until recording is finished
print("Recording complete.")

# Save the recording
filename = "audio_files/utterance_3.wav"
wav.write(filename, sample_rate, audio)
print(f"Saved as {filename}")