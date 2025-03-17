# setup.py
from huggingface_hub import hf_hub_download
from generator import load_csm_1b, Segment
import torchaudio
import torch
import os

# Set device
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("device:", device)

# Define the local path where you want to save the model
local_model_path = "ckpt.pt"

# Check if the model file already exists locally
if os.path.exists(local_model_path):
    print(f"Model file already exists at {local_model_path}. Skipping download.")
    model_path = local_model_path
else:
    # Download the model file if it doesn't exist locally
    print("Downloading model file...")
    model_path = hf_hub_download(
        repo_id="sesame/csm-1b",
        filename="ckpt.pt",
        local_dir=".",  # Save to current directory
        local_dir_use_symlinks=False,  # Copy file instead of symlinking
    )
    print(f"Model downloaded and saved to {model_path}")

# Load the generator with the model
generator = load_csm_1b(model_path, device)

# Define the transcripts, speakers, and audio paths
speakers = [0, 1, 0, 0]
transcripts = [
    "Hey how are you doing.",
    "Pretty good, pretty good.",
    "I'm great.",
    "So happy to be speaking to you.",
]
audio_paths = [
    "utterance_0.wav",
    "utterance_1.wav",
    "utterance_2.wav",
    "utterance_3.wav",
]

# Function to load and resample audio
def load_audio(audio_path):
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
    )
    return audio_tensor

# Create segments
segments = [
    Segment(text=transcript, speaker=speaker, audio=load_audio(audio_path))
    for transcript, speaker, audio_path in zip(transcripts, speakers, audio_paths)
]

# Save only the necessary data
torch.save({
    "model_state_dict": generator._model.state_dict(),
    "segments": segments,
    "sample_rate": generator.sample_rate,
    "device": device,
}, "setup_data.pth")
print("Setup data saved as setup_data.pth")