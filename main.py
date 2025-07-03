import sounddevice as sd
import torch
import whisper

# 1. Record audio
duration = 7  # seconds
sample_rate = 16000  # Whisper expects 16kHz
print("Recording...")
audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
sd.wait()
print("Recording done.")

# 2. Convert to 1D PyTorch tensor
audio_tensor = torch.from_numpy(audio.squeeze())

# 3. Load Whisper model
model = whisper.load_model("small")

# 4. Preprocess: convert to log-mel spectrogram
mel = whisper.log_mel_spectrogram(audio_tensor)

# 5. Transcribe
result = model.transcribe(audio_tensor, task="translate", verbose=True)
print("-"*50)
print("Transcription:", result["text"])