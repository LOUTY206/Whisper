import pyaudio
import numpy as np
import whisper
import torch

# Load Whisper model (you can use 'base' or other models)
model = whisper.load_model("base")

# Convert audio to mel spectrogram
def audio_to_mel(audio_data):
    # Whisper expects the audio to be a numpy array, so convert audio_data to numpy array
    mel = whisper.audio.log_mel_spectrogram(audio_data).to(model.device)
    return mel

def transcribe_audio(mel):
    # Use Whisper to transcribe the mel spectrogram
    result = model.transcribe(mel, 
                 language="fr",)
    print(f"Transcription: {result['text']}")



# Audio setup
p = pyaudio.PyAudio()

# Open the microphone stream
stream = p.open(format=pyaudio.paInt16,  # Audio format (16-bit PCM)
                channels=1,              # Mono audio (1 channel)
                rate=16000,              # Sampling rate (16 kHz)
                input=True,              # We're capturing input (from microphone)
                frames_per_buffer=1024)  # Buffer size (chunk size)

print("Recording...")

# Continuously capture audio in chunks and process it
audio = ""
x = 0
while x < 101:
    # Read audio chunk from microphone
    audio = np.frombuffer(stream.read(1024), dtype=np.int16)
    if not audio.flags['WRITEABLE']:
        audio = np.copy(audio)
    
    # Print raw audio data (just for demonstration)
    print(audio)
    x+=1

audio_tensor = torch.from_numpy(audio).float()
transcribe_audio(audio_to_mel(audio_tensor))