import whisper
import librosa
import noisereduce as nr
import soundfile as sf

# Load your audio file (replace with your file path)
y, sr = librosa.load("audio.mp3", sr=None)

# Reduce noise
y_denoised = nr.reduce_noise(y=y, sr=sr)

# Save the cleaned audio if needed
sf.write("cleaned_audio.mp3", y_denoised, sr)



model = whisper.load_model("small", device="cuda")
result = model.transcribe("cleaned_audio.mp3", 
                 language="fr",         # or "en", "ar", etc.
                 task="transcribe",     # or "translate"
                 verbose=True)
print(result["text"])