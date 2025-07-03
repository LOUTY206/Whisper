import librosa
import librosa.display
import matplotlib.pyplot as plt

y, sr = librosa.load("audio.mp3")
librosa.display.waveshow(y, sr=sr)
plt.title("Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

