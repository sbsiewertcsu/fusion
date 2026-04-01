import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram
import matplotlib.pyplot as plt

sample_rate, audio = wavfile.read("gunshot-with-background.wav")

# If stereo, take one channel
if audio.ndim > 1:
    audio = audio[:, 0]

# default window Hann window size
frequencies, times, Sxx = spectrogram(audio, fs=sample_rate)

#specific Hann window size
#frequencies, times, Sxx = spectrogram(
#    audio,
#    fs=sample_rate,
#    window='hann',
#    nperseg=2048,
#    noverlap=512
#)

plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx + 1e-10), shading="gouraud")
plt.ylabel("Frequency [Hz]")
plt.xlabel("Time [sec]")
plt.title("Spectrogram")
plt.show()
