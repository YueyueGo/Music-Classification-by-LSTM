import librosa
import librosa.display
import matplotlib.pyplot as plt

# load file
audio_path = './gtzan/_train/classical.00030.au'
y, sr = librosa.load(audio_path)

# MFCC
mfcc = librosa.feature.mfcc(y=y, sr=sr)
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()

# Spectral Center
spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
plt.figure(figsize=(10, 4))
plt.semilogy(spectral_centroids.T, label='Spectral centroid')
plt.xlabel('Frame')
plt.ylabel('Hz')
plt.title('Spectral Center')
plt.tight_layout()
plt.show()

# Chroma
chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
plt.figure(figsize=(10, 4))
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', cmap='coolwarm')
plt.colorbar()
plt.title('Chromagram')
plt.tight_layout()
plt.show()

# Spectral Contrast
spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
plt.figure(figsize=(10, 4))
librosa.display.specshow(spectral_contrast, x_axis='time')
plt.colorbar()
plt.title('Spectral Contrast')
plt.tight_layout()
plt.show()