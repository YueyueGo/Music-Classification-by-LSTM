import sys

import librosa
import torch
import numpy as np
import matplotlib.pyplot as plt

from GenreFeatureData import (
    GenreFeatureData,
)
from lstm_genre_classifier_pytorch_ReduceLROnPlateau import (
    LSTM,
)

def load_model(path):
    # Load the model and set model to eval mode
    print("Loading model...")
    batch_size = 35

    model = LSTM(
        input_dim=33, hidden_dim=128, batch_size=batch_size, output_dim=6, num_layers=2
    )
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(path,map_location=torch.device('cuda')))
    else:
        model.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
    model.eval()
    return model

def get_music_features(path):
    # Extract music features
    print("Loading music feature...")
    timeseries_length = 1290
    features = np.zeros((1, timeseries_length, 33), dtype=np.float64)

    y, sr = librosa.load(path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=13)
    spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=512)

    features[0, :, 0:13] = mfcc.T[0:timeseries_length, :]
    features[0, :, 13:14] = spectral_center.T[0:timeseries_length, :]
    features[0, :, 14:26] = chroma.T[0:timeseries_length, :]
    features[0, :, 26:33] = spectral_contrast.T[0:timeseries_length, :]

    return features

def get_probabilities(model,path):
    print("Getting genre...")
    features_minibatch = torch.from_numpy(get_music_features(path))
    features_minibatch = features_minibatch.permute(1, 0, 2)
    features_minibatch = features_minibatch.float()

    outputs, _ = model(features_minibatch)

    # Calculate class probabilities
    print("Calculating class probabilities...")
    log_softmax = torch.nn.LogSoftmax(dim=1)
    outputs = log_softmax(outputs)

    probabilities = torch.exp(outputs)
    probabilities = probabilities.detach().numpy()[0]

    return probabilities


def main():
    PATH = sys.argv[1] if len(sys.argv) == 2 else "./audio/classical_music.mp3"
    MODEL = load_model("./model/model.pt")
    PROBABILITIES = get_probabilities(MODEL, PATH)

    print("--------------------result--------------------")
    # Print predicted class probabilities
    for i in range(len(GenreFeatureData.genre_list)):
        print(f"Probability for {GenreFeatureData.genre_list[i]}: {PROBABILITIES[i]}")

    plt.bar(GenreFeatureData.genre_list, PROBABILITIES)
    plt.title("Music Genre Predict Probabilities")
    plt.xlabel("Genre")
    plt.ylabel("Probability")
    plt.show()

if __name__ == "__main__":
    main()