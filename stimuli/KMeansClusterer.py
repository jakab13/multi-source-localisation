from labplatform.config import get_config
import os
import slab
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from kneed import KneeLocator
import pathlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import librosa


class KMeansClusterer:
    """
    Class for signal selection based on maximum spectral differences. First, we calculate the Mel-frequency-cepstrum-
    coefficients (MFCCs) to label the input sound files based on spectral composition. The MFCCs serve as input for
    k-means clustering. Based on the clusters, we can select one talker in each group.
    MFCC works as follows:
    1. apply FT to time-sliced audio signal (usually 25 ms windows) --> compute spectrogram
    2. apply triangular filter bank on spectrogram --> mel-spectrogram (smoothened)
    3. calculate logarithm of the mel-spectrogram
    4. apply DCT to the output
    """
    def __init__(self, wavfile):
        self.wavfile = wavfile


if __name__ == "__main__":
    # kwargs important for kmeans clustering
    kmeans_kwargs = {"init": "random",
                     "n_init": 10,
                     "random_state": 42}

    # load and sort sound files by talker
    sound_type = "tts-countries_resamp_24414"
    sound_root = pathlib.Path("C:\labplatform\sound_files")
    sound_fp = pathlib.Path(os.path.join(sound_root, sound_type))
    sound_list = slab.Precomputed(slab.Sound.read(pathlib.Path(sound_fp / file)) for file in os.listdir(sound_fp))
    all_talkers = dict()
    talker_id_range = range(225, 377)
    for talker_id in talker_id_range:
        talker_sorted = list()
        for i, sound in enumerate(os.listdir(sound_fp)):
            if str(talker_id) in sound:
                talker_sorted.append(sound_list[i])
        all_talkers[str(talker_id)] = talker_sorted

    centroids = list()
    rolloffs = list()
    f0s = list()
    zcrs = list()

    for k, v in all_talkers.items():
        if all_talkers[k].__len__():
            talker = all_talkers[k]
            centroids.append(np.mean([x.spectral_feature("centroid") for x in talker]))
            rolloffs.append(np.mean([x.spectral_feature("rolloff") for x in talker]))
            # fluxs.append(np.mean([x.spectral_feature("flux") for x in talker]))
            # mean_f0 = np.mean([librosa.yin(x.data, fmin=librosa.note_to_hz('C2'),
                                           #fmax=librosa.note_to_hz('C7'),
                                           #sr=x.samplerate) for x in talker])
            # mean_zcr = np.mean([librosa.feature.zero_crossing_rate(x.data) for x in talker])
        else:
            continue

    centroids = np.reshape(centroids, (-1, 1))
    rolloffs = np.reshape(rolloffs, (-1, 1))

    scaler = StandardScaler()  # basically z-score standardization
    scaled_centroids = np.array(np.round(scaler.fit_transform(centroids), 2))
    scaled_rolloffs = np.array(np.round(scaler.fit_transform(rolloffs), 2))

    # put features together
    X = pd.DataFrame({"centroids": scaled_centroids.reshape(1, -1)[0],
                      "rolloffs": scaled_rolloffs.reshape(1, -1)[0]})

    # PCA
    pca = PCA(2)
    data = pca.fit_transform(X)
    pca1 = [x[0] for x in data]
    pca2 = [x[1] for x in data]
    pcaX = pd.DataFrame({"pca1": pca1,
                         "pca2": pca2})

    # plot features
    plt.scatter(pcaX.pca1, pcaX.pca2)

    # do kmeans clustering and get elbow point
    sse = list()
    for cluster in range(1, 11):
        kmeans = KMeans(n_clusters=cluster, **kmeans_kwargs)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)

    # plot sse
    plt.style.use("fivethirtyeight")
    plt.plot(range(1, 11), sse)
    plt.xticks(range(1, 11))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.show()

    kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
    nclust_opt = kl.elbow  # seems 3 clusters is optimal

    # plot clusters in a scatter plot
    kmeans = KMeans(n_clusters=nclust_opt, **kmeans_kwargs)
    kmeans.fit(data)
    label = kmeans.fit_predict(data)
    filtered_label0 = data[label == 0]
    filtered_label1 = data[label == 1]
    filtered_label2 = data[label == 2]

    # plot results
    plt.scatter(filtered_label0, filtered_label0, color='red')
    plt.scatter(filtered_label1, filtered_label1, color='black')
    plt.scatter(filtered_label2, filtered_label2, color='green')

