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
os.environ["OMP_NUM_THREADS"] = "1"


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
    for k, v in all_talkers.items():
        if all_talkers[k].__len__():
            talker = all_talkers[k]  # first sample of every talker "belgium"
            mean_centroid = np.mean([x.spectral_feature("centroid") for x in talker])
            mean_rolloff = np.mean([x.spectral_feature("rolloff") for x in talker])
            centroids.append(mean_centroid)
            rolloffs.append(mean_rolloff)
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

    # plot features
    plt.scatter(X.centroids, X.rolloffs)

    # do kmeans clustering and get elbow point
    sse = list()
    for cluster in range(1, 11):
        kmeans = KMeans(n_clusters=cluster, **kmeans_kwargs)
        kmeans.fit(X)
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
    kmeans.fit(X)
    label = kmeans.fit_predict(X)
    filtered_label0 = X[label == 0]
    filtered_label1 = X[label == 1]
    filtered_label2 = X[label == 2]

    # plot results
    plt.scatter(filtered_label0.centroids, filtered_label0.rolloffs, color='red')
    plt.scatter(filtered_label1.centroids, filtered_label1.rolloffs, color='black')
    plt.scatter(filtered_label2.centroids, filtered_label2.rolloffs, color='green')

