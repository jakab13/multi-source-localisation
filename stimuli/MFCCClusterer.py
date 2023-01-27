from python_speech_features import mfcc
from labplatform.config import get_config
import os
import slab
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from kneed import KneeLocator
import pathlib
os.environ["OMP_NUM_THREADS"] = "1"


class MFCCClusterer:
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
                     "max_iter": 300,
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

    mfccs = dict()
    for k, v in all_talkers.items():
        if all_talkers[k].__len__():
            first_samp = all_talkers[k][0]  # first sample of every talker
            mfccfeats = mfcc(first_samp, first_samp.samplerate)
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(mfccfeats)
            mfccs[k] = scaled_features
        else:
            continue

    # sse = []
    kmeans = KMeans(n_clusters=10, **kmeans_kwargs)
    labels = dict()
    for k, v in mfccs.items():
        labels[k] = kmeans.fit_predict(mfccs[k])
    # sse.append(kmeans.inertia_)

    # plt.style.use("fivethirtyeight")
    # plt.plot(range(1, 11), sse)
    # plt.xticks(range(1, 11))
    # plt.xlabel("Number of Clusters")
    # plt.ylabel("SSE")
    # plt.show()

    kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
    kl.elbow