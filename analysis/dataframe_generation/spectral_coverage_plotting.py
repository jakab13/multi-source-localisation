import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.ndimage import gaussian_filter
import matplotlib
from matplotlib.colors import ListedColormap

orig = matplotlib.colormaps["Blues"]
new = orig(np.linspace(0.4, 0.8, 256))  # shift bright end darker
blues_darker = ListedColormap(new)

colors = blues_darker(np.linspace(0, 1, 6))

def plot_multisource_coverage_blobs(
    sources,
    sr,
    nperseg=1024,
    noverlap=512,
    coverage_threshold_db=-40,   # per-source, relative to that source's max
    blob_smoothing_sigma=1.0,
    blob_alpha=0.35,
    colors=colors,                 # list of colors for sources; cycles if shorter
    contour_color="black",
    contour_width=1.0,
    contrast=1.0,                # same contrast trick as before
    title=None,
    ax=None,
    show=True
):
    """
    Plot a greyscale spectrogram of the FINAL mixture of all sources and overlay
    one blob per source showing the *newly added* spectral coverage when that
    source is added.

    Parameters
    ----------
    sources : list
        List of audio signals (np.ndarray or slab.Sound), all same sampling rate.
        They should be time-aligned and (roughly) same duration.
    sr : int or float
        Sampling rate in Hz.
    coverage_threshold_db : float
        Threshold relative to each source's own max dB. Example: -40 means
        bins within 40 dB of that source's max are considered 'covered'.
    blob_smoothing_sigma : float
        Gaussian smoothing sigma for the masks.
    blob_alpha : float
        Transparency of each blob overlay.
    colors : list of str or None
        Colors for blobs (one per source). If None, a default palette is used.
    contrast : float
        Visual contrast for the background spectrogram.
    ax : matplotlib.axes.Axes or None
        Axis to draw into. If None, a new figure is created.
    show : bool
        If True, plt.show() is called (if figure created here).

    Returns
    -------
    result : dict
        Contains:
        - f, t : freq/time axes
        - S_mix_db, S_mix_db_plot
        - per_source_masks : list of raw boolean coverage masks per source
        - per_source_unique_masks : list of boolean masks for newly added coverage
        - count_map : int array of how many sources cover each TF bin
    """

    # ----- convert all sources to mono numpy arrays and align lengths -----
    audio_list = []
    for s in sources:
        a = s
        if not isinstance(a, np.ndarray):
            if hasattr(a, "data"):
                a = np.asarray(a.data)
            elif hasattr(a, "waveform"):
                a = np.asarray(a.waveform)
        a = np.asarray(a)

        if a.ndim == 2:  # stereo → mono
            a = a.mean(axis=-1)

        audio_list.append(a)

    # crop all to same length (shortest one) to ensure identical spectrogram grids
    min_len = min(len(a) for a in audio_list)
    audio_list = [a[:min_len] for a in audio_list]

    # ----- final mixture -----
    mix = np.sum(audio_list, axis=0)

    # ----- spectrogram of mixture -----
    f, t, S_mix = spectrogram(
        mix,
        fs=sr,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="spectrum",
        mode="magnitude"
    )

    freq_mask = (f >= f[0]) & (f <= 11000)
    f = f[freq_mask]
    S_mix = S_mix[freq_mask, :]

    S_mix_db = 20 * np.log10(S_mix + 1e-12)

    # fixed display range from mixture
    base_min, base_max = np.percentile(S_mix_db, [1, 99])
    center = 0.5 * (base_min + base_max)
    S_mix_db_plot = (S_mix_db - center) * contrast + center
    vmin, vmax = -90, coverage_threshold_db + 15

    # ----- per-source coverage masks -----
    per_source_masks = []
    for a in audio_list:
        f_i, t_i, S_i = spectrogram(
            a,
            fs=sr,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="spectrum",
            mode="magnitude"
        )

        S_i = S_i[(f_i >= f_i[0]) & (f_i <= 11000)]
        f_i = f_i[(f_i >= f_i[0]) & (f_i <= 11000)]

        S_i_db = 20 * np.log10(S_i + 1e-12)

        # per-source threshold (relative to that source's own max)
        thr_i = S_i_db.max() + coverage_threshold_db
        mask_i = S_i_db >= thr_i
        smooth = gaussian_filter(mask_i.astype(float), sigma=blob_smoothing_sigma)
        mask_i_smooth = smooth >= 0.5
        per_source_masks.append(mask_i_smooth)

    # ----- compute unique "new" coverage for each source -----
    H, W = per_source_masks[0].shape
    cumulative = np.zeros((H, W), dtype=bool)
    per_source_unique_masks = []
    for mask_i in per_source_masks:
        unique_i = mask_i & (~cumulative)       # newly added coverage by this source
        cumulative |= mask_i
        per_source_unique_masks.append(unique_i)

    # also compute how many sources cover each bin (for analysis if needed)
    count_map = np.zeros((H, W), dtype=int)
    for mask_i in per_source_masks:
        count_map += mask_i.astype(int)

    # ----- plotting setup -----
    # if colors is None:
    #     # simple default palette, cycles over sources if needed
    #     colors = ["tab:blue", "tab:orange", "tab:green", "tab:red",
    #               "tab:purple", "tab:brown", "tab:pink", "tab:gray"]

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        created_fig = True
    else:
        fig = ax.figure

    # background spectrogram (final mixture)
    # im = ax.pcolormesh(
    #     t,
    #     f,
    #     S_mix_db_plot,
    #     shading="auto",
    #     cmap="gray_r",
    #     vmin=vmin,
    #     vmax=vmax,
    #     alpha=0.4
    # )
    # ax.set_xlabel("Time [s]")
    # ax.set_ylabel("Frequency [Hz]")
    # if title:
    #     ax.set_title(title)

    # if created_fig:
    #     cbar = fig.colorbar(im, ax=ax)
    #     cbar.set_label("Magnitude [dB]")

    # ----- overlay blobs, one per source, showing *unique* coverage -----
    for idx, unique_mask in enumerate(per_source_unique_masks):
        if not unique_mask.any():
            continue  # nothing new for this source

        # smooth to get blob-y region
        # smooth = gaussian_filter(unique_mask.astype(float), sigma=blob_smoothing_sigma)
        # blob = smooth >= 0.5

        blob = unique_mask

        # color = colors[idx % len(colors)]

        color = blues_darker(idx/len(per_source_unique_masks))

        # filled blob
        # ax.contourf(
        #     t,
        #     f,
        #     blob.astype(float),
        #     levels=[0.5, 1.5],
        #     colors=color,
        #     alpha=0.3
        # )

        # # outline
        # ax.contour(
        #     t,
        #     f,
        #     blob.astype(float),
        #     levels=[0.5],
        #     # colors=[contour_color],
        #     linewidths=contour_width,
        #     colors="tab:blue",
        #     alpha=(idx + 1) * 0.15 + 0.2
        # )

    # ----- overlay blobs, one per source, showing *unique* coverage -----
    for idx, unique_mask in enumerate(per_source_masks):
        if not unique_mask.any():
            continue  # nothing new for this source

        # smooth to get blob-y region
        # smooth = gaussian_filter(unique_mask.astype(float), sigma=blob_smoothing_sigma)
        # blob = smooth >= 0.5

        blob = unique_mask

        color = blues_darker(idx/len(per_source_unique_masks))

        ax.contourf(
            t,
            f,
            blob.astype(float),
            levels=[0.5, 1.5],
            colors=[color],
            alpha=0.5
        )

        # outline
        # ax.contour(
        #     t,
        #     f,
        #     blob.astype(float),
        #     levels=[0.5],
        #     # colors=[contour_color],
        #     linewidths=contour_width,
        #     colors=["white"],
        #     # alpha=0.2
        # )

    fig.tight_layout()
    if show and created_fig:
        plt.show()

    return {
        "f": f,
        "t": t,
        "S_mix_db": S_mix_db,
        "S_mix_db_plot": S_mix_db_plot,
        "per_source_masks": per_source_masks,
        "per_source_unique_masks": per_source_unique_masks,
        "count_map": count_map,
    }
