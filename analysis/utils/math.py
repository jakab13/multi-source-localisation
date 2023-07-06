import numpy as np

_p_ref = 2e-5  # 20 μPa, the standard reference pressure for sound in air


def spectemp_coverage(sound_composition, dyn_range):
    """
    Calculate the spectro-temporal coverage of a sound composition.

    Parameters:
    - sound_composition (list): A list of sounds in the composition.
    - dyn_range (float): The dynamic range in decibels (dB) to consider for coverage calculation.

    Returns:
    - float: The ratio of the number of spectro-temporal points within the specified dynamic range
             to the total number of spectro-temporal points in the composition.

    Raises:
    - ValueError: If the input sound_composition is not a list.

    Note:
    - This function assumes that the sounds in sound_composition have a spectrogram() method that
      returns the spectrogram of the sound.

    """
    assert not isinstance(sound_composition, list), ValueError("Input must be a list of sounds")

    # Combine the sounds in the composition
    sound = sum(sound_composition)

    # Calculate the spectrogram and power
    _, _, power = sound.spectrogram(show=False)

    # Convert power to logarithmic scale for plotting
    power = 10 * np.log10(power / (_p_ref ** 2))

    # Calculate the maximum and minimum dB values
    dB_max = power.max()
    dB_min = dB_max - dyn_range

    # Select the interval of power values within the specified dynamic range
    interval = power[np.where((power > dB_min) & (power < dB_max))]

    # Calculate the ratio of points within the dynamic range to total points
    coverage = interval.shape[0] / power.flatten().shape[0]

    return coverage


def adaptive_mean_threshold(array, window_size, c):
    """
    Apply adaptive mean thresholding to an input array.

    Parameters:
    - array (ndarray): Input array of numeric values.
    - window_size (int): Size of the square neighborhood for local threshold calculation.
                         Must be an odd positive integer.
    - c (float): Constant value subtracted from the local mean to determine the threshold.

    Returns:
    - ndarray: Binary array obtained after thresholding, where 1 indicates values above the threshold
               and 0 indicates values below or equal to the threshold.

    Raises:
    - AssertionError: If the window_size is not an odd positive integer.

    Note:
    - This function applies adaptive mean thresholding to each pixel in the input array.
    - The local threshold is calculated using a square neighborhood centered at each pixel,
      where the size of the neighborhood is determined by the window_size parameter.
    - The binary array is obtained by comparing each pixel value with the local threshold,
      with values above the threshold set to 1 and values below or equal to the threshold set to 0.

    """
    assert window_size % 2 == 1 and window_size > 0, "window_size must be an odd positive integer"

    # Calculate the local threshold for each pixel
    rows, cols = array.shape
    binary = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            # Calculate the local threshold using a square neighborhood centered at (i, j)
            x_min = max(0, i - window_size // 2)
            y_min = max(0, j - window_size // 2)
            x_max = min(rows - 1, i + window_size // 2)
            y_max = min(cols - 1, j + window_size // 2)
            block = array[x_min:x_max+1, y_min:y_max+1]
            thresh = np.mean(block) - c
            if array[i, j] >= thresh:
                binary[i, j] = 1

    return binary


def variance(data):
    """
    Calculate the variance of a list of numeric data.

    Parameters:
    - data (list or array-like): Input data, a list or array-like object containing numeric values.

    Returns:
    - float: The variance of the input data.

    Note:
    - The variance is a measure of how spread out the data values are around the mean.
    - The variance is calculated as the average of the squared differences between each data point
      and the mean of the data.
    - This function assumes that the input data is a list or array-like object containing numeric values.

    """
    # Number of observations
    n = len(data)

    # Mean of the data
    mean = sum(data) / n

    # Square deviations
    deviations = [(x - mean) ** 2 for x in data]

    # Variance
    variance = sum(deviations) / n

    return variance


if __name__ == "__main__":
    pass
