import numpy as np


def normalize_spectrogram(X):
    """Apply normalization from Holtzman et al. 2018 to a spectrogram."""
    return np.maximum(0, np.floor(20 * np.log10(X / np.median(X))))
