"""
baseline.py — RGB appearance-based clustering baseline.

Uses mean RGB value of each object (extracted from pygame render)
as input to Ward agglomerative clustering.
Same algorithm as COG, only the input differs.
"""

import numpy as np
from cluster import fit_ward, predict, evaluate


def collect_rgb_features(env) -> np.ndarray:
    """
    Extract mean RGB value of each object from the rendered scene.

    Returns
    -------
    rgb : (N_OBJECTS, 3) float32 array.
    """
    return env.get_rgb_means()


def normalise_rgb(rgb_list: list[np.ndarray]) -> np.ndarray:
    """
    Z-score normalise RGB features across all episodes.

    Parameters
    ----------
    rgb_list : list of (N_OBJECTS, 3) arrays, one per episode.

    Returns
    -------
    normed : (M, 3) normalised features where M = n_episodes * N_OBJECTS.
    mean   : (3,)
    std    : (3,)
    """
    stacked = np.vstack(rgb_list)   # (M, 3)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0) + 1e-8
    return (stacked - mean) / std, mean, std


def apply_rgb_normalise(rgb: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply pre-computed normalisation to new RGB features."""
    return (rgb - mean) / std
