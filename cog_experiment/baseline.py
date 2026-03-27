# baseline.py
# RGB appearance baseline -- mean colour per object, same Ward clustering as COG
# only the input features differ

import numpy as np
from cluster import fit_ward, predict, evaluate


def collect_rgb_features(env) -> np.ndarray:
    """Get mean RGB of each object from the rendered scene. Returns (N_OBJECTS, 3)."""
    return env.get_rgb_means()


def normalise_rgb(rgb_list: list) -> tuple:
    """Z-score across all training episodes. Returns (normed, mean, std)."""
    stacked = np.vstack(rgb_list)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0) + 1e-8
    return (stacked - mean) / std, mean, std


def apply_rgb_normalise(rgb: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (rgb - mean) / std
