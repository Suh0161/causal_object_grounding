"""
cluster.py — Ward agglomerative clustering, Hungarian-matched accuracy, ARI.

Workflow:
  1. fit_ward(sigs, labels) → cluster centroids + train metrics
  2. predict(sigs, centroids) → predicted cluster labels (0-based)
  3. hungarian_accuracy(pred, true) → accuracy after optimal permutation
  4. compute_ari(pred, true) → adjusted rand index
"""

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from config import N_CLUSTERS


# ── Fitting ─────────────────────────────────────────────────────────────────────

def fit_ward(sigs: np.ndarray, labels: list | np.ndarray):
    """
    Fit Ward agglomerative clustering on training signatures.

    Parameters
    ----------
    sigs   : (M, D) normalised signature matrix.
    labels : (M,) ground truth type labels.

    Returns
    -------
    centroids : (N_CLUSTERS, D) cluster centroids.
    pred      : (M,) predicted cluster indices (0-based, matched to GT).
    metrics   : dict with accuracy and ARI on training data.
    """
    M = len(sigs)
    # Ward linkage on all signatures
    Z = linkage(sigs, method="ward", metric="euclidean")
    raw_pred = fcluster(Z, t=N_CLUSTERS, criterion="maxclust") - 1  # 0-based

    # Compute centroids from cluster assignments
    centroids = np.zeros((N_CLUSTERS, sigs.shape[1]))
    for k in range(N_CLUSTERS):
        mask = raw_pred == k
        if mask.any():
            centroids[k] = sigs[mask].mean(axis=0)
        else:
            centroids[k] = sigs[np.random.randint(M)]

    # Evaluate on training data
    labels_arr = np.array(labels)
    acc, perm = hungarian_accuracy(raw_pred, labels_arr)
    ari = compute_ari(raw_pred, labels_arr)

    # Reorder centroids to match GT label ordering
    # perm[k] = which GT type cluster k was matched to
    reordered = np.zeros_like(centroids)
    for k, gt in enumerate(perm):
        reordered[gt] = centroids[k]
    centroids = reordered

    # Recompute pred with reordered centroids
    pred = predict(sigs, centroids)
    acc, _ = hungarian_accuracy(pred, labels_arr)

    metrics = {"accuracy": acc, "ari": ari}
    return centroids, pred, metrics


# ── Prediction ──────────────────────────────────────────────────────────────────

def predict(sigs: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Assign each signature to the nearest centroid.

    Parameters
    ----------
    sigs      : (M, D)
    centroids : (K, D)

    Returns
    -------
    pred : (M,) cluster indices.
    """
    dists = cdist(sigs, centroids, metric="euclidean")   # (M, K)
    return np.argmin(dists, axis=1)


# ── Hungarian Matching ───────────────────────────────────────────────────────────

def hungarian_accuracy(pred: np.ndarray, true: np.ndarray):
    """
    Find permutation of cluster labels that maximises accuracy.
    Uses the Hungarian algorithm on the confusion matrix.

    Returns
    -------
    accuracy : float
    perm     : list — perm[cluster_k] = matched gt_type
    """
    K = N_CLUSTERS
    # Build cost matrix: cost[i, j] = -(# times pred=i matches true=j)
    cost = np.zeros((K, K), dtype=int)
    for k in range(K):
        for j in range(K):
            cost[k, j] = -np.sum((pred == k) & (true == j))

    row_ind, col_ind = linear_sum_assignment(cost)
    # col_ind[k] = which GT type cluster k is matched to
    perm = col_ind.tolist()

    # Map predictions through permutation
    matched_pred = np.array([perm[p] for p in pred])
    accuracy = float(np.mean(matched_pred == true))
    return accuracy, perm


# ── ARI ─────────────────────────────────────────────────────────────────────────

def compute_ari(pred: np.ndarray, true: np.ndarray) -> float:
    """Adjusted Rand Index between predicted and true labels."""
    return float(adjusted_rand_score(true, pred))


# ── Metrics Bundle ───────────────────────────────────────────────────────────────

def evaluate(sigs: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> dict:
    """
    Predict and compute accuracy + ARI for a set of episodes.

    Parameters
    ----------
    sigs      : (M, D) normalised signatures.
    labels    : (M,) ground truth type labels.
    centroids : (K, D) from training.

    Returns
    -------
    metrics : dict with accuracy, ari.
    """
    pred = predict(sigs, centroids)
    acc, _ = hungarian_accuracy(pred, labels)
    ari = compute_ari(pred, labels)
    return {"accuracy": acc, "ari": ari}
