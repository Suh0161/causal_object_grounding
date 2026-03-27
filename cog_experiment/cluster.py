# cluster.py
# Ward agglomerative clustering + Hungarian-matched accuracy + ARI
#
# basic flow:
#   fit_ward(sigs, labels) -> centroids, pred, metrics
#   predict(sigs, centroids) -> labels
#   hungarian_accuracy(pred, true) -> accuracy, permutation

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from config import N_CLUSTERS


def fit_ward(sigs: np.ndarray, labels):
    """Fit Ward clustering and return centroids aligned to GT label order."""
    M = len(sigs)

    Z = linkage(sigs, method="ward", metric="euclidean")
    raw_pred = fcluster(Z, t=N_CLUSTERS, criterion="maxclust") - 1  # 0-based

    # centroid per cluster
    centroids = np.zeros((N_CLUSTERS, sigs.shape[1]))
    for k in range(N_CLUSTERS):
        mask = raw_pred == k
        if mask.any():
            centroids[k] = sigs[mask].mean(axis=0)
        else:
            centroids[k] = sigs[np.random.randint(M)]

    labels_arr = np.array(labels)
    acc, perm = hungarian_accuracy(raw_pred, labels_arr)
    ari = compute_ari(raw_pred, labels_arr)

    # reorder centroids so centroid[gt_type] = matched cluster centroid
    # this is important -- without it test-time nearest-centroid gives wrong labels
    reordered = np.zeros_like(centroids)
    for k, gt in enumerate(perm):
        reordered[gt] = centroids[k]
    centroids = reordered

    pred = predict(sigs, centroids)
    acc, _ = hungarian_accuracy(pred, labels_arr)

    return centroids, pred, {"accuracy": acc, "ari": ari}


def predict(sigs: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Nearest centroid assignment."""
    dists = cdist(sigs, centroids, metric="euclidean")
    return np.argmin(dists, axis=1)


def hungarian_accuracy(pred: np.ndarray, true: np.ndarray):
    """Best-permutation accuracy via Hungarian algorithm on the confusion matrix."""
    K = N_CLUSTERS
    cost = np.zeros((K, K), dtype=int)
    for k in range(K):
        for j in range(K):
            cost[k, j] = -np.sum((pred == k) & (true == j))

    row_ind, col_ind = linear_sum_assignment(cost)
    perm = col_ind.tolist()

    matched_pred = np.array([perm[p] for p in pred])
    accuracy = float(np.mean(matched_pred == true))
    return accuracy, perm


def compute_ari(pred: np.ndarray, true: np.ndarray) -> float:
    return float(adjusted_rand_score(true, pred))


def evaluate(sigs: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> dict:
    """Run predict + accuracy + ARI for a batch of episodes."""
    pred = predict(sigs, centroids)
    acc, _ = hungarian_accuracy(pred, labels)
    ari = compute_ari(pred, labels)
    return {"accuracy": acc, "ari": ari}
