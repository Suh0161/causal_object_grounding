"""
run_massonly.py - Mass-only baseline for COG Paper 1.

The mass-only baseline uses ONLY the t=1 projected velocity in each of the
4 impulse directions (indices 0, 5, 10, 15 of the 20-dim raw signature).
At t=1, v_proj = I/m -- this separates the 3 mass groups (m=5,2,1) perfectly
but leaves within-group restitution pairs (A/B, C/D, E/F) IDENTICAL.
Expected accuracy: ~0.5 (random split within each mass pair).

Trains on 'train' condition, tests on all 4 conditions.
Saves results to outputs/massonly_results.json.

Usage:
  python run_massonly.py
"""

import os
import sys
import json
import time
import numpy as np

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from config import (
    TRAIN_EPISODES, TEST_EPISODES, N_OBJECTS, SIG_DIM, SAVE_DIR, N_TYPES,
    N_DIRECTIONS,
)
from env import PhysicsEnv
from intervene import EMANormaliser, collect_signatures
from cluster import predict, hungarian_accuracy, compute_ari

# Indices of t=1 in the 20-dim signature: one per direction, spaced 5 apart
MASSONLY_INDICES = [d * (SIG_DIM // N_DIRECTIONS) for d in range(N_DIRECTIONS)]  # [0, 5, 10, 15]
MASSONLY_DIM = len(MASSONLY_INDICES)


def extract_massonly(raw_sigs: np.ndarray) -> np.ndarray:
    """Extract the 4-dim mass-only feature from raw (unnormalised) signatures."""
    return raw_sigs[:, MASSONLY_INDICES]


def fit_massonly_centroids(features: np.ndarray, labels: np.ndarray):
    """Fit Ward clustering on mass-only features. Returns centroids."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.optimize import linear_sum_assignment

    K = N_TYPES
    Z = linkage(features, method="ward", metric="euclidean")
    raw_pred = fcluster(Z, t=K, criterion="maxclust") - 1

    centroids = np.zeros((K, MASSONLY_DIM))
    for k in range(K):
        mask = raw_pred == k
        centroids[k] = features[mask].mean(axis=0) if mask.any() else features[0]

    # Hungarian match to GT
    cost = np.zeros((K, K), dtype=int)
    for k in range(K):
        for j in range(K):
            cost[k, j] = -np.sum((raw_pred == k) & (labels == j))
    _, col_ind = linear_sum_assignment(cost)

    reordered = np.zeros_like(centroids)
    for k, gt in enumerate(col_ind):
        reordered[gt] = centroids[k]

    # Accuracy on training data
    pred = predict(reordered, reordered)  # won't work for eval; use predict(features, reordered)
    # Evaluate manually
    from scipy.spatial.distance import cdist
    dists = cdist(features, reordered, metric="euclidean")
    pred_arr = np.argmin(dists, axis=1)
    acc, _ = hungarian_accuracy(pred_arr, labels)
    ari = compute_ari(pred_arr, labels)

    return reordered, {"accuracy": acc, "ari": ari}


def eval_massonly(features: np.ndarray, labels: np.ndarray, centroids: np.ndarray):
    """Predict and evaluate using mass-only centroids."""
    from scipy.spatial.distance import cdist
    dists = cdist(features, centroids, metric="euclidean")
    pred = np.argmin(dists, axis=1)
    acc, _ = hungarian_accuracy(pred, labels)
    ari = compute_ari(pred, labels)
    return acc, ari


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("=" * 60)
    print("Mass-Only Baseline")
    print(f"Feature: t=1 projected velocity, indices {MASSONLY_INDICES}")
    print(f"Dim: {MASSONLY_DIM}  (one per impulse direction)")
    print("=" * 60)

    # ── Training ──────────────────────────────────────────────────────────────
    normaliser = EMANormaliser(dim=SIG_DIM)
    all_raw_sigs, all_labels = [], []

    t0 = time.time()
    for ep in range(TRAIN_EPISODES):
        env = PhysicsEnv(condition="train", headless=True, seed=ep)
        _, raw_sigs, labels = collect_signatures(env, normaliser, update_norm=True)
        env.close()
        all_raw_sigs.append(raw_sigs)
        all_labels.extend(labels)

    raw_mat = np.vstack(all_raw_sigs)
    labels_arr = np.array(all_labels)

    massonly_train = extract_massonly(raw_mat)

    # Simple z-score normalise the 4-dim feature for clustering
    mo_mean = massonly_train.mean(axis=0)
    mo_std = massonly_train.std(axis=0) + 1e-8
    massonly_normed = (massonly_train - mo_mean) / mo_std

    centroids, train_metrics = fit_massonly_centroids(massonly_normed, labels_arr)
    print(f"\nTrain: acc={train_metrics['accuracy']:.4f}  ARI={train_metrics['ari']:.4f}")

    # ── Testing ───────────────────────────────────────────────────────────────
    norm_frozen = EMANormaliser(dim=SIG_DIM)
    norm_frozen.set_params(normaliser.get_params())

    conditions = ["train", "colour", "texture", "shape"]
    test_results = {}

    for cond in conditions:
        accs, aris = [], []
        n_eps = TEST_EPISODES if cond != "train" else TEST_EPISODES
        seed_off = 1000 if cond != "train" else 2000
        for ep in range(n_eps):
            env = PhysicsEnv(condition=cond, headless=True, seed=seed_off + ep)
            _, raw_sigs, labels = collect_signatures(env, norm_frozen, update_norm=False)
            env.close()

            mo_feat = extract_massonly(raw_sigs)
            mo_feat_normed = (mo_feat - mo_mean) / mo_std
            lbl = np.array(labels)
            acc, ari = eval_massonly(mo_feat_normed, lbl, centroids)
            accs.append(acc)
            aris.append(ari)

        test_results[cond] = {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "ari_mean": float(np.mean(aris)),
            "ari_std": float(np.std(aris)),
        }
        print(f"  {cond:8s}: acc={np.mean(accs):.4f}+-{np.std(accs):.4f}  "
              f"ARI={np.mean(aris):.4f}+-{np.std(aris):.4f}")

    out = {
        "feature": "t1_velocity_4dim",
        "massonly_indices": MASSONLY_INDICES,
        "train": train_metrics,
        "test": test_results,
    }
    out_path = f"{SAVE_DIR}/massonly_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nDone in {time.time()-t0:.1f}s. Saved to {out_path}")
    print("\nNote: expected ~0.50 across all conditions (can separate 3 mass groups")
    print("      but cannot distinguish within-mass restitution pairs A/B, C/D, E/F).")


if __name__ == "__main__":
    main()
