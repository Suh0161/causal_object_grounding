"""
run_jitter.py -- Robustness to continuous within-type property variation.

Each object's mass and restitution are independently perturbed by a
multiplicative factor drawn from Uniform[1-delta, 1+delta], applied
identically at train and test time.

Jitter levels tested: 0%, 5%, 10%, 20%.

Usage:
  python scripts/run_jitter.py
"""

import os
import sys
import json
import time
import numpy as np

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from config import (
    N_TYPES, SIG_DIM, SAVE_DIR, TRAIN_EPISODES, TEST_EPISODES,
)
from env import PhysicsEnv
from intervene import EMANormaliser, collect_signatures


JITTER_LEVELS = [0.0, 0.05, 0.10, 0.20]


def fit_ward_centroids(feats, labels, k=N_TYPES):
    """Ward clustering -> Hungarian-aligned centroids (centroid[i] = GT type i)."""
    Z = linkage(feats, method="ward", metric="euclidean")
    raw_pred = fcluster(Z, t=k, criterion="maxclust") - 1
    centroids = np.zeros((k, feats.shape[1]))
    for c in range(k):
        mask = raw_pred == c
        if mask.any():
            centroids[c] = feats[mask].mean(axis=0)
    cost = np.zeros((k, k), dtype=int)
    for i in range(k):
        for j in range(k):
            cost[i, j] = -np.sum((raw_pred == i) & (labels == j))
    row_ind, col_ind = linear_sum_assignment(cost)
    perm = {r: c for r, c in zip(row_ind, col_ind)}
    reordered = np.zeros_like(centroids)
    for cluster_k, gt_type in perm.items():
        reordered[gt_type] = centroids[cluster_k]
    return reordered


def run_one_level(delta):
    """Train and test COG at a single jitter level. Returns (train_acc, test_acc, test_std)."""
    # -- Train --
    normaliser = EMANormaliser(dim=SIG_DIM)
    all_raw_sigs, all_labels = [], []

    for ep in range(TRAIN_EPISODES):
        env = PhysicsEnv(condition="train", headless=True, seed=ep, jitter=delta)
        _, raw_sigs, labels = collect_signatures(env, normaliser, update_norm=True)
        env.close()
        all_raw_sigs.append(raw_sigs)
        all_labels.extend(labels)

    raw_mat = np.vstack(all_raw_sigs)
    sigs_mat = np.stack([normaliser.normalise(r) for r in raw_mat])
    labels_arr = np.array(all_labels)

    centroids = fit_ward_centroids(sigs_mat, labels_arr)
    train_pred = np.argmin(
        __import__('scipy.spatial.distance', fromlist=['cdist']).cdist(
            sigs_mat, centroids), axis=1)
    train_acc = float(np.mean(train_pred == labels_arr))

    # -- Test (colour condition, same jitter) --
    norm = EMANormaliser(dim=SIG_DIM)
    norm.set_params(normaliser.get_params())

    from scipy.spatial.distance import cdist
    test_accs, test_aris = [], []
    for ep in range(TEST_EPISODES):
        env = PhysicsEnv(condition="colour", headless=True,
                         seed=1000 + ep, jitter=delta)
        test_sigs, _, test_labels = collect_signatures(env, norm, update_norm=False)
        env.close()

        test_labels_arr = np.array(test_labels)
        pred = np.argmin(cdist(test_sigs, centroids), axis=1)
        # Hungarian-match for accurate per-episode acc
        cost = np.zeros((N_TYPES, N_TYPES), dtype=int)
        for i in range(N_TYPES):
            for j in range(N_TYPES):
                cost[i, j] = -np.sum((pred == i) & (test_labels_arr == j))
        ri, ci = linear_sum_assignment(cost)
        mapping = {r: c for r, c in zip(ri, ci)}
        matched = np.array([mapping.get(int(p), -1) for p in pred])
        test_accs.append(float(np.mean(matched == test_labels_arr)))
        test_aris.append(float(adjusted_rand_score(test_labels_arr, pred)))

    return (train_acc,
            float(np.mean(test_accs)), float(np.std(test_accs)),
            float(np.mean(test_aris)), float(np.std(test_aris)))


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("=" * 60)
    print("Jitter Robustness: continuous within-type variation")
    print(f"Jitter levels: {JITTER_LEVELS}")
    print("=" * 60)

    results = []
    t0 = time.time()

    for delta in JITTER_LEVELS:
        print(f"\n  delta={delta:.2f}  -- training {TRAIN_EPISODES} eps ...", flush=True)
        t1 = time.time()
        train_acc, test_mean, test_std, ari_mean, ari_std = run_one_level(delta)
        elapsed = time.time() - t1
        print(f"    train acc={train_acc:.4f}  "
              f"colour acc={test_mean:.4f}+-{test_std:.4f}  "
              f"ARI={ari_mean:.4f}+-{ari_std:.4f}  ({elapsed:.0f}s)", flush=True)
        results.append({
            "delta": delta,
            "train_accuracy": train_acc,
            "colour_accuracy_mean": test_mean,
            "colour_accuracy_std": test_std,
            "colour_ari_mean": ari_mean,
            "colour_ari_std": ari_std,
        })

    print(f"\nDone in {time.time() - t0:.1f}s")
    print("\nSummary:")
    print(f"{'delta':>7}  {'train_acc':>10}  {'colour_acc':>14}  {'ARI':>14}")
    print("-" * 52)
    for r in results:
        print(f"{r['delta']:>7.2f}  {r['train_accuracy']:>10.4f}  "
              f"{r['colour_accuracy_mean']:>10.4f}+-{r['colour_accuracy_std']:.4f}  "
              f"{r['colour_ari_mean']:>10.4f}+-{r['colour_ari_std']:.4f}")

    out_path = f"{SAVE_DIR}/jitter_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
