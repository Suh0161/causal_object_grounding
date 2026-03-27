"""
run_k_sensitivity.py -- Cluster count sensitivity analysis.

Tests COG performance when K is misspecified (K = 4, 5, 6, 7, 8).
Uses the saved training signatures (no re-training needed).
Evaluates on 50 Test-Colour episodes per K value.

Usage:
  python run_k_sensitivity.py
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
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from config import N_TYPES, SIG_DIM, SAVE_DIR, NORM_PARAMS_FILE, TEST_EPISODES
from env import PhysicsEnv
from intervene import EMANormaliser, collect_signatures


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fit_ward_k(sigs: np.ndarray, k: int):
    """Fit Ward clustering with k clusters; return centroids and assignments."""
    Z = linkage(sigs, method="ward", metric="euclidean")
    raw_pred = fcluster(Z, t=k, criterion="maxclust") - 1   # 0-based
    centroids = np.zeros((k, sigs.shape[1]))
    for c in range(k):
        mask = raw_pred == c
        if mask.any():
            centroids[c] = sigs[mask].mean(axis=0)
        else:
            centroids[c] = sigs[np.random.randint(len(sigs))]
    return centroids, raw_pred


def hungarian_accuracy_rect(pred: np.ndarray, true: np.ndarray,
                             n_pred: int, n_true: int) -> float:
    """
    Hungarian-matched accuracy for rectangular (n_pred x n_true) case.
    Handles K != T gracefully.
    """
    cost = np.zeros((n_pred, n_true), dtype=int)
    for i in range(n_pred):
        for j in range(n_true):
            cost[i, j] = -np.sum((pred == i) & (true == j))
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {r: c for r, c in zip(row_ind, col_ind)}
    matched = np.array([mapping.get(int(p), -1) for p in pred])
    return float(np.mean(matched == true))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("K-Sensitivity Analysis")
    print("=" * 60)

    # Load saved training artefacts
    sigs_mat   = np.load(f"{SAVE_DIR}/train_signatures.npy")   # (1200, 20)
    labels_arr = np.load(f"{SAVE_DIR}/train_labels.npy")       # (1200,)
    norm_params = np.load(
        NORM_PARAMS_FILE.replace(".npy", "_cog.npy"), allow_pickle=True
    ).item()

    print(f"Loaded training sigs: {sigs_mat.shape}  labels: {labels_arr.shape}")

    k_values = [4, 5, 6, 7, 8]
    results = {}
    t0 = time.time()

    for k in k_values:
        print(f"\n-- K = {k} " + "-" * 40)

        # Fit Ward with k clusters on training sigs
        centroids, train_pred = fit_ward_k(sigs_mat, k)
        train_acc = hungarian_accuracy_rect(train_pred, labels_arr, k, N_TYPES)
        train_ari = float(adjusted_rand_score(labels_arr, train_pred))
        print(f"  Train  acc={train_acc:.4f}  ARI={train_ari:.4f}")

        # Test on Test-Colour (50 episodes) with frozen normaliser
        norm = EMANormaliser(dim=SIG_DIM)
        norm.set_params(norm_params)

        test_accs, test_aris = [], []
        for ep in range(TEST_EPISODES):
            env = PhysicsEnv(condition="colour", headless=True, seed=1000 + ep)
            test_sigs, _, test_labels = collect_signatures(env, norm, update_norm=False)
            env.close()

            test_labels_arr = np.array(test_labels)
            dists = cdist(test_sigs, centroids, metric="euclidean")
            pred  = np.argmin(dists, axis=1)

            acc = hungarian_accuracy_rect(pred, test_labels_arr, k, N_TYPES)
            ari = float(adjusted_rand_score(test_labels_arr, pred))
            test_accs.append(acc)
            test_aris.append(ari)

        mean_acc = float(np.mean(test_accs))
        std_acc  = float(np.std(test_accs))
        mean_ari = float(np.mean(test_aris))
        std_ari  = float(np.std(test_aris))

        marker = " <-- true K" if k == N_TYPES else ""
        print(f"  Colour acc={mean_acc:.4f}+-{std_acc:.4f}  "
              f"ARI={mean_ari:.4f}+-{std_ari:.4f}{marker}")

        results[k] = {
            "train": {"accuracy": train_acc, "ari": train_ari},
            "test_colour": {
                "accuracy": {"mean": mean_acc, "std": std_acc},
                "ari":      {"mean": mean_ari, "std": std_ari},
            },
        }

    # Summary table
    print("\n" + "=" * 60)
    print(f"{'K':>4}  {'Train acc':>10}  {'Colour acc':>12}  {'Colour ARI':>12}")
    print("-" * 50)
    for k, r in results.items():
        marker = "*" if k == N_TYPES else " "
        print(f"{k:>3}{marker}  {r['train']['accuracy']:>10.4f}  "
              f"{r['test_colour']['accuracy']['mean']:>10.4f}+-"
              f"{r['test_colour']['accuracy']['std']:.4f}  "
              f"{r['test_colour']['ari']['mean']:>10.4f}+-"
              f"{r['test_colour']['ari']['std']:.4f}")
    print(f"  * = true number of types (K={N_TYPES})")

    out_path = f"{SAVE_DIR}/k_sensitivity_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
