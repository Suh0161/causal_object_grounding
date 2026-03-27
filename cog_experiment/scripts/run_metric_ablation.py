"""
run_metric_ablation.py -- Combined ablation with alternative distance metrics.

Addresses reviewer question: does the "appearance corrupts" result persist when
more robust normalisation or distance metrics are used for the combined
(sig + RGB) ablation?

Three variants tested on Test-Colour (50 episodes):
  1. Current  -- EMA-sig + z-RGB, Euclidean distance  (reproduces Table 1)
  2. JointW   -- joint StandardScaler on 23-dim combined, Euclidean distance
  3. Mahal    -- EMA-sig + z-RGB, Mahalanobis distance (within-class cov)

Usage:
  python run_metric_ablation.py
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
from sklearn.preprocessing import StandardScaler

from config import (
    N_TYPES, N_OBJECTS, SIG_DIM, SAVE_DIR,
    NORM_PARAMS_FILE, TRAIN_EPISODES, TEST_EPISODES,
)
from env import PhysicsEnv
from intervene import EMANormaliser, collect_signatures
from baseline import collect_rgb_features, apply_rgb_normalise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fit_ward_centroids(feats: np.ndarray, labels: np.ndarray,
                        k: int = N_TYPES):
    """
    Ward clustering -> k centroids, Hungarian-aligned to GT labels.
    Returns centroids where centroid[i] corresponds to GT type i,
    so test-time nearest-centroid prediction gives directly comparable labels.
    """
    Z = linkage(feats, method="ward", metric="euclidean")
    raw_pred = fcluster(Z, t=k, criterion="maxclust") - 1   # 0-based

    # Raw centroids
    centroids = np.zeros((k, feats.shape[1]))
    for c in range(k):
        mask = raw_pred == c
        if mask.any():
            centroids[c] = feats[mask].mean(axis=0)

    # Hungarian matching: find permutation mapping cluster -> GT type
    cost = np.zeros((k, k), dtype=int)
    for i in range(k):
        for j in range(k):
            cost[i, j] = -np.sum((raw_pred == i) & (labels == j))
    row_ind, col_ind = linear_sum_assignment(cost)
    perm = {r: c for r, c in zip(row_ind, col_ind)}   # cluster_k -> gt_type

    # Reorder centroids: centroid[gt_type] = centroid of matched cluster
    reordered = np.zeros_like(centroids)
    for cluster_k, gt_type in perm.items():
        reordered[gt_type] = centroids[cluster_k]
    return reordered


def hungarian_accuracy(pred: np.ndarray, true: np.ndarray,
                        k: int = N_TYPES) -> float:
    cost = np.zeros((k, k), dtype=int)
    for i in range(k):
        for j in range(k):
            cost[i, j] = -np.sum((pred == i) & (true == j))
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {r: c for r, c in zip(row_ind, col_ind)}
    matched = np.array([mapping[int(p)] for p in pred])
    return float(np.mean(matched == true))


def within_class_cov(feats: np.ndarray, pred: np.ndarray,
                      k: int = N_TYPES) -> np.ndarray:
    """Pooled within-class covariance matrix."""
    d = feats.shape[1]
    cov = np.zeros((d, d))
    n = len(feats)
    for c in range(k):
        mask = pred == c
        if mask.sum() > 1:
            X_c = feats[mask]
            X_c -= X_c.mean(axis=0)
            cov += X_c.T @ X_c
    cov /= n
    # Regularise slightly for numerical stability
    cov += 1e-6 * np.eye(d)
    return cov


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Metric Ablation: Combined (sig + RGB) Distance Variants")
    print("=" * 60)

    # Load saved training sigs and norm params
    sigs_mat   = np.load(f"{SAVE_DIR}/train_signatures.npy")   # (1200, 20)
    labels_arr = np.load(f"{SAVE_DIR}/train_labels.npy")       # (1200,)
    norm_params = np.load(
        NORM_PARAMS_FILE.replace(".npy", "_cog.npy"), allow_pickle=True
    ).item()
    rgb_mean = np.load(NORM_PARAMS_FILE.replace(".npy", "_rgb_mean.npy"))
    rgb_std  = np.load(NORM_PARAMS_FILE.replace(".npy", "_rgb_std.npy"))

    # Re-extract training RGB features (fast: just env spawn + render, no physics)
    print(f"\nRe-extracting RGB from {TRAIN_EPISODES} training episodes...")
    t0 = time.time()
    all_rgb = []
    for ep in range(TRAIN_EPISODES):
        env = PhysicsEnv(condition="train", headless=True, seed=ep)
        all_rgb.append(collect_rgb_features(env))
        env.close()
    rgb_mat = np.vstack(all_rgb)   # (1200, 3)
    rgb_normed = apply_rgb_normalise(rgb_mat, rgb_mean, rgb_std)
    print(f"  Done in {time.time() - t0:.1f}s")

    # Build combined training matrix (current: EMA-sig + z-RGB)
    combined_train = np.hstack([sigs_mat, rgb_normed])   # (1200, 23)

    # -----------------------------------------------------------------------
    # Variant 1: Current  (EMA-sig + z-RGB, Euclidean)
    # -----------------------------------------------------------------------
    print("\n-- Variant 1: Current (EMA-sig + z-RGB, Euclidean) --------")
    cen_current = fit_ward_centroids(combined_train, labels_arr)
    # Verify alignment: predict on training set, compare directly
    train_pred_current = np.argmin(cdist(combined_train, cen_current), axis=1)
    train_acc_current = float(np.mean(train_pred_current == labels_arr))
    train_ari_current = float(adjusted_rand_score(labels_arr, train_pred_current))
    print(f"  Train: acc={train_acc_current:.4f}  ARI={train_ari_current:.4f}")

    # -----------------------------------------------------------------------
    # Variant 2: Joint whitening (StandardScaler on 23-dim, Euclidean)
    # -----------------------------------------------------------------------
    print("\n-- Variant 2: Joint whitening (23-dim StandardScaler) -----")
    scaler = StandardScaler()
    combined_train_jw = scaler.fit_transform(combined_train)
    cen_jw = fit_ward_centroids(combined_train_jw, labels_arr)
    train_pred_jw = np.argmin(cdist(combined_train_jw, cen_jw), axis=1)
    train_acc_jw = float(np.mean(train_pred_jw == labels_arr))
    train_ari_jw = float(adjusted_rand_score(labels_arr, train_pred_jw))
    print(f"  Train: acc={train_acc_jw:.4f}  ARI={train_ari_jw:.4f}")

    # -----------------------------------------------------------------------
    # Variant 3: Mahalanobis (EMA-sig + z-RGB, within-class covariance)
    # -----------------------------------------------------------------------
    print("\n-- Variant 3: Mahalanobis (within-class covariance) -------")
    within_cov = within_class_cov(combined_train, train_pred_current)
    try:
        VI = np.linalg.inv(within_cov)
    except np.linalg.LinAlgError:
        VI = np.linalg.pinv(within_cov)
    # Centroids same as Variant 1; only distance metric changes at test time
    cen_mahal = cen_current.copy()
    train_pred_mahal = np.argmin(
        cdist(combined_train, cen_mahal, metric="mahalanobis", VI=VI), axis=1)
    train_acc_mahal = float(np.mean(train_pred_mahal == labels_arr))
    train_ari_mahal = float(adjusted_rand_score(labels_arr, train_pred_mahal))
    print(f"  Train: acc={train_acc_mahal:.4f}  ARI={train_ari_mahal:.4f}  "
          f"(same Ward fit as Variant 1; metric differs only at test time)")

    # -----------------------------------------------------------------------
    # Test all three variants on Test-Colour
    # -----------------------------------------------------------------------
    norm = EMANormaliser(dim=SIG_DIM)
    norm.set_params(norm_params)

    results = {
        "current": {"accs": [], "aris": []},
        "joint_whitening": {"accs": [], "aris": []},
        "mahalanobis": {"accs": [], "aris": []},
    }

    print(f"\nRunning {TEST_EPISODES} Test-Colour episodes...")
    t1 = time.time()
    for ep in range(TEST_EPISODES):
        seed = 1000 + ep
        env = PhysicsEnv(condition="colour", headless=True, seed=seed)
        test_sigs, _, test_labels = collect_signatures(env, norm, update_norm=False)
        test_labels_arr = np.array(test_labels)

        rgb_test = collect_rgb_features(env)
        env.close()

        rgb_test_normed = apply_rgb_normalise(rgb_test, rgb_mean, rgb_std)
        combined_test = np.hstack([test_sigs, rgb_test_normed])

        # Variant 1: Current (Euclidean)
        d1 = cdist(combined_test, cen_current, metric="euclidean")
        p1 = np.argmin(d1, axis=1)
        results["current"]["accs"].append(float(np.mean(p1 == test_labels_arr)))
        results["current"]["aris"].append(float(adjusted_rand_score(test_labels_arr, p1)))

        # Variant 2: Joint whitening (Euclidean on whitened features)
        combined_test_jw = scaler.transform(combined_test)
        d2 = cdist(combined_test_jw, cen_jw, metric="euclidean")
        p2 = np.argmin(d2, axis=1)
        results["joint_whitening"]["accs"].append(float(np.mean(p2 == test_labels_arr)))
        results["joint_whitening"]["aris"].append(float(adjusted_rand_score(test_labels_arr, p2)))

        # Variant 3: Mahalanobis
        d3 = cdist(combined_test, cen_mahal, metric="mahalanobis", VI=VI)
        p3 = np.argmin(d3, axis=1)
        results["mahalanobis"]["accs"].append(float(np.mean(p3 == test_labels_arr)))
        results["mahalanobis"]["aris"].append(float(adjusted_rand_score(test_labels_arr, p3)))

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{TEST_EPISODES}  |  "
                  f"Current={np.mean(results['current']['accs']):.3f}  "
                  f"JointW={np.mean(results['joint_whitening']['accs']):.3f}  "
                  f"Mahal={np.mean(results['mahalanobis']['accs']):.3f}")

    print(f"  Test time: {time.time() - t1:.1f}s")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS (Test-Colour, 50 episodes):")
    print(f"{'Variant':<22}  {'Accuracy':>12}  {'ARI':>12}")
    print("-" * 50)
    label_map = {
        "current":         "Current (Euclidean)",
        "joint_whitening": "Joint whitening",
        "mahalanobis":     "Mahalanobis",
    }
    summary = {}
    for key, lbl in label_map.items():
        accs = results[key]["accs"]
        aris = results[key]["aris"]
        m_acc, s_acc = float(np.mean(accs)), float(np.std(accs))
        m_ari, s_ari = float(np.mean(aris)), float(np.std(aris))
        print(f"{lbl:<22}  {m_acc:.4f}+-{s_acc:.4f}  {m_ari:.4f}+-{s_ari:.4f}")
        summary[key] = {
            "accuracy": {"mean": m_acc, "std": s_acc},
            "ari":      {"mean": m_ari, "std": s_ari},
        }

    out_path = f"{SAVE_DIR}/metric_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_path}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
