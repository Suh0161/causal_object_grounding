"""
run_noise_sweep.py - Noise sensitivity sweep for COG.

For each noise level sigma in NOISE_LEVELS:
  1. Re-trains COG from scratch (100 episodes, train condition).
  2. Tests on Test-Colour (50 episodes) -- hardest shift condition.
  3. Records mean accuracy.

Saves results to outputs/noise_sweep.json.

Usage:
  python run_noise_sweep.py
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
    TRAIN_EPISODES, TEST_EPISODES, N_OBJECTS, SIG_DIM, SAVE_DIR,
)
from env import PhysicsEnv
from intervene import EMANormaliser, collect_signatures
from cluster import fit_ward, predict, hungarian_accuracy, compute_ari

NOISE_LEVELS = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]


def train_one(sigma, seed_offset=0):
    """Train COG at a given noise level. Returns (centroids, norm_params)."""
    normaliser = EMANormaliser(dim=SIG_DIM)
    all_raw_sigs = []
    all_labels = []

    for ep in range(TRAIN_EPISODES):
        env = PhysicsEnv(condition="train", headless=True, seed=seed_offset + ep)
        _, raw_sigs, labels = collect_signatures(env, normaliser, update_norm=True, noise_sigma=sigma)
        env.close()
        all_raw_sigs.append(raw_sigs)
        all_labels.extend(labels)

    raw_sigs_mat = np.vstack(all_raw_sigs)
    sigs_mat = np.stack([normaliser.normalise(r) for r in raw_sigs_mat])
    labels_arr = np.array(all_labels)

    centroids, _, train_metrics = fit_ward(sigs_mat, labels_arr)
    return centroids, normaliser.get_params(), train_metrics


def test_one(sigma, centroids, norm_params, condition="colour", seed_offset=1000):
    """Test COG at a given noise level on one visual condition."""
    norm = EMANormaliser(dim=SIG_DIM)
    norm.set_params(norm_params)

    accs, aris = [], []
    for ep in range(TEST_EPISODES):
        env = PhysicsEnv(condition=condition, headless=True, seed=seed_offset + ep)
        sigs, _, labels = collect_signatures(env, norm, update_norm=False, noise_sigma=sigma)
        env.close()

        labels_arr = np.array(labels)
        pred = predict(sigs, centroids)
        acc, _ = hungarian_accuracy(pred, labels_arr)
        ari = compute_ari(pred, labels_arr)
        accs.append(acc)
        aris.append(ari)

    return {"mean": float(np.mean(accs)), "std": float(np.std(accs)),
            "ari_mean": float(np.mean(aris)), "ari_std": float(np.std(aris))}


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("=" * 60)
    print("COG Noise Sweep")
    print(f"Noise levels: {NOISE_LEVELS}")
    print("=" * 60)

    results = []
    t0 = time.time()

    for sigma in NOISE_LEVELS:
        print(f"\n  sigma={sigma:.2f}  -- training ...", flush=True)
        centroids, norm_params, train_metrics = train_one(sigma)
        print(f"    train acc={train_metrics['accuracy']:.4f}  -- testing colour ...", flush=True)
        colour_metrics = test_one(sigma, centroids, norm_params, condition="colour")
        print(f"    colour acc={colour_metrics['mean']:.4f}+-{colour_metrics['std']:.4f}", flush=True)

        results.append({
            "sigma": sigma,
            "train_accuracy": train_metrics["accuracy"],
            "colour_accuracy_mean": colour_metrics["mean"],
            "colour_accuracy_std": colour_metrics["std"],
            "colour_ari_mean": colour_metrics["ari_mean"],
            "colour_ari_std": colour_metrics["ari_std"],
        })

    out_path = f"{SAVE_DIR}/noise_sweep.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone in {time.time() - t0:.1f}s. Saved to {out_path}")
    print("\nSummary:")
    print(f"{'sigma':>8}  {'train_acc':>10}  {'colour_acc':>12}")
    print("-" * 36)
    for r in results:
        print(f"{r['sigma']:>8.2f}  {r['train_accuracy']:>10.4f}  "
              f"{r['colour_accuracy_mean']:>10.4f}+-{r['colour_accuracy_std']:.4f}")


if __name__ == "__main__":
    main()
