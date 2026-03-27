"""
run_train.py - Run 100 training episodes under the 'train' visual condition.

What this does:
  1. Runs TRAIN_EPISODES episodes in 'train' condition.
  2. Collects causal signatures + RGB features for all objects.
  3. Fits Ward clustering on signatures (COG) and on RGB (Baseline).
  4. Fits COG+RGB combined (ablation).
  5. Saves: centroids, normaliser params, raw signatures, labels, metrics.
  6. Prints passage check for signature consistency ratio.

Usage:
  python run_train.py
"""

import os
import sys
import json
import time
import numpy as np

# Ensure headless SDL before pygame import
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from config import (
    TRAIN_EPISODES, N_OBJECTS, SIG_DIM, SAVE_DIR,
    CENTROIDS_FILE, NORM_PARAMS_FILE,
    TRAIN_SIGS_FILE, TRAIN_LABELS_FILE,
    CONSISTENCY_THRESHOLD,
)
from env import PhysicsEnv
from intervene import EMANormaliser, collect_signatures, signature_consistency_ratio
from cluster import fit_ward, evaluate
from baseline import collect_rgb_features, normalise_rgb


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("=" * 60)
    print("COG Experiment - Training Phase")
    print(f"Episodes: {TRAIN_EPISODES}  |  Condition: train")
    print("=" * 60)

    normaliser = EMANormaliser(dim=SIG_DIM)

    all_sigs = []
    all_raw_sigs = []
    all_labels = []
    all_rgb = []

    t0 = time.time()
    for ep in range(TRAIN_EPISODES):
        env = PhysicsEnv(condition="train", headless=True, seed=ep)
        sigs, raw_sigs, labels = collect_signatures(env, normaliser, update_norm=True)
        rgb = collect_rgb_features(env)  # rendered
        env.close()

        all_sigs.append(sigs)
        all_raw_sigs.append(raw_sigs)
        all_labels.extend(labels)
        all_rgb.append(rgb)

        if (ep + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (ep + 1) * (TRAIN_EPISODES - ep - 1)
            print(f"  Episode {ep + 1:3d}/{TRAIN_EPISODES}  "
                  f"[{elapsed:.1f}s elapsed, {eta:.1f}s remaining]")

    # Stack raw signatures and re-normalise with the FINAL converged normaliser.
    # (Stacking the incrementally-normalised all_sigs would mix early badly-estimated
    #  z-scores with late well-estimated ones, biasing the Ward centroids.)
    raw_sigs_mat = np.vstack(all_raw_sigs)   # (TRAIN_EPISODES * N_OBJECTS, SIG_DIM)
    sigs_mat = np.stack([normaliser.normalise(r) for r in raw_sigs_mat])
    labels_arr = np.array(all_labels)        # (TRAIN_EPISODES * N_OBJECTS,)
    rgb_mat = np.vstack(all_rgb)             # (TRAIN_EPISODES * N_OBJECTS, 3)

    print(f"\nTotal signature matrix: {sigs_mat.shape}")
    print(f"Total label vector:     {labels_arr.shape}")

    # --Signature consistency ------------------------------------------------
    ratio = signature_consistency_ratio(sigs_mat, labels_arr)
    print(f"\nSignature consistency ratio: {ratio:.4f}  "
          f"  (threshold: {CONSISTENCY_THRESHOLD})")
    if ratio > CONSISTENCY_THRESHOLD:
        print("  [!] WARNING: ratio > threshold -- types may not be separable.")
        print("     Consider extending OBS_WINDOW or increasing IMPULSE_MAGNITUDE.")
    else:
        print("  [OK] Signatures separable -- consistent with core claim.")

    # --COG: Ward on causal signatures --------------------------------------
    print("\n--COG (Ward on signatures) --------------------------------")
    cog_centroids, cog_train_pred, cog_train_metrics = fit_ward(sigs_mat, labels_arr)
    print(f"  Train Accuracy: {cog_train_metrics['accuracy']:.4f}")
    print(f"  Train ARI:      {cog_train_metrics['ari']:.4f}")

    # --Baseline: Ward on RGB -----------------------------------------------
    print("\n--Baseline (Ward on RGB) ---------------------------------")
    rgb_normed, rgb_mean, rgb_std = normalise_rgb(all_rgb)
    rgb_centroids, rgb_train_pred, rgb_train_metrics = fit_ward(rgb_normed, labels_arr)
    print(f"  Train Accuracy: {rgb_train_metrics['accuracy']:.4f}")
    print(f"  Train ARI:      {rgb_train_metrics['ari']:.4f}")

    # --Combined ablation: sig + RGB ----------------------------------------
    print("\n--Combined (sig + RGB, ablation) -------------------------")
    rgb_normed_flat = normalise_rgb(all_rgb)[0]
    combined_mat = np.hstack([sigs_mat, rgb_normed_flat])
    combined_centroids, _, combined_train_metrics = fit_ward(combined_mat, labels_arr)
    print(f"  Train Accuracy: {combined_train_metrics['accuracy']:.4f}")
    print(f"  Train ARI:      {combined_train_metrics['ari']:.4f}")

    # --Save artefacts ------------------------------------------------------
    np.save(CENTROIDS_FILE, cog_centroids)
    np.save(NORM_PARAMS_FILE.replace(".npy", "_cog.npy"),
            normaliser.get_params(), allow_pickle=True)
    np.save(NORM_PARAMS_FILE.replace(".npy", "_rgb_mean.npy"), rgb_mean)
    np.save(NORM_PARAMS_FILE.replace(".npy", "_rgb_std.npy"), rgb_std)
    np.save(f"{SAVE_DIR}/rgb_centroids.npy", rgb_centroids)
    np.save(f"{SAVE_DIR}/combined_centroids.npy", combined_centroids)
    np.save(TRAIN_SIGS_FILE, sigs_mat)
    np.save(TRAIN_LABELS_FILE, labels_arr)

    train_results = {
        "condition": "train",
        "n_episodes": TRAIN_EPISODES,
        "consistency_ratio": ratio,
        "cog": cog_train_metrics,
        "baseline_rgb": rgb_train_metrics,
        "combined": combined_train_metrics,
    }
    with open(f"{SAVE_DIR}/train_results.json", "w") as f:
        json.dump(train_results, f, indent=2)

    print(f"\nSaved to '{SAVE_DIR}/'")
    print(f"Total time: {time.time() - t0:.1f}s")
    print("\n[OK] Training complete.")


if __name__ == "__main__":
    main()
