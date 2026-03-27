"""
run_test.py - Run 50 test episodes per visual condition and report metrics.

What this does:
  1. Loads Ward centroids and normaliser params from training.
  2. For each test condition (colour, texture, shape):
     - Runs TEST_EPISODES episodes.
     - Recomputes causal signatures from scratch (same intervention procedure).
     - Assigns types using trained Ward centroids.
     - Reports accuracy ? std and ARI ? std.
  3. Runs baseline (RGB) identically but using RGB centroids.
  4. Saves all results to outputs/results.json.

Usage:
  python run_test.py
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
    TEST_EPISODES, N_OBJECTS, SIG_DIM, SAVE_DIR,
    CENTROIDS_FILE, NORM_PARAMS_FILE,
    ACCURACY_THRESHOLD, ARI_THRESHOLD,
)
from env import PhysicsEnv
from intervene import EMANormaliser, collect_signatures
from cluster import predict, hungarian_accuracy, compute_ari
from baseline import collect_rgb_features, apply_rgb_normalise


def load_training_artefacts():
    cog_centroids = np.load(CENTROIDS_FILE)
    rgb_centroids = np.load(f"{SAVE_DIR}/rgb_centroids.npy")
    combined_centroids = np.load(f"{SAVE_DIR}/combined_centroids.npy")

    norm_params = np.load(
        NORM_PARAMS_FILE.replace(".npy", "_cog.npy"), allow_pickle=True
    ).item()

    rgb_mean = np.load(NORM_PARAMS_FILE.replace(".npy", "_rgb_mean.npy"))
    rgb_std = np.load(NORM_PARAMS_FILE.replace(".npy", "_rgb_std.npy"))

    return cog_centroids, rgb_centroids, combined_centroids, norm_params, rgb_mean, rgb_std


def run_condition(condition, cog_centroids, rgb_centroids, combined_centroids,
                  norm_params, rgb_mean, rgb_std, seed_offset=1000):
    """Run TEST_EPISODES for one condition, return per-episode metrics."""
    print(f"\n-- Condition: {condition.upper()} ({'x'.join(['-'] * 40)})")

    # Create a frozen normaliser (no update in test phase)
    norm = EMANormaliser(dim=SIG_DIM)
    norm.set_params(norm_params)

    cog_accs, cog_aris = [], []
    base_accs, base_aris = [], []
    comb_accs, comb_aris = [], []

    for ep in range(TEST_EPISODES):
        seed = seed_offset + ep
        env = PhysicsEnv(condition=condition, headless=True, seed=seed)

        # Causal signatures (no normaliser update)
        sigs, _, labels = collect_signatures(env, norm, update_norm=False)
        labels_arr = np.array(labels)

        # COG prediction
        cog_pred = predict(sigs, cog_centroids)
        acc = float(np.mean(cog_pred == labels_arr))
        ari = compute_ari(cog_pred, labels_arr)
        cog_accs.append(acc)
        cog_aris.append(ari)

        # RGB baseline prediction
        rgb = collect_rgb_features(env)
        rgb_normed = apply_rgb_normalise(rgb, rgb_mean, rgb_std)
        base_pred = predict(rgb_normed, rgb_centroids)
        base_acc = float(np.mean(base_pred == labels_arr))
        base_ari = compute_ari(base_pred, labels_arr)
        base_accs.append(base_acc)
        base_aris.append(base_ari)

        # Combined ablation
        combined = np.hstack([sigs, rgb_normed])
        comb_pred = predict(combined, combined_centroids)
        comb_acc = float(np.mean(comb_pred == labels_arr))
        comb_ari = compute_ari(comb_pred, labels_arr)
        comb_accs.append(comb_acc)
        comb_aris.append(comb_ari)

        env.close()

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep + 1:2d}/{TEST_EPISODES}  |  "
                  f"COG acc={np.mean(cog_accs):.3f}  "
                  f"Baseline acc={np.mean(base_accs):.3f}")

    def fmt(vals):
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    results = {
        "condition": condition,
        "cog": {
            "accuracy": fmt(cog_accs),
            "ari": fmt(cog_aris),
            "pass_accuracy": bool(np.mean(cog_accs) >= ACCURACY_THRESHOLD),
            "pass_ari": bool(np.mean(cog_aris) >= ARI_THRESHOLD),
        },
        "baseline_rgb": {
            "accuracy": fmt(base_accs),
            "ari": fmt(base_aris),
        },
        "combined": {
            "accuracy": fmt(comb_accs),
            "ari": fmt(comb_aris),
        },
    }

    print(f"\n  COG      acc={results['cog']['accuracy']['mean']:.4f}+-{results['cog']['accuracy']['std']:.4f}  "
          f"ARI={results['cog']['ari']['mean']:.4f}+-{results['cog']['ari']['std']:.4f}  "
          f"{'[PASS]' if results['cog']['pass_accuracy'] and results['cog']['pass_ari'] else '[FAIL]'}")
    print(f"  Baseline acc={results['baseline_rgb']['accuracy']['mean']:.4f}+-{results['baseline_rgb']['accuracy']['std']:.4f}  "
          f"ARI={results['baseline_rgb']['ari']['mean']:.4f}+-{results['baseline_rgb']['ari']['std']:.4f}")
    print(f"  Combined acc={results['combined']['accuracy']['mean']:.4f}+-{results['combined']['accuracy']['std']:.4f}  "
          f"ARI={results['combined']['ari']['mean']:.4f}+-{results['combined']['ari']['std']:.4f}")

    return results


def main():
    print("=" * 60)
    print("COG Experiment - Test Phase")
    print("=" * 60)

    if not os.path.exists(CENTROIDS_FILE):
        print("ERROR: centroids not found. Run run_train.py first.")
        return

    artefacts = load_training_artefacts()
    cog_centroids, rgb_centroids, combined_centroids, norm_params, rgb_mean, rgb_std = artefacts

    t0 = time.time()
    test_conditions = ["colour", "texture", "shape"]
    all_results = []

    for cond in test_conditions:
        res = run_condition(cond, cog_centroids, rgb_centroids, combined_centroids,
                            norm_params, rgb_mean, rgb_std)
        all_results.append(res)

    # -- Overall verdict ------------------------------------------------------
    print("\n" + "=" * 60)
    print("OVERALL VERDICT")
    print("=" * 60)

    full_pass = all(
        r["cog"]["pass_accuracy"] and r["cog"]["pass_ari"]
        for r in all_results
    )
    partial_pass = (
        all_results[0]["cog"]["pass_accuracy"] and
        all_results[1]["cog"]["pass_accuracy"] and
        not all_results[2]["cog"]["pass_accuracy"]
    )

    if full_pass:
        print("[PASS]  FULL PASS -- COG meets accuracy and ARI thresholds on ALL conditions.")
        verdict = "PASS"
    elif partial_pass:
        print("[PARTIAL]  PARTIAL PASS -- COG passes Colour and Texture but fails Shape.")
        print("      Shape changes alter collision dynamics slightly (polygon vs circle).")
        print("      Core claim supported with scoped shape-invariant caveat.")
        verdict = "PARTIAL"
    else:
        print("[FAIL]  FAIL -- COG falls below threshold on Test-Colour.")
        print("      Diagnose with analyse.py (UMAP + consistency check).")
        verdict = "FAIL"

    # Load train results and merge
    try:
        with open(f"{SAVE_DIR}/train_results.json") as f:
            train_results = json.load(f)
    except FileNotFoundError:
        train_results = {}

    output = {
        "verdict": verdict,
        "train": train_results,
        "test": all_results,
    }
    with open(f"{SAVE_DIR}/results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to '{SAVE_DIR}/results.json'")
    print(f"Total test time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
