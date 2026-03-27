"""
analyse.py - UMAP visualisation, metrics summary, pass/fail verdict, failure diagnosis.

Usage:
  python analyse.py             # full analysis with UMAP
  python analyse.py --no-umap   # skip UMAP (no umap-learn required)
"""

import argparse
import json
import os
import sys

import numpy as np

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from config import (
    ACCURACY_THRESHOLD,
    ARI_THRESHOLD,
    CONSISTENCY_THRESHOLD,
    SAVE_DIR,
    TRAIN_LABELS_FILE,
    TRAIN_SIGS_FILE,
)

TYPE_NAMES = ["A", "B", "C", "D", "E", "F"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-umap", action="store_true", help="Skip UMAP visualisation")
    args = parser.parse_args()

    results_path = f"{SAVE_DIR}/results.json"
    if not os.path.exists(results_path):
        print("ERROR: results.json not found. Run run_train.py and run_test.py first.")
        sys.exit(1)

    with open(results_path) as f:
        results = json.load(f)

    print_summary(results)
    diagnose(results)

    if not args.no_umap:
        plot_umap()


def print_summary(results):
    print("\n" + "=" * 70)
    print("COG EXPERIMENT - RESULTS SUMMARY")
    print("=" * 70)

    train = results.get("train", {})
    print(f"\n{'Phase':<12} {'System':<12} {'Accuracy':>18} {'ARI':>16} {'Status':>6}")
    print("-" * 70)

    def row(phase, system, metrics, threshold=None):
        acc = metrics.get("accuracy", {})
        ari = metrics.get("ari", {})
        if isinstance(acc, dict):
            acc_str = f"{acc['mean']:.4f}+/-{acc['std']:.4f}"
            ari_str = f"{ari['mean']:.4f}+/-{ari['std']:.4f}"
            acc_value = acc["mean"]
        else:
            acc_str = f"{acc:.4f}"
            ari_str = f"{ari:.4f}"
            acc_value = acc
        status = ""
        if threshold is not None:
            status = "OK" if acc_value >= threshold else "FAIL"
        print(f"{phase:<12} {system:<12} {acc_str:>18} {ari_str:>16} {status:>6}")

    if train:
        row("TRAIN", "COG", train.get("cog", {}), ACCURACY_THRESHOLD)
        row("TRAIN", "Baseline", train.get("baseline_rgb", {}))
        row("TRAIN", "Combined", train.get("combined", {}))
        ratio = train.get("consistency_ratio")
        if ratio is not None:
            print(
                f"Consistency ratio: {ratio:.4f} "
                f"(threshold: {CONSISTENCY_THRESHOLD}, ARI target: {ARI_THRESHOLD})"
            )

    print()
    for result in results.get("test", []):
        cond = result["condition"].upper()
        row(cond, "COG", result["cog"], ACCURACY_THRESHOLD)
        row(cond, "Baseline", result["baseline_rgb"])
        row(cond, "Combined", result["combined"])
        print()

    verdict = results.get("verdict", "UNKNOWN")
    verdict_labels = {
        "PASS": "FULL PASS",
        "PARTIAL": "PARTIAL PASS",
        "FAIL": "FAIL",
    }
    print("=" * 70)
    print(f"VERDICT: {verdict_labels.get(verdict, verdict)}")
    print("=" * 70)


def diagnose(results):
    verdict = results.get("verdict", "")
    if verdict == "PASS":
        print("\n[OK] No diagnosis needed - experiment passed.")
        return

    print("\n-- FAILURE DIAGNOSIS --")
    train = results.get("train", {})
    ratio = train.get("consistency_ratio", float("inf"))

    colour_result = next(
        (result for result in results.get("test", []) if result["condition"] == "colour"),
        None,
    )

    if colour_result and colour_result["cog"]["accuracy"]["mean"] < ACCURACY_THRESHOLD:
        acc = colour_result["cog"]["accuracy"]["mean"]
        print(f"COG failed on colour shift (acc={acc:.4f}).")

        if ratio > CONSISTENCY_THRESHOLD:
            print(f"Signatures are too noisy (ratio={ratio:.4f} > {CONSISTENCY_THRESHOLD}).")
            print("Suggested checks:")
            print("  1. Extend OBS_WINDOW in config.py.")
            print("  2. Increase IMPULSE_MAGNITUDE in config.py.")
        else:
            print(f"Signatures are separable (ratio={ratio:.4f} <= {CONSISTENCY_THRESHOLD}).")
            print("Clustering is then the likely bottleneck.")
            print("Suggested checks:")
            print("  1. Run python analyse.py to inspect UMAP clusters.")
            print("  2. Compare Ward clustering against k-means or GMM.")

    for result in results.get("test", []):
        cog_acc = result["cog"]["accuracy"]["mean"]
        rgb_acc = result["baseline_rgb"]["accuracy"]["mean"]
        print(
            f"Condition {result['condition']}: "
            f"COG acc={cog_acc:.4f}  Baseline acc={rgb_acc:.4f}"
        )

    if verdict == "PARTIAL":
        print("Shape shift is the remaining failure mode in this run.")


def plot_umap():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import umap
    except ImportError:
        print("\n[WARN] UMAP or matplotlib not installed. Skipping visualisation.")
        print("Install with: pip install umap-learn matplotlib")
        return

    if not os.path.exists(TRAIN_SIGS_FILE) or not os.path.exists(TRAIN_LABELS_FILE):
        print("\n[WARN] Training signatures not found. Skipping UMAP.")
        return

    sigs = np.load(TRAIN_SIGS_FILE)
    labels = np.load(TRAIN_LABELS_FILE)

    print("\nRunning UMAP on training signatures...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(sigs)

    colours = ["#CC2200", "#0033CC", "#00AA44", "#AA00AA", "#CC8800", "#008888"]
    legend_names = [f"Type {name}" for name in TYPE_NAMES]

    fig, ax = plt.subplots(figsize=(8, 6))
    for type_idx, legend_name in enumerate(legend_names):
        mask = labels == type_idx
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=colours[type_idx],
            label=legend_name,
            alpha=0.65,
            s=20,
            linewidths=0,
        )

    ax.set_title("UMAP of 20-dim Causal Signatures (Train Condition)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(
        handles=[mpatches.Patch(color=colours[i], label=legend_names[i]) for i in range(len(TYPE_NAMES))]
    )
    plt.tight_layout()

    out_path = f"{SAVE_DIR}/umap_signatures.png"
    plt.savefig(out_path, dpi=150)
    print(f"UMAP plot saved to '{out_path}'")
    plt.close()


if __name__ == "__main__":
    main()
