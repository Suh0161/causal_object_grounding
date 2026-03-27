"""
run_gpbaseline.py -- GP-UCB-inspired Physical Inference baseline.

Two sub-methods, both using the same 4-dir x 5-step isolation protocol as COG:

  GP-Ward:     Fit 1D GP (fixed RBF+noise kernel) per direction to denoise
               each object's velocity profile; Ward-cluster on GP-mean 20D
               feature with StandardScaler normalisation.

  GP-Property: Use physics model to extract post-bounce speed from GP-denoised
               curve; Ward-cluster on this 1D property estimate (k=6).
               The post-bounce speed e*I/m is unique across all 6 types:
               A=8, B=32, C=50, E=60, D=90, F=140.

Inspired by:
  Seker, M.Y. and Kroemer, O.
  "Estimating Material Properties of Interacting Objects Using Sum-GP-UCB."
  arXiv:2310.11749, 2023.

Usage:
  python run_gpbaseline.py
"""

import os
import json
import time
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from env import PhysicsEnv
from intervene import EMANormaliser, collect_signatures
from cluster import fit_ward, predict, hungarian_accuracy, compute_ari
from config import (
    TRAIN_EPISODES, TEST_EPISODES, N_OBJECTS, N_DIRECTIONS,
    OBS_WINDOW, SIG_DIM, NOISE_SIGMA, SAVE_DIR,
)

T_STEPS = OBS_WINDOW - 1   # 5

# Fixed kernel: length_scale=2 timesteps, noise matches NOISE_SIGMA^2.
# optimizer=None skips hyperparameter optimisation -- fast on 5-point data.
_KERNEL = (
    RBF(length_scale=2.0, length_scale_bounds="fixed")
    + WhiteKernel(noise_level=NOISE_SIGMA ** 2, noise_level_bounds="fixed")
)


# ── GP utilities ────────────────────────────────────────────────────────────────

def _gp_denoise(v5: np.ndarray) -> np.ndarray:
    """Fit GP to 5-step velocity profile; return GP posterior mean (5,)."""
    t = np.arange(1, T_STEPS + 1, dtype=float).reshape(-1, 1)
    gp = GaussianProcessRegressor(kernel=_KERNEL, optimizer=None,
                                   normalize_y=False)
    gp.fit(t, v5)
    return gp.predict(t)


def _denoise_sig(raw20: np.ndarray) -> np.ndarray:
    """GP-denoise a 20-dim raw signature (4 dirs x 5 steps)."""
    out = np.empty(SIG_DIM)
    for d in range(N_DIRECTIONS):
        sl = slice(d * T_STEPS, (d + 1) * T_STEPS)
        out[sl] = _gp_denoise(raw20[sl])
    return out


def _post_bounce_speed(raw20: np.ndarray) -> float:
    """
    Physics-model feature: post-bounce speed = e * I/m, averaged over dirs.

    For heavy types (A, B): v[0] > 0 (not yet bounced at t=1).
      Post-bounce speed = |v[1]| = e * I/m.
    For light types (C-F): v[0] < 0 (already bounced by t=1).
      Post-bounce speed = |v[0]| = e * I/m.

    Expected values per type: A=8, B=32, C=50, E=60, D=90, F=140.
    """
    speeds = []
    for d in range(N_DIRECTIONS):
        v = raw20[d * T_STEPS: (d + 1) * T_STEPS]
        speeds.append(abs(v[1]) if v[0] > 0 else abs(v[0]))
    return float(np.mean(speeds))


# ── Training ────────────────────────────────────────────────────────────────────

def train_gp_baselines():
    print("=" * 60)
    print("GP Baseline -- Training Phase")
    print(f"Episodes: {TRAIN_EPISODES}  |  Condition: train")
    print("=" * 60)

    dummy_norm = EMANormaliser(dim=SIG_DIM)
    all_raw, all_labels = [], []
    t0 = time.time()

    for ep in range(TRAIN_EPISODES):
        env = PhysicsEnv(condition="train", headless=True, seed=ep)
        _, raw_sigs, labels = collect_signatures(env, dummy_norm,
                                                  update_norm=False)
        env.close()
        all_raw.append(raw_sigs)
        all_labels.extend(labels)
        if (ep + 1) % 20 == 0:
            print(f"  ep {ep + 1}/{TRAIN_EPISODES}  ({time.time()-t0:.0f}s)")

    raw_mat = np.vstack(all_raw)          # (1200, 20)
    labels_arr = np.array(all_labels)

    # -- GP-Ward: denoise -> StandardScale -> Ward ---------------------------
    print(f"\nGP-denoising {len(raw_mat)} signatures...")
    t1 = time.time()
    gpw_feats = np.array([_denoise_sig(raw_mat[i]) for i in range(len(raw_mat))])
    print(f"  done in {time.time()-t1:.1f}s")

    scaler = StandardScaler()
    gpw_scaled = scaler.fit_transform(gpw_feats)
    gpw_centroids, _, gpw_train = fit_ward(gpw_scaled, labels_arr)
    print(f"  GP-Ward   train acc={gpw_train['accuracy']:.4f}  "
          f"ARI={gpw_train['ari']:.4f}")

    # -- GP-Property: post-bounce speed (1D) -> Ward -------------------------
    gpp_feats = np.array([[_post_bounce_speed(raw_mat[i])]
                           for i in range(len(raw_mat))])
    gpp_centroids, _, gpp_train = fit_ward(gpp_feats, labels_arr)
    print(f"  GP-Prop   train acc={gpp_train['accuracy']:.4f}  "
          f"ARI={gpp_train['ari']:.4f}")

    print(f"\nTotal train time: {time.time()-t0:.1f}s")

    artefacts = {
        "gpw_centroids": gpw_centroids,
        "gpp_centroids": gpp_centroids,
        "scaler": scaler,
        "gpw_train": gpw_train,
        "gpp_train": gpp_train,
    }
    return artefacts


# ── Test ────────────────────────────────────────────────────────────────────────

def test_condition(condition, artefacts, seed_offset=1000):
    gpw_c = artefacts["gpw_centroids"]
    gpp_c = artefacts["gpp_centroids"]
    scaler = artefacts["scaler"]
    dummy_norm = EMANormaliser(dim=SIG_DIM)

    gpw_accs, gpw_aris = [], []
    gpp_accs, gpp_aris = [], []

    for ep in range(TEST_EPISODES):
        env = PhysicsEnv(condition=condition, headless=True,
                         seed=seed_offset + ep)
        _, raw_sigs, labels = collect_signatures(env, dummy_norm,
                                                  update_norm=False)
        env.close()
        labels_arr = np.array(labels)

        # GP-Ward
        gpw_feats = np.array([_denoise_sig(raw_sigs[i])
                               for i in range(N_OBJECTS)])
        gpw_scaled = scaler.transform(gpw_feats)
        gpw_pred = predict(gpw_scaled, gpw_c)
        gpw_accs.append(float(np.mean(gpw_pred == labels_arr)))
        gpw_aris.append(compute_ari(gpw_pred, labels_arr))

        # GP-Property
        gpp_feats = np.array([[_post_bounce_speed(raw_sigs[i])]
                               for i in range(N_OBJECTS)])
        gpp_pred = predict(gpp_feats, gpp_c)
        gpp_accs.append(float(np.mean(gpp_pred == labels_arr)))
        gpp_aris.append(compute_ari(gpp_pred, labels_arr))

    def fmt(vals):
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    return {
        "condition": condition,
        "gpward": {"accuracy": fmt(gpw_accs), "ari": fmt(gpw_aris)},
        "gpprop": {"accuracy": fmt(gpp_accs), "ari": fmt(gpp_aris)},
    }


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    t0 = time.time()

    artefacts = train_gp_baselines()

    results = {
        "train": {
            "gpward": artefacts["gpw_train"],
            "gpprop": artefacts["gpp_train"],
        }
    }

    for cond in ["colour", "texture", "shape"]:
        print(f"\n-- {cond.upper()} ({TEST_EPISODES} episodes) ------")
        res = test_condition(cond, artefacts)
        results[cond] = res
        print(f"  GP-Ward  acc={res['gpward']['accuracy']['mean']:.4f}"
              f"+-{res['gpward']['accuracy']['std']:.4f}"
              f"  ARI={res['gpward']['ari']['mean']:.4f}")
        print(f"  GP-Prop  acc={res['gpprop']['accuracy']['mean']:.4f}"
              f"+-{res['gpprop']['accuracy']['std']:.4f}"
              f"  ARI={res['gpprop']['ari']['mean']:.4f}")

    out_path = f"{SAVE_DIR}/gpbaseline_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Saved -> {out_path}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
