"""
run_timing.py -- Measure per-object signature extraction cost.

Runs one episode and times the intervention loop for a single object
across 4 directions, then reports wall-clock cost per object.

Usage:
  python run_timing.py
"""

import os
import sys
import time
import numpy as np

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from config import N_OBJECTS, N_DIRECTIONS, OBS_WINDOW, SIG_DIM, SAVE_DIR, NORM_PARAMS_FILE
from env import PhysicsEnv
from intervene import EMANormaliser, collect_signatures

N_TIMING_EPISODES = 20


def main():
    print("=" * 50)
    print("Computational Cost Measurement")
    print("=" * 50)

    norm_params = np.load(
        NORM_PARAMS_FILE.replace(".npy", "_cog.npy"), allow_pickle=True
    ).item()

    episode_times = []
    for ep in range(N_TIMING_EPISODES):
        norm = EMANormaliser(dim=SIG_DIM)
        norm.set_params(norm_params)

        env = PhysicsEnv(condition="train", headless=True, seed=ep)

        t0 = time.perf_counter()
        sigs, _, labels = collect_signatures(env, norm, update_norm=False)
        elapsed = time.perf_counter() - t0

        env.close()
        episode_times.append(elapsed)

    mean_ep  = np.mean(episode_times)
    std_ep   = np.std(episode_times)
    per_obj  = mean_ep / N_OBJECTS                 # 12 objects per episode
    per_dir  = per_obj / N_DIRECTIONS              # 4 directions per object
    per_step = per_dir / (OBS_WINDOW - 1)          # 5 steps per direction

    print(f"\nResults over {N_TIMING_EPISODES} episodes:")
    print(f"  Mean episode time : {mean_ep*1000:.1f} ms  (std {std_ep*1000:.1f} ms)")
    print(f"  Per object        : {per_obj*1000:.2f} ms")
    print(f"  Per direction     : {per_dir*1000:.2f} ms")
    print(f"  Per physics step  : {per_step*1000:.3f} ms")
    print(f"\n  Objects/second   : {1/per_obj:.0f}")
    print(f"  Signature dims   : {N_DIRECTIONS} directions x {OBS_WINDOW-1} steps = {SIG_DIM}")

    import json
    out = {
        "n_objects": N_OBJECTS,
        "n_directions": N_DIRECTIONS,
        "obs_window_steps": OBS_WINDOW - 1,
        "sig_dim": SIG_DIM,
        "mean_episode_ms": round(mean_ep * 1000, 2),
        "std_episode_ms": round(std_ep * 1000, 2),
        "per_object_ms": round(per_obj * 1000, 2),
        "objects_per_second": round(1 / per_obj, 1),
    }
    with open(f"{SAVE_DIR}/timing_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {SAVE_DIR}/timing_results.json")


if __name__ == "__main__":
    main()
