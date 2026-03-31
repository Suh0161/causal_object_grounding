# Causal Object Grounding (COG)

Code for the paper **"Causal Signatures for Robust Object-Type Discovery Under Visual Domain Shifts"**.

![COG explainer](cog_experiment/assets/COG-I.gif)

COG isolates each object, applies controlled impulses in four directions, and records the projected velocity response over five timesteps — a 20-dimensional kinematic signature with no RGB, texture, or shape information. Ward clustering on these signatures recovers object types under colour, texture, and shape shifts where appearance-based methods fail.

---

## Setup

```bash
pip install -r cog_experiment/requirements.txt
```

Requires Python 3.10+. All experiments run headless (no display needed).

---

## Run Order

All scripts must be run from inside the `cog_experiment/` directory:

```bash
cd cog_experiment
```

### 1. Train
```bash
python scripts/run_train.py
```
Runs 100 training episodes, fits Ward clustering, saves centroids and normaliser params to `outputs/`.

### 2. Test
```bash
python scripts/run_test.py
```
Evaluates on Test-Colour, Test-Texture, Test-Shape (50 episodes each). Saves results to `outputs/results.json`.

### 3. GP Baselines
```bash
python scripts/run_gpbaseline.py
```
Runs GP-Ward and GP-Property baselines on all test conditions.

### 4. Mass-Only Baseline
```bash
python scripts/run_massonly.py
```
Runs the t=1 mass-only ablation.

### 5. Noise Sweep
```bash
python scripts/run_noise_sweep.py
```
Re-trains and evaluates at 8 noise levels (sigma = 0.05 to 10.0).

### 6. Reviewer Experiments (additional ablations)
```bash
python scripts/run_k_sensitivity.py    # K = 4,5,6,7,8 cluster count sensitivity
python scripts/run_metric_ablation.py  # combined ablation: Euclidean / whitening / Mahalanobis
python scripts/run_timing.py           # per-object signature extraction cost
python scripts/run_jitter.py           # within-type property jitter (delta = 0,5,10,20%)
```

### 7. Analysis
```bash
python scripts/analyse.py
```
Prints summary statistics and generates UMAP visualisations (requires `umap-learn`).

---

## File Overview

**Core modules** (in `cog_experiment/`):

| File | Description |
|------|-------------|
| `config.py` | All hyperparameters — single source of truth |
| `env.py` | Pygame + pymunk 2D physics environment |
| `intervene.py` | Intervention loop, signature extraction, EMA normaliser |
| `cluster.py` | Ward clustering, Hungarian matching, ARI |
| `baseline.py` | RGB appearance baseline |

**Run scripts** (in `cog_experiment/scripts/`):

| File | Description |
|------|-------------|
| `run_train.py` | Training phase |
| `run_test.py` | Test phase (colour / texture / shape) |
| `run_gpbaseline.py` | GP-Ward and GP-Property baselines |
| `run_massonly.py` | Mass-only (t=1) ablation |
| `run_noise_sweep.py` | Noise robustness sweep |
| `run_k_sensitivity.py` | Cluster count sensitivity (K = 4-8) |
| `run_metric_ablation.py` | Distance metric ablation for combined sig+RGB |
| `run_timing.py` | Computational cost measurement |
| `run_jitter.py` | Within-type property jitter robustness (delta = 0-20%) |
| `analyse.py` | Summary stats and UMAP visualisation |

---

## Results

| Condition | COG (ours) | RGB | Combined (abl.) |
|-----------|-----------|-----|-----------------|
| Train | 0.999 | 1.000 | 1.000 |
| Test-Colour | **1.000** | 0.167 | 0.828 |
| Test-Texture | **1.000** | 0.830 | 1.000 |
| Test-Shape | **1.000** | 1.000 | 1.000 |

Accuracy (mean over 50 episodes). COG holds at 1.000 across all visual shifts. Noise robustness holds across a 200x range (sigma = 0.05 to 10.0).

---

## Citation

```bibtex
@article{amir2026cog,
  title   = {Causal Signatures for Robust Object-Type Discovery Under Visual Domain Shifts},
  author  = {Amir, Mohd Afif Sabrin},
  year    = {2026},
  url     = {https://github.com/Suh0161/causal_object_grounding}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
