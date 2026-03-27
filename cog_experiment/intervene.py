"""
intervene.py — Intervention loop, 5-step trajectory collection,
20-dim causal signature construction, and EMA normalisation.

Signature layout for one object:
  For each direction d in {right, up, left, down}:
    [v_proj_t1, v_proj_t2, v_proj_t3, v_proj_t4, v_proj_t5]
    where v_proj_tk = dot(velocity_at_step_k, direction_vec)
  Concatenated -> 4 x 5 = 20-dim vector.

OBS_WINDOW=6 steps are recorded; the last step is discarded (deltas[:-1]).
Sensor noise sigma=0.05 is added per step on the projected axis.
"""

import numpy as np
import pymunk
from config import (
    N_OBJECTS, N_DIRECTIONS, IMPULSE_DIRECTIONS, OBS_WINDOW, SIG_DIM,
    INTERVENTIONS_PER_EPISODE, INTERVENTION_EVERY, EMA_DECAY, EPISODE_STEPS,
    NOISE_SIGMA,
)


# ── EMA Normaliser ──────────────────────────────────────────────────────────────

class EMANormaliser:
    """
    Running z-score normaliser using exponential moving average.
    State: (mean, var) per dimension.
    """

    def __init__(self, dim: int, decay: float = EMA_DECAY):
        self.decay = decay
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self._count = 0

    def update(self, x: np.ndarray):
        """Update running stats with a new observation vector x (1-D)."""
        self._count += 1
        delta = x - self.mean
        self.mean = self.decay * self.mean + (1 - self.decay) * x
        self.var = self.decay * self.var + (1 - self.decay) * delta ** 2

    def normalise(self, x: np.ndarray) -> np.ndarray:
        """Z-score normalise x using current running stats."""
        std = np.sqrt(self.var + 1e-8)
        return (x - self.mean) / std

    def get_params(self):
        return {"mean": self.mean.copy(), "var": self.var.copy()}

    def set_params(self, params: dict):
        self.mean = params["mean"].copy()
        self.var = params["var"].copy()


# ── Signature Collector ─────────────────────────────────────────────────────────

def collect_signatures(env, normaliser: EMANormaliser, update_norm: bool = True, noise_sigma: float = None):
    """
    Run one full episode, applying ALL 24 (obj, dir) interventions.

    Each object receives all 4 impulse directions within one episode.
    24 interventions × (OBS_WINDOW+1 gap) steps = ~120 steps max,
    well within the 200-step episode budget.

    Parameters
    ----------
    env : PhysicsEnv
    normaliser : EMANormaliser
    update_norm : bool

    Returns
    -------
    sigs : (N_OBJECTS, SIG_DIM) normalised signature per object.
    raw_sigs : (N_OBJECTS, SIG_DIM) un-normalised.
    labels : list[int] ground truth type labels.
    """
    # All (obj, dir) pairs — full coverage every episode
    schedule = []
    rng = np.random.default_rng()
    pairs = [(o, d) for o in range(N_OBJECTS) for d in range(N_DIRECTIONS)]
    order = rng.permutation(len(pairs))
    schedule = [pairs[i] for i in order]

    # Gap between interventions (steps of free physics to let objects settle)
    SETTLE_GAP = 4   # steps between interventions

    raw_per_dir = [[None for _ in range(N_DIRECTIONS)] for _ in range(N_OBJECTS)]

    step = 0

    for obj_idx, dir_idx in schedule:
        # Settle physics briefly
        env.step(SETTLE_GAP)
        step += SETTLE_GAP
        if step >= EPISODE_STEPS - OBS_WINDOW - 2:
            break

        direction = IMPULSE_DIRECTIONS[dir_idx]
        body = env.bodies[obj_idx]
        shape = env.shapes[obj_idx]

        # Isolate object & guarantee collision using a temporary box
        original_filter = shape.filter
        # Object is category 2, only collides with category 4 (the box)
        shape.filter = pymunk.ShapeFilter(categories=0x2, mask=0x4)

        box_r = 15 + 2  # RADIUS=15 + 2px gap
        px, py = body.position
        
        box_pts = [
            [(px - box_r, py - box_r), (px + box_r, py - box_r)],
            [(px - box_r, py + box_r), (px + box_r, py + box_r)],
            [(px - box_r, py - box_r), (px - box_r, py + box_r)],
            [(px + box_r, py - box_r), (px + box_r, py + box_r)],
        ]
        
        temp_walls = []
        for a, b in box_pts:
            seg = pymunk.Segment(env.space.static_body, a, b, 1)
            seg.friction = 0.5
            seg.elasticity = 1.0  # Perfect bounce from wall
            # Wall is category 4, only collides with category 2 (the object)
            seg.filter = pymunk.ShapeFilter(categories=0x4, mask=0x2)
            env.space.add(seg)
            temp_walls.append(seg)

        # Zero target velocity and gravity for a clean, reproducible I/m measurement.
        body.velocity = (0.0, 0.0)
        body.angular_velocity = 0.0
        
        original_gravity = env.space.gravity
        env.space.gravity = (0, 0)

        # Apply impulse
        env.apply_impulse(obj_idx, direction)

        # Record OBS_WINDOW steps post-impulse.
        # Project onto the impulse axis to keep only the physically meaningful
        # component; this eliminates the orthogonal near-zero dims that would
        # otherwise dominate the EMA-normalised signature with unit-variance noise.
        direction_vec = np.array(direction)
        deltas = []
        for _ in range(OBS_WINDOW):
            env.step(1)
            step += 1
            v_now = np.array(body.velocity)
            _sigma = NOISE_SIGMA if noise_sigma is None else noise_sigma
            v_proj = float(np.dot(v_now, direction_vec)) + rng.normal(0.0, _sigma)
            deltas.append(v_proj)

        # Cleanup box and restore filter and gravity
        for seg in temp_walls:
            env.space.remove(seg)
        shape.filter = original_filter
        env.space.gravity = original_gravity

        # Store t+1, t+2, t+3 etc delta-vs (take first OBS_WINDOW-1 steps)
        raw_per_dir[obj_idx][dir_idx] = deltas[:-1]


    # ── Build raw signature vectors ─────────────────────────────────────────
    raw_sigs = np.zeros((N_OBJECTS, SIG_DIM), dtype=np.float64)
    expected_steps = OBS_WINDOW - 1

    for obj_idx in range(N_OBJECTS):
        sig_parts = []
        for dir_idx in range(N_DIRECTIONS):
            deltas = raw_per_dir[obj_idx][dir_idx]
            if deltas is not None and len(deltas) == expected_steps:
                sig_parts.extend([float(dv) for dv in deltas])
            else:
                sig_parts.extend([0.0] * expected_steps)
        raw_sigs[obj_idx] = sig_parts

    # ── Update normaliser ───────────────────────────────────────────────────
    if update_norm:
        for obj_idx in range(N_OBJECTS):
            normaliser.update(raw_sigs[obj_idx])

    # ── Normalise ───────────────────────────────────────────────────────────
    sigs = np.stack([normaliser.normalise(raw_sigs[obj_idx])
                     for obj_idx in range(N_OBJECTS)])

    labels = list(env.type_labels)
    return sigs, raw_sigs, labels


# ── Signature Consistency Metric ────────────────────────────────────────────────

def signature_consistency_ratio(sigs: np.ndarray, labels: list) -> float:
    """
    Intra-type variance / Inter-type variance.
    Lower = more separable.

    Parameters
    ----------
    sigs : (M, 20) array of normalised signatures across episodes.
    labels : list of M type labels (int).
    """
    sigs = np.array(sigs)
    labels = np.array(labels)
    unique_types = np.unique(labels)

    per_type_means = []
    intra_vars = []
    for t in unique_types:
        mask = labels == t
        sigs_t = sigs[mask]
        mean_t = sigs_t.mean(axis=0)
        per_type_means.append(mean_t)
        intra_vars.append(np.var(sigs_t, axis=0).mean())

    intra_var = np.mean(intra_vars)
    global_mean = np.array(per_type_means).mean(axis=0)
    inter_var = np.var(np.array(per_type_means), axis=0).mean()

    if inter_var < 1e-10:
        return float("inf")
    return float(intra_var / inter_var)
