# intervene.py
# intervention loop: hit each object in 4 directions, record velocity response,
# build the 20-dim causal signature, normalise with EMA running stats
#
# signature layout per object:
#   [right_t1..t5, up_t1..t5, left_t1..t5, down_t1..t5] = 20 values
#   each value is the velocity projected onto the impulse axis (not 2D vector)
#
# why projected? the orthogonal component is near-zero + noise. after EMA
# normalisation it gets scaled to unit variance and drowns out the real signal.
# projection keeps only the physically meaningful part.

import numpy as np
import pymunk
from config import (
    N_OBJECTS, N_DIRECTIONS, IMPULSE_DIRECTIONS, OBS_WINDOW, SIG_DIM,
    INTERVENTIONS_PER_EPISODE, INTERVENTION_EVERY, EMA_DECAY, EPISODE_STEPS,
    NOISE_SIGMA,
)


class EMANormaliser:
    """Running z-score normaliser (exponential moving average of mean and var)."""

    def __init__(self, dim: int, decay: float = EMA_DECAY):
        self.decay = decay
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self._count = 0

    def update(self, x: np.ndarray):
        self._count += 1
        delta = x - self.mean
        self.mean = self.decay * self.mean + (1 - self.decay) * x
        self.var = self.decay * self.var + (1 - self.decay) * delta ** 2

    def normalise(self, x: np.ndarray) -> np.ndarray:
        std = np.sqrt(self.var + 1e-8)
        return (x - self.mean) / std

    def get_params(self):
        return {"mean": self.mean.copy(), "var": self.var.copy()}

    def set_params(self, params: dict):
        self.mean = params["mean"].copy()
        self.var = params["var"].copy()


def collect_signatures(env, normaliser: EMANormaliser, update_norm: bool = True, noise_sigma: float = None):
    """
    Run one episode, apply all 24 (obj, dir) interventions, return signatures.

    Returns sigs (normalised), raw_sigs (un-normalised), labels.
    """
    rng = np.random.default_rng()
    pairs = [(o, d) for o in range(N_OBJECTS) for d in range(N_DIRECTIONS)]
    schedule = [pairs[i] for i in rng.permutation(len(pairs))]

    SETTLE_GAP = 4   # let objects stop wobbling between interventions

    raw_per_dir = [[None for _ in range(N_DIRECTIONS)] for _ in range(N_OBJECTS)]
    step = 0

    for obj_idx, dir_idx in schedule:
        env.step(SETTLE_GAP)
        step += SETTLE_GAP
        if step >= EPISODE_STEPS - OBS_WINDOW - 2:
            break

        direction = IMPULSE_DIRECTIONS[dir_idx]
        body = env.bodies[obj_idx]
        shape = env.shapes[obj_idx]

        # isolate object so only it can interact with the measurement box
        original_filter = shape.filter
        shape.filter = pymunk.ShapeFilter(categories=0x2, mask=0x4)

        # spawn a tight box around the object so bounce timing is fixed
        # regardless of where in the arena the object spawned
        # (without this, objects near walls bounce at different timesteps)
        box_r = 15 + 2   # RADIUS + 2px gap
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
            seg.elasticity = 1.0
            seg.filter = pymunk.ShapeFilter(categories=0x4, mask=0x2)
            env.space.add(seg)
            temp_walls.append(seg)

        # zero velocity and gravity so the only signal is the impulse response
        body.velocity = (0.0, 0.0)
        body.angular_velocity = 0.0
        original_gravity = env.space.gravity
        env.space.gravity = (0, 0)

        env.apply_impulse(obj_idx, direction)

        direction_vec = np.array(direction)
        deltas = []
        for _ in range(OBS_WINDOW):
            env.step(1)
            step += 1
            v_now = np.array(body.velocity)
            _sigma = NOISE_SIGMA if noise_sigma is None else noise_sigma
            v_proj = float(np.dot(v_now, direction_vec)) + rng.normal(0.0, _sigma)
            deltas.append(v_proj)

        # cleanup
        for seg in temp_walls:
            env.space.remove(seg)
        shape.filter = original_filter
        env.space.gravity = original_gravity

        raw_per_dir[obj_idx][dir_idx] = deltas[:-1]   # drop last step

    # build raw signature matrix
    raw_sigs = np.zeros((N_OBJECTS, SIG_DIM), dtype=np.float64)
    expected_steps = OBS_WINDOW - 1
    for obj_idx in range(N_OBJECTS):
        sig_parts = []
        for dir_idx in range(N_DIRECTIONS):
            d = raw_per_dir[obj_idx][dir_idx]
            if d is not None and len(d) == expected_steps:
                sig_parts.extend([float(v) for v in d])
            else:
                sig_parts.extend([0.0] * expected_steps)
        raw_sigs[obj_idx] = sig_parts

    if update_norm:
        for obj_idx in range(N_OBJECTS):
            normaliser.update(raw_sigs[obj_idx])

    sigs = np.stack([normaliser.normalise(raw_sigs[obj_idx]) for obj_idx in range(N_OBJECTS)])
    labels = list(env.type_labels)
    return sigs, raw_sigs, labels


def signature_consistency_ratio(sigs: np.ndarray, labels: list) -> float:
    """Intra-type variance / inter-type variance. Lower = better separated."""
    sigs = np.array(sigs)
    labels = np.array(labels)
    unique_types = np.unique(labels)

    per_type_means = []
    intra_vars = []
    for t in unique_types:
        mask = labels == t
        sigs_t = sigs[mask]
        per_type_means.append(sigs_t.mean(axis=0))
        intra_vars.append(np.var(sigs_t, axis=0).mean())

    intra_var = np.mean(intra_vars)
    inter_var = np.var(np.array(per_type_means), axis=0).mean()

    if inter_var < 1e-10:
        return float("inf")
    return float(intra_var / inter_var)
