"""
config.py — All hyperparameters for the COG Minimal Validation Experiment.
Single source of truth. Import this everywhere.
"""

# ── Scene ──────────────────────────────────────────────────────────────────────
CANVAS_W = 400
CANVAS_H = 400
N_OBJECTS = 12         # total objects per episode (2 per type × 6 types)
N_TYPES = 6            # number of object types (A–F)
N_PER_TYPE = 2         # objects per type

# ── Physics ────────────────────────────────────────────────────────────────────
# Three mass groups (5, 2, 1); two restitution variants per group.
# A and B share mass=5; C and D share mass=2; E and F share mass=1.
# Within each pair only the restitution bounce at t=2,3 separates them.
MASS_A = 5.0
MASS_B = 5.0
MASS_C = 2.0
MASS_D = 2.0
MASS_E = 1.0
MASS_F = 1.0
RESTITUTION_A = 0.2
RESTITUTION_B = 0.8
RESTITUTION_C = 0.5
RESTITUTION_D = 0.9
RESTITUTION_E = 0.3
RESTITUTION_F = 0.7
FRICTION = 0.5         # fixed for all objects and surfaces
GRAVITY = (0, -900)    # pymunk uses y-up; pygame uses y-down — handled in env.py
RADIUS = 15            # pixel radius for circle collision shape
FPS = 60
EPISODE_STEPS = 600    # 48 interventions × (SETTLE_GAP=4 + OBS_WINDOW=6) = 480 < 600

# ── Intervention ───────────────────────────────────────────────────────────────
IMPULSE_MAGNITUDE = 200.0
IMPULSE_DIRECTIONS = [          # (dx, dy) unit vectors
    (1.0,  0.0),   # right
    (0.0,  1.0),   # up
    (-1.0, 0.0),   # left
    (0.0, -1.0),   # down
]
N_DIRECTIONS = len(IMPULSE_DIRECTIONS)
OBS_WINDOW = 6                 # steps recorded after each impulse
# Heavy types (mass=5, Dv=40 units/s) bounce at step 2; lighter types bounce at step 1.
# 5 post-impulse steps capture at least one bounce + up to one re-bounce for all types.
SIG_DIM = N_DIRECTIONS * (OBS_WINDOW - 1)        # 4 dirs × 5 steps × 1 projected axis = 20
INTERVENTIONS_PER_EPISODE = 10  # subsampled from 24 possible
INTERVENTION_EVERY = 20        # apply intervention every N steps

# ── Sensor noise ───────────────────────────────────────────────────────────────
# Gaussian σ added to each velocity reading (raw pymunk units).
# Applied before EMA normalisation.  Tune if B/C separation degrades.
NOISE_SIGMA = 0.05

# ── Normalisation ──────────────────────────────────────────────────────────────
EMA_DECAY = 0.99

# ── Clustering ─────────────────────────────────────────────────────────────────
N_CLUSTERS = N_TYPES           # Ward linkage, Euclidean distance

# ── Training / Evaluation ──────────────────────────────────────────────────────
TRAIN_EPISODES = 100
TEST_EPISODES = 50
CONSISTENCY_THRESHOLD = 0.20   # trigger window extension if exceeded
ACCURACY_THRESHOLD = 0.85      # pass criterion
ARI_THRESHOLD = 0.70           # pass criterion

# ── Visual Conditions ──────────────────────────────────────────────────────────
# Each condition: list of (fill_colour, ...) per type [A, B, C, D, E, F]
# 'train'     → solid circles, canonical colours
# 'colour'    → solid circles, shifted colours (adversarial)
# 'texture'   → same base colours, patterns applied
# 'shape'     → same base colours, polygons (physics stays circular)

CONDITIONS = ["train", "colour", "texture", "shape"]

COLOURS = {
    # red, blue, green, yellow, magenta, orange
    "train":   [(204, 34,   0), (0,   51,  204), (0,  170,  68),
                (204, 170,  0), (170,  0,  170), (255, 102,   0)],
    # cyan, purple, lime, pink, teal, brown  — maximally confuse frozen RGB centroids
    "colour":  [(0,  204, 204), (136,  0,  204), (68,  204,   0),
                (255,  0,  136), (0,  153, 136), (153,  68,   0)],
    # same as train, texture applied on top
    "texture": [(204, 34,   0), (0,   51,  204), (0,  170,  68),
                (204, 170,  0), (170,  0,  170), (255, 102,   0)],
    # same as train, shape changed
    "shape":   [(204, 34,   0), (0,   51,  204), (0,  170,  68),
                (204, 170,  0), (170,  0,  170), (255, 102,   0)],
}

# polygon vertex counts for 'shape' condition (None → circle)
SHAPE_SIDES = {
    "train":   [None, None, None, None, None, None],
    "colour":  [None, None, None, None, None, None],
    "texture": [None, None, None, None, None, None],
    "shape":   [5,    3,    6,    4,    7,    8],    # pentagon, triangle, hexagon, square, heptagon, octagon
}

# texture patterns for 'texture' condition
TEXTURES = {
    "train":   ["solid",   "solid",   "solid",   "solid",   "solid",   "solid"],
    "colour":  ["solid",   "solid",   "solid",   "solid",   "solid",   "solid"],
    "texture": ["striped", "dotted",  "chequered", "striped", "dotted", "chequered"],
    "shape":   ["solid",   "solid",   "solid",   "solid",   "solid",   "solid"],
}

# ── Paths ──────────────────────────────────────────────────────────────────────
SAVE_DIR = "outputs"
CENTROIDS_FILE = "outputs/ward_centroids.npy"
NORM_PARAMS_FILE = "outputs/norm_params.npy"
TRAIN_SIGS_FILE = "outputs/train_signatures.npy"
TRAIN_LABELS_FILE = "outputs/train_labels.npy"
RESULTS_FILE = "outputs/results.json"
