# config.py
# all hyperparameters in one place, import from here everywhere else

# scene
CANVAS_W = 400
CANVAS_H = 400
N_OBJECTS = 12       # 2 per type x 6 types
N_TYPES = 6
N_PER_TYPE = 2

# physics
# three mass groups (5, 2, 1); two restitution variants each
# A/B share mass=5, C/D share mass=2, E/F share mass=1
# within each pair only the bounce timing/height separates them
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
FRICTION = 0.5
GRAVITY = (0, -900)   # pymunk y-up; we zero this during measurements anyway
RADIUS = 15
FPS = 60
EPISODE_STEPS = 600

# intervention
IMPULSE_MAGNITUDE = 200.0
IMPULSE_DIRECTIONS = [
    (1.0,  0.0),   # right
    (0.0,  1.0),   # up
    (-1.0, 0.0),   # left
    (0.0, -1.0),   # down
]
N_DIRECTIONS = len(IMPULSE_DIRECTIONS)
OBS_WINDOW = 6           # steps recorded after impulse; last one gets dropped
# heavy pair (mass=5) bounces at step 2, lighter types at step 1
# 5 usable steps captures at least the first bounce for everyone
SIG_DIM = N_DIRECTIONS * (OBS_WINDOW - 1)   # 4 x 5 = 20
INTERVENTIONS_PER_EPISODE = 10
INTERVENTION_EVERY = 20

# sensor noise -- gaussian added before normalisation
# signal range is 8-140 units/s so SNR is fine up to sigma ~10
NOISE_SIGMA = 0.05

# normalisation
EMA_DECAY = 0.99

# clustering
N_CLUSTERS = N_TYPES   # ward linkage

# training / eval
TRAIN_EPISODES = 100
TEST_EPISODES = 50
CONSISTENCY_THRESHOLD = 0.20
ACCURACY_THRESHOLD = 0.85
ARI_THRESHOLD = 0.70

# visual conditions
# train: canonical colours, solid circles
# colour: adversarial colour swap to break RGB centroids
# texture: same colours + surface patterns
# shape: polygons rendered but physics stays circular

CONDITIONS = ["train", "colour", "texture", "shape"]

COLOURS = {
    "train":   [(204, 34,   0), (0,   51,  204), (0,  170,  68),
                (204, 170,  0), (170,  0,  170), (255, 102,   0)],
    # picked to maximally confuse frozen train centroids
    "colour":  [(0,  204, 204), (136,  0,  204), (68,  204,   0),
                (255,  0,  136), (0,  153, 136), (153,  68,   0)],
    "texture": [(204, 34,   0), (0,   51,  204), (0,  170,  68),
                (204, 170,  0), (170,  0,  170), (255, 102,   0)],
    "shape":   [(204, 34,   0), (0,   51,  204), (0,  170,  68),
                (204, 170,  0), (170,  0,  170), (255, 102,   0)],
}

# None = circle
SHAPE_SIDES = {
    "train":   [None, None, None, None, None, None],
    "colour":  [None, None, None, None, None, None],
    "texture": [None, None, None, None, None, None],
    "shape":   [5, 3, 6, 4, 7, 8],   # pentagon, triangle, hexagon, square, heptagon, octagon
}

TEXTURES = {
    "train":   ["solid",    "solid",     "solid",       "solid",   "solid",   "solid"],
    "colour":  ["solid",    "solid",     "solid",       "solid",   "solid",   "solid"],
    "texture": ["striped",  "dotted",    "chequered",   "striped", "dotted",  "chequered"],
    "shape":   ["solid",    "solid",     "solid",       "solid",   "solid",   "solid"],
}

# paths
SAVE_DIR = "outputs"
CENTROIDS_FILE = "outputs/ward_centroids.npy"
NORM_PARAMS_FILE = "outputs/norm_params.npy"
TRAIN_SIGS_FILE = "outputs/train_signatures.npy"
TRAIN_LABELS_FILE = "outputs/train_labels.npy"
RESULTS_FILE = "outputs/results.json"
