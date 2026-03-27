"""
env.py — Pygame + pymunk physics scene.

Objects (6 types, 2 per type = 12 per episode):
  Type A (idx  0, 1): mass=5.0, restitution=0.2
  Type B (idx  2, 3): mass=5.0, restitution=0.8
  Type C (idx  4, 5): mass=2.0, restitution=0.5
  Type D (idx  6, 7): mass=2.0, restitution=0.9
  Type E (idx  8, 9): mass=1.0, restitution=0.3
  Type F (idx 10,11): mass=1.0, restitution=0.7

A/B share mass; C/D share mass; E/F share mass.
Within each pair only the post-bounce restitution signal separates them.

Visual conditions: train / colour / texture / shape — controlled by config.
Ground truth type labels are always available as env.type_labels.
"""

import math
import random
import pygame
import pymunk
import pymunk.pygame_util
import numpy as np

from config import (
    CANVAS_W, CANVAS_H, N_OBJECTS, N_PER_TYPE, N_TYPES,
    MASS_A, MASS_B, MASS_C, MASS_D, MASS_E, MASS_F,
    RESTITUTION_A, RESTITUTION_B, RESTITUTION_C,
    RESTITUTION_D, RESTITUTION_E, RESTITUTION_F,
    FRICTION, GRAVITY, RADIUS, FPS, EPISODE_STEPS,
    IMPULSE_MAGNITUDE,
    COLOURS, SHAPE_SIDES, TEXTURES,
)

# Physics parameters per type (index = type index 0–5 → A–F)
_MASSES  = [MASS_A, MASS_B, MASS_C, MASS_D, MASS_E, MASS_F]
_RESTITS = [RESTITUTION_A, RESTITUTION_B, RESTITUTION_C,
            RESTITUTION_D, RESTITUTION_E, RESTITUTION_F]
TYPE_NAMES = ["A", "B", "C", "D", "E", "F"]


class PhysicsEnv:
    """
    Thin wrapper around a pymunk Space with a pygame display.

    Parameters
    ----------
    condition : str
        One of 'train', 'colour', 'texture', 'shape'.
    headless : bool
        If True, skip pygame display initialisation (faster for batch runs).
    seed : int | None
        RNG seed for reproducible initial placement.
    """

    def __init__(self, condition: str = "train", headless: bool = True,
                 seed=None, jitter: float = 0.0):
        assert condition in ("train", "colour", "texture", "shape"), \
            f"Unknown condition: {condition}"
        self.condition = condition
        self.headless = headless
        self.jitter = jitter          # multiplicative noise on mass and restitution
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.step_count = 0

        # pymunk coordinate system: y-up.  pygame: y-down.
        # We store pymunk positions; convert only for rendering.
        self._init_pygame()
        self._init_space()
        self._spawn_objects()

    # ── Init helpers ──────────────────────────────────────────────────────────

    def _init_pygame(self):
        if not pygame.get_init():
            pygame.init()
        if self.headless:
            import os
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
            self.screen = pygame.display.set_mode((CANVAS_W, CANVAS_H), flags=pygame.NOFRAME)
        else:
            self.screen = pygame.display.set_mode((CANVAS_W, CANVAS_H))
        self.clock = pygame.time.Clock()

    def _init_space(self):
        self.space = pymunk.Space()
        self.space.gravity = GRAVITY        # y-up: negative = downward in pymunk
        self.space.damping = 1.0            # no air drag

        # Bounding walls (bottom, top, left, right) — static segments
        walls = [
            [(0, 0), (CANVAS_W, 0)],            # bottom
            [(0, CANVAS_H), (CANVAS_W, CANVAS_H)],  # top
            [(0, 0), (0, CANVAS_H)],            # left
            [(CANVAS_W, 0), (CANVAS_W, CANVAS_H)],  # right
        ]
        for a, b in walls:
            seg = pymunk.Segment(self.space.static_body, a, b, 1)
            seg.friction = FRICTION
            seg.elasticity = 0.5
            seg.filter = pymunk.ShapeFilter(categories=0x1, mask=0x1)
            self.space.add(seg)
            
        # Add grid of static "pegs" to induce collisions for B vs C separation
        self.pegs = []
        for px in range(50, CANVAS_W, 100):
            for py in range(50, CANVAS_H, 100):
                peg = pymunk.Circle(self.space.static_body, 10, offset=(px, py))
                peg.friction = FRICTION
                peg.elasticity = 0.5 # mid-way restitution
                peg.filter = pymunk.ShapeFilter(categories=0x1, mask=0x1)
                self.space.add(peg)
                self.pegs.append((px, py))

    def _spawn_objects(self):
        """Create N_OBJECTS bodies; 2 per type, spread over canvas."""
        self.bodies = []      # pymunk Body
        self.shapes = []      # pymunk Shape
        self.type_labels = [] # int type index per object

        colours = COLOURS[self.condition]
        sides = SHAPE_SIDES[self.condition]
        textures = TEXTURES[self.condition]

        positions = self._sample_positions()

        obj_idx = 0
        for type_idx in range(N_TYPES):
            for _ in range(N_PER_TYPE):
                mass = _MASSES[type_idx]
                restitution = _RESTITS[type_idx]
                if self.jitter > 0.0:
                    mass = mass * self.rng.uniform(1 - self.jitter, 1 + self.jitter)
                    restitution = min(0.99, restitution * self.rng.uniform(
                        1 - self.jitter, 1 + self.jitter))
                pos = positions[obj_idx]

                # Create body
                moment = pymunk.moment_for_circle(mass, 0, RADIUS)
                body = pymunk.Body(mass, moment)
                body.position = pos
                # Small random initial velocity
                body.velocity = (
                    self.rng.uniform(-50, 50),
                    self.rng.uniform(-50, 50),
                )

                # Collision shape: always a circle physically to isolate visual-only domain shift
                shape = pymunk.Circle(body, RADIUS)
                
                # We store n_sides on the body purely for the renderer
                n_sides = sides[type_idx]
                body.n_sides = n_sides

                shape.mass = mass
                shape.friction = FRICTION
                shape.elasticity = restitution
                shape.filter = pymunk.ShapeFilter(categories=0x1, mask=0x1)

                self.space.add(body, shape)
                self.bodies.append(body)
                self.shapes.append(shape)
                self.type_labels.append(type_idx)

                obj_idx += 1

        # Store rendering metadata
        self._colours = [COLOURS[self.condition][t] for t in self.type_labels]
        self._textures = [TEXTURES[self.condition][t] for t in self.type_labels]
        self._sides = [SHAPE_SIDES[self.condition][t] for t in self.type_labels]

    def _sample_positions(self):
        """Place N_OBJECTS in non-overlapping positions within canvas.

        Uses module-level np.random (not seeded per-episode) so spawn
        positions vary independently across episodes and runs.
        """
        margin = RADIUS * 3
        positions = []
        attempts = 0
        while len(positions) < N_OBJECTS and attempts < 10000:
            attempts += 1
            x = np.random.uniform(margin, CANVAS_W - margin)
            y = np.random.uniform(margin, CANVAS_H - margin)
            # Check min distance from existing positions
            if all(math.hypot(x - px, y - py) > RADIUS * 4
                   for px, py in positions):
                positions.append((x, y))
        if len(positions) < N_OBJECTS:
            # Fallback: grid placement (4 cols × 3 rows for 12 objects)
            positions = []
            cols = 4
            rows = (N_OBJECTS + cols - 1) // cols
            for i in range(N_OBJECTS):
                r, c = divmod(i, cols)
                x = margin + c * (CANVAS_W - 2 * margin) / max(cols - 1, 1)
                y = margin + r * (CANVAS_H - 2 * margin) / max(rows - 1, 1)
                positions.append((x, y))
        return positions

    @staticmethod
    def _polygon_verts(n: int, radius: float):
        """Regular polygon vertices for pymunk Poly."""
        verts = []
        for i in range(n):
            angle = 2 * math.pi * i / n - math.pi / 2
            verts.append((radius * math.cos(angle), radius * math.sin(angle)))
        return verts

    # ── Core API ──────────────────────────────────────────────────────────────

    def get_state(self) -> np.ndarray:
        """
        Returns array of shape (N_OBJECTS, 4): [x, y, vx, vy] per object.
        Coordinates are in pymunk space (y-up).
        """
        state = []
        for body in self.bodies:
            state.append([body.position.x, body.position.y,
                          body.velocity.x, body.velocity.y])
        return np.array(state, dtype=np.float32)

    def apply_impulse(self, obj_idx: int, direction: tuple):
        """Apply a single instantaneous impulse to one object."""
        dx, dy = direction
        fx = dx * IMPULSE_MAGNITUDE
        fy = dy * IMPULSE_MAGNITUDE
        # Apply in world coordinates so body rotation doesn't scramble the vector
        self.bodies[obj_idx].apply_impulse_at_world_point((fx, fy), self.bodies[obj_idx].position)

    def step(self, n: int = 1):
        """Advance physics simulation by n steps."""
        dt = 1.0 / FPS
        for _ in range(n):
            self.space.step(dt)
            self.step_count += 1
            # Pump events to avoid pygame queue overflow
            if not self.headless:
                pygame.event.pump()

    def reset(self, seed=None):
        """Reset environment to a new episode."""
        # Remove all dynamic bodies+shapes
        for body in self.bodies:
            for shape in body.shapes:
                self.space.remove(shape)
            self.space.remove(body)
        self.bodies.clear()
        self.shapes.clear()
        self.type_labels.clear()
        self.step_count = 0
        if seed is not None:
            self.rng = random.Random(seed)
            self.np_rng = np.random.default_rng(seed)
        self._spawn_objects()

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render(self) -> pygame.Surface:
        """Draw current scene and return the surface."""
        self.screen.fill((50, 50, 50))   # fixed grey background

        for i, body in enumerate(self.bodies):
            # pymunk y-up → pygame y-down
            px = int(body.position.x)
            py = CANVAS_H - int(body.position.y)
            colour = self._colours[i]
            texture = self._textures[i]
            n_sides = self._sides[i]

            if n_sides is None:
                self._draw_circle(px, py, colour, texture)
            else:
                self._draw_polygon(body, px, py, n_sides, colour)

        pygame.display.flip()
        return self.screen

    def _draw_circle(self, cx, cy, colour, texture):
        surf = self.screen
        r = RADIUS
        if texture == "solid":
            pygame.draw.circle(surf, colour, (cx, cy), r)
        elif texture == "striped":
            pygame.draw.circle(surf, colour, (cx, cy), r)
            for offset in range(-r, r, 6):
                y1 = cy + offset
                half = int(math.sqrt(max(0, r**2 - offset**2)))
                pygame.draw.line(surf, (255, 255, 255), (cx - half, y1), (cx + half, y1), 1)
        elif texture == "dotted":
            pygame.draw.circle(surf, colour, (cx, cy), r)
            for dx in range(-r + 4, r, 7):
                for dy in range(-r + 4, r, 7):
                    if dx**2 + dy**2 < (r - 3)**2:
                        pygame.draw.circle(surf, (255, 255, 255),
                                           (cx + dx, cy + dy), 2)
        elif texture == "chequered":
            pygame.draw.circle(surf, colour, (cx, cy), r)
            for dx in range(-r, r, 6):
                for dy in range(-r, r, 6):
                    if dx**2 + dy**2 < r**2:
                        if (dx // 6 + dy // 6) % 2 == 0:
                            pygame.draw.rect(surf, (255, 255, 255),
                                             (cx + dx, cy + dy, 5, 5))

    def _draw_polygon(self, body, cx, cy, n_sides, colour):
        verts = []
        for i in range(n_sides):
            angle = 2 * math.pi * i / n_sides - math.pi / 2 + body.angle
            vx = cx + int(RADIUS * math.cos(angle))
            vy = cy - int(RADIUS * math.sin(angle))   # flip y
            verts.append((vx, vy))
        pygame.draw.polygon(self.screen, colour, verts)

    def get_rgb_means(self) -> np.ndarray:
        """
        Extract mean RGB value inside each object's bounding circle.
        Returns array of shape (N_OBJECTS, 3).
        Used by the RGB baseline.
        """
        self.render()
        pixels = pygame.surfarray.array3d(self.screen)   # (W, H, 3)
        means = []
        for i, body in enumerate(self.bodies):
            cx = int(body.position.x)
            cy = CANVAS_H - int(body.position.y)
            r = RADIUS
            x0 = max(0, cx - r)
            x1 = min(CANVAS_W, cx + r + 1)
            y0 = max(0, cy - r)
            y1 = min(CANVAS_H, cy + r + 1)
            patch = pixels[x0:x1, y0:y1]   # surfarray is (x, y, c)
            means.append(patch.reshape(-1, 3).mean(axis=0))
        return np.array(means, dtype=np.float32)

    def close(self):
        """Clean up pygame resources."""
        pygame.quit()
