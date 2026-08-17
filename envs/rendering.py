"""Pygame renderer — strictly monochrome.

Rendering is not optional in this project: the whole payoff of artificial life
is watching it. The scent field is drawn as a greyscale heat-map so the thing
the worm actually senses is visible on screen, and each pellet gets a faint
ring at its ``scent_radius`` so the sensing footprint is legible next to the
tiny eat radius.

Everything is black and white, so brightness has to carry all the meaning. The
grey ramp is capped below pure white, which is reserved for the two things that
are not scent:

    solid white disc   a food pellet
    white outline ring the worm (dark inside, so it never merges with the field)

Imported lazily by the env, so pygame is only needed when you actually render.
"""

from __future__ import annotations

import numpy as np
import pygame

from .config import EnvConfig
from .food import FoodField
from .metabolism import Metabolism
from .worm import Worm

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
FIELD_LOW = np.array([0, 0, 0], dtype=np.float64)  # no scent
FIELD_HIGH = np.array([190, 190, 190], dtype=np.float64)  # at/above a pellet's peak
RING_GREY = (70, 70, 70)  # scent_radius outlines
BAR_GREY = (60, 60, 60)  # empty part of the energy bar


class PygameRenderer:
    """Draws one frame of the world; also owns the window in ``human`` mode."""

    def __init__(self, config: EnvConfig, render_mode: str):
        self.config = config
        self.render_mode = render_mode

        world = config.world
        self.scale = config.render.window_size / max(world.width, world.height)
        self.size_px = (round(world.width * self.scale), round(world.height * self.scale))

        pygame.init()
        pygame.font.init()
        self.font = pygame.font.Font(None, 22)
        self.surface = pygame.Surface(self.size_px)
        self.window: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        if render_mode == "human":
            pygame.display.init()
            pygame.display.set_caption("AI Worm — world v1")
            self.window = pygame.display.set_mode(self.size_px)
            self.clock = pygame.time.Clock()

        self._scent_points = self._build_scent_grid()

    def draw(
        self, worm: Worm, food: FoodField, metabolism: Metabolism, steps: int
    ) -> np.ndarray | None:
        self._draw_scent_field(food)
        self._draw_food(food)
        self._draw_worm(worm)
        self._draw_hud(food, metabolism, steps)

        if self.render_mode == "human":
            assert self.window is not None and self.clock is not None
            for event in pygame.event.get():  # keep the OS from thinking we hung
                if event.type == pygame.QUIT:
                    self.close()
                    return None
            self.window.blit(self.surface, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.config.render.fps)
            return None

        return np.transpose(np.array(pygame.surfarray.pixels3d(self.surface)), (1, 0, 2))

    def close(self) -> None:
        if self.window is not None:
            pygame.display.quit()
            self.window = None
        pygame.quit()

    # -- world -> screen ---------------------------------------------------

    def to_screen(self, position: np.ndarray) -> tuple[int, int]:
        """World coordinates (y up, origin bottom-left) to pixels (y down)."""
        x, y = np.asarray(position, dtype=np.float64)
        return (round(x * self.scale), round(self.size_px[1] - y * self.scale))

    def _build_scent_grid(self) -> np.ndarray:
        """Sample points for the heat-map, laid out as ``[ix, iy]`` for pygame."""
        grid = self.config.render.scent_grid
        xs = np.linspace(0.0, self.config.world.width, grid)
        ys = np.linspace(self.config.world.height, 0.0, grid)  # top row = high y
        mesh_x, mesh_y = np.meshgrid(xs, ys, indexing="ij")
        return np.stack([mesh_x, mesh_y], axis=-1)

    # -- layers ------------------------------------------------------------

    def _draw_scent_field(self, food: FoodField) -> None:
        if not self.config.render.show_scent_field:
            self.surface.fill(BLACK)
            return

        field = food.scent_at(self._scent_points)
        # Normalise against a single pellet's peak (not the frame's max) so
        # brightness means the same thing from frame to frame, and gamma-lift
        # the faint tail that the worm still has to be able to follow.
        intensity = np.clip(field / max(self.config.food.scent_peak, 1e-9), 0.0, 1.0) ** 0.6
        greys = FIELD_LOW + intensity[..., None] * (FIELD_HIGH - FIELD_LOW)

        small = pygame.surfarray.make_surface(greys.astype(np.uint8))
        pygame.transform.smoothscale(small, self.size_px, self.surface)

    def _draw_food(self, food: FoodField) -> None:
        eat_px = max(3, round(self.config.food.eat_radius * self.scale))
        ring_px = round(self.config.food.scent_radius * self.scale)
        for position in food.positions:
            center = self.to_screen(position)
            if self.config.render.show_scent_rings:
                pygame.draw.circle(self.surface, RING_GREY, center, ring_px, width=1)
            pygame.draw.circle(self.surface, WHITE, center, eat_px)

    def _draw_worm(self, worm: Worm) -> None:
        """A dark dot inside a white ring — legible on black *and* on bright scent."""
        center = self.to_screen(worm.position)
        radius_px = max(4, round(self.config.worm.radius * self.scale))
        pygame.draw.circle(self.surface, BLACK, center, radius_px)
        pygame.draw.circle(self.surface, WHITE, center, radius_px, width=2)

        # A short whisker showing which way the head is pointing.
        tip = worm.position + np.array([np.cos(worm.heading), np.sin(worm.heading)]) * (
            self.config.worm.radius * 3.0
        )
        pygame.draw.line(self.surface, WHITE, center, self.to_screen(tip), 2)

    def _draw_hud(self, food: FoodField, metabolism: Metabolism, steps: int) -> None:
        """Energy has to live here: with no colour, the worm cannot carry it."""
        bar_w, bar_h, margin = 180, 12, 12
        pygame.draw.rect(self.surface, BLACK, (margin, margin, bar_w, bar_h))
        pygame.draw.rect(
            self.surface, WHITE, (margin, margin, round(bar_w * metabolism.energy_fraction), bar_h)
        )
        pygame.draw.rect(self.surface, BAR_GREY, (margin, margin, bar_w, bar_h), width=1)

        lines = [
            f"step {steps}",
            f"energy {metabolism.energy:6.1f} / {self.config.metabolism.max_energy:.0f}",
            f"eaten {food.eaten_total}",
        ]
        for i, line in enumerate(lines):
            # Shadow first, so white text stays readable over a bright patch.
            text_y = margin + bar_h + 6 + i * 20
            self.surface.blit(self.font.render(line, True, BLACK), (margin + 1, text_y + 1))
            self.surface.blit(self.font.render(line, True, WHITE), (margin, text_y))
