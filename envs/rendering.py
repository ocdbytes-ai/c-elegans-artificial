"""Strictly monochrome pygame renderer.

Everything is black and white, so brightness carries all the meaning. Pellets
are solid white discs and the worm is a white body outlined in black, which
keeps it legible whether the background is empty or a bright patch of scent.

The worm's body is decoration, not simulation. The physics is a single point
(:mod:`envs.worm`); the renderer keeps a trail of where that point has been and
lays an undulating body along it. The wave is a function of arc length back
from the head and advances exactly with distance travelled, so it stays locked
to the path: the worm slides along a fixed track rather than swimming through
one, which is how undulatory locomotion actually looks. It follows that the
body is still when the worm is still, and that turning tightly bends the body
rather than sweeping it through space the worm never occupied.

Scent, when enabled, is drawn as dotted iso-concentration rings, each marking a
radius where the concentration halves. The rings are per pellet, which is a
deliberate simplification: the true field is the *sum* over pellets, so where
pellets overlap the real contours are not circles and their merged summit is
not drawn at all. See ``notes/training.md``.

Imported lazily by the environment, so pygame is only needed to render.
"""

from __future__ import annotations

import numpy as np
import pygame

from . import geometry
from .config import EnvConfig
from .food import FoodField
from .metabolism import Metabolism
from .scent import make_scent_profile
from .worm import Worm

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
CONTOUR_NEAR = 225  # grey of the innermost (strongest) scent ring
CONTOUR_FAR = 55  # grey of the outermost (faintest) scent ring
BAR_GREY = (60, 60, 60)  # empty part of the energy bar


class PygameRenderer:
    """Draws frames of the world, and owns the window under ``human`` mode.

    Attributes:
        config: The environment config, for world size and render options.
        render_mode: ``"human"`` or ``"rgb_array"``.
        scale: Pixels per world unit.
        size_px: Surface size in pixels.
        font: HUD font.
        surface: Off-screen surface every frame is drawn onto.
        window: The display surface under ``"human"``, otherwise None.
        clock: Frame-rate limiter under ``"human"``, otherwise None.
    """

    def __init__(self, config: EnvConfig, render_mode: str):
        """Creates surfaces and precomputes the scent contours.

        Args:
            config: The environment config.
            render_mode: ``"human"`` to open a window, ``"rgb_array"`` for
                off-screen frames.
        """
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

        self._contours = self._build_contours()

        # Body state. The trail is kept in unwrapped coordinates so a worm
        # crossing a torus seam keeps one continuous spine; wrapping happens
        # per point at draw time.
        self._trail: list[np.ndarray] = []
        self._last_head: np.ndarray | None = None
        self._phase = 0.0

    def draw(
        self, worm: Worm, food: FoodField, metabolism: Metabolism, steps: int
    ) -> np.ndarray | None:
        """Renders one frame.

        Args:
            worm: The body to draw.
            food: Pellets and the scent field.
            metabolism: Energy state, for the HUD bar.
            steps: Step count, for the HUD.

        Returns:
            An RGB array under ``"rgb_array"``, otherwise None.
        """
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
        """Shuts down pygame and any window it owns."""
        if self.window is not None:
            pygame.display.quit()
            self.window = None
        pygame.quit()

    def to_screen(self, position: np.ndarray) -> tuple[int, int]:
        """Converts a world position to pixel coordinates.

        Args:
            position: World coordinates, y up from the bottom-left.

        Returns:
            Pixel coordinates, y down from the top-left.
        """
        x, y = np.asarray(position, dtype=np.float64)
        return (round(x * self.scale), round(self.size_px[1] - y * self.scale))

    def _to_screen_batch(self, points: np.ndarray) -> np.ndarray:
        """Vectorised :meth:`to_screen`.

        Args:
            points: World positions of shape ``(N, 2)``.

        Returns:
            Integer pixel coordinates of shape ``(N, 2)``.
        """
        screen = np.empty_like(points)
        screen[:, 0] = points[:, 0] * self.scale
        screen[:, 1] = self.size_px[1] - points[:, 1] * self.scale
        return np.rint(screen).astype(np.int32)

    def _build_contours(self) -> list[tuple[np.ndarray, tuple[int, int, int], int]]:
        """Precomputes the dot offsets for each iso-concentration ring.

        Levels halve outwards (peak/2, peak/4, and so on) and the matching
        radius is found by inverting the configured profile numerically, so the
        rings stay correct when the profile is switched. Offsets are in world
        units and identical for every pellet, so drawing a frame only adds the
        pellet position.

        Returns:
            One ``(offsets, grey, dot_radius_px)`` triple per drawable ring.
            Rings whose level the profile never reaches, or that fall below two
            pixels, are omitted.
        """
        render, food = self.config.render, self.config.food
        profile = make_scent_profile(food)

        # The profile is monotonically decreasing, so sampling it once and
        # interpolating backwards inverts it for any profile shape.
        probe = np.linspace(0.0, food.scent_radius * 2.0, 4096)
        values = profile(probe)

        contours = []
        for index in range(render.scent_contours):
            level = food.scent_peak * 0.5 ** (index + 1)
            if level >= values[0] or level <= values[-1]:
                continue  # this level is not reachable on this profile
            radius = float(np.interp(level, values[::-1], probe[::-1]))
            radius_px = radius * self.scale
            if radius_px < 2:
                continue

            # Constant arc-length spacing, so outer rings do not look sparse.
            count = max(8, round(2 * np.pi * radius_px / render.contour_dot_spacing))
            angles = np.linspace(0.0, 2 * np.pi, count, endpoint=False)
            offsets = np.stack([np.cos(angles), np.sin(angles)], axis=-1) * radius

            fade = index / max(render.scent_contours - 1, 1)
            grey = round(CONTOUR_NEAR + (CONTOUR_FAR - CONTOUR_NEAR) * fade)
            dot_px = 2 if fade < 0.5 else 1
            contours.append((offsets, (grey, grey, grey), dot_px))
        return contours

    def _draw_scent_field(self, food: FoodField) -> None:
        """Clears the surface and draws each pellet's contour rings.

        Args:
            food: The pellet field.
        """
        self.surface.fill(BLACK)
        render = self.config.render
        if not (render.show_scent_field or render.debug):
            return

        world = self.config.world
        for position in food.positions:
            for offsets, grey, dot_px in self._contours:
                points = position + offsets
                if world.boundary == "wrap":
                    # Wrap each dot individually, so a ring running off one edge
                    # reappears on the other, as the real field does.
                    points = geometry.apply_boundary(points, world.size, world.boundary)
                else:
                    inside = np.all((points >= 0) & (points <= np.array(world.size)), axis=-1)
                    points = points[inside]
                for x, y in self._to_screen_batch(points):
                    pygame.draw.circle(self.surface, grey, (int(x), int(y)), dot_px)

    def _draw_food(self, food: FoodField) -> None:
        """Draws each pellet as a solid disc.

        Under ``debug`` the disc is the true ``food.eat_radius``, so what is on
        screen is the boundary that actually eats. Otherwise it is scaled down
        by ``render.food_dot_scale`` to a marker, which keeps the frame legible
        at the cost of no longer showing where contact happens.

        Args:
            food: The pellet field.
        """
        render, food_config = self.config.render, self.config.food
        scale = 1.0 if render.debug else render.food_dot_scale
        dot_px = max(2, round(food_config.eat_radius * scale * self.scale))
        for position in food.positions:
            pygame.draw.circle(self.surface, WHITE, self.to_screen(position), dot_px)

    def _track_head(self, worm: Worm) -> None:
        """Extends the trail by this frame's motion and advances the wave phase.

        A reset teleports the worm, which would otherwise leave a body stretched
        across the arena, so any hop larger than the episode could physically
        take restarts the trail. Under ``wrap`` the minimum-image displacement
        makes a seam crossing read as the short hop it is, so crossings extend
        the trail rather than restarting it.

        Args:
            worm: The body, for its position and this episode's speed ceiling.
        """
        world = self.config.world
        head = np.asarray(worm.position, dtype=np.float64)

        if self._last_head is None:
            self._trail = [head.copy()]
            self._last_head = head.copy()
            return

        delta = geometry.displacement(self._last_head, head, world.size, world.boundary)
        self._last_head = head.copy()

        travelled = float(np.linalg.norm(delta))
        if travelled > 3.0 * worm.max_speed * world.dt:
            self._trail = [head.copy()]
            self._phase = 0.0
            return

        self._trail.append(self._trail[-1] + delta)
        self._phase += 2.0 * np.pi * travelled / self.config.render.worm_wavelength

        # Keep only enough history to cover the body, plus a little slack for
        # the resampling to interpolate against.
        budget = self.config.render.worm_body_length * 1.5
        kept, run = 1, 0.0
        for index in range(len(self._trail) - 1, 0, -1):
            run += float(np.linalg.norm(self._trail[index] - self._trail[index - 1]))
            kept += 1
            if run >= budget:
                break
        if kept < len(self._trail):
            self._trail = self._trail[-kept:]

    def _spine(self, worm: Worm, samples: int) -> np.ndarray:
        """Resamples the trail to evenly spaced points running head to tail.

        Even spacing is what lets the undulation have a constant wavelength: the
        raw trail is spaced by per-step travel, which varies with throttle. A
        trail shorter than the body — the first frames of an episode, or a worm
        that has barely moved — is extended straight backwards so the worm is
        never drawn as a stub.

        Args:
            worm: The body, for the heading used to extend a short trail.
            samples: Number of points to produce.

        Returns:
            Positions of shape ``(samples, 2)``, head first, in unwrapped world
            coordinates.
        """
        length = self.config.render.worm_body_length
        points = np.array(self._trail[::-1], dtype=np.float64)

        if len(points) < 2:
            backward = -geometry.heading_vector(worm.heading)
            points = np.stack([points[0], points[0] + backward * length])

        segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(segments)])

        if arc[-1] < length:
            # Continue in the direction the tail was already heading. Any
            # non-degenerate segment will do; the last one is the freshest.
            usable = segments > 1e-12
            direction = (
                (points[-1] - points[-2]) / segments[-1]
                if usable.any() and segments[-1] > 1e-12
                else -geometry.heading_vector(worm.heading)
            )
            points = np.vstack([points, points[-1] + direction * (length - arc[-1] + 1e-6)])
            arc = np.concatenate([arc, [length + 1e-6]])

        targets = np.linspace(0.0, length, samples)
        return np.stack(
            [np.interp(targets, arc, points[:, 0]), np.interp(targets, arc, points[:, 1])],
            axis=-1,
        )

    def _draw_worm(self, worm: Worm) -> None:
        """Draws the worm as an undulating body tapering from head to tail.

        Drawn as discs rather than a polyline, which keeps it correct under
        ``wrap``: each disc wraps independently, so a worm straddling a seam
        appears at both edges instead of streaking across the frame. Black
        discs go down first as an outline, since interleaving the two passes
        would let a later outline erase body already drawn.

        Args:
            worm: The body to draw.
        """
        render = self.config.render
        self._track_head(worm)

        # Spaced well under a pixel-radius apart, so the discs still overlap
        # where the tail tapers to its thinnest and the body stays continuous.
        samples = max(8, round(render.worm_body_length * self.scale / 1.2))
        spine = self._spine(worm, samples)
        arc = np.linspace(0.0, render.worm_body_length, samples)
        fraction = arc / render.worm_body_length

        # Tangents along the spine, and the left-hand normal the wave rides on.
        tangents = np.gradient(spine, axis=0)
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        tangents = np.divide(tangents, norms, out=np.zeros_like(tangents), where=norms > 1e-12)
        normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=-1)

        # The wave travels head to tail, and grows towards the tail: the head
        # end stays comparatively steady and does the steering.
        swing = render.worm_wave_amplitude * (0.25 + 0.75 * fraction)
        offset = swing * np.sin(2.0 * np.pi * arc / render.worm_wavelength - self._phase)
        body = spine + normals * offset[:, None]

        # A rounded nose over the first tenth, then a long taper to a point.
        nose = np.sqrt(np.clip(1.0 - (1.0 - np.clip(fraction / 0.1, 0.0, 1.0)) ** 2, 0.0, 1.0))
        tail = np.clip((1.0 - fraction) / 0.55, 0.0, 1.0) ** 0.7
        half_width = self.config.worm.radius * np.minimum(nose, tail)
        radii = np.maximum(1, np.rint(half_width * self.scale).astype(int))

        # The outline scales with the body, or a thin worm would be mostly
        # outline. It only earns its keep against the debug contours; over bare
        # black it costs nothing.
        outline_px = max(1, min(2, round(self.config.worm.radius * self.scale * 0.25)))

        world = self.config.world
        drawn = geometry.apply_boundary(body, world.size, world.boundary)
        screen = self._to_screen_batch(drawn)
        for colour, pad in ((BLACK, outline_px), (WHITE, 0)):
            for (x, y), radius in zip(screen, radii, strict=True):
                pygame.draw.circle(self.surface, colour, (int(x), int(y)), int(radius) + pad)

        # A dark eye just behind the nose. Taper alone leaves the two ends
        # ambiguous at a glance, and a white whisker would read as a tail tip;
        # black on the widest part of the body cannot be mistaken for either.
        eye = body[min(round(0.18 * samples), samples - 1)]
        eye_px = max(1, round(self.config.worm.radius * 0.32 * self.scale))
        pygame.draw.circle(self.surface, BLACK, self.to_screen(eye), eye_px)

        if render.debug:
            # The body is decoration; this ring is the point the physics
            # integrates, which is what senses scent and what eats.
            pygame.draw.circle(self.surface, WHITE, self.to_screen(worm.position), 5, width=1)

    def _draw_hud(self, food: FoodField, metabolism: Metabolism, steps: int) -> None:
        """Draws the energy bar and text overlay.

        Energy lives here because with no colour available the worm itself
        cannot carry it.

        Args:
            food: The pellet field, for the eaten count.
            metabolism: Energy state, for the bar.
            steps: Step count.
        """
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
