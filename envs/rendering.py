"""Dot-matrix renderer for the pygame world.

The world is drawn onto a low-resolution grid where every display pixel is
either on or off, then scaled up with a gap between pixels, like a monochrome
LCD. There is no anti-aliasing, glow or shading: a frame holds exactly three
colours, the lit pixel, the unlit pixel and the gap. Grey is faked by dithering,
which is how hazards are drawn.

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
not drawn at all.

Imported lazily by the environment, so pygame is only needed to render.
"""

from __future__ import annotations

import numpy as np
import pygame

from . import geometry
from .config import EnvConfig
from .metabolism import Metabolism
from .scent import make_scent_profile
from .sources import FoodField, ScentField
from .worm import Worm

BACKGROUND = (2, 8, 11)
PIXEL_OFF = (5, 22, 27)
PIXEL_ON = (40, 220, 255)
PIXEL = 5  # frame pixels per display pixel, including a 1px gap
WINDOW_PIXEL = 3  # the same for the human window, which keeps it compact
HUD_ROWS = 10

# 3x5 bitmap glyphs, read left to right then top to bottom.
FONT = {
    "0": "111101101101111", "1": "010110010010111", "2": "111001111100111",
    "3": "111001111001111", "4": "101101111001001", "5": "111100111001111",
    "6": "111100111101111", "7": "111001001001001", "8": "111101111101111",
    "9": "111101111001111", "E": "111100110100111", "G": "011100101101011",
    "N": "110101101101101", "R": "110101110101101", "T": "111010010010010",
    "Y": "101101010010010",
}


class PygameRenderer:
    """Draws frames of the world, and owns the window under ``human`` mode.

    Attributes:
        config: The environment config, for world size and render options.
        render_mode: ``"human"`` or ``"rgb_array"``.
        scale: Display pixels per world unit.
        size_px: Surface size in screen pixels: ``render.window_size`` for
            ``"rgb_array"`` frames, smaller for the ``"human"`` window.
        surface: Off-screen surface each finished frame is scaled onto.
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
        window = config.render.window_size
        side = window // PIXEL
        # One pixel short on each axis, so a position on the far wall rounds
        # onto the last pixel rather than just past it.
        self.scale = min((side - 1) / world.width, (side - HUD_ROWS - 1) / world.height)
        self.play_origin = ((side - world.width * self.scale) / 2.0, HUD_ROWS)
        # Both modes share one grid of display pixels; the window only draws
        # each of them smaller.
        human = render_mode == "human"
        pixel = WINDOW_PIXEL if human else PIXEL
        self.size_px = (side * pixel, side * pixel) if human else (window, window)

        pygame.init()
        self.surface = pygame.Surface(self.size_px)
        self.surface.fill(BACKGROUND)
        self._screen = pygame.Surface((side, side))
        self._gaps = self._build_gaps(pixel)
        self.window: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self.closed = False
        if render_mode == "human":
            pygame.display.init()
            pygame.display.set_caption("AI Worm")
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
        self,
        worm: Worm,
        food: FoodField,
        toxin: ScentField,
        metabolism: Metabolism,
        steps: int,
    ) -> np.ndarray | None:
        """Renders one frame.

        The counters live in the terminal now (:mod:`ppo.tui`), so nothing here
        reads energy or step state and the window stays purely the world.

        Args:
            worm: The body to draw.
            food: Pellets and the scent field.
            toxin: Hazards, drawn as dithered discs.
            metabolism: Current energy for the HUD.
            steps: Current episode step for the HUD.

        Returns:
            An RGB array under ``"rgb_array"``, otherwise None, including once
            the window has been closed.
        """
        if self.closed:
            return None

        self._screen.fill(PIXEL_OFF)
        self._draw_scent_field(food)
        self._draw_food(food)
        self._draw_toxins(toxin)
        self._draw_worm(worm)
        self._draw_hud(metabolism, steps)
        # transform.scale is nearest-neighbour, so no in-between colours appear.
        self.surface.blit(pygame.transform.scale(self._screen, self._gaps.get_size()), (0, 0))
        self.surface.blit(self._gaps, (0, 0))

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
        """Shuts down pygame and any window it owns.

        Idempotent: a window closed by the user is closed again by
        ``env.close()``, and tearing pygame down twice raises.
        """
        if self.closed:
            return
        self.closed = True
        if self.window is not None:
            pygame.display.quit()
            self.window = None
        pygame.quit()

    def _build_gaps(self, pixel: int) -> pygame.Surface:
        """Builds the transparent overlay that draws the gap between pixels.

        Args:
            pixel: Screen pixels per display pixel, including the gap.

        Returns:
            A surface covering the scaled screen, opaque only along the last
            row and column of every display pixel.
        """
        side = self._screen.get_width() * pixel
        gaps = pygame.Surface((side, side), pygame.SRCALPHA)
        for edge in range(pixel - 1, side, pixel):
            pygame.draw.line(gaps, BACKGROUND, (edge, 0), (edge, side))
            pygame.draw.line(gaps, BACKGROUND, (0, edge), (side, edge))
        return gaps

    def _draw_text(self, text: str, left: int, top: int) -> None:
        """Draws text in the 3x5 :data:`FONT`, one blank column between glyphs.

        Args:
            text: Characters to draw; any not in the font draw as a blank.
            left: Display-pixel column of the first glyph.
            top: Display-pixel row of the glyphs' top edge.
        """
        for offset, char in enumerate(text):
            x = left + 4 * offset
            for index, bit in enumerate(FONT.get(char, "")):
                if bit == "1":
                    self._screen.set_at((x + index % 3, top + index // 3), PIXEL_ON)

    def _draw_hud(self, metabolism: Metabolism, steps: int) -> None:
        """Draws the energy bar, elapsed-step counter and playfield divider."""
        margin, side = 2, self._screen.get_width()
        self._screen.fill(PIXEL_OFF, (0, 0, side, HUD_ROWS))  # erase world drawn past the top
        self._draw_text("ENERGY", margin, 2)

        bar = pygame.Rect(margin + 4 * 6 + 2, 1, round(side * 0.23), 7)
        pygame.draw.rect(self._screen, PIXEL_ON, bar, width=1)
        fill = round((bar.width - 4) * metabolism.energy_fraction)
        if fill:
            pygame.draw.rect(self._screen, PIXEL_ON, (bar.x + 2, bar.y + 2, fill, bar.height - 4))

        timer = f"T {steps:05d}"
        self._draw_text(timer, side - margin - (4 * len(timer) - 1), 2)

        for x in range(margin, side - margin, 2):
            self._screen.set_at((x, HUD_ROWS - 2), PIXEL_ON)

    def to_screen(self, position: np.ndarray) -> tuple[int, int]:
        """Converts a world position to display-pixel coordinates.

        Args:
            position: World coordinates, y up from the bottom-left.

        Returns:
            Display-pixel coordinates, y down from the top-left.
        """
        x, y = np.asarray(position, dtype=np.float64)
        left, top = self.play_origin
        return (
            round(left + x * self.scale),
            round(top + (self.config.world.height - y) * self.scale),
        )

    def _to_screen_batch(self, points: np.ndarray) -> np.ndarray:
        """Vectorised :meth:`to_screen`.

        Args:
            points: World positions of shape ``(N, 2)``.

        Returns:
            Integer pixel coordinates of shape ``(N, 2)``.
        """
        left, top = self.play_origin
        screen = np.empty_like(points)
        screen[:, 0] = left + points[:, 0] * self.scale
        screen[:, 1] = top + (self.config.world.height - points[:, 1]) * self.scale
        return np.rint(screen).astype(np.int32)

    def _build_contours(self) -> list[np.ndarray]:
        """Precomputes the dot offsets for each iso-concentration ring.

        Levels halve outwards (peak/2, peak/4, and so on) and the matching
        radius is found by inverting the configured profile numerically, so the
        rings stay correct when the profile is switched. Offsets are in world
        units and identical for every pellet, so drawing a frame only adds the
        pellet position. Pixels cannot dim, so outer rings fade by spacing their
        dots further apart instead.

        Returns:
            One ``(N, 2)`` offset array per drawable ring. Rings whose level the
            profile never reaches, or that fall below two pixels, are omitted.
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

            # Arc-length spacing, in display pixels, doubling by the outermost ring.
            fade = index / max(render.scent_contours - 1, 1)
            spacing = render.contour_dot_spacing / PIXEL * (1.0 + fade)
            count = max(8, round(2 * np.pi * radius_px / spacing))
            angles = np.linspace(0.0, 2 * np.pi, count, endpoint=False)
            contours.append(np.stack([np.cos(angles), np.sin(angles)], axis=-1) * radius)
        return contours

    def _draw_scent_field(self, food: FoodField) -> None:
        """Draws each pellet's contour rings, one lit pixel per dot.

        Args:
            food: The pellet field.
        """
        render = self.config.render
        if not (render.show_scent_field or render.debug):
            return

        world = self.config.world
        for position in food.positions:
            for offsets in self._contours:
                points = position + offsets
                if world.boundary == "wrap":
                    # Wrap each dot individually, so a ring running off one edge
                    # reappears on the other, as the real field does.
                    points = geometry.apply_boundary(points, world.size, world.boundary)
                else:
                    inside = np.all((points >= 0) & (points <= np.array(world.size)), axis=-1)
                    points = points[inside]
                for x, y in self._to_screen_batch(points):
                    self._screen.set_at((int(x), int(y)), PIXEL_ON)

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
            pygame.draw.circle(self._screen, PIXEL_ON, self.to_screen(position), dot_px)

    def _draw_toxins(self, toxin: ScentField) -> None:
        """Draws each harmful region as a disc lighting every other pixel.

        The dither is aligned to the screen, not the toxin, so overlapping
        regions merge into one even pattern.

        Args:
            toxin: The hazard field.
        """
        if toxin.count == 0:
            return

        config = self.config.toxin
        # Concentration where damage == basal_cost, inverted through the profile
        # the same way the contours are.
        probe = np.linspace(0.0, config.scent_radius * 2.0, 2048)
        values = toxin.profile(probe)
        parity = self.config.metabolism.basal_cost / max(config.damage, 1e-9)
        radius = (
            float(np.interp(parity, values[::-1], probe[::-1]))
            if values[-1] < parity < values[0]
            else config.scent_radius * 0.5
        )

        radius_px = max(2.0, radius * self.scale)
        side = self._screen.get_width()
        xs, ys = np.ogrid[:side, :side]  # surfarray indexes (x, y)
        inside = np.zeros((side, side), dtype=bool)
        for x, y in self._to_screen_batch(toxin.positions):
            inside |= (xs - x) ** 2 + (ys - y) ** 2 <= radius_px**2

        pixels = pygame.surfarray.pixels3d(self._screen)
        pixels[inside & (xs % 2 == 0) & (ys % 2 == 0)] = PIXEL_ON
        del pixels  # releases the surface lock

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
        appears at both edges instead of streaking across the frame. Unlit
        discs go down first as an outline, since interleaving the two passes
        would let a later outline erase body already drawn.

        Args:
            worm: The body to draw.
        """
        render = self.config.render
        self._track_head(worm)

        # Half a display pixel apart, so the body has no gaps between discs.
        samples = max(8, round(render.worm_body_length * self.scale / 0.5))
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

        world = self.config.world
        drawn = geometry.apply_boundary(body, world.size, world.boundary)
        screen = self._to_screen_batch(drawn)
        discs = [((int(x), int(y)), int(r)) for (x, y), r in zip(screen, radii, strict=True)]
        for centre, radius in discs:
            pygame.draw.circle(self._screen, PIXEL_OFF, centre, radius + 1)
        for centre, radius in discs:
            pygame.draw.circle(self._screen, PIXEL_ON, centre, radius)

        if render.debug:
            # The body is decoration; this ring is the point the physics
            # integrates, which is what eats, and the dot is where it smells.
            # The two coincide unless the head sweeps.
            pygame.draw.circle(self._screen, PIXEL_ON, self.to_screen(worm.position), 3, width=1)
            self._screen.set_at(self.to_screen(worm.head_position), PIXEL_ON)
