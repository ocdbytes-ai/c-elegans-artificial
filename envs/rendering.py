"""Retro cyan phosphor renderer for the pygame world.

The display uses a dark dotted grid, glowing cyan organisms and hazards, and a
compact pixel-style HUD inspired by old monochrome laboratory terminals.

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
GRID = (5, 22, 27)
CYAN = (40, 220, 255)
CYAN_DIM = (12, 82, 100)
CYAN_FAINT = (7, 42, 51)
HUD_HEIGHT = 58
CONTOUR_NEAR = 185
CONTOUR_FAR = 45


class PygameRenderer:
    """Draws frames of the world, and owns the window under ``human`` mode.

    Attributes:
        config: The environment config, for world size and render options.
        render_mode: ``"human"`` or ``"rgb_array"``.
        scale: Pixels per world unit.
        size_px: Surface size in pixels.
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
        window = config.render.window_size
        self.scale = min(window / world.width, (window - HUD_HEIGHT) / world.height)
        play_width = world.width * self.scale
        play_height = world.height * self.scale
        self.play_origin = ((window - play_width) / 2.0, HUD_HEIGHT)
        self.size_px = (window, window)

        pygame.init()
        pygame.font.init()
        self.surface = pygame.Surface(self.size_px)
        self._background = self._build_background()
        self._glow = pygame.Surface(self.size_px, pygame.SRCALPHA)
        self._font = pygame.font.Font(None, max(20, round(window * 0.045)))
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
            toxin: Hazards, drawn as dotted fields.
            metabolism: Current energy for the HUD.
            steps: Current episode step for the HUD.

        Returns:
            An RGB array under ``"rgb_array"``, otherwise None, including once
            the window has been closed.
        """
        if self.closed:
            return None

        self.surface.blit(self._background, (0, 0))
        self._glow.fill((0, 0, 0, 0))
        self._draw_scent_field(food)
        self._draw_food(food)
        self._draw_toxins(toxin)
        self._draw_worm(worm)
        self.surface.blit(self._glow, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
        self._draw_hud(metabolism, steps)

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

    def _build_background(self) -> pygame.Surface:
        """Builds the static dotted screen and playfield divider."""
        background = pygame.Surface(self.size_px)
        background.fill(BACKGROUND)
        spacing = max(4, round(self.config.render.window_size / 140))
        for y in range(2, self.size_px[1], spacing):
            for x in range(2, self.size_px[0], spacing):
                background.set_at((x, y), GRID)
        for x in range(8, self.size_px[0] - 8, spacing * 2):
            pygame.draw.circle(background, CYAN_DIM, (x, HUD_HEIGHT - 3), 1)
        return background

    def _glow_circle(self, centre: tuple[int, int], radius: int, colour=CYAN) -> None:
        """Draws one crisp phosphor dot with a soft halo."""
        pygame.draw.circle(self._glow, (*CYAN_DIM, 28), centre, radius + 5)
        pygame.draw.circle(self._glow, (*CYAN, 55), centre, radius + 2)
        pygame.draw.circle(self.surface, colour, centre, radius)

    def _draw_hud(self, metabolism: Metabolism, steps: int) -> None:
        """Draws the energy bar and elapsed-step counter."""
        energy = metabolism.energy_fraction
        margin = 12
        label = self._font.render("ENERGY", False, CYAN)
        self.surface.blit(label, (margin, 10))

        bar_x = margin + label.get_width() + 10
        bar_y, bar_h = 13, max(14, label.get_height() - 7)
        bar_w = max(80, round(self.size_px[0] * 0.23))
        pygame.draw.rect(self.surface, CYAN_DIM, (bar_x, bar_y, bar_w, bar_h), width=2)
        inset = 4
        fill_w = round((bar_w - 2 * inset) * energy)
        if fill_w:
            pygame.draw.rect(
                self.surface,
                CYAN,
                (bar_x + inset, bar_y + inset, fill_w, bar_h - 2 * inset),
            )

        timer = self._font.render(f"T {steps:05d}", False, CYAN)
        self.surface.blit(timer, (self.size_px[0] - timer.get_width() - margin, 10))

    def to_screen(self, position: np.ndarray) -> tuple[int, int]:
        """Converts a world position to pixel coordinates.

        Args:
            position: World coordinates, y up from the bottom-left.

        Returns:
            Pixel coordinates, y down from the top-left.
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
            intensity = round(CONTOUR_NEAR + (CONTOUR_FAR - CONTOUR_NEAR) * fade)
            colour = (round(intensity * 0.18), round(intensity * 0.82), intensity)
            dot_px = 2 if fade < 0.5 else 1
            contours.append((offsets, colour, dot_px))
        return contours

    def _draw_scent_field(self, food: FoodField) -> None:
        """Clears the surface and draws each pellet's contour rings.

        Args:
            food: The pellet field.
        """
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
            self._glow_circle(self.to_screen(position), dot_px)

    def _draw_toxins(self, toxin: ScentField) -> None:
        """Draws each harmful region as a dotted phosphor field.

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

        radius_px = max(4, round(radius * self.scale))
        spacing = max(10, round(self.scale * 0.4))
        rings = max(3, round(radius_px / spacing))
        for position in toxin.positions:
            centre = self.to_screen(position)
            for ring in range(1, rings + 1):
                ring_radius = radius_px * ring / rings
                count = max(8, round(2.0 * np.pi * ring_radius / spacing))
                intensity = 0.75 - 0.4 * ring / rings
                colour = (
                    round(CYAN[0] * intensity),
                    round(CYAN[1] * intensity),
                    round(CYAN[2] * intensity),
                )
                # Offset alternating rings so dots form circles, not spokes.
                offset = (ring % 2) * np.pi / count
                for angle in np.linspace(0.0, 2.0 * np.pi, count, endpoint=False) + offset:
                    point = (
                        round(centre[0] + ring_radius * np.cos(angle)),
                        round(centre[1] + ring_radius * np.sin(angle)),
                    )
                    pygame.draw.circle(self.surface, colour, point, 1)

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

        outline_px = max(1, min(2, round(self.config.worm.radius * self.scale * 0.25)))

        world = self.config.world
        drawn = geometry.apply_boundary(body, world.size, world.boundary)
        screen = self._to_screen_batch(drawn)
        for (x, y), radius in zip(screen, radii, strict=True):
            centre = (int(x), int(y))
            pygame.draw.circle(self.surface, BACKGROUND, centre, int(radius) + outline_px)
            self._glow_circle(centre, int(radius))

        if render.debug:
            # The body is decoration; this ring is the point the physics
            # integrates, which is what eats, and the dot is where it smells.
            # The two coincide unless the head sweeps.
            pygame.draw.circle(self.surface, CYAN, self.to_screen(worm.position), 5, width=1)
            pygame.draw.circle(self.surface, CYAN, self.to_screen(worm.head_position), 2)
