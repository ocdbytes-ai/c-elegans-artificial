"""Terminal visualiser for the actor network driving a worm.

Draws one forward pass per frame: the stacked observation as the network sees
it, every hidden activation, and the action that comes out, alongside the
energy, food and step counters that used to sit in the pygame window.

Two things it makes visible that a rendered worm cannot:

- **The frame stack is a time axis.** Smell is a single scalar with no
  direction, so the only way to tell "climbing a gradient" from "falling off
  one" is to compare frames. The sparkline per channel *is* that comparison,
  which is the signal chemotaxis is built on.
- **Saturation.** A tanh layer sitting at ±1 has no gradient and is passing a
  constant, so a policy that looks busy can be mostly frozen.

Everything is greyscale, matching the renderer: intensity carries magnitude, so
it reads the same on any terminal theme and in a screenshot. Frames are written
with a cursor-home rather than a clear, which avoids flicker.
"""

from __future__ import annotations

import shutil
import sys

import numpy as np
import torch
from torch import nn

# Eight levels of vertical fill, for sparklines and activation grids alike.
BLOCKS = " ▁▂▃▄▅▆▇█"
BAR_FULL, BAR_EMPTY = "█", "░"

# xterm-256 greyscale runs 232 (black) to 255 (white). Starting above the floor
# keeps the faintest activation legible on a black terminal.
GREY_LOW, GREY_HIGH = 237, 255

HOME = "\x1b[H"
CLEAR_BELOW = "\x1b[J"
HIDE_CURSOR = "\x1b[?25l"
SHOW_CURSOR = "\x1b[?25h"
RESET = "\x1b[0m"
DIM = "\x1b[2m"
BOLD = "\x1b[1m"


def grey(level: float) -> str:
    """Returns the escape code for a greyscale level.

    Args:
        level: Intensity in ``[0, 1]``, clamped.

    Returns:
        An ANSI foreground colour escape.
    """
    span = GREY_HIGH - GREY_LOW
    return f"\x1b[38;5;{GREY_LOW + round(float(np.clip(level, 0.0, 1.0)) * span)}m"


def block(level: float) -> str:
    """Picks a fill character for a magnitude in ``[0, 1]``."""
    return BLOCKS[round(float(np.clip(level, 0.0, 1.0)) * (len(BLOCKS) - 1))]


# Below this the channel is treated as unchanging, so numerical dither is not
# stretched into a dramatic-looking waveform.
FLAT = 1e-3


def sparkline(values: np.ndarray) -> str:
    """Draws a sequence as shaded blocks, oldest first.

    Scaled to the series' own range rather than an absolute one: what matters
    here is whether a channel is climbing or falling between frames, and the
    absolute level is printed alongside anyway. A channel that does not move is
    drawn flat rather than amplified.

    Args:
        values: The series.

    Returns:
        One coloured character per value.
    """
    low, high = float(values.min()), float(values.max())
    if high - low < FLAT:
        return f"{grey(0.5)}{'▄' * len(values)}{RESET}"
    out = []
    for value in values:
        level = (float(value) - low) / (high - low)
        # Never a fully empty cell: the trough is still a reading.
        out.append(f"{grey(0.35 + 0.65 * level)}{block(0.15 + 0.85 * level)}")
    return "".join(out) + RESET


def meter(fraction: float, width: int) -> str:
    """Draws a proportion as a filled bar.

    Args:
        fraction: Proportion filled, clamped to ``[0, 1]``.
        width: Bar width in characters.

    Returns:
        A coloured bar.
    """
    filled = round(float(np.clip(fraction, 0.0, 1.0)) * width)
    return f"{BOLD}{BAR_FULL * filled}{RESET}{DIM}{BAR_EMPTY * (width - filled)}{RESET}"


def dial(value: float, width: int) -> str:
    """Draws a signed value as a marker on a centred axis.

    Args:
        value: The value, clamped to ``[-1, 1]``.
        width: Axis width in characters; the centre is the zero point.

    Returns:
        A coloured axis with the marker in place.
    """
    centre = width // 2
    position = centre + round(float(np.clip(value, -1.0, 1.0)) * centre)
    low, high = min(centre, position), max(centre, position)
    cells = []
    for index in range(width + 1):
        if index == position:
            cells.append(f"{BOLD}●{RESET}")
        elif index == centre:
            cells.append(f"{DIM}│{RESET}")
        elif low <= index <= high:
            cells.append(f"{grey(0.6)}─{RESET}")
        else:
            cells.append(f"{DIM}·{RESET}")
    return "".join(cells)


def actor_activations(
    actor: nn.Module, obs: torch.Tensor
) -> tuple[list[np.ndarray], np.ndarray]:
    """Replays a forward pass, keeping every hidden activation.

    Args:
        actor: A :class:`~ppo.networks.GaussianActor`.
        obs: A single flattened observation.

    Returns:
        The post-activation output of each hidden layer, and the action mean.
    """
    hidden: list[np.ndarray] = []
    x = obs
    with torch.no_grad():
        for layer in actor.mu_net:
            x = layer(x)
            # build_mlp puts an activation after every Linear but the last, so
            # the non-Linear modules are exactly the hidden-layer outputs.
            if not isinstance(layer, nn.Linear):
                hidden.append(x.cpu().numpy().copy())
    return hidden, x.cpu().numpy().copy()


class ActorTUI:
    """Renders the actor's forward pass to the terminal.

    Attributes:
        labels: Observation channel names, innermost frame last.
        frames: How many stacked frames the observation holds.
        every: Steps between redraws.
        width: Columns available.
    """

    def __init__(self, labels: list[str], frames: int, every: int = 3):
        """Sizes the display against the terminal.

        Args:
            labels: Channel names for one frame.
            frames: Stacked frames per observation.
            every: Redraw every N steps; the simulation runs far faster than a
                terminal can usefully be repainted.
        """
        self.labels = labels
        self.frames = frames
        self.every = max(1, every)
        self.width = max(56, min(shutil.get_terminal_size((80, 24)).columns, 100))
        self._grid_cols = 32
        self._started = False

    def start(self) -> None:
        """Clears the screen and hides the cursor."""
        if not self._started:
            sys.stdout.write(HIDE_CURSOR + "\x1b[2J")
            self._started = True

    def stop(self) -> None:
        """Restores the cursor and leaves the last frame on screen."""
        if self._started:
            sys.stdout.write(SHOW_CURSOR + "\n")
            sys.stdout.flush()
            self._started = False

    def _rule(self, title: str) -> str:
        """A titled horizontal rule."""
        return f"{DIM}── {RESET}{title}{DIM} {'─' * max(0, self.width - len(title) - 5)}{RESET}"

    def _senses(self, obs: np.ndarray) -> list[str]:
        """Renders each channel's recent history as a sparkline."""
        per_frame = len(self.labels)
        lines = []
        for channel, name in enumerate(self.labels):
            # FlattenObservation lays the stack out frame-major, so channel c of
            # frame f sits at f * per_frame + c.
            series = np.array([obs[f * per_frame + channel] for f in range(self.frames)])
            delta = float(series[-1] - series[0])
            arrow = "▲" if delta > FLAT else "▼" if delta < -FLAT else " "
            lines.append(
                f"  {name:<12}{sparkline(series)}  {series[-1]:+6.2f}  "
                f"{arrow}{DIM}{delta:+6.3f}{RESET}"
            )
        return lines

    def _layer(self, values: np.ndarray, index: int) -> list[str]:
        """Renders one hidden layer as a shaded grid plus its summary."""
        magnitude = np.abs(values)
        saturated = float((magnitude > 0.95).mean())
        head = self._rule(
            f"hidden {index} ({values.size})  "
            f"{DIM}mean |a| {magnitude.mean():.2f}   saturated {100 * saturated:.0f}%"
        )
        rows = [head]
        cols = self._grid_cols
        for start in range(0, values.size, cols):
            chunk = magnitude[start : start + cols]
            rows.append("  " + "".join(f"{grey(v)}{block(v)}" for v in chunk) + RESET)
        return rows

    def frame(self, obs: np.ndarray, hidden, action, energy, eaten, steps) -> str:
        """Builds one complete frame of output.

        Args:
            obs: The flattened observation the network was given.
            hidden: Per-layer activations from :func:`actor_activations`.
            action: The action mean.
            energy: ``(current, maximum)`` energy.
            eaten: Pellets eaten this life.
            steps: Steps survived this life.

        Returns:
            The frame, with a leading cursor-home.
        """
        current, maximum = energy
        counters = (
            f"  {DIM}energy{RESET}  {meter(current / maximum, 28)}  "
            f"{current:5.1f}/{maximum:.0f}    "
            f"{DIM}eaten{RESET} {BOLD}{eaten:<4}{RESET} {DIM}step{RESET} {BOLD}{steps}{RESET}"
        )
        lines = [
            f"{BOLD}ACTOR{RESET}{DIM} · the network choosing every move{RESET}",
            "",
            counters,
            "",
            self._rule(f"senses  {DIM}{self.frames} frames, oldest to newest, normalised"),
            *self._senses(obs),
            "",
        ]
        for index, layer in enumerate(hidden, start=1):
            lines += self._layer(layer, index)
            lines.append("")

        lines.append(self._rule("action"))
        for name, value in zip(("turn", "throttle"), action, strict=False):
            # The env clips to [-1, 1], so a larger mean is effort with nowhere
            # to go: worth seeing, since it means the policy is at its bound.
            clipped = f"{DIM} clipped{RESET}" if abs(float(value)) > 1.0 else ""
            lines.append(
                f"  {name:<10}{dial(float(value), 28)}  "
                f"{BOLD}{float(value):+6.2f}{RESET}{clipped}"
            )
        return HOME + "\n".join(line + "\x1b[K" for line in lines) + "\n" + CLEAR_BELOW

    def draw(self, actor, obs_tensor, obs, energy, eaten, steps) -> None:
        """Renders a frame, subject to the redraw interval.

        Args:
            actor: The :class:`~ppo.networks.GaussianActor` to visualise.
            obs_tensor: The observation as handed to the network.
            obs: The same observation as a numpy array.
            energy: ``(current, maximum)`` energy.
            eaten: Pellets eaten this life.
            steps: Steps survived this life.
        """
        if steps % self.every:
            return
        self.start()
        hidden, action = actor_activations(actor, obs_tensor)
        sys.stdout.write(self.frame(obs, hidden, action, energy, eaten, steps))
        sys.stdout.flush()
