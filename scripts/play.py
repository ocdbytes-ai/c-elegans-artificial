"""Watch the world, or measure a baseline in it.

    uv run python scripts/play.py --policy greedy --render
    uv run python scripts/play.py --policy freeze --episodes 20

These are the baselines a trained worm has to beat. Two of them are traps PPO
will find on its own if the world allows it, and neither senses anything:

- ``freeze`` — the local optimum when the metabolism is mistuned and moving
  costs meaningfully more than idling.
- ``straight`` — the local optimum on a torus, where a straight line sweeps the
  whole world. Measured at 1051 steps on a 20x20 wrap world against 320 for
  freeze; harmless under ``boundary: clamp``.

``greedy`` cheats by reading pellet positions and gives the ceiling. A trained
policy that does not clear both traps has not learned chemotaxis, whatever its
lifespan says.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym

from envs import ENV_ID
from envs.config import EnvConfig
from envs.episodes import episode_line, run_episodes, summary_line
from envs.geometry import wrap_angle


def random_policy(env: gym.Env):
    """Samples uniformly from the action space.

    Args:
        env: The environment, for its action space.

    Returns:
        A policy callable.
    """
    return lambda _obs: env.action_space.sample()


def freeze_policy(_env: gym.Env):
    """Does nothing and coasts on the initial energy.

    Scores ``initial_energy / basal_cost`` exactly, and is the local optimum
    when moving costs meaningfully more than idling.

    Args:
        _env: Unused; present for a uniform factory signature.

    Returns:
        A policy callable.
    """
    return lambda _obs: np.zeros(2, dtype=np.float32)


def straight_policy(env: gym.Env):
    """Commits to the spawn heading and never turns, sensing nothing.

    The ballistic counterpart to ``freeze``, and the one that matters on a
    torus: with no walls a straight line eventually sweeps the whole world, so
    blind running becomes real foraging. Measured on a 20x20 wrap world with 10
    pellets it survived 1051 steps and ate 16, against 320 for freeze. Under
    ``boundary: clamp`` it is worthless (328 steps) because the worm parks
    against a wall.

    A wrap-mode policy that cannot beat this has learned to run, not to smell.

    Args:
        env: Unused; present for a uniform factory signature.

    Returns:
        A policy callable.
    """
    return lambda _obs: np.array([0.0, 1.0], dtype=np.float32)


def greedy_policy(env: gym.Env):
    """Reads the true nearest pellet from the simulation and steers at it.

    Deliberately uses information the worm cannot sense and which never appears
    in an observation, so it is an upper reference rather than a baseline: how
    long could a worm live if chemotaxis were already solved.

    Args:
        env: The environment, read directly for pellet positions.

    Returns:
        A policy callable.
    """

    def act(_obs: np.ndarray) -> np.ndarray:
        """Steers at the nearest pellet at full throttle."""
        world = env.unwrapped
        _, delta, _ = world.food.nearest(world.worm.position)
        error = wrap_angle(np.arctan2(delta[1], delta[0]) - world.worm.heading)
        # This episode's drawn turn rate, not the nominal one: with
        # randomisation on, steering against the config value would
        # consistently over- or undershoot.
        max_turn_per_step = world.worm.max_turn_rate * world.config.world.dt
        return np.array([np.clip(error / max_turn_per_step, -1.0, 1.0), 1.0], dtype=np.float32)

    return act


POLICIES = {
    "random": random_policy,
    "freeze": freeze_policy,
    "straight": straight_policy,
    "greedy": greedy_policy,
}


def main() -> None:
    """Runs the selected baseline and prints its statistics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="random", choices=sorted(POLICIES))
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_argument("--render", action="store_true", help="open a pygame window")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="draw pellets at their true eat radius, with scent contours",
    )
    args = parser.parse_args()

    # Resolved here rather than inside the env so --debug can be applied on top
    # of whatever --config names, without either flag having to know about the
    # other.
    config = EnvConfig.resolve(args.config)
    config.render.debug = config.render.debug or args.debug

    env = gym.make(
        ENV_ID,
        config=config,
        render_mode="human" if args.render else None,
    )
    print(f"obs={env.unwrapped.observation_labels}  policy={args.policy}")

    stats = run_episodes(
        env,
        POLICIES[args.policy](env),
        episodes=args.episodes,
        seed=args.seed,
        on_episode=lambda index, episode: print(episode_line(index, episode)),
    )
    env.close()
    print(summary_line(stats))


if __name__ == "__main__":
    main()
