"""Watch a trained worm, or measure one.

    uv run python scripts/watch.py experiments/ppo_s0/checkpoints/latest.pt
    uv run python scripts/watch.py <checkpoint> --episodes 20 --no-render

The world is rebuilt from the config stored inside the checkpoint, not from
``configs/`` — so what you watch is the world the policy actually trained in,
even if the config files have moved on since.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs import UNLIMITED_EPISODE_STEPS
from envs.episodes import EpochStats, episode_line, run_episodes, summary_line
from ppo import load_policy


def main() -> None:
    """Loads a checkpoint and rolls its policy out, rendering by default."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="path to a .pt checkpoint")
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="episodes to run, or 0 to keep going until the window is closed "
        "or Ctrl-C is pressed",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-render", action="store_true", help="measure instead of watch")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="sample actions instead of taking the policy mean",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="draw pellets at their true eat radius, with scent contours",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="override the step cap per episode, or 0 to let a life run until "
        "the worm starves",
    )
    args = parser.parse_args()

    ac, env, state = load_policy(
        args.checkpoint,
        render_mode=None if args.no_render else "human",
        env_overrides={"render": {"debug": True}} if args.debug else None,
        max_episode_steps=(
            UNLIMITED_EPISODE_STEPS if args.max_steps == 0 else args.max_steps
        ),
    )
    print(
        f"epoch {state['epoch']}  ({state['total_steps']:,} env steps)  "
        f"{'stochastic' if args.stochastic else 'deterministic'}"
    )

    # Accumulated here rather than taken from the return value: an unbounded
    # run is ended by Ctrl-C or a closed window, and neither reaches a return.
    stats = EpochStats()

    def report(index: int, episode) -> None:
        stats.add(episode)
        print(episode_line(index, episode), flush=True)

    try:
        run_episodes(
            env,
            lambda obs: ac.act(
                torch.as_tensor(obs, dtype=torch.float32), deterministic=not args.stochastic
            ),
            episodes=args.episodes or None,
            seed=args.seed,
            on_episode=report,
            stop=None if args.no_render else lambda: env.unwrapped.window_closed,
        )
    except KeyboardInterrupt:
        print()
    env.close()

    if not stats.episodes:
        print("\nno episode finished")
        return
    print(summary_line(stats))

    # Computed from this checkpoint's own world, not hardcoded — the freeze
    # baseline moves with initial_energy and basal_cost, and a stale number
    # here is worse than none.
    metabolism = env.unwrapped.config.metabolism
    freeze = metabolism.initial_energy / metabolism.basal_cost
    print(
        f"freeze baseline for this world: {freeze:.0f} steps "
        f"(initial_energy {metabolism.initial_energy:g} / basal_cost {metabolism.basal_cost:g}). "
        f"Compare against scripts/play.py --policy straight too."
    )
    if args.episodes < 20:
        print(
            f"note: {args.episodes} episodes is not a measurement — lifespan here has a "
            f"standard deviation of roughly its own mean. Use --episodes 50 to compare runs."
        )


if __name__ == "__main__":
    main()
