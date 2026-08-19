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

from envs.episodes import episode_line, run_episodes, summary_line
from ppo import load_policy


def main() -> None:
    """Loads a checkpoint and rolls its policy out, rendering by default."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="path to a .pt checkpoint")
    parser.add_argument("--episodes", type=int, default=5)
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
    args = parser.parse_args()

    ac, env, state = load_policy(
        args.checkpoint,
        render_mode=None if args.no_render else "human",
        env_overrides={"render": {"debug": True}} if args.debug else None,
    )
    print(
        f"epoch {state['epoch']}  ({state['total_steps']:,} env steps)  "
        f"{'stochastic' if args.stochastic else 'deterministic'}"
    )

    stats = run_episodes(
        env,
        lambda obs: ac.act(
            torch.as_tensor(obs, dtype=torch.float32), deterministic=not args.stochastic
        ),
        episodes=args.episodes,
        seed=args.seed,
        on_episode=lambda index, episode: print(episode_line(index, episode)),
    )
    env.close()

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
