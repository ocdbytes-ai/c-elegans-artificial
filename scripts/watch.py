"""Watch a trained worm, or measure one.

    uv run python scripts/watch.py experiments/ppo_s0/checkpoints/latest.pt
    uv run python scripts/watch.py <checkpoint> --episodes 20 --no-render
    uv run python scripts/watch.py <checkpoint> --env-config configs/bigger.yaml

The world is rebuilt from the config stored inside the checkpoint, not from
``configs/``, so what you watch is the world the policy actually trained in
even if the config files have moved on since. ``--env-config`` is the way to
override that deliberately: the file is merged over the stored config, so a
YAML naming only what it changes leaves the rest of the trained world alone.

Note that ``experiments/<run>/env.yaml`` is a record of a finished run, not an
input. Nothing reads it back. Edit ``configs/`` and pass it here instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs import UNLIMITED_EPISODE_STEPS
from envs.episodes import EpochStats, episode_line, run_episodes, summary_line
from ppo import load_policy
from ppo.tui import ActorTUI


def describe_overrides(stored: dict, overrides: dict) -> list[str]:
    """Lists the settings an override file actually changes.

    Silently watching a policy in a world it never trained in is the failure
    this exists to prevent, so what changed is printed rather than assumed.

    Args:
        stored: The env config saved inside the checkpoint.
        overrides: Nested ``{section: {key: value}}`` to be merged over it.

    Returns:
        One ``section.key: old -> new`` line per genuine change.
    """

    def norm(value: Any) -> Any:
        """YAML gives lists where the config holds tuples."""
        return tuple(value) if isinstance(value, list) else value

    changes = []
    for section, values in overrides.items():
        for key, value in (values or {}).items():
            was = (stored.get(section) or {}).get(key)
            if norm(was) != norm(value):
                changes.append(f"    {section}.{key}: {was} -> {value}")
    return changes


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
    parser.add_argument(
        "--env-config",
        default=None,
        help="YAML merged over the world stored in the checkpoint; may name "
        "only the settings it changes",
    )
    parser.add_argument(
        "--no-tui",
        action="store_true",
        help="do not draw the actor network in the terminal",
    )
    args = parser.parse_args()

    overrides: dict[str, Any] = {}
    if args.env_config:
        with open(args.env_config) as handle:
            overrides = yaml.safe_load(handle) or {}
    if args.debug:
        overrides["render"] = {**overrides.get("render", {}), "debug": True}

    ac, env, state = load_policy(
        args.checkpoint,
        render_mode=None if args.no_render else "human",
        env_overrides=overrides or None,
        max_episode_steps=(
            UNLIMITED_EPISODE_STEPS if args.max_steps == 0 else args.max_steps
        ),
    )

    if args.env_config:
        changes = describe_overrides(state["env_config"], overrides)
        if changes:
            print(f"world overridden from {args.env_config}, so this is NOT the")
            print("world the policy trained in; scores are not comparable:")
            print("\n".join(changes))
        else:
            print(f"{args.env_config} changes nothing about the stored world")
    print(
        f"epoch {state['epoch']}  ({state['total_steps']:,} env steps)  "
        f"{'stochastic' if args.stochastic else 'deterministic'}"
    )

    # Accumulated here rather than taken from the return value: an unbounded
    # run is ended by Ctrl-C or a closed window, and neither reaches a return.
    stats = EpochStats()

    # The TUI owns the terminal while it runs, so per-episode lines are held
    # back and printed once it has released the screen.
    tui: ActorTUI | None = None
    if not (args.no_render or args.no_tui):
        world = env.unwrapped
        tui = ActorTUI(world.observation_labels, state["ppo_config"]["rollout"]["frame_stack"])

    def report(index: int, episode) -> None:
        stats.add(episode)
        if tui is None:
            print(episode_line(index, episode), flush=True)

    def policy(obs):
        """Chooses an action, drawing the forward pass that produced it."""
        tensor = torch.as_tensor(obs, dtype=torch.float32)
        action = ac.act(tensor, deterministic=not args.stochastic)
        if tui is not None:
            world = env.unwrapped
            tui.draw(
                ac.pi,
                tensor,
                obs,
                energy=(world.metabolism.energy, world.config.metabolism.max_energy),
                eaten=world.food.eaten_total,
                steps=world.steps,
            )
        return action

    try:
        run_episodes(
            env,
            policy,
            episodes=args.episodes or None,
            seed=args.seed,
            on_episode=report,
            stop=None if args.no_render else lambda: env.unwrapped.window_closed,
        )
    except KeyboardInterrupt:
        print()
    finally:
        if tui is not None:
            tui.stop()
    env.close()

    if not stats.episodes:
        print("\nno episode finished")
        return
    if tui is not None:
        # Held back while the TUI owned the screen.
        for index, episode in enumerate(stats.episodes):
            print(episode_line(index, episode))
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
