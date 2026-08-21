"""Command-line entry point for training a worm with PPO.

Resolves a world config and a learner config, runs the training loop, and
reports a deterministic evaluation. Command-line flags override the YAML files,
which override the dataclass defaults. Each run writes both resolved configs,
``progress.csv`` and its checkpoints into ``experiments/<name>_s<seed>/``, and
never overwrites an existing directory.

The two configs have separate jobs: ``--env-config`` describes the world (arena,
food, toxins, metabolism, and which channels the worm senses) while
``--ppo-config`` describes the learner (network shape, discount, and the
update). See :mod:`envs.config` and :mod:`ppo.config` for the schemas.

Typical usage example:

  uv run python train.py --env-config configs/world_v2.yaml --name toxins
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ppo import PPOConfig, PPOTrainer

# Used when the flags are omitted, so editing these files is enough to change a
# run. Falls back to the dataclass defaults if a file is missing.
DEFAULT_ENV_CONFIG = Path("configs/world_v1.yaml")
DEFAULT_PPO_CONFIG = Path("configs/ppo.yaml")

# Shown below the flag list by --help. This is operator guidance rather than a
# description of the module, so it lives here instead of in the docstring: the
# curriculum and --resume both make the task harder over time and are routinely
# mistaken for one another.
EPILOG = """\
two ways the task gets harder:

  The curriculum (the `curriculum` section of the PPO config) anneals the
  world within a single run, from its *_start values toward whatever the env
  config specifies. It needs nothing on the command line and happens on its
  own whenever it is enabled.

  --resume continues one run's weights in a new run, against whatever world
  --env-config names. Use it for a handoff the curriculum cannot express: a
  different boundary, a retuned metabolism, or more epochs on a plateau.

    uv run python train.py --env-config configs/world_v1.yaml --name easy
    uv run python train.py --env-config configs/world_v1.yaml --name hard \\
        --resume experiments/easy_s0/checkpoints/latest.pt

  Resume works only when the observation width matches. Weights are a matrix
  sized to the input, so a checkpoint cannot be loaded into a run that senses
  a different number of things. Changing frame_stack or any
  observation.include_* flag, which is what separates world v1 from world v2,
  means training from scratch. The run fails immediately and says so rather
  than loading something mismatched.

watch a finished run:

  uv run python scripts/watch.py experiments/<run>/checkpoints/latest.pt
"""


def _default(path: Path) -> str | None:
    """Returns the path as a string if it exists, else None.

    Args:
        path: Candidate config file.

    Returns:
        The path, or None to fall back to the dataclass defaults.
    """
    return str(path) if path.exists() else None


def main() -> None:
    """Parses arguments, trains, and reports a deterministic evaluation."""
    # Raw, so the line breaks in the docstring and the epilog survive into
    # --help; argparse otherwise reflows both into single paragraphs.
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--env-config",
        default=_default(DEFAULT_ENV_CONFIG),
        help=f"YAML for the world (default: {DEFAULT_ENV_CONFIG} if present)",
    )
    parser.add_argument(
        "--ppo-config",
        default=_default(DEFAULT_PPO_CONFIG),
        help=f"YAML for the hyperparameters (default: {DEFAULT_PPO_CONFIG} if present)",
    )
    parser.add_argument("--name", default=None, help="run name (default: from config)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument(
        "--resume",
        default=None,
        metavar="CHECKPOINT",
        help="continue from a checkpoint's weights in a new run directory, leaving the "
        "original alone. The world comes from --env-config, not from the checkpoint, so "
        "the two can differ — but only if they produce the same observation width. "
        "Adding or removing an observation channel means training from scratch.",
    )
    parser.add_argument(
        "--reset-optimizers",
        action="store_true",
        help="with --resume, start Adam fresh instead of restoring its moment estimates. "
        "Those moments summarise recent gradients, so they help when continuing the same "
        "task and mislead when the new world makes them describe a task that no longer exists.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="deterministic episodes to run after training (0 to skip)",
    )
    args = parser.parse_args()

    config = PPOConfig.resolve(args.ppo_config)
    # CLI flags win over the YAML, which wins over the dataclass defaults.
    for value, section, field in [
        (args.name, config.run, "name"),
        (args.seed, config.run, "seed"),
        (args.device, config.run, "device"),
        (args.epochs, config.rollout, "epochs"),
        (args.steps_per_epoch, config.rollout, "steps_per_epoch"),
    ]:
        if value is not None:
            setattr(section, field, value)
    config.run.__post_init__()
    config.rollout.__post_init__()

    print(f"env config: {args.env_config or '<defaults>'}   ppo config: {args.ppo_config or '<defaults>'}")
    trainer = PPOTrainer(env_config=args.env_config, ppo_config=config)
    if config.curriculum.enabled:
        target, start = trainer.target_env_config.food, trainer.env_config.food
        print(
            f"curriculum: food {start.count}->{target.count}, "
            f"eat_radius {start.eat_radius:.2f}->{target.eat_radius:.2f}, "
            f"scent_radius {start.scent_radius:.1f}->{target.scent_radius:.1f} "
            f"over {round(config.rollout.epochs * config.curriculum.anneal_fraction)} epochs"
        )
    if args.resume:
        trainer.load_checkpoint(args.resume, reset_optimizers=args.reset_optimizers)
        print(
            f"resumed from {args.resume} "
            f"(epoch {trainer.epoch}, {trainer.total_steps:,} prior env steps"
            f"{', optimizers reset' if args.reset_optimizers else ''})"
        )
    run_dir = trainer.run()

    if args.eval_episodes > 0:
        summary = trainer.evaluate(episodes=args.eval_episodes)
        print("\ndeterministic evaluation:")
        for key in ("lifespan_mean", "lifespan_min", "lifespan_max", "eaten_mean", "death_rate"):
            print(f"  {key:16s} {summary[key]:.2f}")
    print(f"\nrun saved to {run_dir}")


if __name__ == "__main__":
    main()
