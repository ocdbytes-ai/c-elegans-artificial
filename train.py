"""Train a worm with PPO.

    uv run python train.py                                  # defaults
    uv run python train.py --epochs 500 --seed 3
    uv run python train.py --env-config configs/world_v1.yaml --ppo-config configs/ppo.yaml

Curriculum handoff — train somewhere forgiving, then continue the same weights
against the real world::

    uv run python train.py --env-config configs/curriculum_stage1.yaml --name stage1
    uv run python train.py --env-config configs/world_v1.yaml --name stage2 \
        --resume experiments/stage1_s0/checkpoints/latest.pt

Everything a run needs is written into ``experiments/<name>_s<seed>/``: both
configs, ``progress.csv``, and checkpoints. Watch the result with::

    uv run python scripts/watch.py experiments/ppo_s0/checkpoints/latest.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ppo import PPOConfig, PPOTrainer

# Used when the flags are omitted, so editing these files is enough to change a
# run. Falls back to the dataclass defaults if a file is missing.
DEFAULT_ENV_CONFIG = Path("configs/world_v1.yaml")
DEFAULT_PPO_CONFIG = Path("configs/ppo.yaml")


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
    parser = argparse.ArgumentParser(description=__doc__)
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
        help="continue from a checkpoint's weights. The world comes from --env-config, "
        "not from the checkpoint, so this is how you hand a policy to a harder curriculum "
        "stage. Results go to a new run directory; the original is left alone.",
    )
    parser.add_argument(
        "--reset-optimizers",
        action="store_true",
        help="with --resume, drop Adam's moment estimates (advisable when the new world "
        "differs a lot from the one the checkpoint trained in)",
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
