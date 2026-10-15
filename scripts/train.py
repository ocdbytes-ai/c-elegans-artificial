"""Command-line entry point for training a worm with PPO.

Resolves a world config and a learner config, runs the training loop, and
reports a deterministic evaluation. Command-line flags override the YAML files,
which override the dataclass defaults. Each run writes both resolved configs,
``progress.csv`` and its checkpoints into ``experiments/<name>_s<seed>/``, and
never overwrites an existing directory.

Typical usage example:

    uv run python scripts/train.py --env-config configs/world_v2.yaml --name toxins

For command reference:

    uv run python scripts/train.py --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.cli import positive_int
from ppo import PPOConfig, PPOTrainer
from ppo.config import DEVICES

DEFAULT_ENV_CONFIG = ROOT / "configs" / "world_v1.yaml"
DEFAULT_PPO_CONFIG = ROOT / "configs" / "ppo.yaml"
DEFAULT_EVAL_EPISODES = 10

GREEN = "\033[32m"
RESET = "\033[0m"

NO_BRAIN_ART = r""" __          __               _______        _
 \ \        / /              |__   __|      (_)
  \ \  /\  / /__  _ __ _ __ ___ | |_ __ __ _ _ _ __   ___ _ __
   \ \/  \/ / _ \| '__| '_ ` _ \| | '__/ _` | | '_ \ / _ \ '__|
    \  /\  / (_) | |  | | | | | | | | | (_| | | | | |  __/ |
     \/  \/ \___/|_|  |_| |_| |_|_|_|  \__,_|_|_| |_|\___|_|

⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣤⠶⠞⠛⠛⠉⠉⠉⠉⠉⠙⠛⢦⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⢀⡴⠞⠉⠀⠀⠀⠆⠀⠀⠀⠀⠀⠀⠀⠀⠐⢹⠀⠀⠀
⠀⠀⠀⠀⠀⠀⢠⡾⠋⠀⠀⠀⠀⢀⣀⣤⣤⠴⠶⠶⠶⠶⠶⡶⡾⠀⠀⠀⠀
⠀⠀⠀⠀⠀⣴⠋⠀⠀⢀⡴⠾⠛⠉⠉⠀⠀⠀⠀⠀⠀⠀⣼⠁⠀⠀⠀⠀⠀
⠀⠀⠀⢀⡼⠁⠀⠀⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡼⠃⠀⠀⠀⠀⠀⠀Train this....
⠀⠀⢀⡼⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⡇⠀⠀⠀⠀⠀⠀⠀
⠀⠀⣼⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀
⠀⢠⡇⠀⠀⠀⠀⠀⠀⠀⠀⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀
⠀⣸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣆⠀⠀⠀⠀⠀⠀⠀⢀⡇⠀⣰⣏🧠⣀⣀
⠀⡿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣆⠀⠀⠀⠀⠀⠀⢸⡇⣠⠏⠉⠛⠛⠋⠁
⢀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠹⡄⠀⠀⠀⠀⠀⢸⣿⠋⠀⠀⠀⠀⠀⠀
⢸⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⡄⠀⠀⠀⠀⣼⡇⠀⠀⠀⠀⠀⠀⠀
⣼⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⡀⠀⠀⣼⠃⣧⠀⠀⠀⠀⠀⠀⠀"""


class _TrainArgumentParser(argparse.ArgumentParser):
    def format_help(self) -> str:
        return f"{GREEN}{NO_BRAIN_ART}{RESET}\n\n{super().format_help()}"


EPILOG = """\
MAKING TRAINING HARDER
----------------------

Curriculum: make one run harder over time
  Enable `curriculum` in the PPO config. Training starts with the *_start
  settings and gradually moves toward the environment config. No command-line
  option is needed.

Resume: continue training in a new run
  Use --resume to load weights from a checkpoint. The new run uses the world
  from --env-config. Use this to change the boundary or metabolism, or to keep
  training after progress has stopped.

  Example:

    uv run python scripts/train.py --env-config configs/world_v1.yaml --name easy
    uv run python scripts/train.py --env-config configs/world_v1.yaml --name hard \\
        --resume experiments/easy_s0/checkpoints/latest.pt

WHEN RESUME WILL NOT WORK
-------------------------

The checkpoint and the new world must use the same observation width. Train
from scratch after changing frame_stack or an observation.include_* setting.
This means you cannot resume a world v1 checkpoint in world v2. The command
stops with an error when the widths do not match.

WATCH A TRAINED WORM
--------------------

  uv run python scripts/watch.py experiments/<run>/checkpoints/best.pt

MORE HELP
---------

  See scripts/README.md for all scripts and options.
"""


def _default(path: Path) -> str | None:
    """Returns the path as a string if it exists, else None.

    Args:
        path: Candidate config file.

    Returns:
        The path, or None to fall back to the dataclass defaults.
    """
    return str(path) if path.exists() else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses the command line.

    Args:
        argv: Arguments without the program name, or None for ``sys.argv``.

    Returns:
        The parsed flags. Overrides left unset are None.
    """
    # Raw, so the line breaks in the docstring and the epilog survive into
    # --help; argparse otherwise reflows both into single paragraphs.
    parser = _TrainArgumentParser(
        description=__doc__,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--env-config",
        default=_default(DEFAULT_ENV_CONFIG),
        help="world config file (default: configs/world_v1.yaml if present)",
    )
    parser.add_argument(
        "--ppo-config",
        default=_default(DEFAULT_PPO_CONFIG),
        help="training config file (default: configs/ppo.yaml if present)",
    )
    parser.add_argument(
        "--name", default=None, help="name for this run (default: from config)"
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--epochs", type=positive_int, default=None, help="number of epochs"
    )
    parser.add_argument(
        "--steps-per-epoch", type=positive_int, default=None, help="steps in each epoch"
    )
    parser.add_argument(
        "--device", default=None, choices=DEVICES, help="device to train on"
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="CHECKPOINT",
        help="start a new run with weights from a checkpoint. The new run uses "
        "--env-config. The observation width must match the checkpoint.",
    )
    parser.add_argument(
        "--reset-optimizers",
        action="store_true",
        help="with --resume, reset the Adam optimizers instead of loading them from the "
        "checkpoint",
    )
    # To add case for conflicted args : if one is present no one can use other
    evaluation = parser.add_mutually_exclusive_group()
    evaluation.add_argument(
        "--eval-episodes",
        type=positive_int,
        default=None,
        help=f"test episodes to run after training (default: {DEFAULT_EVAL_EPISODES})",
    )
    evaluation.add_argument(
        "--no-eval", action="store_true", help="do not test after training"
    )
    return parser.parse_args(argv)


def resolve_config(args: argparse.Namespace) -> PPOConfig:
    """Applies the command-line overrides on top of the PPO YAML.

    Rebuilt from a dict rather than patched in place, so every section is
    validated once, against its final values.

    Args:
        args: Flags from :func:`parse_args`.

    Returns:
        The config the run will use.
    """
    data = PPOConfig.resolve(args.ppo_config).to_dict()
    for section, key, value in (
        ("run", "name", args.name),
        ("run", "seed", args.seed),
        ("run", "device", args.device),
        ("rollout", "epochs", args.epochs),
        ("rollout", "steps_per_epoch", args.steps_per_epoch),
    ):
        if value is not None:
            data[section][key] = value
    return PPOConfig.from_dict(data)


def main(argv: list[str] | None = None) -> None:
    """Trains, then reports a deterministic evaluation.

    Args:
        argv: Arguments without the program name, or None for ``sys.argv``.
    """
    args = parse_args(argv)
    config = resolve_config(args)

    print(
        f"env config: {args.env_config or '<defaults>'}   ppo config: {args.ppo_config or '<defaults>'}"
    )
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
    print(f"{GREEN}{NO_BRAIN_ART}{RESET}")
    run_dir = trainer.run()

    if not args.no_eval:
        summary = trainer.evaluate(episodes=args.eval_episodes or DEFAULT_EVAL_EPISODES)
        print("\ndeterministic evaluation:")
        for key in (
            "lifespan_mean",
            "lifespan_min",
            "lifespan_max",
            "eaten_mean",
            "death_rate",
        ):
            print(f"  {key:16s} {summary[key]:.2f}")
    print(f"\nrun saved to {run_dir}")


if __name__ == "__main__":
    main()
