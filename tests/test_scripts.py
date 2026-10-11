"""Command-line contracts of the scripts in ``scripts/``."""

from __future__ import annotations

import pytest

from ppo import PPOConfig
from scripts import play, train, watch


@pytest.mark.parametrize(
    ("parse", "argv"),
    [
        # Each repeats the flag's default value, which argparse's mutual
        # exclusion check would ignore if the parser's default were that value.
        (play.parse_args, ["--loop", "--episodes", str(play.DEFAULT_EPISODES)]),
        (watch.parse_args, ["x.pt", "--loop", "--episodes", str(watch.DEFAULT_EPISODES)]),
        (train.parse_args, ["--no-eval", "--eval-episodes", str(train.DEFAULT_EVAL_EPISODES)]),
        (watch.parse_args, ["x.pt", "--max-steps", "10", "--no-step-limit"]),
        # 0 is a typo, never "forever" or "no limit".
        (play.parse_args, ["--episodes", "0"]),
        (play.parse_args, ["--max-steps", "0"]),
        (train.parse_args, ["--epochs", "0"]),
    ],
)
def test_bad_flags_are_rejected(parse, argv):
    with pytest.raises(SystemExit):
        parse(argv)


def test_train_flags_override_only_what_they_name():
    config = train.resolve_config(train.parse_args(["--epochs", "7", "--name", "x"]))
    yaml = PPOConfig.resolve(train.DEFAULT_PPO_CONFIG)

    assert (config.rollout.epochs, config.run.name) == (7, "x")
    assert config.rollout.steps_per_epoch == yaml.rollout.steps_per_epoch
    assert config.network == yaml.network
