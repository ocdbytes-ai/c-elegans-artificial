"""Watch the world, or measure a baseline in it.

    uv run python scripts/play.py --policy greedy --render
    uv run python scripts/play.py --policy freeze --episodes 20

The three policies here are the baselines a trained worm has to beat
(PROJECT_PLAN.md §3.6). ``freeze`` is the important one: it is the local
optimum PPO will find if the metabolism is mistuned, so if a trained policy
cannot beat it, the problem is the config, not the algorithm.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym

import envs  # noqa: F401  (registers the ids)
from envs.geometry import wrap_angle


def random_policy(env: gym.Env, _obs: np.ndarray) -> np.ndarray:
    return env.action_space.sample()


def freeze_policy(_env: gym.Env, _obs: np.ndarray) -> np.ndarray:
    """Do nothing and coast on the initial energy. The bar to beat."""
    return np.zeros(2, dtype=np.float32)


def greedy_policy(env: gym.Env, _obs: np.ndarray) -> np.ndarray:
    """Cheat: read the true nearest pellet out of the sim and steer at it.

    This deliberately uses information the worm cannot sense — it never appears
    in an observation. It is the upper reference for "how long could you live
    if chemotaxis were already solved".
    """
    world = env.unwrapped
    _, delta, _ = world.food.nearest(world.worm.position)
    error = wrap_angle(np.arctan2(delta[1], delta[0]) - world.worm.heading)
    max_turn_per_step = world.config.worm.max_turn_rate * world.config.world.dt
    return np.array([np.clip(error / max_turn_per_step, -1.0, 1.0), 1.0], dtype=np.float32)


POLICIES = {"random": random_policy, "freeze": freeze_policy, "greedy": greedy_policy}


def run_episode(env: gym.Env, policy, seed: int | None) -> dict[str, float]:
    observation, info = env.reset(seed=seed)
    total_reward, steps = 0.0, 0
    while True:
        observation, reward, terminated, truncated, info = env.step(policy(env, observation))
        total_reward += reward
        steps += 1
        if terminated or truncated:
            return {
                "steps": steps,
                "return": total_reward,
                "eaten": info["food_eaten_total"],
                "death": "starved" if terminated else "timeout",
            }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="random", choices=sorted(POLICIES))
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_argument("--render", action="store_true", help="open a pygame window")
    args = parser.parse_args()

    env = gym.make(
        "WormWorld-v1",
        config=args.config,
        render_mode="human" if args.render else None,
    )
    print(f"obs={env.unwrapped.observation_labels}  policy={args.policy}")

    results = []
    for episode in range(args.episodes):
        result = run_episode(env, POLICIES[args.policy], seed=args.seed + episode)
        results.append(result)
        print(
            f"  episode {episode:>3}  steps {result['steps']:>5}  "
            f"return {result['return']:>7.1f}  eaten {result['eaten']:>3}  ({result['death']})"
        )
    env.close()

    lifespans = np.array([r["steps"] for r in results], dtype=np.float64)
    eaten = np.array([r["eaten"] for r in results], dtype=np.float64)
    print(
        f"\nmean lifespan {lifespans.mean():.1f} +/- {lifespans.std():.1f} steps"
        f"   mean eaten {eaten.mean():.2f}"
    )


if __name__ == "__main__":
    main()
