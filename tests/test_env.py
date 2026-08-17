"""Contract tests for the worm world.

These cover the things that are easy to break silently and expensive to debug
later: the Gymnasium API contract, config/YAML drift, the shape of the scent
field, and the metabolism invariants the plan depends on.
"""

from __future__ import annotations

import os

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import envs  # noqa: F401  (registers the ids)
from envs import geometry
from envs.config import EnvConfig
from envs.worm_world import WormWorldEnv

CONFIG_DIR = "configs"


# -- Gymnasium API ---------------------------------------------------------


def test_passes_gymnasium_env_checker():
    env = WormWorldEnv()
    check_env(env, skip_render_check=True)
    env.close()


def test_registered_id_runs():
    env = gym.make("WormWorld-v1")
    observation, _info = env.reset(seed=0)
    assert env.observation_space.contains(observation)
    for _ in range(50):
        observation, _reward, terminated, truncated, _info = env.step(env.action_space.sample())
        assert env.observation_space.contains(observation)
        if terminated or truncated:
            break
    env.close()


def test_observation_is_smell_only():
    env = WormWorldEnv()
    assert env.observation_labels == ["energy", "food_smell", "heading_sin", "heading_cos"]
    # The worm must never be handed a direction or a distance to food.
    assert not any("dx" in label or "dy" in label or "dist" in label for label in env.observation_labels)


def test_energy_channel_can_be_ablated():
    env = WormWorldEnv(config={"observation": {"include_energy": False}})
    assert env.observation_labels == ["food_smell", "heading_sin", "heading_cos"]
    assert env.observation_space.shape == (3,)


def test_same_seed_gives_identical_trajectories():
    actions = [np.array([0.4, 0.9], dtype=np.float32)] * 40

    def rollout():
        env = WormWorldEnv()
        observations = [env.reset(seed=7)[0]]
        for action in actions:
            observations.append(env.step(action)[0])
        env.close()
        return np.array(observations)

    np.testing.assert_array_equal(rollout(), rollout())


# -- config ----------------------------------------------------------------


def test_shipped_yaml_matches_dataclass_defaults():
    """configs/world_v1.yaml is the documented mirror of the defaults."""
    assert EnvConfig.from_yaml(f"{CONFIG_DIR}/world_v1.yaml") == EnvConfig()


def test_partial_config_overrides_only_what_it_names():
    config = EnvConfig.from_dict({"food": {"scent_radius": 6.0}})
    assert config.food.scent_radius == 6.0
    assert config.food.count == EnvConfig().food.count
    assert config.world == EnvConfig().world


def test_unknown_config_key_is_an_error():
    with pytest.raises(ValueError, match="unknown config key"):
        EnvConfig.from_dict({"food": {"scent_radius": 5.0, "scnet_peak": 2.0}})


def test_invalid_values_are_rejected():
    with pytest.raises(ValueError, match="eat_radius must not exceed"):
        EnvConfig.from_dict({"food": {"eat_radius": 9.0, "scent_radius": 4.0}})
    with pytest.raises(ValueError, match="initial_energy"):
        EnvConfig.from_dict({"metabolism": {"initial_energy": 500.0}})


# -- scent field -----------------------------------------------------------


def _single_pellet_env(**food_overrides) -> WormWorldEnv:
    env = WormWorldEnv(config={"food": {"count": 1, **food_overrides}})
    env.reset(seed=3)
    return env


@pytest.mark.parametrize("profile", ["gaussian", "linear", "inverse_square"])
def test_scent_peaks_at_the_centre_and_decays_outward(profile):
    env = _single_pellet_env(scent_profile=profile)
    pellet = env.food.positions[0]
    direction = np.array([1.0, 0.0])

    radii = np.linspace(0.0, env.config.food.scent_radius, 12)
    samples = env.food.scent_at(pellet + direction * radii[:, None])

    assert samples[0] == pytest.approx(env.config.food.scent_peak, rel=1e-6)
    assert np.all(np.diff(samples) < 0), "scent must fall off monotonically"


def test_scent_is_summed_over_pellets():
    env = WormWorldEnv(config={"food": {"count": 2, "min_spawn_distance": 0.0}})
    env.reset(seed=1)
    env.food.positions[:] = np.array([[10.0, 10.0], [10.5, 10.0]])
    midpoint = np.array([10.25, 10.0])

    one = env.food.scent_at(np.array([[10.0, 10.0]]))[0]
    both = float(env.food.scent_at(midpoint))
    assert both > one * 0.9  # two overlapping pellets build a richer landscape


def test_scent_crosses_the_wrap_seam():
    """On a torus a pellet just past the edge must smell near, not far."""
    env = _single_pellet_env()
    env.food.positions[0] = np.array([0.2, 10.0])
    near_seam = float(env.food.scent_at(np.array([19.8, 10.0])))
    far_away = float(env.food.scent_at(np.array([10.0, 10.0])))
    assert near_seam > far_away


def test_scalar_and_batched_scent_queries_agree():
    env = _single_pellet_env()
    points = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    batched = env.food.scent_at(points)
    assert batched.shape == (3,)
    assert np.shape(env.food.scent_at(points[0])) == ()
    assert float(env.food.scent_at(points[0])) == pytest.approx(batched[0])

    grid = env.food.scent_at(points.reshape(3, 1, 2))
    assert grid.shape == (3, 1)


# -- metabolism and the life loop -----------------------------------------


def test_eating_restores_energy_and_respawns_the_pellet():
    env = _single_pellet_env(min_spawn_distance=0.0)
    env.metabolism.energy = 40.0
    env.food.positions[0] = env.worm.position.copy()
    before = env.food.positions[0].copy()

    _, _, terminated, _, info = env.step(np.zeros(2, dtype=np.float32))

    assert info["food_eaten"] == 1
    assert info["energy"] == pytest.approx(40.0 - env.config.metabolism.basal_cost + 25.0)
    assert not terminated
    assert not np.array_equal(env.food.positions[0], before)


def test_energy_is_capped_at_max():
    env = _single_pellet_env(min_spawn_distance=0.0)
    env.metabolism.energy = env.config.metabolism.max_energy
    env.food.positions[0] = env.worm.position.copy()
    env.step(np.zeros(2, dtype=np.float32))
    assert env.metabolism.energy <= env.config.metabolism.max_energy


def test_starvation_terminates_and_reward_counts_surviving_steps():
    env = WormWorldEnv(config={"food": {"count": 1, "scent_radius": 0.5, "eat_radius": 0.01}})
    env.reset(seed=11)
    env.metabolism.energy = 3 * env.config.metabolism.basal_cost

    rewards = []
    for _ in range(10):
        _, reward, terminated, _, _ = env.step(np.zeros(2, dtype=np.float32))
        rewards.append(reward)
        if terminated:
            break

    assert terminated
    assert rewards[:-1] == [1.0] * (len(rewards) - 1)
    assert rewards[-1] == 0.0, "the step it dies on is not a step it survived"


def test_moving_costs_more_than_idling_but_not_much_more():
    """The ratio that decides whether freezing is the optimal policy."""
    config = EnvConfig()
    idle = config.metabolism.basal_cost
    full_effort = idle + config.metabolism.move_cost * 2.0  # |[1, 1]|^2
    assert full_effort > idle
    assert full_effort < 1.5 * idle


def test_speed_factor_is_a_smooth_ramp_never_zero():
    env = WormWorldEnv()
    env.reset(seed=0)
    factors = []
    for fraction in np.linspace(0.0, 1.0, 51):
        env.metabolism.energy = fraction * env.config.metabolism.max_energy
        factors.append(env.metabolism.speed_factor)

    factors = np.array(factors)
    assert factors.min() >= env.config.metabolism.min_speed_factor
    assert factors.max() == pytest.approx(1.0)
    assert np.all(np.diff(factors) >= -1e-12), "speed must never increase as energy drops"
    assert np.max(np.abs(np.diff(factors))) < 0.1, "no cliffs — that is a death spiral"


def test_starving_worm_still_moves():
    env = WormWorldEnv()
    env.reset(seed=5)
    env.metabolism.energy = 0.01
    start = env.worm.position.copy()
    env.step(np.array([0.0, 1.0], dtype=np.float32))
    assert geometry.distance(start, env.worm.position, env.config.world.size, "wrap") > 0.0


# -- geometry --------------------------------------------------------------


def test_wrap_keeps_the_worm_in_bounds_and_clamp_stops_it():
    for boundary in ("wrap", "clamp"):
        env = WormWorldEnv(config={"world": {"boundary": boundary}})
        env.reset(seed=2)
        for _ in range(200):
            env.step(np.array([0.0, 1.0], dtype=np.float32))
            if env.metabolism.is_dead:
                env.reset()
            assert np.all(env.worm.position >= 0.0)
            assert np.all(env.worm.position <= np.array(env.config.world.size))


def test_rendered_frames_are_black_and_white_only():
    """No colour anywhere: every pixel must have R == G == B."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    env = WormWorldEnv(render_mode="rgb_array")
    env.reset(seed=4)
    for _ in range(15):
        env.step(np.array([0.2, 1.0], dtype=np.float32))

    frame = env.render()
    env.close()

    assert frame.shape == (700, 700, 3)
    assert np.array_equal(frame[..., 0], frame[..., 1])
    assert np.array_equal(frame[..., 1], frame[..., 2])
    assert frame.max() == 255 and frame.min() == 0  # the full range is in use


def test_minimum_image_displacement_takes_the_short_way():
    size = (20.0, 20.0)
    delta = geometry.displacement(np.array([19.0, 10.0]), np.array([1.0, 10.0]), size, "wrap")
    np.testing.assert_allclose(delta, [2.0, 0.0])

    delta = geometry.displacement(np.array([19.0, 10.0]), np.array([1.0, 10.0]), size, "clamp")
    np.testing.assert_allclose(delta, [-18.0, 0.0])
