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

import envs
from envs import geometry
from envs.config import EnvConfig
from envs.episodes import run_episodes
from envs.step_info import StepInfo
from envs.worm_world import WormWorldEnv

CONFIG_DIR = "configs"


# -- Gymnasium API ---------------------------------------------------------


def test_passes_gymnasium_env_checker():
    env = WormWorldEnv()
    check_env(env, skip_render_check=True)
    env.close()


def test_registered_id_runs():
    env = gym.make(envs.ENV_ID)
    observation, _info = env.reset(seed=0)
    assert env.observation_space.contains(observation)
    for _ in range(50):
        observation, _reward, terminated, truncated, _info = env.step(env.action_space.sample())
        assert env.observation_space.contains(observation)
        if terminated or truncated:
            break
    env.close()


FULL_OBSERVATION = ["energy", "food_smell", "touch", "heading_sin", "heading_cos"]


def test_food_is_sensed_only_as_smell():
    env = WormWorldEnv()
    assert env.observation_labels == FULL_OBSERVATION
    # The worm must never be handed a direction or a distance to food.
    assert not any(
        "dx" in label or "dy" in label or "dist" in label for label in env.observation_labels
    )


def test_channels_can_be_ablated():
    env = WormWorldEnv(config={"observation": {"include_energy": False}})
    assert env.observation_labels == ["food_smell", "touch", "heading_sin", "heading_cos"]
    assert env.observation_space.shape == (4,)

    env = WormWorldEnv(config={"observation": {"include_touch": False}})
    assert env.observation_labels == ["energy", "food_smell", "heading_sin", "heading_cos"]
    assert env.observation_space.shape == (4,)


# -- mechanosensation ------------------------------------------------------


def _touch_of(env) -> float:
    return env._observe()[env.observation_labels.index("touch")]


def _walled_env() -> WormWorldEnv:
    """Walls exist only under `clamp` — the dataclass default is `wrap`."""
    env = WormWorldEnv(
        config={"world": {"boundary": "clamp"}, "randomization": {"enabled": False}}
    )
    env.reset(seed=0)
    return env


def test_open_water_registers_no_touch():
    env = _walled_env()
    env.worm.position[:] = [10.0, 10.0]
    env.worm.heading = 0.0
    _, _, _, _, info = env.step(np.array([0.0, 1.0], dtype=np.float32))
    assert info["touch"] == 0.0
    assert _touch_of(env) == 0.0


def test_swimming_into_a_wall_registers_touch():
    """The signal that makes a wall learnable instead of invisible."""
    env = _walled_env()
    env.worm.position[:] = [env.config.world.width, 10.0]  # against the east wall
    env.worm.heading = 0.0  # pushing straight into it

    _, _, _, _, info = env.step(np.array([0.0, 1.0], dtype=np.float32))
    assert info["touch"] == pytest.approx(1.0), "head-on into a wall is fully blocked"
    assert _touch_of(env) == pytest.approx(1.0)


def test_sliding_along_a_wall_registers_partial_touch():
    env = _walled_env()
    env.worm.position[:] = [env.config.world.width, 10.0]
    env.worm.heading = np.pi / 4  # 45 degrees: one component blocked, one free

    env.step(np.array([0.0, 1.0], dtype=np.float32))
    touch = _touch_of(env)
    assert 0.0 < touch < 1.0, f"a glancing collision is partial, got {touch}"


def test_wrap_worlds_never_register_touch():
    """There is nothing to collide with on a torus, seam crossings included."""
    env = WormWorldEnv(config={"world": {"boundary": "wrap"}, "randomization": {"enabled": False}})
    env.reset(seed=0)
    env.worm.position[:] = [env.config.world.width - 0.01, 10.0]
    env.worm.heading = 0.0  # about to cross the seam

    for _ in range(20):
        _, _, _, _, info = env.step(np.array([0.0, 1.0], dtype=np.float32))
        assert info["touch"] == 0.0, "crossing the seam is not a collision"


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


def test_info_dict_matches_the_declared_contract():
    """StepInfo is not enforced at runtime, so assert the real keys against it.

    This is what turns renaming an info field into a test failure instead of a
    KeyError halfway through a rollout, since the consumer
    (envs.episodes.EpisodeAccumulator) reads these keys by name.
    """
    declared = set(StepInfo.__annotations__)
    env = WormWorldEnv()

    _, reset_info = env.reset(seed=0)
    assert set(reset_info) == declared, "reset() info drifted from StepInfo"

    _, _, _, _, step_info = env.step(np.zeros(2, dtype=np.float32))
    assert set(step_info) == declared, "step() info drifted from StepInfo"
    env.close()


def test_episode_statistics_only_read_declared_fields():
    """The consumer must not depend on a key the contract does not promise."""
    consumed = {"energy_intake", "basal_cost", "move_cost", "distance_moved", "food_eaten_total"}
    assert consumed <= set(StepInfo.__annotations__)

    env = WormWorldEnv()
    stats = run_episodes(env, lambda _obs: np.zeros(2, dtype=np.float32), episodes=2, seed=0)
    env.close()
    assert len(stats.episodes) == 2
    assert all(episode.died for episode in stats.episodes)


# -- config ----------------------------------------------------------------


@pytest.mark.parametrize("name", ["world_v1.yaml", "world_v2.yaml"])
def test_shipped_yaml_documents_every_field(assert_yaml_covers_config, name):
    assert_yaml_covers_config(f"{CONFIG_DIR}/{name}", EnvConfig)


def test_partial_config_overrides_only_what_it_names():
    config = EnvConfig.from_dict({"food": {"scent_radius": 6.0}})
    assert config.food.scent_radius == 6.0
    assert config.food.count == EnvConfig().food.count
    assert config.world == EnvConfig().world


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"food": {"scent_radius": 5.0, "scnet_peak": 2.0}}, "unknown config key"),
        ({"food": {"eat_radius": 9.0, "scent_radius": 4.0}}, "eat_radius must not exceed"),
        ({"metabolism": {"initial_energy": 500.0}}, "initial_energy"),
        ({"randomization": {"speed_scale": [1.5, 0.5]}}, "speed_scale low must not exceed high"),
        ({"randomization": {"speed_scale": [0.0, 1.0]}}, "speed_scale low must be positive"),
        ({"randomization": {"turn_rate_scale": 1.5}}, "turn_rate_scale must be a"),
    ],
)
def test_bad_config_is_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        EnvConfig.from_dict(overrides)


# -- domain randomisation --------------------------------------------------


def _episode_actuations(env: WormWorldEnv, episodes: int = 30) -> np.ndarray:
    return np.array(
        [(env.reset(seed=i)[1]["max_speed"], env.reset(seed=i)[1]["max_turn_rate"])
         for i in range(episodes)]
    )


def test_actuation_is_redrawn_each_episode():
    env = WormWorldEnv()
    draws = _episode_actuations(env)
    assert len(np.unique(draws[:, 0])) > 1, "max_speed must vary between episodes"
    assert len(np.unique(draws[:, 1])) > 1, "max_turn_rate must vary between episodes"


def test_actuation_draws_stay_inside_the_configured_band():
    env = WormWorldEnv()
    cfg, rand = env.config.worm, env.config.randomization
    draws = _episode_actuations(env)
    assert np.all(draws[:, 0] >= cfg.max_speed * rand.speed_scale[0])
    assert np.all(draws[:, 0] <= cfg.max_speed * rand.speed_scale[1])
    assert np.all(draws[:, 1] >= cfg.max_turn_rate * rand.turn_rate_scale[0])
    assert np.all(draws[:, 1] <= cfg.max_turn_rate * rand.turn_rate_scale[1])


def test_randomisation_can_be_disabled():
    env = WormWorldEnv(config={"randomization": {"enabled": False}})
    draws = _episode_actuations(env, episodes=5)
    assert np.all(draws[:, 0] == env.config.worm.max_speed)
    assert np.all(draws[:, 1] == env.config.worm.max_turn_rate)
    assert all(env.reset(seed=i)[1]["food_count"] == env.config.food.count for i in range(5))


def test_food_count_is_redrawn_each_episode_within_bounds():
    env = WormWorldEnv()
    low, high = env.config.randomization.food_count_bounds(env.config.food.count)
    counts = [env.reset(seed=i)[1]["food_count"] for i in range(40)]

    assert len(set(counts)) > 1, "pellet count must vary between episodes"
    assert min(counts) >= low and max(counts) <= high

    drawn = env.reset(seed=0)[1]["food_count"]
    assert len(env.food.positions) == drawn, "positions array must match the draw"


def test_smell_bound_covers_the_richest_possible_episode():
    """The space is fixed for the env's lifetime; a rich episode must still fit."""
    env = WormWorldEnv()
    _, max_count = env.config.randomization.food_count_bounds(env.config.food.count)
    assert env.observation_space.high[1] == pytest.approx(
        max_count * env.config.food.scent_peak
    )

    # Stack every pellet of the richest draw on the head: still inside the space.
    seeds = (seed for seed in range(200) if env.reset(seed=seed)[1]["food_count"] == max_count)
    assert next(seeds, None) is not None, f"no seed under 200 drew {max_count} pellets"
    env.food.positions[:] = env.worm.position
    observation = env.step(np.zeros(2, dtype=np.float32))[0]
    assert env.observation_space.contains(observation)


def test_actuation_draw_is_seeded():
    """Randomisation must not cost reproducibility."""
    first = WormWorldEnv().reset(seed=99)[1]
    second = WormWorldEnv().reset(seed=99)[1]
    assert first["max_speed"] == second["max_speed"]
    assert first["max_turn_rate"] == second["max_turn_rate"]


def test_randomisation_is_not_observable():
    """The worm must not be handed its own calibration."""
    env = WormWorldEnv()
    assert env.observation_labels == FULL_OBSERVATION
    assert env.observation_space.shape == (len(FULL_OBSERVATION),)


def test_yaml_lists_become_validated_tuples():
    config = EnvConfig.from_dict({"randomization": {"speed_scale": [0.5, 1.5]}})
    assert config.randomization.speed_scale == (0.5, 1.5)


# -- scent field -----------------------------------------------------------


def _fixed_env(**food_overrides) -> WormWorldEnv:
    """An env with randomisation off, for tests that pin pellets by hand.

    Randomisation redraws the pellet count every reset, so anything that
    assigns into ``food.positions`` has to opt out or the array size shifts
    underneath it.
    """
    return WormWorldEnv(
        config={"food": food_overrides, "randomization": {"enabled": False}}
    )


def _single_pellet_env(**food_overrides) -> WormWorldEnv:
    env = _fixed_env(count=1, **food_overrides)
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
    env = _fixed_env(count=2, min_spawn_distance=0.0)
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


def test_world_can_be_emptied_when_food_does_not_respawn():
    """The env must stay steppable after the last pellet is gone."""
    env = _fixed_env(count=2, min_spawn_distance=0.0, respawn_on_eat=False)
    env.reset(seed=0)
    assert env.food.count == 2

    for expected in (1, 0):
        env.food.positions[0] = env.worm.position.copy()
        _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        assert env.food.count == expected
        assert info["food_count"] == expected

    # Emptied. Stepping on must work, and the sensors must report honestly.
    observation, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert env.observation_space.contains(observation)
    assert info["food_smell"] == 0.0
    assert info["nearest_food_distance"] == float("inf")
    assert env.food.nearest(env.worm.position)[0] == -1


def test_depleting_world_caps_lifespan_by_arithmetic():
    env = _fixed_env(count=3, respawn_on_eat=False)
    env.reset(seed=1)
    metabolism = env.config.metabolism
    cap = (metabolism.initial_energy + 3 * metabolism.food_value) / metabolism.basal_cost

    steps = 0
    while steps < cap + 50:
        _, _, terminated, _, _ = env.step(np.zeros(2, dtype=np.float32))
        steps += 1
        if terminated:
            break
    assert terminated and steps <= cap, "nothing may outlive the food it can eat"


def test_energy_is_capped_at_max():
    env = _single_pellet_env(min_spawn_distance=0.0)
    env.metabolism.energy = env.config.metabolism.max_energy
    env.food.positions[0] = env.worm.position.copy()
    env.step(np.zeros(2, dtype=np.float32))
    assert env.metabolism.energy <= env.config.metabolism.max_energy


def test_starvation_terminates_and_reward_counts_surviving_steps():
    env = _fixed_env(count=1, scent_radius=0.5, eat_radius=0.01)
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
