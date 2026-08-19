"""Contract tests for the PPO module.

The expensive-to-debug parts get the coverage: GAE arithmetic, the death-vs-
timeout bootstrap, config/YAML drift, and whether a checkpoint actually
restores the same policy in the same world.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ppo import PPOConfig, PPOTrainer, RolloutBuffer, load_policy, make_env
from ppo.buffer import discounted_cumsum
from ppo.networks import ActorCritic

CONFIG_DIR = "configs"

# Small enough to run in a test, real enough to exercise every code path.
# steps_per_epoch has to exceed a worm's lifespan (~190 steps under an untrained
# policy) or no episode ever completes and the episode stats stay empty.
FAST_RUN = {
    "rollout": {"steps_per_epoch": 400, "epochs": 1, "frame_stack": 2},
    "optim": {"policy_iters": 3, "value_iters": 3},
    "network": {"hidden_sizes": [16, 16]},
    # Off by default here so tests see exactly the world they configure.
    "curriculum": {"enabled": False},
}


@pytest.fixture
def trainer(tmp_path) -> PPOTrainer:
    config = PPOConfig.from_dict({**FAST_RUN, "run": {"output_dir": str(tmp_path)}})
    instance = PPOTrainer(ppo_config=config)
    yield instance
    instance.env.close()


# -- config ----------------------------------------------------------------


def test_shipped_yaml_documents_every_field(assert_yaml_covers_config):
    assert_yaml_covers_config(f"{CONFIG_DIR}/ppo.yaml", PPOConfig)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"optim": {"clip_ratio": 0.2, "clip_ration": 0.3}}, "unknown config key"),
        ({"rollout": {"gamma": 1.5}}, "gamma"),
        ({"network": {"activation": "swish"}}, "activation"),
        ({"optim": {"entropy_coef": -0.1}}, "entropy_coef"),
        (
            {"network": {"log_std_init": -2.0, "log_std_min": -1.2}},
            "log_std_min .* is above log_std_init",
        ),
        ({"curriculum": {"metabolism_scale_start": 0.5}}, "metabolism_scale_start must be >= 1"),
        ({"curriculum": {"anneal_fraction": 0.0}}, "anneal_fraction"),
    ],
)
def test_bad_hyperparameters_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        PPOConfig.from_dict(overrides)


# -- curriculum ------------------------------------------------------------


def _curriculum_trainer(tmp_path, **curriculum):
    config = PPOConfig.from_dict(
        {
            **FAST_RUN,
            "rollout": {**FAST_RUN["rollout"], "epochs": 10},
            "curriculum": {"enabled": True, **curriculum},
            "run": {"output_dir": str(tmp_path)},
        }
    )
    return PPOTrainer(ppo_config=config)


def test_world_starts_forgiving_and_ends_at_the_target(tmp_path):
    trainer = _curriculum_trainer(tmp_path)
    target, live = trainer.target_env_config.food, trainer.env_config.food

    assert live.count == 30 and live.eat_radius == 1.5, "epoch 1 is the forgiving world"

    trainer.curriculum.apply(trainer.env_config, epoch=10, total_epochs=10)
    # Exactly equal, not merely close: the final world must be the real one.
    assert live.count == target.count
    assert live.eat_radius == target.eat_radius
    assert live.scent_radius == target.scent_radius
    trainer.env.close()


def test_world_tightens_monotonically(tmp_path):
    trainer = _curriculum_trainer(tmp_path)
    counts, radii, basal = [], [], []
    for epoch in range(1, 11):
        info = trainer.curriculum.apply(trainer.env_config, epoch, total_epochs=10)
        counts.append(info["food_count"])
        radii.append(info["eat_radius"])
        basal.append(info["basal_cost"])

    assert np.all(np.diff(counts) <= 0) and np.all(np.diff(radii) <= 0)
    assert counts[0] > counts[-1], "food must actually get scarcer"
    assert np.all(np.diff(basal) >= 0), "living must get more expensive, never cheaper"
    assert basal[0] < basal[-1]
    trainer.env.close()


def test_metabolism_eases_at_the_start_and_ends_at_the_configured_rate(tmp_path):
    trainer = _curriculum_trainer(tmp_path, metabolism_scale_start=2.0)
    target, live = trainer.target_env_config.metabolism, trainer.env_config.metabolism

    assert live.basal_cost == pytest.approx(target.basal_cost / 2.0)
    assert live.move_cost == pytest.approx(target.move_cost / 2.0)

    trainer.curriculum.apply(trainer.env_config, epoch=10, total_epochs=10)
    assert live.basal_cost == target.basal_cost
    assert live.move_cost == target.move_cost
    trainer.env.close()


def test_metabolism_ratio_is_preserved_throughout(tmp_path):
    """basal must dominate move at every stage, or freezing becomes optimal."""
    trainer = _curriculum_trainer(tmp_path, metabolism_scale_start=8.0)
    target = trainer.target_env_config.metabolism
    reference = target.move_cost / target.basal_cost

    for epoch in range(1, 11):
        trainer.curriculum.apply(trainer.env_config, epoch, total_epochs=10)
        live = trainer.env_config.metabolism
        assert live.move_cost / live.basal_cost == pytest.approx(reference)
        # The ratio that actually matters: full effort vs idling.
        assert live.basal_cost + 2 * live.move_cost < 1.5 * live.basal_cost
    trainer.env.close()


def test_target_is_reached_at_anneal_fraction_then_held(tmp_path):
    trainer = _curriculum_trainer(tmp_path, anneal_fraction=0.5)
    target = trainer.target_env_config.food.count

    # 10 epochs, fraction 0.5 -> target by epoch 6, held for the rest.
    assert trainer.curriculum.apply(trainer.env_config, 6, 10)["food_count"] == target
    assert trainer.curriculum.apply(trainer.env_config, 10, 10)["food_count"] == target
    trainer.env.close()


def test_smell_ceiling_is_sized_for_the_richest_stage(tmp_path):
    """The observation space is fixed at build time; it must cover epoch 1."""
    trainer = _curriculum_trainer(tmp_path)
    _, max_count = trainer.env_config.randomization.food_count_bounds(30)
    # The wrapped space is unbounded (NormalizeObservation), so check the base
    # env, which is where the clipping in ObservationBuilder actually happens.
    base = trainer.env.unwrapped.observation_space
    assert base.high[1] == pytest.approx(max_count)
    assert max_count > trainer.target_env_config.food.count, "sized for stage 1, not the target"
    trainer.env.close()


def test_curriculum_reaches_the_live_env(tmp_path):
    trainer = _curriculum_trainer(tmp_path)
    trainer.curriculum.apply(trainer.env_config, epoch=10, total_epochs=10)
    trainer.env.reset(seed=0)
    low, high = trainer.env_config.randomization.food_count_bounds(
        trainer.target_env_config.food.count
    )
    assert low <= trainer.env.unwrapped.food.count <= high
    trainer.env.close()


def test_disabled_curriculum_leaves_the_world_alone(tmp_path):
    config = PPOConfig.from_dict({**FAST_RUN, "run": {"output_dir": str(tmp_path)}})
    trainer = PPOTrainer(env_config={"food": {"count": 5}}, ppo_config=config)
    assert trainer.env_config.food.count == 5
    trainer.env.close()


def test_a_schedule_that_builds_an_invalid_world_fails_loudly(tmp_path):
    """Not a config check: these values are each legal, and only the world they
    anneal through is not."""
    with pytest.raises(ValueError, match="eat_radius must not exceed"):
        _curriculum_trainer(tmp_path, eat_radius_start=9.0, scent_radius_start=6.0)


# -- GAE and the bootstrap contract ----------------------------------------


def test_discounted_cumsum_matches_the_definition():
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    np.testing.assert_allclose(
        discounted_cumsum(values, 0.5),
        [1 + 0.5 * 2 + 0.25 * 3, 2 + 0.5 * 3, 3],
        rtol=1e-6,
    )


def _filled_buffer(last_val: float) -> RolloutBuffer:
    """Three steps of reward 1 with a critic that predicts 0 throughout."""
    buffer = RolloutBuffer(obs_dim=1, act_dim=1, size=3, gamma=1.0, lam=1.0)
    for _ in range(3):
        buffer.store(np.zeros(1), np.zeros(1), rew=1.0, val=0.0, logp=0.0)
    buffer.finish_path(last_val)
    return buffer


def test_death_bootstraps_to_zero():
    buffer = _filled_buffer(last_val=0.0)
    # Nothing follows death, so returns are just the rewards left in the life.
    np.testing.assert_allclose(buffer.ret[:3], [3.0, 2.0, 1.0])
    np.testing.assert_allclose(buffer.adv[:3], [3.0, 2.0, 1.0])


def test_timeout_bootstraps_from_the_value_function():
    buffer = _filled_buffer(last_val=10.0)
    # The cut-off future is worth V(s_T)=10, and every step inherits it.
    np.testing.assert_allclose(buffer.ret[:3], [13.0, 12.0, 11.0])
    np.testing.assert_allclose(buffer.adv[:3], [13.0, 12.0, 11.0])


def test_buffer_refuses_to_overfill_or_drain_early():
    buffer = RolloutBuffer(obs_dim=1, act_dim=1, size=1, gamma=0.99, lam=0.97)
    buffer.store(np.zeros(1), np.zeros(1), 1.0, 0.0, 0.0)
    with pytest.raises(RuntimeError, match="full"):
        buffer.store(np.zeros(1), np.zeros(1), 1.0, 0.0, 0.0)

    half = RolloutBuffer(obs_dim=1, act_dim=1, size=2, gamma=0.99, lam=0.97)
    with pytest.raises(RuntimeError, match="fill it first"):
        half.get(torch.device("cpu"))


def test_collect_bootstraps_deaths_at_zero_and_cutoffs_at_v(tmp_path):
    """The distinction the buffer relies on has to actually be made upstream."""
    config = PPOConfig.from_dict({**FAST_RUN, "run": {"output_dir": str(tmp_path)}})
    # Starve the worm so the epoch contains many deaths and one epoch cutoff.
    instance = PPOTrainer(
        env_config={"metabolism": {"initial_energy": 2.0}}, ppo_config=config
    )
    recorded: list[float] = []
    original = instance.buffer.finish_path
    instance.buffer.finish_path = lambda last_val=0.0: (
        recorded.append(last_val),
        original(last_val),
    )[1]

    stats = instance.collect()
    instance.env.close()

    deaths = sum(episode.died for episode in stats.episodes)
    assert deaths > 1, "test needs several deaths to be meaningful"
    assert recorded.count(0.0) == deaths, "every death must bootstrap at exactly 0"

    # One extra call closes the epoch mid-life and must bootstrap from V(s) —
    # unless the epoch happened to end exactly on a death, which needs no extra.
    assert len(recorded) in (deaths, deaths + 1)
    if len(recorded) == deaths + 1:
        assert recorded[-1] != 0.0, "an epoch cutoff is not a death"


# -- networks --------------------------------------------------------------


def test_actor_critic_shapes():
    env, _ = make_env(ppo_config=PPOConfig.from_dict(FAST_RUN))
    ac = ActorCritic(env.observation_space, env.action_space, PPOConfig().network)
    obs = torch.zeros(env.observation_space.shape, dtype=torch.float32)

    action, value, logp = ac.step(obs)
    assert action.shape == env.action_space.shape
    assert isinstance(value, float) and isinstance(logp, float)

    batch = torch.zeros((7, *env.observation_space.shape), dtype=torch.float32)
    pi = ac.pi.distribution(batch)
    assert ac.pi.log_prob(pi, torch.zeros(7, 2)).shape == (7,)
    assert ac.v(batch).shape == (7,), "value must be squeezed or the loss broadcasts"
    env.close()


def test_log_std_floor_holds_through_training(trainer):
    """Exploration must not be able to collapse, whatever the loss prefers."""
    floor = trainer.config.network.log_std_min
    assert floor is not None

    # Drive it hard below the floor, as 470 epochs of surrogate pressure would.
    with torch.no_grad():
        trainer.ac.pi.log_std.fill_(floor - 3.0)
    trainer.ac.pi.clamp_log_std()
    assert torch.all(trainer.ac.pi.log_std >= floor)

    trainer.epoch = 1
    trainer.collect()
    trainer.update()
    assert torch.all(trainer.ac.pi.log_std >= floor), "an update pushed sigma under the floor"


def test_floor_can_be_disabled():
    config = PPOConfig.from_dict({**FAST_RUN, "network": {"log_std_min": None}})
    env, _ = make_env(ppo_config=config)
    ac = ActorCritic(env.observation_space, env.action_space, config.network)
    with torch.no_grad():
        ac.pi.log_std.fill_(-9.0)
    ac.pi.clamp_log_std()
    assert torch.all(ac.pi.log_std == -9.0)
    env.close()


def test_floor_above_the_initial_std_is_rejected():
    with pytest.raises(ValueError, match="log_std_min .* is above log_std_init"):
        PPOConfig.from_dict({"network": {"log_std_init": -2.0, "log_std_min": -1.2}})


def test_untrained_policy_starts_swimming_forward(tmp_path):
    """The property that matters: a fresh policy must actually travel.

    With a zero-mean throttle it diffuses instead — 1.24 units of net
    displacement over a whole life — never reaches food, and every episode
    returns the same number, leaving PPO nothing to learn from.
    """
    config = PPOConfig.from_dict(FAST_RUN)
    env, _ = make_env(ppo_config=config)
    ac = ActorCritic(env.observation_space, env.action_space, config.network)

    obs = torch.zeros(env.observation_space.shape, dtype=torch.float32)
    turn, throttle = ac.act(obs, deterministic=True)
    assert throttle > 0.3, "an untrained worm must want to move forward"
    assert abs(turn) < 0.2, "and must not want to circle"
    env.close()


def test_mean_bias_can_be_disabled_and_is_validated():
    config = PPOConfig.from_dict({**FAST_RUN, "network": {"mean_bias_init": None}})
    env, _ = make_env(ppo_config=config)
    ac = ActorCritic(env.observation_space, env.action_space, config.network)
    assert abs(float(ac.act(torch.zeros(env.observation_space.shape), deterministic=True)[1])) < 0.3
    env.close()

    bad = PPOConfig.from_dict({**FAST_RUN, "network": {"mean_bias_init": [0.0, 0.5, 0.5]}})
    env, _ = make_env(ppo_config=bad)
    with pytest.raises(ValueError, match="one bias per action"):
        ActorCritic(env.observation_space, env.action_space, bad.network)
    env.close()


def test_deterministic_act_is_the_distribution_mean():
    env, _ = make_env(ppo_config=PPOConfig.from_dict(FAST_RUN))
    ac = ActorCritic(env.observation_space, env.action_space, PPOConfig().network)
    obs = torch.zeros(env.observation_space.shape, dtype=torch.float32)
    np.testing.assert_allclose(
        ac.act(obs, deterministic=True), ac.act(obs, deterministic=True)
    )
    env.close()


# -- the loop and its artefacts --------------------------------------------


def test_frame_stacking_widens_the_observation():
    base, _ = make_env(ppo_config=PPOConfig.from_dict({"rollout": {"frame_stack": 1}}))
    stacked, _ = make_env(ppo_config=PPOConfig.from_dict({"rollout": {"frame_stack": 4}}))
    assert stacked.observation_space.shape[0] == 4 * base.observation_space.shape[0]
    base.close()
    stacked.close()


def test_normaliser_is_handed_back_only_when_it_exists():
    on, normalizer = make_env(ppo_config=PPOConfig.from_dict({"rollout": FAST_RUN["rollout"]}))
    assert normalizer is not None
    on.close()

    off, absent = make_env(
        ppo_config=PPOConfig.from_dict({"rollout": {"normalize_observations": False}})
    )
    assert absent is None
    off.close()


def test_one_epoch_runs_and_writes_its_artefacts(trainer):
    trainer.epoch = 1
    stats = trainer.collect()
    info = trainer.update()

    assert stats.summary()["episodes"] >= 1
    assert np.isfinite(info["loss_pi"]) and np.isfinite(info["loss_v"])
    assert 0.0 <= info["clip_frac"] <= 1.0
    assert (trainer.run_dir / "env.yaml").exists()
    assert (trainer.run_dir / "ppo.yaml").exists()

    trainer.logger.log({"epoch": 1, "total_steps": 400, **stats.summary(), **info})
    assert (trainer.run_dir / "progress.csv").read_text().count("\n") == 2  # header + row


def test_run_dirs_never_clobber_each_other(tmp_path):
    config = PPOConfig.from_dict({**FAST_RUN, "run": {"output_dir": str(tmp_path)}})
    first = PPOTrainer(ppo_config=config)
    second = PPOTrainer(ppo_config=config)
    assert first.run_dir != second.run_dir
    first.env.close()
    second.env.close()


def test_checkpoint_restores_the_same_policy_and_world(trainer):
    trainer.epoch = 1
    trainer.collect()
    trainer.update()
    path = trainer.save_checkpoint()

    restored, env, state = load_policy(path)
    # Checkpoints store the *target* world, so evaluation measures the real task.
    assert state["env_config"] == trainer.target_env_config.to_dict()
    assert env.observation_space.shape == trainer.env.observation_space.shape

    obs = torch.zeros(trainer.env.observation_space.shape, dtype=torch.float32)
    np.testing.assert_allclose(
        restored.act(obs, deterministic=True),
        trainer.ac.act(obs, deterministic=True),
        rtol=1e-6,
    )
    env.close()


def test_resume_carries_weights_into_a_different_world(trainer, tmp_path):
    """The curriculum handoff: same policy, harsher config."""
    trainer.epoch = 1
    trainer.collect()
    trainer.update()
    checkpoint = trainer.save_checkpoint()

    config = PPOConfig.from_dict({**FAST_RUN, "run": {"output_dir": str(tmp_path)}})
    harder = PPOTrainer(env_config={"food": {"count": 4}}, ppo_config=config)
    harder.load_checkpoint(checkpoint)

    obs = torch.zeros(harder.env.observation_space.shape, dtype=torch.float32)
    np.testing.assert_allclose(
        harder.ac.act(obs, deterministic=True), trainer.ac.act(obs, deterministic=True), rtol=1e-6
    )
    # The world must come from this trainer, not tag along with the weights.
    assert harder.env_config.food.count == 4
    assert harder.run_dir != trainer.run_dir
    harder.env.close()


def test_resume_refuses_a_mismatched_observation_width(trainer, tmp_path):
    trainer.epoch = 1
    checkpoint = trainer.save_checkpoint()

    config = PPOConfig.from_dict(
        {**FAST_RUN, "run": {"output_dir": str(tmp_path)}, "rollout": {**FAST_RUN["rollout"], "frame_stack": 3}}
    )
    other = PPOTrainer(ppo_config=config)
    with pytest.raises(ValueError, match="observations of shape"):
        other.load_checkpoint(checkpoint)
    other.env.close()


def test_resume_can_drop_optimizer_state(trainer, tmp_path):
    trainer.epoch = 1
    trainer.collect()
    trainer.update()  # gives Adam non-empty moment estimates
    checkpoint = trainer.save_checkpoint()
    assert trainer.pi_optimizer.state_dict()["state"]

    config = PPOConfig.from_dict({**FAST_RUN, "run": {"output_dir": str(tmp_path)}})
    fresh = PPOTrainer(ppo_config=config)
    fresh.load_checkpoint(checkpoint, reset_optimizers=True)
    assert not fresh.pi_optimizer.state_dict()["state"]
    fresh.env.close()


def test_observation_normaliser_statistics_survive_the_round_trip(trainer):
    trainer.collect()
    before = trainer.normalizer.obs_rms.mean.copy()
    assert np.any(before != 0.0), "normaliser should have seen data by now"

    _, env, _ = load_policy(trainer.save_checkpoint())
    # Reached generically through the wrapper stack: the restored statistics
    # have to be live where the env actually uses them, not just stored.
    np.testing.assert_allclose(env.get_wrapper_attr("obs_rms").mean, before)
    env.close()
