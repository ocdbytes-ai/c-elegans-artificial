"""The PPO training loop.

Structure follows OpenAI Spinning Up's PPO — collect a fixed-size on-policy
epoch, compute GAE advantages, then take many small clipped policy steps with
KL early stopping, followed by value regression.

Two things are specific to this project and easy to get wrong:

1. **Death and timeout bootstrap differently.** See
   :meth:`ppo.buffer.RolloutBuffer.finish_path`.
2. **The env is wrapped before the policy ever sees it.** Smell is a single
   scalar with no time axis, so frames are stacked; and it is far quieter than
   the heading channels, so observations are normalised. Both live in
   :func:`make_env` and both are saved with the checkpoint, because a policy
   restored without them is being fed a different input distribution than it
   trained on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import (
    FlattenObservation,
    FrameStackObservation,
    NormalizeObservation,
)
from torch.optim import Adam

from envs import ENV_ID
from envs.config import EnvConfig
from envs.episodes import EpisodeAccumulator, EpochStats, run_episodes

from .buffer import RolloutBuffer
from .config import PPOConfig
from .curriculum import Curriculum
from .metrics import RunLogger
from .networks import ActorCritic

# Kept away from the training seeds so evaluation always measures the same
# worlds, and never one the policy has already been trained on.
EVAL_SEED = 10_000


def make_env(
    env_config: EnvConfig | dict[str, Any] | str | Path | None = None,
    ppo_config: PPOConfig | None = None,
    render_mode: str | None = None,
    max_episode_steps: int | None = None,
) -> tuple[gym.Env, NormalizeObservation | None]:
    """Builds the training environment with the wrappers the policy expects.

    Order matters: the raw observation is normalised first so the running
    statistics stay interpretable per sense, then frames are stacked, then
    flattened for the MLP. Stacking first would hand the normaliser k copies of
    every channel.

    Args:
        env_config: An :class:`~envs.config.EnvConfig`, a nested dict, a path to
            a YAML file, or None for the defaults.
        ppo_config: Supplies ``frame_stack`` and ``normalize_observations``.
            None uses the defaults.
        render_mode: Passed through to the environment.
        max_episode_steps: Overrides the registered truncation limit, or None
            to keep it. :data:`envs.UNLIMITED_EPISODE_STEPS` lets an episode run
            until the worm starves, which is for watching rather than training.

    Returns:
        The outermost wrapper and the normaliser. The normaliser is returned
        because its running statistics are learned state that must be saved,
        restored and frozen, and it ends up buried several layers into the
        stack. It is None when ``normalize_observations`` is off.
    """
    ppo_config = ppo_config or PPOConfig()
    env = gym.make(
        ENV_ID,
        config=env_config,
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
    )

    normalizer: NormalizeObservation | None = None
    if ppo_config.rollout.normalize_observations:
        normalizer = NormalizeObservation(env)
        env = normalizer
    if ppo_config.rollout.frame_stack > 1:
        env = FrameStackObservation(env, ppo_config.rollout.frame_stack)
        env = FlattenObservation(env)
    return env, normalizer


class PPOTrainer:
    """Owns the environment, networks and run directory for one training run.

    Attributes:
        target_env_config: The world the policy must ultimately handle, and the
            one saved with checkpoints.
        env_config: The world it trains in right now, which the curriculum walks
            toward the target.
        config: PPO hyperparameters.
        env: The wrapped training environment.
        normalizer: The observation normaliser, or None.
        ac: The actor-critic.
        buffer: One epoch of on-policy storage.
        curriculum: Schedules ``env_config`` toward ``target_env_config``.
        run_dir: Where configs, logs and checkpoints are written.
        epoch: Epochs completed.
        total_steps: Environment steps taken.
    """

    def __init__(
        self,
        env_config: EnvConfig | dict[str, Any] | str | Path | None = None,
        ppo_config: PPOConfig | dict[str, Any] | str | Path | None = None,
    ):
        """Builds the environment, networks and run directory.

        Args:
            env_config: The target world. Accepts a config object, a nested
                dict, a YAML path, or None for the defaults.
            ppo_config: Hyperparameters, in the same forms.
        """
        # target_env_config is the world the policy must ultimately handle, and
        # the one saved with checkpoints. env_config is the world it trains in
        # right now, which the curriculum walks toward the target.
        self.target_env_config = EnvConfig.resolve(env_config)
        self.config = PPOConfig.resolve(ppo_config)
        run, rollout = self.config.run, self.config.rollout

        torch.manual_seed(run.seed)
        np.random.seed(run.seed)
        self.device = torch.device(_resolve_device(run.device))

        self.curriculum = Curriculum(self.config.curriculum, self.target_env_config)
        self.env_config = self.curriculum.starting_world()
        self.env, self.normalizer = make_env(self.env_config, self.config)
        obs_dim = int(np.prod(self.env.observation_space.shape))
        act_dim = int(np.prod(self.env.action_space.shape))

        self.ac = ActorCritic(
            self.env.observation_space, self.env.action_space, self.config.network
        ).to(self.device)
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.config.optim.policy_lr)
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=self.config.optim.value_lr)
        self.buffer = RolloutBuffer(
            obs_dim, act_dim, rollout.steps_per_epoch, rollout.gamma, rollout.gae_lambda
        )

        self.run_dir = _unique_run_dir(Path(run.output_dir), f"{run.name}_s{run.seed}")
        self.run_dir.mkdir(parents=True)
        (self.run_dir / "checkpoints").mkdir()
        # Save both configs up front: a checkpoint whose world you cannot
        # reconstruct is a checkpoint you cannot evaluate. The *target* world is
        # saved, not the current curriculum stage, so evaluating a mid-training
        # checkpoint measures it against the real task.
        self.target_env_config.to_yaml(self.run_dir / "env.yaml")
        self.config.to_yaml(self.run_dir / "ppo.yaml")
        self.logger = RunLogger(self.run_dir)

        self.epoch = 0
        self.total_steps = 0
        self._obs, _ = self.env.reset(seed=run.seed)
        self._episode = EpisodeAccumulator()

    # -- rollout -----------------------------------------------------------

    def _tensor(self, obs: np.ndarray) -> torch.Tensor:
        """Moves an observation onto the training device."""
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device)

    def collect(self) -> EpochStats:
        """Fills the buffer with one epoch of on-policy experience.

        Returns:
            Statistics for every episode that finished inside the epoch. Empty
            when episodes outlast it.
        """
        stats = EpochStats()
        for _ in range(self.config.rollout.steps_per_epoch):
            action, value, logp = self.ac.step(self._tensor(self._obs))
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            self.buffer.store(self._obs, action, reward, value, logp)
            self._episode.add(action, reward, info)
            self._obs = next_obs
            self.total_steps += 1

            epoch_ended = self.buffer.ptr == self.buffer.max_size
            if not (terminated or truncated or epoch_ended):
                continue

            # Death excludes all future reward, so its bootstrap is exactly 0.
            last_val = 0.0 if terminated else self.ac.value(self._tensor(self._obs))
            self.buffer.finish_path(last_val)

            if terminated or truncated:
                stats.add(self._episode.close(died=terminated))
                self._obs, _ = self.env.reset()
                self._episode = EpisodeAccumulator()
        return stats

    # -- update ------------------------------------------------------------

    def _policy_loss(self, data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        """Computes the clipped surrogate loss and its diagnostics.

        Args:
            data: One drained epoch, as returned by ``RolloutBuffer.get``.

        Returns:
            The differentiable loss, and diagnostics that are reported but never
            differentiated.
        """
        clip = self.config.optim.clip_ratio
        pi = self.ac.pi.distribution(data["obs"])
        logp = self.ac.pi.log_prob(pi, data["act"])

        ratio = torch.exp(logp - data["logp"])
        clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * data["adv"]
        surrogate = -torch.min(ratio * data["adv"], clipped).mean()
        entropy = pi.entropy().sum(axis=-1).mean()

        loss = surrogate - self.config.optim.entropy_coef * entropy
        with torch.no_grad():
            diagnostics = {
                # Unbiased but high variance, and can come out negative on a
                # finite sample. Used only for early stopping.
                "kl": float((data["logp"] - logp).mean()),
                "entropy": float(entropy),
                "clip_frac": float(((ratio - 1.0).abs() > clip).float().mean()),
                "surrogate": float(surrogate),
            }
        return loss, diagnostics

    def _value_loss(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        """Computes the critic's mean squared error against rewards-to-go.

        Args:
            data: One drained epoch.

        Returns:
            The scalar loss.
        """
        return ((self.ac.v(data["obs"]) - data["ret"]) ** 2).mean()

    def _clip_grads(self, module: torch.nn.Module) -> None:
        """Clips a module's gradient norm in place, if configured.

        Args:
            module: The module whose gradients to clip.
        """
        if self.config.optim.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(module.parameters(), self.config.optim.max_grad_norm)

    def update(self) -> dict[str, float]:
        """Runs one PPO update over the collected epoch.

        Returns:
            Losses and diagnostics for the progress log.
        """
        data = self.buffer.get(self.device)
        optim = self.config.optim

        with torch.no_grad():
            loss_pi_start = float(self._policy_loss(data)[0])
            loss_v_start = float(self._value_loss(data))
        diagnostics: dict[str, float] = {}

        stop_iter = optim.policy_iters
        # Gradient descent for Actor
        for iteration in range(optim.policy_iters):
            self.pi_optimizer.zero_grad()
            loss, diagnostics = self._policy_loss(data)
            if diagnostics["kl"] > 1.5 * optim.target_kl:
                # The surrogate is a local approximation: from the TRPO bound
                # (Schulman et al. 2015), true improvement equals the surrogate
                # minus an error term growing with the KL to the collecting
                # policy. Past that, maximising it can reduce real performance.
                # Clipping approximates the trust region; this is the backstop.
                stop_iter = iteration
                break
            loss.backward()
            self._clip_grads(self.ac.pi)
            self.pi_optimizer.step()
            self.ac.pi.clamp_log_std()  # exploration floor, applied every step

        # Gradient descent for Critic 
        for _ in range(optim.value_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self._value_loss(data)
            loss_v.backward()
            self._clip_grads(self.ac.v)
            self.vf_optimizer.step()

        with torch.no_grad():
            loss_v_end = float(self._value_loss(data))
        return {
            "loss_pi": loss_pi_start,
            "loss_v": loss_v_start,
            "delta_loss_v": loss_v_end - loss_v_start,
            "stop_iter": stop_iter,
            "value_mean": float(data["ret"].mean()),
            "log_std": float(self.ac.pi.log_std.detach().mean()),
            **diagnostics,
        }

    # -- run ---------------------------------------------------------------

    def run(self) -> Path:
        """Trains for the configured number of epochs.

        Returns:
            The run directory, holding both configs, ``progress.csv`` and the
            checkpoints.
        """
        print(f"run dir: {self.run_dir}")
        print(f"obs {self.env.observation_space.shape}  act {self.env.action_space.shape}")
        try:
            for epoch in range(1, self.config.rollout.epochs + 1):
                self.epoch = epoch
                # Lands on the next env.reset(), so an episode already in
                # progress finishes under the previous epoch's world.
                world_info = self.curriculum.apply(
                    self.env_config, epoch, self.config.rollout.epochs
                )
                stats = self.collect()
                update_info = self.update()

                if epoch % self.config.run.log_every == 0:
                    self.logger.log(
                        {
                            "epoch": epoch,
                            "total_steps": self.total_steps,
                            **world_info,
                            **stats.summary(),
                            **update_info,
                        }
                    )
                if epoch % self.config.run.save_every == 0 or epoch == self.config.rollout.epochs:
                    self.save_checkpoint()
        finally:
            self.env.close()
        return self.run_dir

    def evaluate(self, episodes: int = 10, deterministic: bool = True) -> dict[str, float]:
        """Rolls out the current policy without training.

        Observation statistics are frozen for the duration, so evaluation does
        not shift the normaliser the policy trained against.

        Args:
            episodes: How many episodes to run.
            deterministic: Take the policy mean instead of sampling.

        Returns:
            Aggregate statistics over the episodes.
        """
        frozen = self.normalizer is not None
        if frozen:
            self.normalizer.update_running_mean = False
        try:
            stats = run_episodes(
                self.env,
                lambda obs: self.ac.act(self._tensor(obs), deterministic=deterministic),
                episodes=episodes,
                seed=EVAL_SEED,
            )
            return stats.summary()
        finally:
            if frozen:
                self.normalizer.update_running_mean = True
            # Evaluation left the env mid-stream; restart the training episode.
            self._obs, _ = self.env.reset()
            self._episode = EpisodeAccumulator()

    # -- checkpoints -------------------------------------------------------

    def save_checkpoint(self, name: str | None = None) -> Path:
        """Writes weights, optimiser state, normaliser statistics and configs.

        Both configs are stored so a checkpoint is self-describing and the world
        it trained in cannot drift out from under it when ``configs/`` changes.
        The *target* world is saved rather than the current curriculum stage, so
        a mid-training checkpoint is evaluated against the real task.

        Args:
            name: File name inside ``checkpoints/``. Defaults to the epoch.

        Returns:
            Path to the written checkpoint. A copy is always left at
            ``latest.pt``.
        """
        state = {
            "epoch": self.epoch,
            "total_steps": self.total_steps,
            "model": self.ac.state_dict(),
            "pi_optimizer": self.pi_optimizer.state_dict(),
            "vf_optimizer": self.vf_optimizer.state_dict(),
            "obs_norm": _normalizer_state(self.normalizer),
            # Saved together so a checkpoint is self-describing: the world it
            # trained in cannot drift out from under it when configs/ changes.
            "env_config": self.target_env_config.to_dict(),
            "ppo_config": self.config.to_dict(),
        }
        path = self.run_dir / "checkpoints" / (name or f"epoch_{self.epoch:05d}.pt")
        torch.save(state, path)
        torch.save(state, self.run_dir / "checkpoints" / "latest.pt")
        return path

    def load_checkpoint(self, path: str | Path, reset_optimizers: bool = False) -> None:
        """Continues from a saved policy, possibly into a different world.

        This is the curriculum handoff: train somewhere forgiving, then resume
        the same weights against a harsher config. Only learned state is
        restored; the configs come from this trainer rather than the checkpoint,
        which is what makes changing the world on resume possible.

        Args:
            path: Checkpoint to load.
            reset_optimizers: Drop Adam's moment estimates. Worth doing when the
                new world differs sharply, since those moments describe
                gradients from a task that no longer exists.

        Raises:
            ValueError: If the checkpoint's observation width does not match
                what this run builds.
        """
        state = torch.load(path, map_location=self.device, weights_only=False)

        saved_shape = _observation_shape(state["env_config"], state["ppo_config"])
        if saved_shape != self.env.observation_space.shape:
            raise ValueError(
                f"checkpoint expects observations of shape {saved_shape}, this run builds "
                f"{self.env.observation_space.shape}. frame_stack and include_energy change "
                f"the observation width, so weights from one cannot be loaded into the other."
            )

        self.ac.load_state_dict(state["model"])
        if not reset_optimizers:
            self.pi_optimizer.load_state_dict(state["pi_optimizer"])
            self.vf_optimizer.load_state_dict(state["vf_optimizer"])
        # Carried over as a starting point; NormalizeObservation keeps updating,
        # so the statistics re-adapt to the new world over a few thousand steps.
        _restore_normalizer(self.normalizer, state.get("obs_norm"))
        self.epoch = state.get("epoch", 0)
        self.total_steps = state.get("total_steps", 0)


def load_policy(
    path: str | Path,
    device: str = "cpu",
    render_mode: str | None = None,
    env_overrides: dict[str, Any] | None = None,
    max_episode_steps: int | None = None,
) -> tuple[ActorCritic, gym.Env, dict]:
    """Rebuilds a policy and its world from a checkpoint.

    The environment is reconstructed from the config stored inside the
    checkpoint rather than from ``configs/``, so a retuned config file cannot
    silently change what a saved policy is measured against. Observation
    statistics are restored and frozen, since this is an evaluation path and
    letting them adapt would feed the policy an input distribution it never
    trained on.

    Args:
        path: Checkpoint to load.
        device: Where to place the weights.
        render_mode: Passed through to the environment.
        env_overrides: Nested dict merged over the stored env config, section by
            section. Mainly for loading checkpoints that predate an observation
            channel.
        max_episode_steps: Overrides the registered truncation limit, or None to
            keep it. Unlike the env config this is not pinned to the checkpoint,
            since it bounds the measurement rather than the world.

    Returns:
        The policy, its environment, and the raw checkpoint dict.

    Raises:
        ValueError: If the checkpoint's observation width does not match what
            the current code builds from the same config.
    """
    state = torch.load(path, map_location=device, weights_only=False)

    # Copied, not aliased: `state` is returned to the caller and asserted
    # against elsewhere, so the config it reports must stay as it was saved.
    stored = dict(state["env_config"])
    # Rendering is the one section not pinned to the checkpoint. Nothing in
    # `render` reaches the simulation or the observation, so it cannot change
    # what a policy is measured against — and pinning it would mean watching an
    # old policy through whatever the renderer looked like when it was trained.
    stored.pop("render", None)
    for section, values in (env_overrides or {}).items():
        stored[section] = {**stored.get(section, {}), **values}
    env_config = EnvConfig.from_dict(stored)
    ppo_config = PPOConfig.from_dict(state["ppo_config"])

    env, normalizer = make_env(
        env_config, ppo_config, render_mode=render_mode, max_episode_steps=max_episode_steps
    )
    _restore_normalizer(normalizer, state.get("obs_norm"))
    if normalizer is not None:
        normalizer.update_running_mean = False

    # Checkpoints predate any observation channel added since they were saved,
    # and a stored config that never mentioned the channel picks up today's
    # default. Say so plainly rather than surfacing a raw size mismatch.
    trained_width = int(state["model"]["pi.mu_net.0.weight"].shape[1])
    current_width = int(np.prod(env.observation_space.shape))
    if trained_width != current_width:
        env.close()
        raise ValueError(
            f"this checkpoint was trained on {trained_width}-dim observations; the current "
            f"code builds {current_width} from the same config "
            f"({env.unwrapped.observation_labels} x {ppo_config.rollout.frame_stack} frames). "
            f"An observation channel was added or removed since it was saved, which "
            f"invalidates the weights. To load it anyway, switch the newer channels off, e.g. "
            f'load_policy(path, env_overrides={{"observation": {{"include_touch": False}}}}).'
        )

    ac = ActorCritic(env.observation_space, env.action_space, ppo_config.network).to(device)
    ac.load_state_dict(state["model"])
    ac.eval()
    return ac, env, state


def _observation_shape(env_config: dict[str, Any], ppo_config: dict[str, Any]) -> tuple[int, ...]:
    """Derives the observation shape a pair of stored configs implies.

    Args:
        env_config: Stored environment config, as a dict.
        ppo_config: Stored PPO config, as a dict.

    Returns:
        The observation space shape those configs produce.
    """
    env, _ = make_env(EnvConfig.from_dict(env_config), PPOConfig.from_dict(ppo_config))
    shape = env.observation_space.shape
    env.close()
    return shape


def _normalizer_state(normalizer: NormalizeObservation | None) -> dict[str, Any] | None:
    """Extracts the normaliser's running statistics for a checkpoint.

    Args:
        normalizer: The normaliser, or None.

    Returns:
        Its mean, variance and sample count, or None.
    """
    if normalizer is None:
        return None
    rms = normalizer.obs_rms
    return {"mean": rms.mean.copy(), "var": rms.var.copy(), "count": rms.count}


def _restore_normalizer(
    normalizer: NormalizeObservation | None, state: dict[str, Any] | None
) -> None:
    """Restores running statistics onto a normaliser.

    Args:
        normalizer: The normaliser, or None to do nothing.
        state: Statistics from :func:`_normalizer_state`, or None.
    """
    if normalizer is None or state is None:
        return
    normalizer.obs_rms.mean = np.asarray(state["mean"])
    normalizer.obs_rms.var = np.asarray(state["var"])
    normalizer.obs_rms.count = state["count"]


def _resolve_device(requested: str) -> str:
    """Resolves ``"auto"`` to a concrete torch device string.

    Args:
        requested: A device name, or ``"auto"``.

    Returns:
        The device to use. ``"auto"`` never picks MPS, which loses to CPU on
        networks this small.
    """
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _unique_run_dir(root: Path, stem: str) -> Path:
    """Finds a run directory name that does not already exist.

    Args:
        root: Directory holding all runs.
        stem: Preferred name.

    Returns:
        ``root/stem``, or the first free numbered variant, so a previous run's
        logs are never clobbered.
    """
    candidate = root / stem
    index = 1
    while candidate.exists():
        candidate = root / f"{stem}_{index}"
        index += 1
    return candidate
