"""Rollout storage and GAE-Lambda advantage estimation.

Holds one epoch of on-policy experience, sliced into trajectories. The subtle
part is :meth:`RolloutBuffer.finish_path`, where the bootstrap value must
distinguish death from a time limit.
"""

from __future__ import annotations

import numpy as np
import torch


def discounted_cumsum(values: np.ndarray, discount: float) -> np.ndarray:
    """Computes ``out[t] = values[t] + discount * out[t + 1]`` backwards.

    Spinning Up uses ``scipy.signal.lfilter`` for this. A reverse loop gives the
    same result without the dependency, and these arrays are one epoch long.

    Args:
        values: Sequence to accumulate.
        discount: Per-step discount factor.

    Returns:
        The discounted cumulative sums, same shape as ``values``.
    """
    out = np.zeros_like(values)
    running = 0.0
    for index in reversed(range(len(values))):
        running = values[index] + discount * running
        out[index] = running
    return out


class RolloutBuffer:
    """Fixed-size on-policy buffer for one epoch of transitions.

    Attributes:
        obs: Observations, shape ``(size, obs_dim)``.
        act: Actions sampled from the policy, unclipped.
        rew: Rewards received.
        val: Critic estimates at each state.
        logp: Log-probabilities of the stored actions under the collecting
            policy. Frozen at collection time; this is the only record of the
            old policy, and what the importance ratio is measured against.
        adv: GAE advantages, filled in by :meth:`finish_path`.
        ret: Rewards-to-go, the critic's regression targets.
        gamma: Discount factor.
        lam: GAE lambda.
        max_size: Capacity, one epoch of steps.
        ptr: Next free slot.
        path_start: Index where the current trajectory began.
    """

    def __init__(self, obs_dim: int, act_dim: int, size: int, gamma: float, lam: float):
        """Allocates the buffer.

        Args:
            obs_dim: Width of a flattened observation.
            act_dim: Width of an action.
            size: Steps per epoch.
            gamma: Discount factor.
            lam: GAE lambda.
        """
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.act = np.zeros((size, act_dim), dtype=np.float32)
        self.rew = np.zeros(size, dtype=np.float32)
        self.val = np.zeros(size, dtype=np.float32)
        self.logp = np.zeros(size, dtype=np.float32)
        self.adv = np.zeros(size, dtype=np.float32)
        self.ret = np.zeros(size, dtype=np.float32)

        self.gamma, self.lam = gamma, lam
        self.max_size = size
        self.ptr = 0
        self.path_start = 0

    def store(
        self, obs: np.ndarray, act: np.ndarray, rew: float, val: float, logp: float
    ) -> None:
        """Appends one transition.

        Args:
            obs: Observation before the action.
            act: Action taken.
            rew: Reward received.
            val: Critic estimate for ``obs``.
            logp: Log-probability of ``act`` under the collecting policy.

        Raises:
            RuntimeError: If the buffer is already full.
        """
        if self.ptr >= self.max_size:
            raise RuntimeError("rollout buffer is full; call get() before storing more")
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.val[self.ptr] = val
        self.logp[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val: float = 0.0) -> None:
        """Closes the current trajectory and computes its advantages and returns.

        Args:
            last_val: Value of whatever follows the trajectory. Must be ``0.0``
                when the worm died, and ``V(s_T)`` when the trajectory was
                merely cut short by the time limit or the epoch boundary.
                Conflating the two biases the critic, and under a survival
                reward ``V(s)`` is approximately the expected remaining
                lifespan — the headline metric itself. Treating a timeout as
                death teaches the critic that surviving to the cap is worth
                nothing beyond it.
        """
        path = slice(self.path_start, self.ptr)
        rewards = np.append(self.rew[path], last_val)
        values = np.append(self.val[path], last_val)

        # GAE-Lambda: an exponentially weighted average of n-step TD residuals.
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.adv[path] = discounted_cumsum(deltas, self.gamma * self.lam)
        self.ret[path] = discounted_cumsum(rewards, self.gamma)[:-1]

        self.path_start = self.ptr

    def get(self, device: torch.device) -> dict[str, torch.Tensor]:
        """Drains the buffer into tensors, normalising the advantages.

        Args:
            device: Where to place the tensors.

        Returns:
            Keys ``obs``, ``act``, ``ret``, ``adv`` and ``logp``.

        Raises:
            RuntimeError: If the buffer is not full.
        """
        if self.ptr != self.max_size:
            raise RuntimeError(f"buffer holds {self.ptr}/{self.max_size} steps; fill it first")
        self.ptr, self.path_start = 0, 0

        advantages = self.adv
        normalised = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        data = {
            "obs": self.obs,
            "act": self.act,
            "ret": self.ret,
            "adv": normalised,
            "logp": self.logp,
        }
        return {
            key: torch.as_tensor(value, dtype=torch.float32, device=device)
            for key, value in data.items()
        }
