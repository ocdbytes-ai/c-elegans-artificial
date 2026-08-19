"""The actor-critic network.

A plain MLP with a Gaussian policy, matching Spinning Up's ``MLPActorCritic``:
the policy standard deviation is a state-independent learned parameter rather
than a network output, and actions are sampled unsquashed, leaving the
environment to clip them to its own bounds.

Kept faithful to that reference so a misbehaving run can be compared against a
known-good implementation. Orthogonal initialisation with a small policy-head
gain is a common improvement and is deliberately not applied, so the baseline
stays comparable.
"""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces
from torch import nn
from torch.distributions import Normal

from .config import NetworkConfig

_ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU}


def build_mlp(sizes: list[int], activation: type[nn.Module]) -> nn.Sequential:
    """Builds a stack of linear layers with a linear output.

    Args:
        sizes: Layer widths, from input to output.
        activation: Activation inserted between layers, but not after the last.

    Returns:
        The assembled network.
    """
    layers: list[nn.Module] = []
    for index in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[index], sizes[index + 1]))
        if index < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    """Diagonal Gaussian policy with a learned, state-independent std.

    Because the std does not depend on the observation, the policy can modulate
    how hard it turns but not how randomly. That makes klinotaxis expressible
    and klinokinesis, which works by modulating the *rate* of random
    reorientation, reachable only as a chaotic proxy.

    Attributes:
        mu_net: Maps an observation to the action mean.
        log_std: Log standard deviation, one entry per action dimension.
        log_std_min: Floor applied by :meth:`clamp_log_std`, or None.
    """

    def __init__(self, obs_dim: int, act_dim: int, config: NetworkConfig):
        """Builds the policy and applies the configured initial biases.

        Args:
            obs_dim: Width of a flattened observation.
            act_dim: Width of an action.
            config: Network hyperparameters.

        Raises:
            ValueError: If ``config.mean_bias_init`` does not have one entry per
                action dimension.
        """
        super().__init__()
        activation = _ACTIVATIONS[config.activation]
        self.mu_net = build_mlp([obs_dim, *config.hidden_sizes, act_dim], activation)
        self.log_std = nn.Parameter(
            torch.full((act_dim,), float(config.log_std_init), dtype=torch.float32)
        )
        self.log_std_min = config.log_std_min

        if config.mean_bias_init is not None:
            bias = torch.tensor(config.mean_bias_init, dtype=torch.float32)
            if bias.shape != (act_dim,):
                raise ValueError(
                    f"network.mean_bias_init has {tuple(bias.shape)} entries but the action "
                    f"space has {act_dim} dimensions; give one bias per action."
                )
            # A starting point only. Training moves it like any other weight.
            with torch.no_grad():
                self.mu_net[-1].bias.copy_(bias)

    def distribution(self, obs: torch.Tensor) -> Normal:
        """Returns the action distribution for an observation.

        Args:
            obs: Observation, batched or single.

        Returns:
            A diagonal Gaussian over actions.
        """
        return Normal(self.mu_net(obs), torch.exp(self.log_std))

    @torch.no_grad()
    def clamp_log_std(self) -> None:
        """Holds the policy std above its floor. Call after every policy step.

        Applied in place on the parameter rather than inside
        :meth:`distribution`. Clamping the forward pass would zero the gradient
        below the floor, so a parameter that had already drifted under it could
        never climb back: it would report the floor forever while the underlying
        value kept sinking. Clamping the parameter keeps it genuinely in range,
        so the entropy bonus can still raise it when exploration is worth more.
        """
        if self.log_std_min is not None:
            self.log_std.clamp_(min=self.log_std_min)

    def log_prob(self, pi: Normal, action: torch.Tensor) -> torch.Tensor:
        """Computes the log-probability of an action.

        Args:
            pi: Distribution from :meth:`distribution`.
            action: Action to score.

        Returns:
            Log-probability, summed over action dimensions because torch's
            ``Normal`` reports it per component.
        """
        return pi.log_prob(action).sum(axis=-1)


class Critic(nn.Module):
    """State-value function ``V(s)``.

    Attributes:
        v_net: Maps an observation to a scalar value.
    """

    def __init__(self, obs_dim: int, config: NetworkConfig):
        """Builds the critic.

        Args:
            obs_dim: Width of a flattened observation.
            config: Network hyperparameters.
        """
        super().__init__()
        activation = _ACTIVATIONS[config.activation]
        self.v_net = build_mlp([obs_dim, *config.hidden_sizes, 1], activation)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Estimates the value of an observation.

        Args:
            obs: Observation, batched or single.

        Returns:
            Value estimates with the trailing dimension squeezed away, without
            which the value loss would silently broadcast.
        """
        return torch.squeeze(self.v_net(obs), -1)


class ActorCritic(nn.Module):
    """Policy and value function sharing an observation but no weights.

    Attributes:
        pi: The policy.
        v: The critic.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        config: NetworkConfig,
    ):
        """Builds both heads to match the given spaces.

        Args:
            observation_space: The environment's observation space.
            action_space: The environment's action space.
            config: Network hyperparameters.
        """
        super().__init__()
        obs_dim = int(np.prod(observation_space.shape))
        act_dim = int(np.prod(action_space.shape))
        self.pi = GaussianActor(obs_dim, act_dim, config)
        self.v = Critic(obs_dim, config)

    @torch.no_grad()
    def step(self, obs: torch.Tensor) -> tuple[np.ndarray, float, float]:
        """Samples an action and reports what the buffer needs to store.

        Args:
            obs: A single observation.

        Returns:
            ``(action, value, log_prob)``. The action is unclipped; the
            environment applies its own bounds.
        """
        pi = self.pi.distribution(obs)
        action = pi.sample()
        return (
            action.cpu().numpy(),
            float(self.v(obs)),
            float(self.pi.log_prob(pi, action)),
        )

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        """Chooses an action without the extras needed for training.

        Args:
            obs: A single observation.
            deterministic: Take the distribution's mean instead of sampling.

        Returns:
            The chosen action.
        """
        pi = self.pi.distribution(obs)
        return (pi.mean if deterministic else pi.sample()).cpu().numpy()

    @torch.no_grad()
    def value(self, obs: torch.Tensor) -> float:
        """Estimates the value of a single observation.

        Args:
            obs: A single observation.

        Returns:
            The critic's estimate.
        """
        return float(self.v(obs))
