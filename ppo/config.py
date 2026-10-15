"""PPO configuration and default values.

``configs/ppo.yaml`` can override any setting. Defaults follow OpenAI Spinning
Up's PPO implementation unless noted otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from common import config_io

ACTIVATIONS = ("tanh", "relu")
DEVICES = ("auto", "cpu", "cuda", "mps")


@dataclass
class NetworkConfig:
    """Actor-critic architecture and initialisation.

    A plain MLP is the right starting point for world v1.

    Attributes:
        hidden_sizes: Widths of the shared hidden-layer shape, used for both the
            policy and the critic.
        activation: One of :data:`ACTIVATIONS`.
        log_std_init: Starting log standard deviation. The default gives sigma
            ~0.61.
        log_std_min: Lowest log standard deviation allowed, or None to disable.
            Sampling is the only exploration PPO has and our network calculates the
            [turn, throttle] and then the separate learned vector of log_std values
            is defined so in order to keep exploration alive we clamp the values to
            a min so that the gaussian distribution of actions sampled remains
            "creative" and not deterministic too soon while training for large
            number of epochs.
        mean_bias_init: Initial bias per action dimension, here
            ``[turn, throttle]``, or None to leave the network's own init alone.
            It is the bias values in the start of the environment. Eg :
            [0.5, 0.0] --> [half throttle, no turns] --> Head straight with slight
            forward throttle
        tanh_mean: Pass the action mean through tanh, holding it inside the
            ``[-1, 1]`` action bounds. If not then we can get the values of
            actions beyond what we expect. Eg : [2.5, ..] but we only accept
            the throttle value of [-1, 1] now we will get some values of throttle
            and the average will shift. So we constraint the values if required.
    """

    hidden_sizes: tuple[int, ...] = (64, 64)
    activation: str = "tanh"
    log_std_init: float = -0.5
    log_std_min: float | None = -1.2
    mean_bias_init: tuple[float, ...] | None = (0.0, 0.5)
    tanh_mean: bool = False

    def __post_init__(self) -> None:
        self.hidden_sizes = tuple(int(size) for size in self.hidden_sizes)
        if self.mean_bias_init is not None:
            self.mean_bias_init = tuple(float(b) for b in self.mean_bias_init)
        if not self.hidden_sizes or any(size < 1 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must be a non-empty list of positive ints")
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}")
        if self.log_std_min is not None and self.log_std_min > self.log_std_init:
            raise ValueError(
                f"log_std_min ({self.log_std_min}) is above log_std_init "
                f"({self.log_std_init}); the policy would start clamped"
            )
        if self.tanh_mean and any(abs(b) >= 1 for b in self.mean_bias_init or ()):
            raise ValueError(
                f"tanh_mean keeps the action mean inside (-1, 1), so mean_bias_init "
                f"{self.mean_bias_init} is unreachable"
            )


@dataclass
class RolloutConfig:
    """How experience is collected and turned into advantages.

    Attributes:
        steps_per_epoch: Transitions gathered before each update.
        epochs: Update rounds in the run.
        gamma: Discount factor. Its horizon ``1/(1-gamma)`` must stay within a
            few multiples of the freeze lifespan, or eating changes ``V(s)`` by
            too little to learn from. (PPO)
        gae_lambda: GAE lambda, trading advantage bias against variance. configures
            how far into the future the PPO looks when deciding the consequence
            of the actions (worse / better) (PPO)
        frame_stack: Observation frames concatenated. Smell alone has no time
            axis, so stacking is what turns "how much" into "more than before"
            based on the sensors.
        normalize_observations: Track running mean and variance per channel.
            ``food_smell`` averages ~0.13 against sin/cos spanning [-1, 1], so
            without this the signal that matters is the quietest input. It is
            like normalising the features in a model or adding layer norms in
            model architecture
    """

    steps_per_epoch: int = 4000
    epochs: int = 300
    gamma: float = 0.99
    gae_lambda: float = 0.97
    frame_stack: int = 4
    normalize_observations: bool = True

    def __post_init__(self) -> None:
        if self.steps_per_epoch < 1 or self.epochs < 1:
            raise ValueError("steps_per_epoch and epochs must be >= 1")
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma must be in (0, 1] and gae_lambda in [0, 1]")
        if self.frame_stack < 1:
            raise ValueError("frame_stack must be >= 1 (1 disables stacking)")


@dataclass
class OptimConfig:
    """The PPO Optimizer update

    Attributes:
        clip_ratio: Surrogate clipping range (PPO).
        policy_lr: Adam learning rate for the policy.
        value_lr: Adam learning rate for the critic.
        policy_iters: Maximum gradient steps on the policy per epoch.
        value_iters: Gradient steps on the critic per epoch.
        target_kl: Policy iterations stop early once the approximate KL exceeds
            1.5x this, past which the clipped surrogate stops approximating the
            true objective. Checks the KL divergence basically if KL exceeds
            given value it starts with fresh experience from next run.
        entropy_coef: Weight on the entropy bonus. A deviation from spinup,
            which computes entropy for logging only. Under a survival reward the
            freeze policy is a strong early attractor for our worm so in order to
            keep just enough exploration this is another knob. Set to 0.0 to match
            spinup exactly. In simple terms it is subtracted from the PPO loss
            by this formula :
            L = L_PPO - entropy_coef * H(pi)
            Here, H(pi) : policy entropy calculated in our implementation
        max_grad_norm: Gradient-norm clip, or 0 to disable.
        anneal_lr: Scale both learning rates linearly from their full value on
            the first epoch to ``1/epochs`` of it on the last, as CleanRL's PPO
            does by default, so late updates cannot undo a good policy. Restarts
            from the full rate on ``--resume``, since each run counts its own
            epochs.
    """

    clip_ratio: float = 0.2
    policy_lr: float = 3e-4
    value_lr: float = 1e-3
    policy_iters: int = 80
    value_iters: int = 80
    target_kl: float = 0.015
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    anneal_lr: bool = False

    def __post_init__(self) -> None:
        if self.clip_ratio <= 0:
            raise ValueError("clip_ratio must be positive")
        if min(self.policy_lr, self.value_lr) <= 0:
            raise ValueError("learning rates must be positive")
        if self.policy_iters < 1 or self.value_iters < 1:
            raise ValueError("policy_iters and value_iters must be >= 1")
        if self.entropy_coef < 0 or self.max_grad_norm < 0:
            raise ValueError("entropy_coef and max_grad_norm must be non-negative")

    def lr_fraction(self, epoch: int, total_epochs: int) -> float:
        """Computes the share of the configured learning rates used this epoch.

        Args:
            epoch: One-based epoch number.
            total_epochs: Total epochs in the run.

        Returns:
            1.0 on the first epoch, falling linearly to ``1/total_epochs`` on
            the last, and 1.0 throughout when annealing is off.
        """
        if not self.anneal_lr:
            return 1.0
        return 1.0 - (epoch - 1) / total_epochs


@dataclass
class CurriculumConfig:
    """Starts the worm in a forgiving world and tightens toward the real one.

    Each ``*_start`` value anneals toward whatever the *env* config specifies:
    the env config is the target, and these only say where training begins.
    None pins a parameter at its target for the whole run.

    Anneal downward only. The ``food_smell`` observation ceiling is fixed when
    the environment is built, from the starting count, so density that rose
    later would be silently clipped.

    Attributes:
        enabled: Whether to anneal at all. When off, the world is left exactly
            as configured.
        anneal_fraction: Fraction of training after which the target world is
            reached and held, so the final policy is trained and evaluated on
            the real thing.
        food_count_start: Starting pellet count, or None to pin at the target.
        eat_radius_start: Starting contact radius, or None to pin.
        scent_radius_start: Starting sensory radius, or None to pin. Raising it
            helps early sensing but merges neighbouring pellets into single
            hills whose summits hold no food.
        metabolism_scale_start: Divides ``basal_cost`` and ``move_cost`` by this
            at the start, annealing to 1.0, or None to pin. Scaling both
            together makes it a pure time dilation that leaves the food
            economics unchanged and keeps basal dominating move, so freezing
            never becomes optimal. 2.0 gave the widest spread of outcomes on the
            starting world (cv 0.49 against 0.32 at 1.0); larger is not better,
            since at 8.0 the spread collapses to 0.09 with 85% of episodes
            hitting the step cap.
    """

    enabled: bool = True
    anneal_fraction: float = 0.7
    food_count_start: int | None = 30
    eat_radius_start: float | None = 1.5
    scent_radius_start: float | None = 6.0
    metabolism_scale_start: float | None = 2.0

    def __post_init__(self) -> None:
        if not 0 < self.anneal_fraction <= 1:
            raise ValueError("anneal_fraction must be in (0, 1]")
        for name in ("food_count_start", "eat_radius_start", "scent_radius_start"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive or null, got {value}")
        if self.metabolism_scale_start is not None and self.metabolism_scale_start < 1:
            raise ValueError(
                "metabolism_scale_start must be >= 1 (it only ever eases the "
                f"metabolism, never tightens it), got {self.metabolism_scale_start}"
            )

    def progress(self, epoch: int, total_epochs: int) -> float:
        """Computes how far along the ramp a given epoch sits.

        Args:
            epoch: One-based epoch number.
            total_epochs: Total epochs in the run.

        Returns:
            0.0 on the first epoch, rising to 1.0 once the target world is
            reached, and 1.0 throughout when the curriculum is disabled.
        """
        if not self.enabled:
            return 1.0
        anneal_until = max(1, round(total_epochs * self.anneal_fraction))
        return min(1.0, (epoch - 1) / anneal_until)


@dataclass
class RunConfig:
    """Where a run lives and how often it records itself.

    Attributes:
        name: Run name, combined with the seed to form the directory.
        seed: Seeds torch, numpy and the environment.
        device: One of :data:`DEVICES`. Networks this small are latency-bound,
            so accelerators rarely beat the CPU.
        output_dir: Directory holding all runs.
        save_every: Epochs between checkpoints.
        log_every: Epochs between progress rows.
        eval_every: Epochs between fixed-world evaluations, or 0 to skip them.
            These are the only trustworthy progress signal in the run: the
            training log is scored on whatever world the curriculum has reached,
            so it can fall while the policy improves. Each evaluation costs
            roughly ``eval_episodes * lifespan`` extra environment steps.
        eval_episodes: Episodes per evaluation, on the target world with fixed
            seeds. Lifespan here has a standard deviation close to its own mean,
            so small samples are noise: 20 gives a standard error of 20-37 steps
            against effects worth acting on of 30-60.
    """

    name: str = "ppo"
    seed: int = 0
    device: str = "cpu"
    output_dir: str = "experiments"
    save_every: int = 10
    log_every: int = 1
    eval_every: int = 25
    eval_episodes: int = 20

    def __post_init__(self) -> None:
        if self.device not in DEVICES:
            raise ValueError(f"device must be one of {DEVICES}")
        if self.save_every < 1 or self.log_every < 1:
            raise ValueError("save_every and log_every must be >= 1")
        if self.eval_every < 0 or self.eval_episodes < 1:
            raise ValueError(
                "eval_every must be >= 0 (0 disables) and eval_episodes >= 1"
            )


@dataclass
class PPOConfig(config_io.ConfigRoot):
    """Root config handed to :class:`ppo.trainer.PPOTrainer`.

    Attributes:
        network: Actor-critic architecture and initialisation.
        rollout: How experience is collected.
        optim: The PPO update.
        curriculum: How the world tightens over the run.
        run: Naming, seeding and checkpoint cadence.
    """

    network: NetworkConfig = field(default_factory=NetworkConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    run: RunConfig = field(default_factory=RunConfig)

    # from_dict / from_yaml / resolve / to_dict / to_yaml come from ConfigRoot.
