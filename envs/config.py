"""Configuration schema for the worm world.

Every tunable constant lives here (and in ``configs/*.yaml``)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from common import config_io

BOUNDARIES = ("wrap", "clamp")
SCENT_PROFILES = ("gaussian", "linear", "inverse_square")


@dataclass
class WorldConfig:
    """The arena. Units are arbitrary, roughly body lengths.

    Attributes:
        width: Arena width in world units.
        height: Arena height in world units.
        dt: Seconds of simulated time per step.
        boundary: ``"wrap"`` for a torus with no walls, or ``"clamp"`` for hard
            edges. The choice decides whether blind straight-line sweeping is a
            viable foraging strategy; see ``notes/training.md``.
    """

    width: float = 20.0
    height: float = 20.0
    dt: float = 0.1
    boundary: str = "wrap"

    def __post_init__(self) -> None:
        if self.boundary not in BOUNDARIES:
            raise ValueError(f"boundary must be one of {BOUNDARIES}, got {self.boundary!r}")
        if min(self.width, self.height, self.dt) <= 0:
            raise ValueError("width, height and dt must be positive")

    @property
    def size(self) -> tuple[float, float]:
        """The arena as ``(width, height)``."""
        return (self.width, self.height)

    @property
    def diagonal(self) -> float:
        """Length of the arena's diagonal."""
        return float((self.width**2 + self.height**2) ** 0.5)


@dataclass
class WormConfig:
    """The body. For world v1 it is a single point with a heading.

    Attributes:
        max_speed: World units per second at full throttle.
        max_turn_rate: Radians per second at full turn.
        allow_reverse: Whether negative throttle is honoured. C. elegans
            reverses as part of the pirouette.
        radius: Drawn size only; eating uses ``food.eat_radius``.
    """

    max_speed: float = 2.0
    max_turn_rate: float = 3.0
    allow_reverse: bool = True
    radius: float = 0.25

    def __post_init__(self) -> None:
        if min(self.max_speed, self.max_turn_rate, self.radius) <= 0:
            raise ValueError("max_speed, max_turn_rate and radius must be positive")


@dataclass
class RandomizationConfig:
    """Per-episode domain randomisation, redrawn on every reset.

    Every entry is a ``[low, high]`` multiplier on the corresponding nominal
    value, so the base config stays the calibration and these express the spread
    around it. The worm is never told what it drew, which is the point: a policy
    trained on one exact calibration bakes in a fixed relationship between how
    hard it turns and how far it travels, and overshoots silently once that
    relationship changes.

    Attributes:
        enabled: Whether to randomise at all. When off, every draw returns the
            nominal value.
        speed_scale: Multiplier range on ``worm.max_speed``.
        turn_rate_scale: Multiplier range on ``worm.max_turn_rate``.
        food_count_scale: Multiplier range on ``food.count``. Unlike the
            actuation draws this one is sensable, since a denser world smells
            stronger everywhere, so the worm can in principle search differently
            when food is scarce.
    """

    enabled: bool = True
    speed_scale: tuple[float, float] = (0.75, 1.25)
    turn_rate_scale: tuple[float, float] = (0.75, 1.25)
    food_count_scale: tuple[float, float] = (0.5, 1.5)

    def __post_init__(self) -> None:
        self.speed_scale = config_io.validate_range("speed_scale", self.speed_scale)
        self.turn_rate_scale = config_io.validate_range("turn_rate_scale", self.turn_rate_scale)
        self.food_count_scale = config_io.validate_range(
            "food_count_scale", self.food_count_scale
        )

    def food_count_bounds(self, nominal: int) -> tuple[int, int]:
        """Computes the smallest and largest pellet count an episode can draw.

        Args:
            nominal: The configured pellet count.

        Returns:
            ``(low, high)``. The upper bound is what the ``food_smell``
            observation must be sized for, since the space is fixed for the
            environment's lifetime and has to cover the richest world the worm
            could wake up in.
        """
        if not self.enabled:
            return (nominal, nominal)
        low = max(1, round(nominal * self.food_count_scale[0]))
        high = max(low, round(nominal * self.food_count_scale[1]))
        return (low, high)


@dataclass
class FoodConfig:
    """Pellets and the scent they emit.

    Scent is a scalar field peaking at a pellet's centre and falling off with
    distance, so ``scent_radius`` is the sensory footprint while ``eat_radius``
    is the much smaller contact one.

    Attributes:
        count: Nominal pellet count, before per-episode randomisation.
        eat_radius: Contact radius. Touching a pellet eats it.
        scent_radius: Sensory scale of the falloff. Pellets closer together than
            twice the profile's sigma merge into a single hill whose summit
            holds no food; see ``notes/training.md``.
        scent_peak: Concentration at a pellet's exact centre.
        scent_profile: One of :data:`SCENT_PROFILES`; see :mod:`envs.scent`.
        gaussian_sigma_scale: Gaussian sigma as a fraction of ``scent_radius``.
        respawn_on_eat: True makes an eaten pellet reappear elsewhere, holding
            density constant so the world can sustain a worm indefinitely. False
            depletes the world, capping lifespan at
            ``(initial_energy + count * food_value) / basal_cost`` regardless of
            policy, and forcing an exploration/exploitation trade-off.
        min_spawn_distance: Clearance kept between new pellets and the worm.
    """

    count: int = 8
    eat_radius: float = 0.4
    scent_radius: float = 4.0
    scent_peak: float = 1.0
    scent_profile: str = "gaussian"
    gaussian_sigma_scale: float = 0.35
    respawn_on_eat: bool = True
    min_spawn_distance: float = 2.0

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError("food.count must be >= 1")
        if self.scent_profile not in SCENT_PROFILES:
            raise ValueError(
                f"scent_profile must be one of {SCENT_PROFILES}, got {self.scent_profile!r}"
            )
        if self.eat_radius <= 0 or self.scent_radius <= 0:
            raise ValueError("eat_radius and scent_radius must be positive")
        if self.eat_radius > self.scent_radius:
            raise ValueError("eat_radius must not exceed scent_radius (nothing to follow)")


@dataclass
class ToxinConfig:
    """Hazards that emit a scent and cost energy to be near.

    Structurally food's mirror image, with one deliberate difference: a toxin is
    never touched or consumed. Damage is a *dose*, proportional to the
    concentration at the worm's head, so harm is graded rather than binary. Two
    reasons. It arrives in the same step as the action that caused it, so it
    sits well inside the credit-assignment window; and it leaves a gradient to
    descend, where a contact hazard would be an invisible cliff edge.

    Attributes:
        count: Toxin sources in the world. 0 disables them entirely, which is
            the default and leaves the world exactly as world v1 describes it.
        damage: Energy lost per step at full concentration, scaled by the
            concentration actually sensed. Judge it against
            ``metabolism.basal_cost``: at 1.5 against a basal of 0.2, sitting on
            a source is roughly 8x the cost of existing.
        scent_radius: Sensory scale of the falloff.
        scent_peak: Concentration at a source's exact centre.
        scent_profile: One of :data:`SCENT_PROFILES`; see :mod:`envs.scent`.
        gaussian_sigma_scale: Gaussian sigma as a fraction of ``scent_radius``.
        min_spawn_distance: Clearance kept between a source and the worm at
            reset, so no episode opens already inside one.
    """

    count: int = 0
    damage: float = 1.5
    scent_radius: float = 4.0
    scent_peak: float = 1.0
    scent_profile: str = "gaussian"
    gaussian_sigma_scale: float = 0.35
    min_spawn_distance: float = 3.0

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("toxin.count must be >= 0 (0 disables toxins)")
        if self.scent_profile not in SCENT_PROFILES:
            raise ValueError(
                f"scent_profile must be one of {SCENT_PROFILES}, got {self.scent_profile!r}"
            )
        if self.scent_radius <= 0:
            raise ValueError("toxin.scent_radius must be positive")
        if self.damage < 0:
            raise ValueError("toxin.damage is subtracted; give it a positive magnitude")

    @property
    def enabled(self) -> bool:
        """Whether any toxin exists in the world."""
        return self.count > 0


@dataclass
class MetabolismConfig:
    """Energy bookkeeping, and the rates that decide what strategy pays.

    ``basal_cost`` must dominate ``move_cost``: if moving cost meaningfully more
    than idling, freezing would maximise lifespan and training would converge
    there instead of on foraging.

    Attributes:
        max_energy: Cap on stored energy.
        initial_energy: Energy at reset. With ``basal_cost`` this sets the
            freeze baseline, ``initial_energy / basal_cost``, which every policy
            must beat.
        basal_cost: Unconditional cost per step.
        move_cost: Cost per step, multiplied by ``|action|^2``.
        food_value: Energy per pellet eaten.
        min_speed_factor: Speed floor for a starving worm. Never 0, which would
            make low-energy states unrecoverable.
        speed_knee: Energy fraction above which speed is unimpaired.
    """

    max_energy: float = 100.0
    initial_energy: float = 50.0
    basal_cost: float = 0.25
    move_cost: float = 0.02
    food_value: float = 25.0
    min_speed_factor: float = 0.35
    speed_knee: float = 0.25

    def __post_init__(self) -> None:
        if self.max_energy <= 0:
            raise ValueError("max_energy must be positive")
        if not 0 < self.initial_energy <= self.max_energy:
            raise ValueError("initial_energy must be in (0, max_energy]")
        if not 0 < self.min_speed_factor <= 1:
            raise ValueError("min_speed_factor must be in (0, 1] — 0 is a death spiral")
        if not 0 < self.speed_knee <= 1:
            raise ValueError("speed_knee must be in (0, 1]")
        if self.basal_cost <= 0:
            raise ValueError("basal_cost must be positive — existing is not free")


@dataclass
class RewardConfig:
    """Survival is the only reward. Resist adding sub-rewards here.

    Attributes:
        survival: Reward per surviving step. The whole reward function.
        death_penalty: Subtracted on the terminal step. Optional sharpening at
            most: death already forfeits all future reward, and when every
            episode ends in death this is a constant that advantage
            normalisation removes exactly.
    """

    survival: float = 1.0
    death_penalty: float = 0.0

    def __post_init__(self) -> None:
        if self.death_penalty < 0:
            raise ValueError("death_penalty is subtracted; give it a positive magnitude")


@dataclass
class ObservationConfig:
    """Which optional channels the worm senses.

    Food is always a single scalar concentration at the head, so chemotaxis must
    be learned rather than handed over. See :mod:`envs.observations`.

    Attributes:
        include_energy: Interoception, required for hunger-modulated behaviour.
        include_toxin: A second chemical sense, reporting toxin concentration on
            its own channel. Separate from ``food_smell`` because a summed or
            signed channel cannot be disentangled: summed, food-beside-toxin is
            numerically identical to a strong food source; signed, the two
            cancel and the most dangerous spot in the arena reads as open water.
            A separate line supplies discriminability while leaving valence to
            be learned, which is also how the animal is wired. Constant 0 when
            ``toxin.count`` is 0.
        include_touch: Mechanosensation, reporting how much of the last step's
            commanded motion the world refused. Under ``boundary: clamp`` a wall
            is otherwise invisible: the worm cannot tell it has stopped moving,
            so it cannot learn to turn away while it burns ``move_cost`` pushing
            into stone. Constant 0 under ``boundary: wrap``.
    """

    include_energy: bool = True
    include_touch: bool = True
    include_toxin: bool = False


@dataclass
class RenderConfig:
    """How the world is drawn.

    The worm's drawn body is decoration over a point-mass simulation: the
    undulating shape follows the path the point actually took, but nothing here
    reaches the physics or the observation. See :mod:`envs.rendering`.

    Two looks are available. The default is clean: small pellet markers, no
    scent, just the worm. ``debug`` switches to a truthful one that draws the
    quantities the simulation actually uses, at the size it actually uses them.

    Attributes:
        window_size: Pixels along the world's longer axis.
        fps: Frame-rate cap under ``human`` render mode.
        debug: Draw the world as the simulation sees it: pellets at their true
            ``food.eat_radius``, the scent contours on, and a marker on the
            point the physics actually integrates. Overrides
            ``show_scent_field`` and ``food_dot_scale``, which are the clean
            look's settings.
        show_scent_field: Whether to draw the scent contours at all. Off by
            default: being per-pellet they cannot show the merged summits that
            actually matter, so they clutter more than they explain.
        scent_contours: Dotted iso-concentration rings drawn per pellet.
        contour_dot_spacing: Target pixels between dots along a ring.
        food_dot_scale: Pellet disc as a fraction of ``food.eat_radius``. Below
            1.0 the drawn pellet is smaller than the radius that actually eats
            it, trading a truthful contact zone for a less cluttered frame.
        worm_body_length: Drawn body length in world units, trailing back from
            the simulated point.
        worm_wave_amplitude: Peak lateral displacement of the undulation, in
            world units.
        worm_wavelength: Undulation wavelength along the body, in world units.
    """

    window_size: int = 700
    fps: int = 30
    debug: bool = False
    show_scent_field: bool = False
    scent_contours: int = 5
    contour_dot_spacing: float = 12.0
    food_dot_scale: float = 0.3
    worm_body_length: float = 2.5
    worm_wave_amplitude: float = 0.35
    worm_wavelength: float = 1.6

    def __post_init__(self) -> None:
        if self.window_size < 64 or self.fps < 1:
            raise ValueError("window_size >= 64 and fps >= 1 required")
        if self.scent_contours < 1:
            raise ValueError("scent_contours must be >= 1")
        if self.contour_dot_spacing < 2:
            raise ValueError("contour_dot_spacing must be >= 2 pixels")
        if not 0 < self.food_dot_scale <= 1:
            raise ValueError("food_dot_scale must be in (0, 1]")
        if min(self.worm_body_length, self.worm_wavelength) <= 0:
            raise ValueError("worm_body_length and worm_wavelength must be positive")
        if self.worm_wave_amplitude < 0:
            raise ValueError("worm_wave_amplitude must be non-negative (0 draws it straight)")


@dataclass
class EnvConfig(config_io.ConfigRoot):
    """Root config handed to :class:`envs.worm_world.WormWorldEnv`.

    Attributes:
        world: Arena dimensions and boundary convention.
        worm: Nominal body parameters.
        randomization: Per-episode spread on body and food count.
        food: Pellets and the scent they emit.
        toxin: Hazards and the scent they emit.
        metabolism: Energy rates and limits.
        reward: The reward function.
        observation: Which optional channels the worm senses.
        render: How the world is drawn.
    """

    world: WorldConfig = field(default_factory=WorldConfig)
    worm: WormConfig = field(default_factory=WormConfig)
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)
    food: FoodConfig = field(default_factory=FoodConfig)
    toxin: ToxinConfig = field(default_factory=ToxinConfig)
    metabolism: MetabolismConfig = field(default_factory=MetabolismConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    render: RenderConfig = field(default_factory=RenderConfig)

    # from_dict / from_yaml / resolve / to_dict / to_yaml come from ConfigRoot.
