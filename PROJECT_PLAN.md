# AI Worm — An Artificial-Life Organism That Learns to Survive

> A simulated worm that lives in a small world, must **eat to keep existing**, and
> learns both **how to move** and **how to forage** from a single reward signal:
> stay alive. Built on the RL algorithms from OpenAI Spinning Up (PPO / SAC).

**Elevator pitch:** Most RL projects solve a task. This one grows an *organism*.
By coupling learned locomotion to a metabolism (moving costs energy, food restores
it, energy = 0 means death), foraging, gradient-following, predator avoidance and
energy-efficient movement all *emerge* from the pressure to survive — no
hand-crafted sub-rewards.

---

## Guiding principles

- **Survival is the only reward.** Reward = staying alive (`+1`/step). We never
  directly reward "eating" or "avoiding danger." Everything interesting must
  emerge from that pressure.
  - ⚠️ Note: `Δenergy` is **not** a neutral alternative to `+1`/step — it spikes
    by `+food_value` at the exact moment of eating, which *is* a direct eating
    reward. Keep `+1`/step as the headline result; if `Δenergy` is used, label it
    as shaped. (Also: `Δenergy` telescopes to `final − initial` energy, so at
    γ=1 it is nearly a pure terminal reward.)
- **The worm may be naive, but it must not be blind.** It starts with no
  *innate* knowledge of what anything means — but anything it must learn to
  react to has to be **present in its observation**. You cannot learn to avoid
  what you cannot sense.
- **Biological plausibility earns the cool behavior.** The more we constrain the
  worm's senses to what a real worm has (local chemical concentration, not a
  god's-eye map), the more genuinely lifelike the emergent behavior.
- **Ship the loop first, then make it hard.** Get a dumb worm surviving in the
  easiest possible world, *then* remove crutches (sensors, body, world) one at a
  time.
- **Test hard learning problems on simple bodies.** Avoidance and chemotaxis are
  learning problems; an N-link body is a physics problem. Don't debug both at once.
- **Every phase produces something watchable.** Rendering is not optional; the
  payoff of ALife is seeing it.

---

## Part 0 — Prerequisites & setup

- [ ] Reuse the Spinning Up repo's algos (`spinup/algos/pytorch/{ppo,sac}`) or
      copy the core loop out so this project is standalone.
- [ ] Python env: `numpy`, `gymnasium`, `torch`, `matplotlib`/`pygame` for render.
- [ ] Decide sim backend: **custom NumPy 2D sim** for v1–v3 (no MuJoCo dependency,
      trivial to add metabolism). Revisit for the N-link body — see §2.4.
- [ ] Repo skeleton:
      ```
      ai-worm/
        PROJECT_PLAN.md        <- this file
        notes/biology.md       <- Part 1 findings
        envs/worm_world.py     <- the environment
        configs/               <- metabolism / world ratios as YAML, not constants
        policies/              <- policy networks
        train.py               <- training entrypoint
        render.py              <- watch a trained worm
        experiments/           <- saved runs, configs, logs
      ```

---

## Part 1 — Learn the biology (understand before you model) [DONE]

Goal: build an accurate-enough mental model of a real worm so the design choices
are principled, not arbitrary. Target organism: **_C. elegans_** (the most-studied
nervous system on Earth — 302 neurons, fully mapped connectome).

## Part 2 — Design the environment

Build the world in increasing difficulty. Each sub-phase is a working, watchable env.

**Ordering note:** adversaries (v3) come *before* the N-link body (v4). Avoidance
is a learning problem and is far cheaper to train on a point body; the N-link body
is a physics problem. Test them separately.

### 2.0 The metabolism (shared by all phases)

```python
# per step
energy -= basal_cost                            # existing is not free
energy -= move_cost * |action|**2               # keep SMALL relative to basal (see below)
energy -= contact_damage                        # v3+, on adversary contact
energy += food_value                            # on eat
max_speed = base_speed * speed_factor(energy)   # smooth ramp, NOT a cliff
done    = energy <= 0                           # death
reward  = +1 per surviving step
```

- [ ] **Critical ratio — `basal_cost` vs `move_cost`.** If moving costs meaningfully
      more than idling, *standing perfectly still maximises lifespan*, and PPO will
      find that local optimum within a few epochs and stop exploring. Set
      `basal_cost` high enough that freezing is clearly fatal, and `move_cost` small
      enough that it only shapes efficiency later.
- [ ] **`speed_factor(energy)` must be a smooth multiplier, not a threshold.** A hard
      cliff creates a death spiral (low energy → slow → can't reach food → dead),
      making all states below it unrecoverable and useless for learning. Verify food
      is still reachable at reduced speed.
- [ ] Death is already maximally punished under `+1`/step (the worm forfeits all
      future reward). An extra `-X` terminal penalty is **optional sharpening**, not
      a requirement — never let it become the primary signal.
- [ ] Put every constant above in `configs/`, not in the env source.

### 2.1 World v1 — point-worm (validate the life loop)
- [ ] **Body:** single point on a 2D continuous plane with a heading θ.
- [ ] **Action space:** continuous `[turn, forward_speed]`.
- [ ] **Food:** discrete pellets at random positions; eaten when head is within
      radius `r`; respawn elsewhere on eat.
- [ ] **Observation (easy mode):** `[energy, vector_to_nearest_food, sin θ, cos θ]`.
      (Use `sin/cos`, not raw θ — avoids the 0/2π discontinuity.)
- [ ] **Render:** matplotlib/pygame — worm dot, food pellets, energy bar.
- [ ] **Success criterion:** trained worm survives far longer than a random policy;
      you can visibly see it heading toward food.

### 2.2 World v2 — realistic senses (earn chemotaxis)
- [ ] Replace `vector_to_nearest_food` with a **scalar "smell"**: total food
      concentration sampled at the head (sum of Gaussian bumps around pellets).
- [ ] Now the worm sees only *how much it smells here*, not *where food is*.
- [ ] Give it temporal information — **frame-stack the last k observations first**
      (see §3.2; recurrence is a much larger lift).
- [ ] **Success criterion:** emergent gradient-climbing (klinokinesis/klinotaxis-
      like behavior) — turning more when smell drops, steering up-gradient.

### 2.3 World v3 — adversaries (earn avoidance)
- [ ] Add adversaries that emit a **second chemical channel**, sensed as an
      unlabeled scalar `adversary_smell`. The worm is told nothing about what it
      means; it must discover that the channel predicts harm.
- [ ] **Start with graded, survivable damage** — contact costs a large chunk of
      energy (e.g. 30% of max) but is not instantly fatal. Lethal-on-touch yields
      exactly one training example per episode, delivered as a terminal
      catastrophe: the sparsest, highest-variance signal possible.
- [ ] Tighten toward lethal *after* avoidance emerges, as a difficulty knob.
- [ ] Adversaries may be static hazards first, then mobile.
- [ ] **Do not start this until M3 is green.** A worm that cannot yet feed itself
      dies constantly of starvation, and death-by-adversary is indistinguishable
      from death-by-hunger in the reward signal.
- [ ] **Success criterion:** measurable negative chemotaxis — the worm's turning
      rate and heading correlate with the adversary gradient, and contact events
      per lifetime drop over training.

### 2.4 World v4 — a real body (earn locomotion)
- [ ] Upgrade body to an **N-link chain** (e.g. 3–8 segments) with joint torques as
      actions. Fluid drag model (§1.3) so undulation produces forward motion.
- [ ] Proprioception in the observation (joint angles + velocities).
- [ ] **Reconsider the backend here.** A *stable* N-link body with correct drag is
      real physics-engine work; MuJoCo `Swimmer` plus a metabolism wrapper may be
      less total effort than debugging a custom solver. MuJoCo is already installed
      in the spinningup env.
- [ ] **Success criterion:** a serpentine gait *emerges* purely from the survival
      reward — this is the "mesmerizing visuals" milestone.

### 2.5 World v5 — richer ecology (stretch)
- [ ] Multiple food types (high/low value, some toxic → negative energy).
- [ ] Walls & obstacles; mobile/pursuing predators.
- [ ] Depleting patches → forces exploration vs. exploitation trade-offs.
- [ ] **"Lives inside your computer" twist:** spawn food from real system events
      (disk I/O, new files, CPU spikes); render as an ambient desktop window.

---

## Part 3 — Design the policy & training

### 3.1 Algorithm selection
- [ ] **v1–v3 (simple continuous control):** start with **PPO** — matches the
      Spinning Up policy-gradient work already done; on-policy, stable.
- [ ] **v4 (continuous joint torques):** **SAC** — best sample efficiency for
      continuous locomotion; already in `spinup/algos/pytorch/sac`.
  - ⚠️ SAC is **not** in spinup's `MPI_COMPATIBLE_ALGOS` (`['vpg','trpo','ppo']`),
    so v4 runs single-process. Also `sac/core.py` sets `act_limit =
    action_space.high[0]` — keep joint torque limits **symmetric**.
- [ ] Keep both wired up: this env is a clean **on-policy vs off-policy** testbed.

### 3.2 Network architecture
- [ ] v1: simple MLP actor-critic.
- [ ] v2/v3: **frame-stacked MLP** for the temporal comparison chemotaxis needs.
  - ⚠️ **Do not start with a recurrent policy.** spinup's `PPOBuffer` stores flat
    transitions with no sequence batching or hidden-state handling; a GRU/LSTM
    actor means rewriting the buffer plus hidden-state plumbing through rollout
    and episode resets. Frame stacking is ~10 lines and answers "is smell higher
    than k steps ago?", which is all klinokinesis requires. Reach for recurrence
    only after demonstrating stacking fails.
- [ ] v4: MLP over proprioception + smell; consider larger hidden sizes.
- [ ] Stretch: a **connectome-constrained** network (sparse topology inspired by
      the 302-neuron *C. elegans* wiring) — compare vs. a dense MLP. See §1.5 for
      prior art to read first.

### 3.3 Observation design (v3 target)
```python
obs = [
    energy / max_energy,          # interoception - required for hunger-modulated behavior
    food_smell,                   # scalar concentration at head
    adversary_smell,              # scalar, UNLABELED - worm learns what it means
    sin(heading), cos(heading),
]
# frame-stacked over the last k steps
```

### 3.4 Reward & curriculum
- [ ] Default reward: survival (`+1`/step). Resist adding sub-rewards — see the
      `Δenergy` caveat in Guiding principles.
- [ ] If learning stalls, use a **curriculum**: start with dense food / cheap
      metabolism, then gradually make the world harsher. Evaluate on the *hard*
      setting throughout, so curriculum overfitting is visible.
- [ ] Log reward *decomposition* (energy in vs. out) for debugging even though the
      agent only optimizes the scalar.

### 3.5 Failure modes to watch for
- [ ] **The freeze trap.** Log mean `|action|` every epoch. If it decays toward
      zero while lifespan plateaus, the worm has learned to stand still — rebalance
      `basal_cost` vs `move_cost` (§2.0).
- [ ] **Exploration collapse.** spinup's PPO has **no entropy bonus** — `loss_pi =
      -(min(ratio*adv, clip_adv)).mean()`, with entropy computed for logging only.
      Expect to add an entropy term to survive the early phase.
- [ ] **Death vs. timeout bootstrapping.** GAE needs `last_val = 0` on true death
      but `last_val = V(s)` on episode-cap timeout. Conflating them biases the value
      function — and under a survival reward `V(s)` ≈ expected remaining lifespan,
      i.e. exactly the headline metric. Mirror spinup PPO's `timeout or epoch_ended`
      branch.
- [ ] **Invisible adversaries.** If avoidance never emerges, first confirm
      `adversary_smell` is actually in the observation and has non-trivial variance.

### 3.6 Experiments & evaluation
- [ ] **Metrics:** mean lifespan, food eaten per life, energy efficiency
      (distance per unit energy), path tortuosity, adversary contacts per life.
      (Note: under `+1`/step, mean lifespan *is* the return — the others are the
      independent signals.)
- [ ] **Baselines:** random policy; hand-coded greedy "walk toward nearest food";
      "freeze" policy (the local optimum — beating it is the real bar for M2).
- [ ] **Ablations:** vector-sensor vs. smell-only; frame-stack vs. recurrent;
      PPO vs. SAC; with vs. without `adversary_smell` in the observation.
- [ ] **Behavior analysis:** does the worm behave differently when hungry vs. full?
      Plot turning rate vs. smell gradient (klinokinesis test) and vs. adversary
      gradient (negative chemotaxis test).

---

## Milestones (definition of done per stage)

| # | Milestone | Done when… |
|---|-----------|-----------|
| M0 | Setup | Repo skeleton + deps + render stub run end-to-end. |
| M1 | Biology notes | `notes/biology.md` written with design implications, predictions + sources. |
| M2 | Life loop works | v1 point-worm trained (PPO) survives ≫ random **and ≫ the freeze baseline**. |
| M3 | Chemotaxis emerges | v2 smell-only worm climbs gradients using stacked frames. |
| M4 | Avoidance emerges | v3 worm learns negative chemotaxis to an unlabeled channel; contacts/life fall. |
| M5 | Locomotion emerges | v4 segmented worm learns an undulatory gait (SAC). |
| M6 | Experiments | Ablation table + behavior plots written up. |
| M7 (stretch) | Ecology / computer-life | v5 hazards or system-fed food + desktop render. |

---

## Decisions made

- **Reward:** `+1` per surviving step. `Δenergy` is shaped, not a neutral variant.
- **Adversary sensing:** separate unlabeled chemical channel in the observation.
  Naive ≠ blind.
- **Adversary damage:** graded and survivable first; lethal later as a difficulty knob.
- **Phase order:** adversaries (v3) before the N-link body (v4).
- **Memory:** frame stacking before recurrence.
- **Locomotion model:** swimming (fluid drag), not crawling.
- **World:** continuous, not grid.
- **Episodes:** episodic with reset on death — GAE and PPO assume episode
  boundaries; a single lifelong episode pushes toward an average-reward
  formulation, which is a different algorithm.
- **Sim backend:** custom NumPy for v1–v3; re-evaluate MuJoCo at v4.

## Open questions

- [ ] Exact `basal_cost : move_cost : food_value : food_density` ratios — needs
      empirical tuning against the freeze baseline.
- [ ] Contact damage magnitude, and when to tighten it toward lethal.
- [ ] Frame-stack depth `k` for the temporal gradient signal.
- [ ] Are adversaries static or mobile at v3? (Static first is easier to diagnose.)
- [ ] Add reproduction/evolution later (population + mutation) for a full ALife arc?
      Deferred past M7 — it is an outer optimization loop, not an extension of the
      inner one.

---

## References to gather (fill in during Part 1)

- [ ] *C. elegans* chemotaxis strategies (klinokinesis / klinotaxis) — primary papers.
- [ ] *C. elegans* repulsive/negative chemotaxis — primary papers.
- [ ] *C. elegans* undulatory locomotion & drag models.
- [ ] OpenWorm project (whole-organism simulation).
- [ ] Neural Circuit Policies (Lechner, Hasani et al.) & Liquid Time-constant Networks.
- [ ] WormBook (open-access review of *C. elegans* biology).
- [ ] Spinning Up docs for PPO & SAC (algorithm references).

---

*Plan created as the roadmap for the AI Worm artificial-life project. Built on
OpenAI Spinning Up (`../spinningup`).*
