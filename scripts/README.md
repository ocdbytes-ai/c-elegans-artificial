# Scripts

Command-line entry points. Run them from anywhere with `uv run python scripts/<name>.py`. Every script prints its full flag list with `--help`.

| Script | Purpose |
|---|---|
| [`train.py`](#trainpy) | Train a policy with PPO and save a run to `experiments/` |
| [`watch.py`](#watchpy) | Watch or measure a trained checkpoint |
| [`play.py`](#playpy) | Run a hand-written baseline policy (no training) |

**Flag conventions shared by all scripts**

- Counts (`--episodes`, `--max-steps`, `--epochs`, `--eval-episodes`, ...) must be at least 1. A 0 is rejected as a typo; it never means "forever" or "no limit".
- Unbounded modes are their own flags: `--loop` runs episodes until you close the window or press Ctrl-C, and `--no-step-limit` lets each life run until the worm dies.
- A count and its unbounded flag cannot be used together (`--episodes` with `--loop`, `--max-steps` with `--no-step-limit`).

---

## train.py

Trains a worm with PPO. Command-line flags override the YAML files, which override the dataclass defaults. Each run writes both resolved configs, `progress.csv` and its checkpoints to `experiments/<name>_s<seed>/`, and never overwrites an existing run.

```
uv run python scripts/train.py [--env-config YAML] [--ppo-config YAML]
                               [--name NAME] [--seed N] [--epochs N] [--steps-per-epoch N]
                               [--device {auto,cpu,cuda,mps}]
                               [--resume CHECKPOINT [--reset-optimizers]]
                               [--eval-episodes N | --no-eval]
```

| Flag | Default | Meaning |
|---|---|---|
| `--env-config` | `configs/world_v1.yaml` | The world: arena, food, toxins, metabolism, sensed channels |
| `--ppo-config` | `configs/ppo.yaml` | The learner: network, rollout, optimiser, curriculum, run |
| `--name` | `run.name` in the PPO config | Run directory name (the seed is appended) |
| `--seed` | `run.seed` | Seeds torch, numpy and the environment |
| `--epochs` | `rollout.epochs` | Update rounds |
| `--steps-per-epoch` | `rollout.steps_per_epoch` | Env steps collected before each update |
| `--device` | `run.device` | Where the networks run |
| `--resume` | none | Continue a checkpoint's weights in a new run directory. The world comes from `--env-config`, which must produce the same observation width |
| `--reset-optimizers` | off | With `--resume`, start Adam fresh instead of restoring its state |
| `--eval-episodes` | 10 | Deterministic episodes evaluated after training |
| `--no-eval` | off | Skip that evaluation |

Examples:

```
# Train on world v2 with the PPO settings in configs/ppo.yaml
uv run python scripts/train.py --env-config configs/world_v2.yaml --name toxins

# Rerun an earlier experiment's exact world with a different seed
uv run python scripts/train.py --env-config experiments/head_sweep_v5_s0/env.yaml --name v5_repeat --seed 1

# Continue a run for 500 more epochs in a new run directory
uv run python scripts/train.py --env-config experiments/head_sweep_v5_s0/env.yaml --name v5_more --epochs 500 \
    --resume experiments/head_sweep_v5_s0/checkpoints/best.pt
```

## watch.py

Loads a checkpoint and runs its policy, rendering by default. The world is rebuilt from the config stored inside the checkpoint, not from `configs/`, so you see the world the policy trained in.

```
uv run python scripts/watch.py CHECKPOINT
                               [--episodes N | --loop] [--max-steps N | --no-step-limit]
                               [--seed N] [--no-render] [--stochastic] [--debug]
                               [--env-config YAML] [--no-tui]
```

| Flag | Default | Meaning |
|---|---|---|
| `CHECKPOINT` | required | A `.pt` file, e.g. `experiments/<run>/checkpoints/best.pt` |
| `--episodes` | 5 | Episodes to run |
| `--loop` | off | Keep running episodes until the window is closed or Ctrl-C |
| `--max-steps` | 4000 | Step cap per episode |
| `--no-step-limit` | off | No step cap; each life runs until the worm dies |
| `--seed` | 0 | Seed of the first episode; episode `i` uses `seed + i` |
| `--no-render` | off | No window; measure only |
| `--stochastic` | off | Sample actions instead of taking the policy mean |
| `--debug` | off | Draw true eat radii, scent contours and the swept head position |
| `--env-config` | none | YAML merged over the stored world. It may name only what it changes; the changed settings are printed, and scores are then not comparable to training |
| `--no-tui` | off | Do not draw the actor network in the terminal |

Examples:

```
# Watch the best checkpoint of a run
uv run python scripts/watch.py experiments/head_sweep_v5_s0/checkpoints/best.pt

# Measure it: 100 deterministic episodes, no window
uv run python scripts/watch.py experiments/head_sweep_v5_s0/checkpoints/best.pt \
    --episodes 100 --no-render --seed 20000

# Watch indefinitely, with no step cap, and debug overlays
uv run python scripts/watch.py experiments/head_sweep_v5_s0/checkpoints/best.pt \
    --loop --no-step-limit --debug
```

Lifespan has a standard deviation close to its own mean, so use at least 50 episodes (100 is better) and the same `--seed` when comparing checkpoints.

## play.py

Runs a fixed baseline policy, to watch the world or to get a score a trained worm must beat.

```
uv run python scripts/play.py [--policy {random,freeze,straight,greedy}]
                              [--episodes N | --loop] [--max-steps N | --no-step-limit]
                              [--seed N] [--config YAML] [--render] [--debug]
```

| Flag | Default | Meaning |
|---|---|---|
| `--policy` | `random` | `random`: uniform actions. `freeze`: never moves; scores `initial_energy / basal_cost`. `straight`: full throttle, never turns. `greedy`: reads the true nearest pellet position and steers at it (an upper reference, not a fair baseline) |
| `--episodes` | 3 | Episodes to run |
| `--loop` | off | Keep running episodes until the window is closed or Ctrl-C |
| `--max-steps` | 4000 | Step cap per episode |
| `--no-step-limit` | off | No step cap; each life runs until the worm dies |
| `--seed` | 0 | Seed of the first episode |
| `--config` | dataclass defaults | World YAML, e.g. `configs/world_v2.yaml` |
| `--render` | off | Open a pygame window |
| `--debug` | off | Draw true eat radii and scent contours |

Examples:

```
# Watch the greedy upper reference on world v2
uv run python scripts/play.py --policy greedy --config configs/world_v2.yaml --render

# Measure the freeze baseline
uv run python scripts/play.py --policy freeze --config configs/world_v2.yaml --episodes 20
```
