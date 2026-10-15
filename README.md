# C. Elegans. Simulation

<img src="assets/banner-retro-worm.svg" width="100%">

This repository is an experiment to create an artificial worm that senses its environment and acts on it.
The only reward is +1 per step survived, so everything the worm does is learned for survival.

## Current status

A point worm in a 20 × 20 arena eats pellets to keep its energy up and avoids toxin patches. The policy is an MLP trained with PPO.

**Senses:** energy, food smell and its change, toxin smell and its change, touch, heading, and head-sweep phase. Smell is read at a single point; the head sweep lets the worm tell which side the food is on.

**Actions:** turn and throttle, each in [−1, 1].

**Results** (world v2, 100 episodes):

| Policy | Mean lifespan (steps) | Pellets eaten |
|---|---|---|
| freeze | 199 | 0 |
| straight | 230 | 0.6 |
| **trained (mean for 100 runs)** | **3573** | **33** |
| greedy (knows pellet positions) | 4000 (cap) | 219 |

**Emerged behaviour:** klinokinesis for food and toxins, and klinotaxis after the head sweep was added.

## Quick start

```
uv sync
uv run python scripts/train.py --env-config configs/world_v2.yaml --name <RUN_NAME>
uv run python scripts/watch.py experiments/<RUN_NAME>/checkpoints/best.pt
```

## Project layout

```
envs/       environment
ppo/        PPO trainer
common/     config loading and CLI helpers
configs/    world and PPO configs
scripts/    train, watch, play
tests/      tests
notes/      biology and analysis notes
notebooks/  training report, chemotaxis behaviours
```

## Docs

- [Biology](./notes/biology.md)
- [Stage log](./notes/log.md)
- [Klinokinesis and klinotaxis](./notes/klinokinesis-and-klinotaxis.md)
- [Scripts](./scripts/README.md)

## Checklist

- [x] Biological understanding
- [x] Environment creation (base environment)
- [ ] Train the behaviour
    - [x] Stage 1: Food source sensory movement
    - [x] Stage 2: Other chemical stimulation (toxins)
    - [x] Stage 3: Klinokinesis and klinotaxis analysis
    - [x] Stage 4: Head sweep, so klinotaxis can emerge
    - [ ] Stage 5: Reproduction and population dynamics
    - [ ] Stage 6: Dauer stage experimentation
- [ ] Create the environment in Rust for ESP32
    - [ ] Check memory constraints
    - [ ] Check population growth
    - [ ] Create inference model
- [ ] Transfer the environment to hardware

## License

[MIT](./LICENSE)
