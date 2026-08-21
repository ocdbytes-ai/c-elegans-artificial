# Log

## Stage 1

![worm](../assets/worm.png)

### Signals

Stage 1 is defined as training the worm for finding food in order to survive. Current inputs to the policy are : 

- Energy
- Food Smell (chemotaxis)
- Touch (mechanotaxis)
- heading sin
- heading cos

heading sin and cos are basically a compass on where the worm is headed :

```
┌───────────┬──────┬─────┬─────┐
│ direction │  θ   │ sin │ cos │
├───────────┼──────┼─────┼─────┤
│ east      │ 0    │ 0   │ +1  │
├───────────┼──────┼─────┼─────┤
│ north     │ π/2  │ +1  │ 0   │
├───────────┼──────┼─────┼─────┤
│ west      │ π    │ 0   │ −1  │
├───────────┼──────┼─────┼─────┤
│ south     │ 3π/2 │ −1  │ 0   │
└───────────┴──────┴─────┴─────┘
```

These all inputs to the policy gives us the actions to actually steer the worm towards survival. We get two actions :

- Throttle (speed of movement)
- Turn (direction of movement)

Each of these action values are in range : [-1, 1]. 

### Training

Stage 1 training infra includes a writen environemnt in gymnasium framework and a PPO implementation inspired from open ai's spinningup repository. Here is the [Modernised Fork]() of spinning up. So currently we have a simple PPO training loop running over stacked frames but I have added a small modification to the actor policy (defined as `pi` in code) loss function by adding an entropy constant to it. 

**Role of Entropy :**
As PPO is on-policy (it can only learn from the latest sampling except for the pi_old not the whole history) so we need to prevent from the sampling to approach deterministic behaviour because after that the policy doesn't look into any alternative approaches and thus stopping the improvement. Here I introduced a constant and named it entropy just to prevent this behaviour.

I have also clamped the `log_std` value by setting a `min_log_std` value in ppo config because `log_std` kept falling and then there was nothing to actually update in the policy for (because it keeps falling and we see no improvement in the policy). For `log_std` to be increasing thus to add a more creative sampling (more noise !! more noise !!!) I added a `min_log_std` clamp. Also want to clear that the entropy coeffieicnt pushes back on `log_std`'s complete falling but it is a constant so it doesn't help that much while experimentation. That matters beacuase sampling is the only exploration PPO has, and because KL scales as 1/(std)^2, a collapsed policy also hits target_kl sooner and loses the ability to move its mean at all.

**Modified Policy Loss :**

$$
L^{\pi}(\theta) \;=\; -\,\mathbb{E}_t\!\left[\min\!\Big(r_t(\theta)\,\hat{A}_t,\;
\operatorname{clip}\big(r_t(\theta),\,1-\epsilon,\,1+\epsilon\big)\,\hat{A}_t\Big)\right]
\;-\; c_{H}\,\mathbb{E}_t\!\left[\mathcal{H}\big[\pi_\theta(\cdot\mid s_t)\big]\right]
$$

$$
r_t(\theta) \;=\; \exp\!\Big(\log \pi_\theta(a_t\mid s_t) \;-\; \log \pi_{\theta_{\text{old}}}(a_t\mid s_t)\Big),
\qquad
\log \pi_\theta(a\mid s) \;=\; \sum_{i=1}^{k} \log \mathcal{N}\!\big(a_i;\, \mu_{\theta,i}(s),\, \sigma_i\big)
$$

## Stage 2 

Stage 2 of the project was to add a sense for toxins and that was very simple compared to Stage 1 as I only had to add a channel to sense toxins (this is how it happens in a real life worm btw the worm has set of channels for different things and all work through chemotaxis). I added the channel and worm detects it automatically that toxin is bad for it without hard coding a real logic into the worm's programming. I mean it is cool !! (atleast for me).

Problem during the implementation and training :

I added the damage from toxins as a huge multiplier of `1.5` now the problem is that the worm kind of is scared and not bold enough to take risks and actually go look out for food in order to survive so I tried decreasing the multiplier to `0.8` to see if worm starts taking risk or not --> I saw increase in lifespan by making the worm more confident and actually risk taking. If talking about numbers I saw about 24 % increase in lifespan in the trained policy.  
