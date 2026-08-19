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