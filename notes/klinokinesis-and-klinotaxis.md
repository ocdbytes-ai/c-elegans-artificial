## Scent field:

Food pellets sit at positions $(p_1 \dots p_n)$

```
                                     scent
              _______   - - - - - - - -^- - - -
            /         \                |
           /           \             .-+-.
          |      •      |           /  |  \
           \           /          /    |    \
            \_______/       __.-'      |      '-.__
    ↗ - - - - - - - - - -  ------------+----------------> x
   /
food pellet               → How food scent works in
with certain                gaussian mode
scent radius
```

Each food pellet is a Gaussian bump, and the field is the **sum**:

$$
C(x) = A \cdot \sum_{i=1}^{n} \exp\left(-\frac{\lVert x - p_i \rVert^2}{2s^2}\right)
$$

$A$ : scent peak = 1.0

(S.D. of the Gaussian bump) $s$ = sigma = gaussian_sigma_scale × scent radius.

This is a **scalar field**. (So it has only single value)

What worm gets:

$C(x_{worm})$, concentration at the worm head.
This is the **food-smell channel**

---

Now to move uphill (to eat the food because it has a radius) we need a gradient:

$$
\text{grad}\,(C(x)) = \left(\frac{\partial C}{\partial x}, \frac{\partial C}{\partial y}\right)
$$

- for one pellet:

$$
\text{grad}\,(C(x)) = -\left(\frac{A}{s^2}\right) \cdot \exp\left(-\frac{r^2}{2s^2}\right) \cdot \overrightarrow{(x - p)}
$$

> $\overrightarrow{(x - p)}$ → This is the vector component $(x_x - p_x,\; x_y - p_y)$

**Problem:**

$C(x)$ is a scalar quantity so we get only one number (a scalar) that tells the concentration.

Here we have the situation if let's suppose we have a pellet:

| position | C(x) sensed | true gradient | points toward |
|---|---|---|---|
| [12.00 10.00] | 0.360448 | [-0.3678 0.0000] | 180 deg |
| [11.00 11.73] | 0.360448 | [-0.1839 -0.3185] | 240 deg |
| [ 9.00 11.73] | 0.360448 | [ 0.1839 -0.3185] | 300 deg |
| [ 8.00 10.00] | 0.360448 | [ 0.3678 0.0000] | 0 deg |
| [ 9.00 8.27] | 0.360448 | [ 0.1839 0.3185] | 60 deg |
| [11.00 8.27] | 0.360448 | [-0.1839 0.3185] | 120 deg |

at (10,10) we have 6 "correct" directions :(

To solve this I implemented a stacked frame approach.

as let's say the worm is at 4 different locations:

time series $\{C(t_0),\ C(t_1),\ C(t_2),\ C(t_3)\}$

so we give these 4 "frames" as an input to the model to figure out where it is (the worm).

### What the moving scalar sensor measures:

let say $C(x(t))$

$$
\frac{dC}{dt} = \text{grad}\,C \cdot \frac{dx}{dt}
$$

$$
\frac{\delta C}{\delta t} = \text{grad}\,C \cdot v
$$

we can write velocity as:

$v = \text{speed} \cdot \vec{u}$ , $\vec{u}$ : heading vector

$$
\frac{\delta C}{\delta t} = \text{speed} \cdot \lVert \text{grad}\,C \rVert \cdot \cos\theta
$$

$\theta$ : angle where worm is pointing and gradient is pointing.

**Cost 1:** Problem is $\cos(\theta) = \cos(-\theta)$ so heading 135° = heading 225° (Eg.)

**Cost 2:** Magnitude is several things multiplied together

In practice, we get difference b/w the frame stack instead of derivative

$$
\Delta C(t) = C(t) - C(t-1) \approx dt \cdot \text{speed} \cdot \lVert \text{grad}\,C \rVert \cdot \cos\theta
$$

→ if we have frame stack: 4 then it is kind of enough to see if the gradient is changing (slope change).

Here we have answers to some of the questions:

- Is worm improving? : **YES**
- How strongly? : Partly
- which way to turn : **NO**
- where is the food : **NO**

## Klinokinesis:

Strategy:

Do only this:

- Run in straight line every step, with probability $\lambda$, tumble (This picks a new random heading)

Now we make the $\lambda$ dependent on the $\Delta C$ (which was the concentration difference defined above)

$\Delta C > 0$, (improving) → $\lambda$ → low
$\Delta C < 0$, (worsening) → $\lambda$ → high

---

### Uncertainty observed in the trained model:

This is what happens when we use only one channel and steer decision is based on only one channel

If $\Delta C > 0$, it (worm) keeps heading
$\Delta C < 0$, tries something new

**In practice** → worm gets clamped and keeps clamped till the end of its life in certain simulations as $\Delta C$ might not be that strong and the action is *nothing*. leads to worm **dying**.

---

So the worm never decides where to go but it still climbs:

In practice the worm accumulates more in "good" directions when simulated and less in "bad" directions

put gradient along x-axis and heading should be defined as $u = (\cos\theta, \sin\theta)$ is uniform on the circle.

$$
\lambda(\theta) = \lambda_0 \left(1 - \alpha \cos\theta\right), \quad \alpha \in [0, 1)
$$

---

How tumble rate: $\lambda(\theta) = \lambda_0 (1 - \alpha \cos\theta)$ ?

As till now we only have one fixed quantity $\delta C / \delta t$ so we make tumble rate $\lambda$ depend upon that:

$$
\lambda = f\left(\frac{\delta C}{\delta t}\right)
$$

[ $f$ is a decreasing function — if $\delta C / \delta t$ increase the $f$ decreases ]

if $f$ → decreasing → tumble less

$f(0) = \lambda_0$ (if no signal tumble should be baseline)

Amazing thing is we don't know $f$ so how should we go about that?

→ **Claude** came for rescue here

Any smooth function (assuming) can be represented as Taylor expanded about zero:

$$
f(x) = f(0) + f'(0)\,x + \frac{1}{2} f''(0)\,x^2 + \dots
$$

write $f(0) = \lambda_0$ and $f'(0) = -k$

$$
\lambda \approx \lambda_0 - k \cdot \left(\frac{\delta C}{\delta t}\right)
$$

let's put the projection on here:

$$
\lambda \approx \lambda_0 - k \cdot v \cdot \lVert \text{grad}\,C \rVert \cdot \cos\theta
$$

bundle all the fixed quantity in the given situation

$$
\alpha \equiv \frac{k \cdot v \cdot \lVert \text{grad}\,C \rVert}{\lambda_0}
$$

$$
\Rightarrow \boxed{\lambda(\theta) = \lambda_0 \cdot (1 - \alpha \cos\theta)}
$$

---

Coming back

$\cos\theta = +1$ means head straight towards the grad.
$\cos\theta = -1$ means short runs

As tumbling is a Poisson process (memoryless), so run process is reciprocable:

$$
\tau(\theta) = 1/\lambda(\theta) \approx \left(\frac{1}{\lambda_0}\right)(1 + \alpha \cos\theta)
$$

(assuming $\alpha$ is close to zero).

Displacement during the run is $v \cdot \tau(\theta)$ in $\vec{u}$

Averaging the component along all directions:

$$
\mathbb{E}[\Delta x] = \frac{1}{2\pi} \int_0^{2\pi} v \cdot \tau(\theta) \cdot \cos\theta \, d\theta
= \frac{1}{2\pi\lambda_0} \int_0^{2\pi} (1 + \alpha \cos\theta) \cdot \cos\theta \, d\theta
$$

$$
\mathbb{E}[\Delta x] = \frac{v\alpha}{2\lambda_0}
$$

$$
V_{drift} = \frac{\mathbb{E}[\Delta x]}{\mathbb{E}[\tau]} = \frac{v\alpha}{2}
$$

$v$ : speed
$\alpha$ : modulation depth

→ I mean why not just give worm a proper sense of direction?

Because I am trying to mimic the actual biology in this project otherwise things would have been pretty easy and performance would be awesome, and an "ideal condition".

---

Actions are defined as:

$$
a \sim \mathcal{N}(\mu(obs), \sigma)
$$

- $\mu(obs)$ → we get from network output (worm can change how to change direction based on smell)
- $\sigma$ → fixed (randomness cannot be changed)

→ So we are actually not using the full potential of klinokinesis here because recalling:

$$
\lambda(\theta) = \lambda_0 (1 - \alpha \cos\theta)
$$

$\alpha$ → This should vary with signal

> **In code $\alpha$ is basically 0**

---

## Klinotaxis.

As we saw earlier $\cos\theta = \cos(-\theta)$ so we need a direction.

```
                  ↗ gradient
                 /
                /
               /
              /  θ
             /  )
      ~~~~~~●──────────────────→ heading
      (worm)

      θ : positive counterclockwise
```

if a worm rotates its heading by $\delta$ then heading is $\theta - \delta$.

we measure two headings:

$h_1 = v \cdot \lVert \text{grad}\,C \rVert \cdot \cos\theta$ — Before turn
$h_2 = v \cdot \lVert \text{grad}\,C \rVert \cdot \cos(\theta - \delta)$ — After turn

as, $\cos(\theta - \delta) = \cos\theta \cos\delta + \sin\theta \sin\delta$

$$
\Rightarrow h_2 = v \cdot \lVert \text{grad}\,C \rVert \cos\theta \cos\delta + v \cdot \lVert \text{grad}\,C \rVert \sin\theta \sin\delta
$$

$$
h_2 = h_1 \cos\delta + v \cdot \lVert \text{grad}\,C \rVert \sin\theta \sin\delta
$$

$$
v \cdot \lVert \text{grad}\,C \rVert \cdot \sin\theta = \frac{h_2 - h_1 \cos\delta}{\sin\delta}
$$

```
                               ↗ grad C
                             / |
                           /   |
                         /     |
                       /       |   cross track (how much better)
                     /         |
                   /  θ        |
                 /  )          |
               ●───────────────┴────→
                along track (heading)
```

$$
\text{along-track} = v \cdot \lVert \text{grad}\,C \rVert \cdot \cos\theta = h_1
$$

$$
\text{cross-track} = v \cdot \lVert \text{grad}\,C \rVert \cdot \sin\theta = \frac{h_2 - h_1 \cos\delta}{\sin\delta}
$$

$$
\theta = \tan(\text{cross track},\ \text{along track})
$$

By using frame stack we get the $\delta$ by difference between the heading.

While learning in code the network basically gets

$$
\text{cross} = (h_2 - h_1) / \delta
$$

> **Note:** this is the small-$\delta$ approximation of the exact cross-track expression derived above, $\dfrac{h_2 - h_1 \cos\delta}{\sin\delta}$, using $\cos\delta \approx 1$ and $\sin\delta \approx \delta$.

---

## Some Stats

| policy | mean C felt | dist/pellet | vs blind | pellets/1k steps |
|---|---|---|---|---|
| trained | 0.2769 | 13.9 | 1.20x | 8.57 |
| straight | 0.0626 | 23.6 | 0.71x | 1.99 |
| random | 0.1719 | 400.4 | 0.04x | 0.21 |
