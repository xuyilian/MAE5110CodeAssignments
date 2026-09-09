# Assignment 1: Rimless-Wheel Dynamics and Stability

## 1. Model description and parameters

The rimless wheel represents a passive walker with a point mass at its hub and equally spaced, rigid, massless spokes. During each stance phase, one spoke is pinned to the ground without slipping. The hub rotates about this contact point under gravity. When an adjacent spoke touches the ground, the contact point changes through an instantaneous plastic impact. The model therefore combines continuous motion with discrete changes in state.

Figure 1 shows the wheel geometry and angle conventions used to define the state, continuous dynamics, and contact conditions below.

![Rimless-wheel model showing the hub mass, spoke length, spoke spacing, stance angle, slope, and gravity](figures/rimless_wheel_model.svg)

*Figure 1. Rimless-wheel geometry and parameter definitions. The stance angle is measured from the upward vertical and is positive in the downhill direction. Diagram from Russ Tedrake's [Underactuated Robotics, “Simple Models of Walking and Running”](https://underactuated.mit.edu/simple_legs.html), reproduced from the [original illustration](https://underactuated.mit.edu/figures/rimlessWheel.svg).*

### 1.1. Physical parameters

The reference configuration below is the one specified in [state_space_basins.py](state_space_basins.py). The slope and number of spokes can be varied for the parameter studies.

| Symbol | Definition | Reference value | Code parameter |
| --- | --- | --- | --- |
| $`m`$ | Point mass at the hub | $`1.0\ \mathrm{kg}`$ | `hub_mass` |
| $`l`$ | Length of each spoke | $`1.0\ \mathrm{m}`$ | `spoke_length` |
| $`g`$ | Gravitational acceleration | $`9.81\ \mathrm{m/s^2}`$ | `gravity` |
| $`N`$ | Number of equally spaced spokes | $`8`$ | `number_of_spokes` |
| $`\alpha`$ | Half the angle between adjacent spokes | $`\pi/N=\pi/8=22.5^\circ`$ | `half_spoke_angle` |
| $`\gamma`$ | Downhill inclination of the ground relative to horizontal | $`5^\circ`$ | `slope_angle` |

The angular separation between adjacent spokes is

```math
2\alpha=\frac{2\pi}{N}.
```

The number of spokes and the half-spoke angle are consequently linked: changing $`N`$ requires recalculating $`\alpha`$. Angles are stored in radians in the implementation; degrees are used where convenient for presentation. Earth gravity is held fixed throughout the study.

### 1.2. State and modeling assumptions

The continuous state is

```math
\mathbf{x}=\begin{bmatrix}\theta\\\omega\end{bmatrix},
\qquad \omega=\dot\theta.
```

Here, $`\theta`$ is the stance-spoke angle measured from the upward vertical, positive downhill. The variable $`\omega`$ is the angular velocity about the active ground-contact point, in $`\mathrm{rad/s}`$; it is not an independent spin of the point mass. Positive $`\omega`$ indicates forward rotation, while negative $`\omega`$ indicates backward rotation.

The model assumes:

- A point-mass hub and rigid, massless spokes of equal length.
- A straight, rigid slope and a stance contact that acts as a pin without slipping or detaching.
- No applied torque, air resistance, or continuous viscous damping.
- Instantaneous plastic impacts, with angular momentum conserved about the new contact point and kinetic energy dissipated.

The single-spoke stance interval is

```math
\gamma-\alpha\leq\theta\leq\gamma+\alpha.
```

For the reference parameters, this becomes $`-17.5^\circ\leq\theta\leq27.5^\circ`$. The angle coordinate is reset when the supporting spoke changes; this coordinate jump does not represent an instantaneous displacement of the hub.

## 2. Hybrid dynamics

### 2.1. Continuous stance dynamics

Between impacts, the wheel behaves as an inverted pendulum. The moment of inertia about the stance contact is $`I=ml^2`$, and the gravitational torque in the positive angular direction is $`mgl\sin\theta`$. Therefore,

```math
ml^2\ddot\theta=mgl\sin\theta,
```

which gives

```math
\boxed{
\dot{\mathbf{x}}=
\begin{bmatrix}
\omega\\
\dfrac{g}{l}\sin\theta
\end{bmatrix}.
}
```

The mass cancels from the angular-acceleration equation. The slope does not appear explicitly in this equation because $`\theta`$ is measured relative to the vertical; instead, $`\gamma`$ determines the contact angles and reset configurations. This continuous vector field is used for both forward and backward motion until a contact event occurs.

With a fixed stance contact, the mechanical energy is

```math
E=\frac{1}{2}ml^2\omega^2+mgl\cos\theta+C,
```

where $`C`$ accounts for the height of that contact point relative to a common reference. This energy is conserved between impacts. When comparing energy across different stance contacts, the change in contact height must be included.

### 2.2. Forward contact and reset

A forward impact occurs when the downhill adjacent spoke reaches the ground. Its guard is

```math
\boxed{
\mathcal{G}_{\mathrm{f}}=
\left\{(\theta,\omega):\theta=\gamma+\alpha,\ \omega>0\right\}.
}
```

The velocity condition ensures that the contact surface is crossed in the forward direction. Superscripts $`-`$ and $`+`$ denote the states immediately before and after the same impact.

Switching to the next spoke changes the angle coordinate by $`-2\alpha`$:

```math
\theta^+=\theta^- -2\alpha=\gamma-\alpha.
```

Conservation of angular momentum about the new contact point gives

```math
ml^2\omega^+=ml^2\omega^-\cos(2\alpha).
```

Thus, the forward reset is

```math
\boxed{
\mathbf{x}^+=R_{\mathrm{f}}(\mathbf{x}^-)=
\begin{bmatrix}
\gamma-\alpha\\
\omega^-\cos(2\alpha)
\end{bmatrix}.
}
```

The corresponding implementation functions in [models/rimless_wheel.py](models/rimless_wheel.py) are `detect_contact()` and `apply_impact_reset()`.

### 2.3. Backward contact and reset

If the wheel lacks enough energy to pass the upright position, it can turn back. The previous, uphill spoke then contacts the ground. The backward guard is

```math
\boxed{
\mathcal{G}_{\mathrm{b}}=
\left\{(\theta,\omega):\theta=\gamma-\alpha,\ \omega<0\right\}.
}
```

Switching to the previous spoke changes the angle coordinate by $`+2\alpha`$:

```math
\theta^+=\theta^-+2\alpha=\gamma+\alpha.
```

The angular-momentum calculation gives the same velocity factor as for forward contact. The backward reset is therefore

```math
\boxed{
\mathbf{x}^+=R_{\mathrm{b}}(\mathbf{x}^-)=
\begin{bmatrix}
\gamma+\alpha\\
\omega^-\cos(2\alpha)
\end{bmatrix}.
}
```

For the assignment's range $`N=6,\ldots,12`$, the factor $`\cos(2\alpha)`$ is positive and less than one. A backward impact consequently preserves the negative sign of $`\omega`$ while reducing its magnitude. These operations are implemented by `detect_backward_contact()` and `apply_backward_impact_reset()`.

### 2.4. Impact energy loss and resting contact

For either impact direction, the change in kinetic energy follows directly from the velocity reset:

```math
K^+=K^-\cos^2(2\alpha),
\qquad
K^--K^+=\frac{1}{2}ml^2(\omega^-)^2\sin^2(2\alpha).
```

For $`N=8`$, the post-impact angular velocity is approximately $`0.7071`$ times its pre-impact value, and the kinetic energy is reduced by $`50\%`$. Repeated alternating impacts can therefore dissipate the energy of rocking motion.

For $`0<\gamma<\alpha`$, a two-contact resting configuration is possible. It can be represented as $`(\gamma-\alpha,0)`$ using one contact as the reference, or $`(\gamma+\alpha,0)`$ using the other. These are two coordinate representations of the same physical resting configuration. Rest requires both contact constraints; it is not generally an equilibrium of the single-contact inverted-pendulum equation.

The numerical model approximates the limiting sequence of small rocking impacts by a resting mode once a convergence criterion is satisfied. In that mode, the configuration is held fixed and $`\omega=0`$. A momentary zero velocity at an ordinary turning point is not sufficient to identify rest. The upright state $`(0,0)`$, by contrast, is an unstable single-contact equilibrium.

The mathematical guards above use equality at contact. Numerical detectors use $`\theta\geq\gamma+\alpha`$ for forward crossings and $`\theta\leq\gamma-\alpha`$ for backward crossings to identify a crossed boundary. In the basin calculations, bisection then locates the contact within the integration step before applying the reset.

## 3. Sanity checks

The checks below compare the implemented dynamics with physical expectations and analytical predictions. They use the reference parameters $`N=8`$, $`\gamma=5^\circ`$, $`l=1\ \mathrm{m}`$, $`m=1\ \mathrm{kg}`$, and $`g=9.81\ \mathrm{m/s^2}`$. All angular velocities are in $`\mathrm{rad/s}`$.

The measurements and assertions are reproducible with [sanity_checks.py](sanity_checks.py). From the repository root, with the project dependencies installed, run:

```console
uv run python sanity_checks.py
```

This writes the measured values to [figures/sanity_checks.json](figures/sanity_checks.json) and regenerates Figure 2. The script checks the model's guard/reset functions, RK4 integrator, contact-location routine, and the convergence classifier used by the basin calculations.

Figure 2 summarizes three checks: energy conservation during stance, convergence toward steady walking, and decay of rocking motion. The experiments and their measured results are discussed below.

![Sanity checks: relative energy error during continuous stance, convergence of forward post-impact velocities, and decay of rocking speed](figures/sanity_checks.png)

*Figure 2. Left: energy conservation during a stance interval without impacts. Center: successive forward post-impact velocities approach the analytical walking value. Right: the magnitude of the post-impact velocity decreases during alternating forward/backward rocking. Index zero denotes the initial state. The rocking trace is recorded without artificially setting its velocity to zero.*

### 3.1. Energy conservation between impacts

**Experiment.** Start from $`(\theta_0,\omega_0)=(0.1\ \mathrm{rad},0.2\ \mathrm{rad/s})`$ and integrate for $`0.5\ \mathrm{s}`$ with RK4 and a timestep of $`0.01\ \mathrm{s}`$. Verify that neither contact guard is triggered during this interval.

**Expected result.** With no actuation, damping, or impacts, mechanical energy should be conserved. The relative energy error should remain small:

```math
\varepsilon_E(t)=\left|\frac{E(t)-E(0)}{E(0)}\right|.
```

**Observed result.** No impact occurred, and the maximum relative energy error was $`1.2781\times10^{-10}`$, below the check's acceptance tolerance of $`10^{-7}`$. The left panel of Figure 2 shows this small energy drift. This supports the sign and implementation of the continuous dynamics and the accuracy of RK4 for this stance trajectory.

### 3.2. Contact direction, coordinate reset, and impact losses

**Experiment.** For each contact direction, test a state exactly at the contact angle moving toward the boundary, a state moving in the opposite direction, a zero-velocity state, and a state just inside the boundary. Then apply each reset to a prescribed pre-impact velocity of magnitude $`2\ \mathrm{rad/s}`$.

**Expected result.** Only a contact-angle state moving in the appropriate direction should trigger its guard. Since $`\cos(2\alpha)=\cos45^\circ=1/\sqrt{2}`$, each reset should multiply velocity by $`1/\sqrt{2}`$ and retain half the kinetic energy. It should switch the angle to the opposite endpoint of the stance interval without changing the original input array or immediately retriggering the same guard.

**Observed result.** All eight guard cases passed, as did the input-preservation and reset checks. The measured impact results were:

| Impact direction | $`\theta^-`$ | $`\theta^+`$ | $`\omega^-`$ | $`\omega^+`$ | $`K^+/K^-`$ |
| --- | --- | --- | --- | --- | --- |
| Forward | $`27.5^\circ`$ | $`-17.5^\circ`$ | $`2.000000`$ | $`1.414214`$ | $`0.500000`$ |
| Backward | $`-17.5^\circ`$ | $`27.5^\circ`$ | $`-2.000000`$ | $`-1.414214`$ | $`0.500000`$ |

The residual $`ml^2[\omega^+-\omega^-\cos(2\alpha)]`$ was zero to the reported floating-point precision in both tests. These checks verify the prescribed impact law; the full-step experiment below also checks its interaction with continuous integration and event detection.

### 3.3. Whole-step accuracy and timestep refinement

**Experiment.** Start just after a forward impact at $`(\theta_0,\omega_0)=(\gamma-\alpha,2)`$, integrate to the next forward contact, and apply the reset. Repeat with progressively smaller timesteps, using bisection to locate contact within each crossing step.

**Expected result.** Energy conservation during stance and the collision law predict

```math
\omega_{\mathrm{next}}^+
=\cos(2\alpha)\sqrt{\omega_0^2+\frac{4g}{l}\sin\alpha\sin\gamma}
=1.629228962842\ \mathrm{rad/s}.
```

The numerical result should approach this value as the timestep decreases.

**Observed result.** Every run reached the forward guard and produced the following errors relative to that prediction:

| Timestep (s) | First-contact time (s) | Absolute post-impact velocity error (rad/s) |
| --- | --- | --- |
| $`0.010`$ | $`0.413778106952`$ | $`1.1221\times10^{-9}`$ |
| $`0.005`$ | $`0.413778104461`$ | $`6.6074\times10^{-11}`$ |
| $`0.002`$ | $`0.413778104294`$ | $`2.2220\times10^{-12}`$ |
| $`0.001`$ | $`0.413778104289`$ | $`2.4403\times10^{-12}`$ |

The velocity error falls substantially before reaching approximately $`2\times10^{-12}\ \mathrm{rad/s}`$. At that scale, finite event-location precision and floating-point arithmetic limit further improvement; the last refinement does not reduce the error monotonically. The $`0.002\ \mathrm{s}`$ timestep is sufficiently accurate for this particular trajectory relative to the convergence tolerances used below.

### 3.4. Walking convergence and decay of rocking motion

**Experiment.** Starting from $`\theta_0=\gamma-\alpha`$, compare an initial velocity of $`2\ \mathrm{rad/s}`$ with an initial velocity of $`0.1\ \mathrm{rad/s}`$. Record 20 impacts for the first trajectory and 30 for the second, without imposing a resting cutoff on either recorded sequence.

**Expected result.** The larger initial velocity should produce sustained walking. Successive forward post-impact velocities should approach the value obtained by balancing the gain during stance against collision losses:

```math
\omega^*=\sqrt{
\frac{4g}{l}
\frac{\cos^2(2\alpha)}{1-\cos^2(2\alpha)}
\sin\alpha\sin\gamma
}
=1.144016619951\ \mathrm{rad/s}.
```

The smaller initial velocity should fail to cross the upright position, reverse direction, and produce alternating backward and forward impacts with decreasing speed.

**Observed result.** All 20 impacts in the walking trajectory were forward. Its final post-impact velocity was $`1.144017741681\ \mathrm{rad/s}`$, approximately $`1.12\times10^{-6}\ \mathrm{rad/s}`$ above the analytical value, as illustrated in the center panel of Figure 2. The rocking trajectory alternated impact directions throughout, and the post-impact speed decreased monotonically from $`0.1`$ to $`3.0517\times10^{-6}\ \mathrm{rad/s}`$ after 30 impacts; the right panel of Figure 2 shows this decay. The observed behavior agrees with both expectations.

### 3.5. Convergence classification and unresolved trajectories

**Experiment.** Pass the two initial conditions from Section 3.4 to the actual basin classifier. Then repeat with a deliberately short horizon of $`0.001\ \mathrm{s}`$. Finally, compare basin classifications on a $`21\times20`$ validation grid spanning $`\theta_0\in[\gamma-\alpha,\gamma+\alpha]`$ and $`\omega_0\in[-3,3]\ \mathrm{rad/s}`$ at timesteps of $`0.002`$ and $`0.001\ \mathrm{s}`$, with a $`30\ \mathrm{s}`$ time limit.

**Expected result.** Walking and rest should be reported only after their respective convergence tests pass. For successive post-impact velocities, the comparison tolerance is

```math
|\omega_{k+1}-\omega_k|
\leq 10^{-4}+10^{-3}\max(|\omega_k|,|\omega_{k+1}|).
```

Walking requires five consecutive successful comparisons between forward impacts, with a backward impact resetting the counter. Rest requires three consecutive successful comparisons between alternating impacts, together with decreasing speed no greater than $`10^{-4}\ \mathrm{rad/s}`$ and a geometry supporting two-contact rest. A short timeout should remain unresolved. Reducing the integration timestep should preserve the classifications on the validation grid.

**Observed result.** The first trajectory was classified as walking, with a converged post-impact velocity of $`1.144088408351\ \mathrm{rad/s}`$. The second was classified as rest and assigned the limiting value zero. Both remained unresolved with the deliberately short horizon. All 420 validation-grid classifications agreed between the two timesteps.

These results support the numerical choices for the tested conditions. They do not establish exact basin boundaries or prove convergence for every possible initial state. Later basin plots retain an unresolved category, and boundary resolution must also be checked against the initial-state grid spacing.

## 4. Regions of attraction

### 4.1. Attractors and the sampled state space

The region of attraction (RoA), or basin of attraction, of an attractor is the set of initial states whose trajectories approach it. Here the state is $`(\theta,\omega)`$, with the stance-spoke coordinate relabeled at every impact. Walking is therefore periodic in this reduced state space even though the wheel continues to translate downhill.

For the reference parameters $`N=8`$, $`\alpha=22.5^\circ`$, and $`\gamma=5^\circ`$, we expect two stable attractors:

- **Two-contact rest:** rocking decays through repeated impacts until the wheel rests on adjacent spokes. The states $`(\gamma-\alpha,0)`$ and $`(\gamma+\alpha,0)`$ describe the same resting configuration using different stance spokes; they are not two distinct attractors.
- **Periodic forward walking:** the continuous stance motion and forward impact reset repeat from step to step. Angular velocity varies throughout each step, but the post-impact velocity approaches a fixed value, approximately $`1.14402\ \mathrm{rad/s}`$ for these parameters.

The upright state $`(0,0)`$ is also stationary, but it is unstable and is classified separately rather than counted as an attractor. In particular, $`\gamma<\alpha`$ permits stable two-contact rest; it does not exclude a coexisting walking cycle.

To estimate both basins, [state_space_basins.py](state_space_basins.py) samples

```math
\theta_0\in[\gamma-\alpha,\gamma+\alpha]
=[-17.5^\circ,27.5^\circ],
\qquad
\omega_0\in[-6,3]\ \mathrm{rad/s}.
```

Each axis initially contains 401 uniformly spaced values, with spacings $`0.1125^\circ`$ and $`0.0225\ \mathrm{rad/s}`$, respectively. Adding zero explicitly to each axis gives a $`402\times402`$ grid containing **161,604 initial states**. Including negative velocities allows the map to capture uphill motion, reversal, and backward impacts. The velocity interval is a finite observation window, not a physical bound on the model's state space.

### 4.2. Simulation and convergence criteria

Every nonstationary grid point is integrated using RK4 with a maximum step of $`0.002\ \mathrm{s}`$. When a step crosses either contact guard, 30 bisection iterations locate the contact within that step before the corresponding reset is applied. The two exact resting-coordinate states and the exact upright equilibrium are assigned their known labels directly; the remaining 161,601 states are simulated.

The convergence tests from Section 3.5 determine when each trajectory can stop:

| Classification | Required evidence |
| --- | --- |
| Walking | Five consecutive comparisons of forward post-impact velocities satisfy the absolute/relative tolerance; a backward impact resets the walking test. |
| Rest | Three consecutive comparisons of alternating-impact velocities satisfy the tolerance, with decreasing post-impact speed no greater than $`10^{-4}\ \mathrm{rad/s}`$. |
| Unresolved | Neither convergence test passes before the allowed simulation time. |

The comparison uses an absolute tolerance of $`10^{-4}\ \mathrm{rad/s}`$ and a relative tolerance of $`10^{-3}`$. Rest is interpreted as the limiting two-contact configuration, not as an equilibrium of the single-contact differential equation. Walking convergence is tested at a consistent phase of the cycle, rather than requiring the instantaneous velocity to become constant.

The initial time limit is $`30\ \mathrm{s}`$. If needed, unresolved initial conditions are simulated again from their original states with limits of 60 and then 120 seconds. A timeout is never automatically labeled as rest. Computation is accelerated by vectorizing independent trajectories and removing already-classified states from further integration.

### 4.3. Basin map and numerical results

Figure 3 compares the initial-state attraction map in panel (a) with representative trajectories in panel (b), connecting the basin classifications to the underlying motion.

![Side-by-side comparison of the attraction map and phase portrait](figures/basins_and_phase_portrait.png)

*Figure 3. Side-by-side views of attraction and motion for the same reference parameters. (a) Left: estimated regions of attraction. Lavender denotes convergence to two-contact rest, and teal denotes convergence to periodic forward walking. The dark-green curve is the stance phase of the walking limit cycle, not a basin boundary; its reset connector is omitted. (b) Right: phase portrait showing walking and resting transients, constant-energy contours, and the red walking limit cycle, including its reset. The left angle axis is in degrees and the right in radians; the velocity ranges differ to show the full sampled basin window on the left and trajectory detail on the right.*

In panel (a), the horizontal limits are exactly the admissible stance-angle interval, $`[\gamma-\alpha,\gamma+\alpha]=[-17.5^\circ,27.5^\circ]`$. Basin boundaries are the interfaces between the background colors. The resting configurations are marked in panel (b), and the classification counts are reported below.

To read the overlay, distinguish an **initial condition** from an **instantaneous state**. The background color at $`(\theta_0,\omega_0)`$ tells which attractor a simulation starting there approaches. The solid curve instead consists of the instantaneous states $`(\theta(t),\omega(t))`$ visited during steady walking, drawn on the same angle–velocity axes. A trajectory starting elsewhere in the teal region generally does not follow this curve immediately; it approaches the repeating cycle after a transient.

On the steady cycle, a step begins near $`(-17.5^\circ,1.144\ \mathrm{rad/s})`$. The state moves along the curve from left to right, slowing as the hub rises toward $`\theta=0`$ and accelerating after it passes the upright position. Just before the next impact it reaches approximately $`(27.5^\circ,1.618\ \mathrm{rad/s})`$. Impact switches the stance spoke and reduces velocity, returning the state to approximately $`(-17.5^\circ,1.144\ \mathrm{rad/s})`$, where the next step begins. The curve therefore repeats from step to step. It is a **hybrid limit cycle**: the continuous curve is closed by an instantaneous reset, even though that reset is not drawn. This is a state-space trajectory, not the hub's path through physical space or a velocity-versus-time plot.

All simulated states satisfied a convergence test within the first 30-second pass; no longer-horizon retries were required.

| Final classification | Grid points | Fraction of sampled points |
| --- | ---: | ---: |
| Two-contact rest | 99,137 | 61.3456% |
| Periodic forward walking | 62,466 | 38.6537% |
| Unstable upright equilibrium | 1 | 0.0006% |
| Unresolved | 0 | 0% |
| **Total** | **161,604** | **100%** |

The walking trajectories had terminal post-impact velocities between $`1.143939`$ and $`1.144095\ \mathrm{rad/s}`$, with a median of $`1.144060\ \mathrm{rad/s}`$, consistent with convergence to the same cycle within the selected tolerances. These small terminal differences are numerical stopping differences, not evidence of distinct walking attractors.

### 4.4. Physical interpretation and limitations

The large teal region at positive initial velocity shows that sufficient downhill motion can carry the hub past the upright position and establish repeated forward steps. Gravity supplies energy during each downhill step, while impacts dissipate energy. Their balance supports the walking cycle. States in the resting basin instead lose enough energy to enter decaying rocking motion between adjacent contacts.

The thin teal bands at negative initial velocity show that initially moving uphill does not necessarily imply eventual rest. After backward steps and a reversal, some trajectories retain enough energy to enter the forward-walking basin. The outcome depends on both initial angle and velocity, because these determine the sequence of contacts and associated energy losses. Thus neither the sign nor the magnitude of initial velocity alone classifies the attractor. The separated bands reflect different impact histories; the plot alone does not establish chaotic or fractal basin structure.

This is a numerical estimate for one parameter set and a finite grid. The percentages describe sampled-point counts, not universal probabilities of walking or exact basin areas. Narrow regions and trajectories close to basin boundaries may require finer initial-state sampling, smaller timesteps, tighter convergence tolerances, or longer horizons. The timestep comparison in Section 3.5 supports the classifications on its smaller validation grid, but is not a full-resolution boundary-convergence study.

To reproduce the map and save its classification data, run from the repository root:

```console
uv run python state_space_basins.py
```

The outputs are [the basin figure](figures/state_space_basins.png) and [the grid and classification data](figures/state_space_basins.npz). Section 6 examines how these basins and the walking cycle change with slope and number of spokes.

### 4.5. Phase portrait and the walking limit cycle

Figure 3(b) shows the phase portrait alongside the attraction map. Gray dashed curves are constant-energy contours during stance. Blue solid curves show five successive stance phases starting from the blue star R3 at $`(\gamma-\alpha,1.5\ \mathrm{rad/s})`$. Green (R1) and brown (R2) curves show eight impacts of two trajectories approaching rest. Colored dashed connectors represent impact resets, and stars labeled R1, R2, and R3 mark the three initial states. The red solid curve and red dashed reset together represent the steady walking cycle; red arrows indicate its direction. Black squares mark two coordinate representations of the same resting configuration, and black dashed vertical lines mark the contact-coordinate limits. Angles are in radians in this panel.

Read each blue walking stance curve from left to right, then follow its reset connector back to the left boundary. The connectors represent instantaneous jumps, not intermediate states visited by the wheel. Successive blue stance curves approach the red curve as the post-impact velocity approaches $`1.14401662\ \mathrm{rad/s}`$. After the five illustrated transient steps, it is $`1.15680047\ \mathrm{rad/s}`$; the transient has not yet reached the exact cycle.

The additional initial states illustrate the other stable attractor:

| Trajectory | Initial angle | Initial angular velocity | Converged attractor |
| --- | --- | --- | --- |
| Blue (R3) | $`-17.5^\circ`$ | $`1.5\ \mathrm{rad/s}`$ | Forward walking |
| Green (R1) | $`-17.5^\circ`$ | $`0.35\ \mathrm{rad/s}`$ | Two-contact rest |
| Brown (R2) | $`0^\circ`$ | $`-0.5\ \mathrm{rad/s}`$ | Two-contact rest |

R1 initially moves forward but cannot reach upright; it reverses and hits the backward-contact boundary. R2 starts at the upright angle with nonzero backward velocity, so it is not the unstable equilibrium $`(0,0)`$; it moves toward backward contact. Both subsequently alternate backward and forward impacts while their rocking amplitude decreases. Continuous motion goes rightward when $`\omega>0`$ and leftward when $`\omega<0`$, reversing at a turning point where $`\omega=0`$. After the eight plotted impacts, their post-impact speed magnitudes are approximately $`0.021875`$ and $`0.067259\ \mathrm{rad/s}`$, respectively. These finite traces illustrate decay rather than exact rest; separate simulations using the convergence tests in Section 3.5 confirm both resting classifications. The two endpoint squares are equivalent contact-coordinate representations, not separate resting attractors.

The gray contours use $`E_{\mathrm{stance}}=\tfrac12 ml^2\omega^2+mgl\cos\theta`$. This energy is constant between impacts, but its reference height changes when the stance contact is relabeled. Consequently, contour values on opposite sides of a reset should not be directly interpreted as absolute mechanical-energy loss without accounting for the contact-height offset.

The highlighted cycle is initialized at the analytical post-impact fixed velocity from Section 3.4 and traced numerically using RK4 and refined contact detection. Its velocity closure error after one impact is approximately $`1.62\times10^{-12}\ \mathrm{rad/s}`$, and the maximum within-stance energy variation over the plotted trajectories is $`1.35\times10^{-11}\ \mathrm{J}`$. Figure 3(b) shows evolution toward the attractors, whereas Figure 3(a) classifies the long-term outcomes of different initial conditions.

To reproduce the standalone phase portrait:

```console
uv run python phase_portrait.py
```

To regenerate the side-by-side Figure 3 using the saved basin data:

```console
uv run python compare_attractors.py
```

The combined image is saved as [basins_and_phase_portrait.png](figures/basins_and_phase_portrait.png). Both panels have equal plotting-area dimensions, with their legends placed below; the individual figures remain available separately.

## 5. Poincaré section and impact-to-impact return map

### 5.1. Sampling after every contact

Record the state immediately after **every** impact, forward or backward. The post-impact section is the union of two components:

```math
\Sigma^+=
\{(\theta,\dot\theta):\theta=\gamma-\alpha,\ \dot\theta>0\}
\quad\cup\quad
\{(\theta,\dot\theta):\theta=\gamma+\alpha,\ \dot\theta<0\}.
```

For the spoke counts considered here, $`\cos(2\alpha)>0`$, so the reset preserves the sign of the pre-impact velocity. Positive post-impact velocity identifies a forward impact and negative velocity identifies a backward impact. The signed angular velocity determines the post-impact angle:

```math
\theta_k=\gamma-\operatorname{sgn}(\dot\theta_k)\alpha,\qquad
\dot\theta_{k+1}=P(\dot\theta_k).
```

Here $`\dot\theta_k`$ means the angular velocity immediately after impact $`k`$; the superscript $`+`$ is omitted for readability. The simulator starts at the corresponding endpoint, integrates until the **very next contact of either direction**, and applies exactly one reset. It does not skip backward impacts or wait specifically for a forward impact.

This produces a one-dimensional signed map even though the section has two components. For the studied regime $`0<\gamma<\alpha`$, at zero velocity the two endpoint coordinates describe the same two-contact resting configuration; no new impact occurs there, so rest is treated as a limiting state rather than an observed event return.

### 5.2. Crossing thresholds and analytical map

For $`0<\gamma<\alpha`$, the hub must overcome a potential-energy barrier to pass upright. The required speed magnitudes differ according to the initial contact:

```math
\dot\theta_{\min,\mathrm{f}}
=\sqrt{\frac{2g}{l}[1-\cos(\gamma-\alpha)]}
=0.9529288674\ \mathrm{rad/s},
```

```math
\dot\theta_{\min,\mathrm{b}}
=\sqrt{\frac{2g}{l}[1-\cos(\gamma+\alpha)]}
=1.4889081412\ \mathrm{rad/s}.
```

The corresponding signed thresholds are $`+\dot\theta_{\min,\mathrm{f}}`$ and $`-\dot\theta_{\min,\mathrm{b}}`$. At either threshold, the trajectory approaches upright asymptotically and does not produce a next impact in finite time. The map is undefined at these two inputs.

**Extension to steeper slopes.** The positive forward threshold above applies only when the post-forward-impact angle $`\gamma-\alpha`$ is negative, so the hub must first climb toward upright. For ordinary downhill inclinations $`0<\gamma<\pi/2`$, the forward threshold can be written more generally as

```math
\dot\theta_{\min,\mathrm{f}}=
\begin{cases}
\sqrt{\dfrac{2g}{l}[1-\cos(\gamma-\alpha)]}, & 0<\gamma<\alpha,\\
0, & \gamma\geq\alpha.
\end{cases}
```

When $`\gamma>\alpha`$, the new stance angle satisfies $`\theta^+=\gamma-\alpha>0`$: the hub is already beyond upright in the downhill direction. There is no potential-energy barrier ahead before the next forward contact. Even with zero initial angular velocity, the pinned-stance model gives $`\ddot\theta=(g/l)\sin(\gamma-\alpha)>0`$, so the wheel starts moving forward. Thus no positive initial angular velocity is required, and $`\dot\theta_{\min,\mathrm{f}}=0`$. Substituting $`\gamma>\alpha`$ into the square-root expression alone would incorrectly impose an uphill barrier that is not encountered by the forward trajectory.

The boundary case $`\gamma=\alpha`$ is different: the post-impact configuration is exactly upright. The threshold is still zero as an infimum, but an exactly zero initial velocity remains at the unstable equilibrium in the ideal model; any strictly positive initial velocity initiates a forward step.

This extension concerns the forward threshold only. The signed return-map branches derived below, the two-contact-rest interpretation, and the numerical studies in this report remain restricted to $`0<\gamma<\alpha`$; they must not be extrapolated unchanged to $`\gamma\geq\alpha`$.

Define the dimensionless impact factor and squared-speed increment

```math
c=\cos(2\alpha),\qquad A=\frac{4g}{l}\sin\alpha\sin\gamma.
```

The energy and reset laws then give

```math
\boxed{
P(\dot\theta_k)=
\begin{cases}
-c\sqrt{\dot\theta_k^2-A},
&\dot\theta_k<-\dot\theta_{\min,\mathrm{b}},\\
-c\,\dot\theta_k,
&-\dot\theta_{\min,\mathrm{b}}<\dot\theta_k<\dot\theta_{\min,\mathrm{f}},
\quad \dot\theta_k\ne0,\\
c\sqrt{\dot\theta_k^2+A},
&\dot\theta_k>\dot\theta_{\min,\mathrm{f}}.
\end{cases}
}
```

Each branch corresponds to a different contact sequence:

| Branch | Motion before the next impact | Sign of the next post-impact velocity |
| --- | --- | --- |
| Large negative input | Continues uphill to the next backward contact; stance motion loses kinetic energy while gaining height | Negative |
| Intermediate nonzero input | Fails to pass upright, reverses, and returns to the same contact-angle boundary before switching spokes | Opposite to the input |
| Large positive input | Continues downhill to the next forward contact; stance motion gains kinetic energy while losing height | Positive |

For reversal, the trajectory returns to its starting contact angle with equal speed magnitude but opposite direction. The single reset then gives $`P(\dot\theta_k)=-c\dot\theta_k`$. For $`N=8`$, this is approximately $`-0.707107\dot\theta_k`$, **not** $`+0.5\dot\theta_k`$. The latter describes two successive reversing impacts, when the trajectory remains on the rocking branch.

Near zero, alternating signs and shrinking magnitudes describe decay toward rest. Setting $`P(0)=0`$ would extend this branch continuously to the resting configuration, but the numerical event map stores NaN at zero because there is no actual next impact. Negative inputs are otherwise fully included. At the two crossing thresholds, NaN separates discontinuous branches so the plotted curves do not incorrectly join them. Timeouts raise errors instead of being assigned a mapped speed.

### 5.3. Numerical map and attractors

[poincare_map.py](poincare_map.py) samples the full displayed interval $`\dot\theta_k\in[-3,3]\ \mathrm{rad/s}`$. Both axes use this finite display window; it is not a physical bound on angular velocity. The grid contains 367 inputs: 121 continuing-backward samples, 61 negative reversing samples, zero, 61 positive reversing samples, 121 continuing-forward samples, and the two thresholds. The 364 valid returns are integrated with RK4 at $`\Delta t=0.002\ \mathrm{s}`$ and bisection-refined contact detection. Every evaluation includes exactly one impact.

Figure 4(a) compares the numerical next-impact map with its analytical branches and identifies the walking fixed point. Figure 4(b) examines how perturbations around that fixed point change after one step, providing the local stability estimate discussed in Section 5.4.

![Signed impact-to-impact return map and local walking convergence](figures/poincare_map.png)

*Figure 4. (a) Signed next-impact map for forward and backward motion. Pale teal vertical strips mark initial post-impact velocities that converge to the walking cycle. Blue numerical samples (every fourth sample displayed) agree with the red analytical branches. Dotted vertical lines mark the two undefined crossing thresholds. The central descending branch reverses the sign of angular velocity. The red intersection with the dashed identity line is the walking fixed point; the hollow origin is the limiting resting state. (b) Local perturbations around the walking fixed point shrink approximately by half per forward walking step.*

The background shading is obtained from a separate convergence classification of 1,801 post-impact states over $`\dot\theta_k\in[-3,3]\ \mathrm{rad/s}`$, spaced by approximately $`0.003333\ \mathrm{rad/s}`$. Positive velocities start at $`\theta=\gamma-\alpha`$ and negative velocities at $`\theta=\gamma+\alpha`$. These trajectories use the same convergence tests as Section 6, with time limits of 30, 60, and 120 seconds; unresolved samples, if present, are shown in pale orange rather than assigned to rest. Strip boundaries are estimated midway between neighboring samples. Shading depends only on the horizontal coordinate: it indicates the eventual outcome from that initial state, not a two-dimensional basin in independently chosen current and next velocities. Negative shaded bands represent trajectories that initially move backward but eventually reach forward walking. The classifications are saved in [poincare_map_basins.npz](figures/poincare_map_basins.npz).

The positive walking fixed point is unchanged, since a walking step still contains one forward impact:

```math
\dot\theta^*
=\sqrt{\frac{4g}{l}\frac{\cos^2(2\alpha)}{1-\cos^2(2\alpha)}
\sin\alpha\sin\gamma}
=1.144016619951\ \mathrm{rad/s}.
```

It is physically admissible because $`\dot\theta^*>\dot\theta_{\min,\mathrm{f}}`$. The numerical fixed point is located independently by 36 bisection iterations on the simulated residual $`P(\dot\theta)-\dot\theta`$ within the forward-walking branch.

| Quantity | Numerical result |
| --- | --- |
| Walking fixed point | $`1.144016619947\ \mathrm{rad/s}`$ |
| Fixed-point residual $`\lvert P(\dot\theta^*)-\dot\theta^*\rvert`$ | $`6.28\times10^{-12}\ \mathrm{rad/s}`$ |
| Maximum numerical/analytical map difference over 364 valid samples | $`6.92\times10^{-12}\ \mathrm{rad/s}`$ |

The walking branch lies above the identity line below its fixed point and below the identity line above it, consistent with attraction from both sides. Continuing backward steps reduce speed magnitude; there is no negative walking fixed point on this positive slope. Subsequent reversal can lead either to decaying rocking or to forward walking, depending on the returned velocity. Thus not every point on the reversal branch necessarily remains there forever. Sufficiently small velocities, however, stay in the reversing regime and contract toward rest with an alternating sign.

Unlike a forward-contact-only map, this signed map explicitly displays intermediate backward impacts. Its local event-map slope near rest is $`-c`$, while the multiplier of the forward-walking cycle is $`c^2`$, as estimated next.

### 5.4. Floquet multiplier and local convergence

The perturbations plotted in Figure 4(b) return closer to zero than the dashed identity line would predict. Their local input–output slope quantifies contraction toward the walking cycle.

Write the post-impact velocity as $`\dot\theta_k=\dot\theta^*+\delta \dot\theta_k`$. Linearizing the return map gives

```math
\delta \dot\theta_{k+1}=\mu\,\delta \dot\theta_k+O(\delta \dot\theta_k^2),
\qquad \mu=P'(\dot\theta^*).
```

This scalar derivative is the nontrivial Floquet multiplier of the walking cycle. Sampling at a fixed event removes the arbitrary phase along the periodic motion. To estimate it numerically, perturb the simulated fixed point by $`\pm\varepsilon`$ and calculate

```math
\mu_-\approx\frac{P(\dot\theta^*)-P(\dot\theta^*-\varepsilon)}{\varepsilon},\qquad
\mu_+\approx\frac{P(\dot\theta^*+\varepsilon)-P(\dot\theta^*)}{\varepsilon},
```

```math
\mu_{\mathrm{centered}}\approx
\frac{P(\dot\theta^*+\varepsilon)-P(\dot\theta^*-\varepsilon)}{2\varepsilon}.
```

All perturbed inputs remain above $`\dot\theta_{\min,\mathrm{f}}`$. The numerical estimates are:

| $`\varepsilon`$ (rad/s) | Left slope $`\mu_-`$ | Right slope $`\mu_+`$ | Centered slope |
| --- | --- | --- | --- |
| $`10^{-2}`$ | 0.498902568 | 0.501087881 | 0.499995224 |
| $`10^{-3}`$ | 0.499890689 | 0.500109211 | 0.499999950 |
| $`10^{-4}`$ | 0.499989073 | 0.500010899 | 0.499999986 |
| $`10^{-5}`$ | 0.499998975 | 0.500000566 | 0.499999770 |

As an independent check, differentiating the forward-walking branch gives

```math
P'(\dot\theta)=\cos(2\alpha)\frac{\dot\theta}{\sqrt{\dot\theta^2+\frac{4g}{l}\sin\alpha\sin\gamma}},
\qquad
\boxed{\mu=P'(\dot\theta^*)=\cos^2(2\alpha)=0.5}.
```

Since $`|\mu|<1`$, the walking cycle is locally attracting. Its positive multiplier means that small velocity errors retain their sign while shrinking: to first order, their magnitude halves each step. This describes convergence **per step**, not per second; step duration also matters for convergence measured in time.

Halving the integration timestep to $`0.001\ \mathrm{s}`$ gives a centered estimate of $`0.5000000011`$ at $`\varepsilon=10^{-4}\ \mathrm{rad/s}`$, with the same numerical fixed point to the reported precision. The smallest perturbation does not give the most accurate estimate because finite event-location and floating-point errors are amplified by division by $`\varepsilon`$. The agreement across perturbations, timestep refinement, and the analytical derivative supports the estimate $`\mu\approx0.5`$.

To reproduce the figure, numerical data, and validation measurements:

```console
uv run python poincare_map.py
```

Outputs are [the return-map figure](figures/poincare_map.png), [the sampled map data](figures/poincare_map.npz), and [the numerical measurements](figures/poincare_map.json). Section 6 examines how slope and spoke count affect the admissible walking domain, the basins of attraction, and the local convergence multiplier.

## 6. Effects of slope and number of spokes

### 6.1. Controlled numerical experiments

Two sweeps separate the effects of terrain and wheel geometry. Gravity, mass, and spoke length remain fixed at $`9.81\ \mathrm{m/s^2}`$, $`1\ \mathrm{kg}`$, and $`1\ \mathrm{m}`$.

| Study | Varied parameter | Fixed parameter |
| --- | --- | --- |
| Slope sweep | 15 inclinations between $`1^\circ`$ and $`14^\circ`$, with extra samples near the walking threshold | $`N=8`$, $`\alpha=22.5^\circ`$ |
| Spoke-count sweep | Every integer $`N=6,\ldots,12`$, updating $`\alpha=\pi/N`$ | $`\gamma=5^\circ`$ |

All cases satisfy $`0<\gamma<\alpha`$, where adjacent spokes can support two-contact rest. The experiments do not address the different contact geometry at $`\gamma\geq\alpha`$.

To compare initial conditions consistently, define a normalized stance angle

```math
q_0=\frac{\theta_0-\gamma}{\alpha}\in[-1,1],
\qquad \theta_0=\gamma+\alpha q_0.
```

Each case uses 61 equally spaced normalized angles and 81 initial angular velocities in $`[-3,3]\ \mathrm{rad/s}`$, giving 4,941 grid points. Thus every wheel is sampled over its full admissible stance-angle interval, including forward and backward initial motion. The walking fraction is the number of points classified as walking divided by 4,941. It measures the sampled basin within this window, not a universal probability or an absolute physical state-space area.

The RK4 timestep is $`0.002\ \mathrm{s}`$, with refined contact times. The convergence logic remains the same as in Section 3, but the absolute and relative velocity tolerances are tightened to $`10^{-6}\ \mathrm{rad/s}`$ and $`10^{-5}`$ to reduce premature classification near the walking threshold. Walking requires five successful consecutive comparisons; rest requires three quiet, decreasing, alternating-impact comparisons. Exact resting states and any sampled upright equilibrium are handled separately. Unresolved trajectories are retried from their original states with time limits of 30, 60, and 120 seconds; a timeout is never assigned to rest automatically.

For each existing walking cycle, its fixed point is found by bisection of the numerically simulated map. Perturbations on both sides then estimate the Floquet multiplier with the centered difference from Section 5. The perturbation is $`10^{-4}\ \mathrm{rad/s}`$ or smaller if necessary to remain within the walking branch. A nonexistent walking cycle has **undefined** walking speed and multiplier, rather than zero values.

### 6.2. Analytical predictions

A finite-period walking cycle requires its post-impact fixed velocity to exceed the forward upright-crossing threshold:

```math
(\dot\theta^*)^2=
\frac{4g}{l}\frac{\cos^2(2\alpha)}{1-\cos^2(2\alpha)}\sin\alpha\sin\gamma
>
\frac{2g}{l}[1-\cos(\gamma-\alpha)].
```

Solving the equality for the onset slope gives

```math
\boxed{\gamma_{\mathrm{crit}}(N)
=2\arctan\!\left[\tan\left(\frac{\alpha}{2}\right)\tan^2\alpha\right],
\qquad \alpha=\frac{\pi}{N}.}
```

Walking requires $`\gamma>\gamma_{\mathrm{crit}}`$ in the studied regime. At equality, the candidate trajectory approaches upright asymptotically; it is not a finite-period walking cycle. For $`N=8`$, $`\gamma_{\mathrm{crit}}=3.909259664^\circ`$. The slope sweep includes points $`0.20^\circ`$ and $`0.05^\circ`$ on either side of this prediction, without treating the exact singular case as a periodic orbit.

For an existing walking cycle,

```math
\boxed{\mu=\cos^2(2\alpha)=\cos^2\left(\frac{2\pi}{N}\right).}
```

Thus slope changes the walking speed and its existence condition, but not the local multiplier at fixed $`N`$. Spoke count changes both geometry and collision losses, and therefore changes the multiplier. These are predictions to compare against the simulated maps, not substitutes for the basin integrations.

### 6.3. Changes in the attraction basins

Figure 5 compares representative attraction maps from the two sweeps. Its top row isolates the effect of slope, while its bottom row isolates the effect of spoke count.

![Representative normalized attraction maps for slope and spoke-count sweeps](figures/parameter_basins.png)

*Figure 5. Representative basin maps on the common normalized grid. Top row: varying slope with $`N=8`$. Bottom row: varying spoke count at $`\gamma=5^\circ`$. Lavender denotes rest and teal denotes forward walking. Small white cells, where present, mark sampled exact upright equilibria and are not additional attractors. The duplicated $`N=8`$, $`\gamma=5^\circ`$ panel provides a reference between the two sweeps. Panel percentages count sampled initial states.*

The slope sweep found no walking trajectories at any sampled slope below $`\gamma_{\mathrm{crit}}`$. Just above it, at $`3.95926^\circ`$, about 40.01% of sampled states converged to walking. The top row of Figure 5 illustrates the emergence and subsequent growth of the walking basin as slope increases, including regions with initially negative velocity. Selected measurements are:

| Slope $`\gamma`$ | Walking fraction | Walking post-impact $`\dot\theta^*`$ (rad/s) | Numerical $`\mu`$ |
| --- | ---: | ---: | ---: |
| $`3.85926^\circ`$ | 0.00% | Undefined | Undefined |
| $`3.95926^\circ`$ | 40.01% | 1.018256 | 0.500000 |
| $`5^\circ`$ | 48.29% | 1.144017 | 0.500000 |
| $`8^\circ`$ | 65.65% | 1.445646 | 0.500000 |
| $`12^\circ`$ | 82.66% | 1.766948 | 0.500000 |
| $`14^\circ`$ | 88.95% | 1.905996 | 0.500000 |

A steeper slope supplies more gravitational energy during each forward step and lowers the barrier encountered just after a forward impact. This supports a faster cycle and allows more initial states to reach it. The apparent sharp basin change near onset should not be interpreted as a resolved continuous transition: the two nearest samples bracket the analytical threshold with a $`0.1^\circ`$ gap. Long transients also make this region particularly sensitive to stopping criteria.

At fixed $`\gamma=5^\circ`$, the six- and seven-spoke wheels have no admissible walking cycle. Their required slopes exceed $`5^\circ`$, and their sampled non-equilibrium states converged to rest. Walking appears for $`N\geq8`$, and its sampled basin grows as $`N`$ increases, as illustrated by the representative maps in the bottom row of Figure 5:

| $`N`$ | $`\gamma_{\mathrm{crit}}`$ (deg) | Walking fraction at $`5^\circ`$ | Walking $`\dot\theta^*`$ (rad/s) | Numerical $`\mu`$ |
| --- | ---: | ---: | ---: | ---: |
| 6 | 10.2078 | 0.00% | Undefined | Undefined |
| 7 | 6.0600 | 0.00% | Undefined | Undefined |
| 8 | 3.9093 | 48.29% | 1.144017 | 0.500000 |
| 9 | 2.6762 | 62.03% | 1.288917 | 0.586824 |
| 10 | 1.9159 | 73.87% | 1.414955 | 0.654509 |
| 11 | 1.4204 | 78.63% | 1.527387 | 0.707708 |
| 12 | 1.0831 | 83.91% | 1.629563 | 0.750000 |

More spokes reduce the angle between successive stance spokes. Each collision retains a larger fraction of kinetic energy, and the geometric barrier is smaller. These effects lower the minimum walking slope and enlarge the walking basin in the sampled cases. The banded regions at negative initial velocity reflect different backward-contact and reversal histories, not additional walking attractors.

The reference walking fraction here, 48.29%, differs from Section 4's 38.65% because this controlled sweep uses $`\dot\theta_0\in[-3,3]`$ instead of $`[-6,3]\ \mathrm{rad/s}`$, as well as a different grid resolution and tighter tolerances. Fractions from different initial-state windows are not directly comparable.

### 6.4. Walking speed versus local convergence

Figure 6 brings together the basin fractions, walking velocities, and Floquet multipliers for every sampled parameter value, allowing global attraction and local convergence to be compared directly.

![Basin fractions, walking velocities, and numerical Floquet multipliers across both sweeps](figures/parameter_trends.png)

*Figure 6. Top: slope sweep at $`N=8`$. Bottom: spoke-count sweep at $`\gamma=5^\circ`$. The first column reports sampled basin fractions, the second the steady post-impact walking velocity, and the third the walking Floquet multiplier. Open blue circles are numerical fixed-point or perturbation measurements; gray curves are analytical predictions. Shaded areas in the last two columns indicate cases without a walking cycle, not zero speed or zero multiplier. The dotted line in the slope panels marks $`\gamma_{\mathrm{crit}}`$. Lines between samples guide the eye; spoke count is discrete.*

In the top row of Figure 6, the numerical walking multiplier remains approximately 0.5 at fixed $`N=8`$, even as the walking speed and basin fraction increase with slope. A larger basin therefore does not imply a smaller local multiplier: basin size concerns which initial conditions reach the attractor, while the multiplier concerns small disturbances already near it.

In the bottom row of Figure 6, increasing $`N`$ from 8 to 12 at fixed slope increases the multiplier from 0.5 to 0.75. The wheel loses less energy per collision, but perturbations also decay less per step. For example, a sufficiently small post-impact velocity error is approximately halved each step for $`N=8`$, while about 75% remains after each step for $`N=12`$. Both cycles are locally stable because $`|\mu|<1`$. This comparison is per walking step; it does not establish convergence rates per second, which also depend on step duration.

Across all existing cycles in these sweeps, the maximum difference between numerical and analytical fixed velocities was $`1.34\times10^{-11}\ \mathrm{rad/s}`$, and the maximum multiplier difference was $`3.04\times10^{-8}`$. No walking multiplier is assigned to $`N=6`$ or 7 at $`5^\circ`$, despite the fact that the expression $`\cos^2(2\pi/N)`$ can be evaluated algebraically.

### 6.5. Convergence and resolution checks

All integrated trajectories in the reported sweeps were classified within 60 seconds; no 120-second retry was needed. Near onset, at $`N=8`$ and $`\gamma=3.95926^\circ`$, 1,737 coarse-grid trajectories remained unresolved at 30 seconds and required the 60-second pass. This demonstrates why reaching a fixed time limit alone is not sufficient evidence of steady state.

Three refinement experiments checked the reported results:

| Check | Original walking fraction | Refined walking fraction | Classification disagreements at common grid points |
| --- | ---: | ---: | ---: |
| $`N=8`$, $`\gamma=5^\circ`$: halve timestep to $`0.001\ \mathrm{s}`$ | 48.2898% | 48.2898% | 0 / 4,941 |
| $`N=8`$, $`\gamma=3.95926^\circ`$: refine grid to $`121\times161`$ | 40.0121% | 40.0955% | 0 / 4,941 |
| $`N=12`$, $`\gamma=5^\circ`$: refine grid to $`121\times161`$ | 83.9101% | 84.0306% | 0 / 4,941 |

The finer grids contain 19,481 points each and halve both initial-state spacings. Walking fractions changed by approximately 0.083 and 0.120 percentage points, respectively, with no unresolved refined trajectories. These checks support the observed trends for the tested cases, but do not prove exact basin boundaries or convergence for every unsampled state. Particularly thin bands and conditions arbitrarily close to the walking threshold may need further refinement.

In summary, **steeper slopes and more spokes make walking accessible from more sampled initial states**, but their effects on local convergence differ: slope leaves the walking multiplier unchanged at fixed $`N`$, whereas adding spokes increases it and slows error decay per step. Stable two-contact rest coexists with walking throughout the sampled walking regimes.

### 6.6. Reproducing the study

Run both sweeps and the refinement checks with:

```console
uv run python -u parameter_study.py
```

To redraw the figures without repeating the integrations:

```console
uv run python parameter_study.py --plot-only
```

The results are saved as [basin panels](figures/parameter_basins.png), [parameter trends](figures/parameter_trends.png), [grid classifications](figures/parameter_study.npz), and [measurements and refinement results](figures/parameter_study.json).
