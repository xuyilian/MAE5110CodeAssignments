"""Plot stance-energy contours, a transient, and the hybrid walking cycle.

Run from the repository root: uv run python assignment_1/codes/phase_portrait.py
"""

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from gamma_sweep import REST, WALKING, classify_initial_states, locate_impacts
from integrators import rk4_step
from models import rimless_wheel as model
from state_space_basins import trace_walking_step

NUMBER_OF_SPOKES = 8
SLOPE_DEGREES = 5.0
INITIAL_POST_IMPACT_VELOCITY = 1.5  # rad/s
TRANSIENT_STEPS = 5
RESTING_IMPACTS = 8
OUTPUT_DIRECTORY = Path(__file__).resolve().parents[1] / "figures"


def calculate_walking_fixed_point(params):
    """Return the forward post-impact fixed point if a finite step exists."""
    alpha = params["half_spoke_angle"]
    gamma = params["slope_angle"]
    factor = np.cos(2 * alpha)
    if not (0 < factor < 1 and 0 < gamma < alpha):
        raise ValueError("This portrait requires 0 < gamma < alpha and 0 < cos(2 alpha) < 1")
    velocity_squared = (
        4 * params["gravity"] / params["spoke_length"]
        * factor**2 / (1 - factor**2) * np.sin(alpha) * np.sin(gamma)
    )
    barrier_squared = (
        2 * params["gravity"] / params["spoke_length"]
        * (1 - np.cos(gamma - alpha))
    )
    if velocity_squared <= barrier_squared:
        raise ValueError("The fixed point cannot cross upright in finite time")
    return np.sqrt(velocity_squared)


def calculate_stance_energy(states, params):
    """Energy relative to the current contact; its height offset changes at reset."""
    return params["hub_mass"] * (
        0.5 * params["spoke_length"]**2 * states[..., 1]**2
        + params["gravity"] * params["spoke_length"] * np.cos(states[..., 0])
    )


def trace_next_impact(initial_state, params, timestep=0.002, time_limit=30.0):
    """Trace continuous motion to either contact, without imposing a rest cutoff."""
    state = np.asarray(initial_state, dtype=float).copy()
    orbit = [state.copy()]
    elapsed = 0.0
    while elapsed < time_limit:
        step = min(timestep, time_limit - elapsed)
        trial = rk4_step(model.dynamics, elapsed, state, step, params)
        forward = model.detect_contact(trial, params)
        backward = model.detect_backward_contact(trial, params)
        if forward or backward:
            contact, _ = locate_impacts(
                state[:, None], np.array([step]), params, np.array([forward])
            )
            before = contact[:, 0]
            orbit.append(before.copy())
            reset = model.apply_impact_reset if forward else model.apply_backward_impact_reset
            return np.asarray(orbit), reset(before, params)
        state = trial
        orbit.append(state.copy())
        elapsed += step
    raise RuntimeError("No impact before the phase-portrait timeout")


def plot_phase_portrait(params, *, axis=None):
    alpha, gamma = params["half_spoke_angle"], params["slope_angle"]
    lower_angle, upper_angle = gamma - alpha, gamma + alpha
    fixed_velocity = calculate_walking_fixed_point(params)
    cycle, reset = trace_walking_step(fixed_velocity, params)
    closure_error = float(np.max(np.abs(reset - cycle[0])))
    if closure_error > 1e-7:
        raise RuntimeError(f"Walking cycle does not close: residual {closure_error:g}")

    transient = []
    velocity = INITIAL_POST_IMPACT_VELOCITY
    for _ in range(TRANSIENT_STEPS):
        orbit, after = trace_walking_step(velocity, params)
        transient.append((orbit, after))
        velocity = after[1]

    # Two different initial states in the resting basin, including backward motion.
    resting_examples = [
        (np.array([lower_angle, 0.35]), "#15803d", "R1"),
        (np.array([0.0, -0.5]), "#b45309", "R2"),
    ]
    initial_states = np.vstack([
        [lower_angle, INITIAL_POST_IMPACT_VELOCITY],
        *[initial for initial, _, _ in resting_examples],
    ])
    labels = classify_initial_states(initial_states[:, 0], initial_states[:, 1], params)
    if not np.array_equal(labels, [WALKING, REST, REST]):
        raise RuntimeError(f"Example states have unexpected attractor labels: {labels}")
    resting_traces = []
    for initial, color, name in resting_examples:
        state = initial.copy()
        segments = []
        for _ in range(RESTING_IMPACTS):
            orbit, state = trace_next_impact(state, params)
            segments.append((orbit, state))
        resting_traces.append((initial, color, name, segments))
        print(f"{name}: confirmed rest; speed after {RESTING_IMPACTS} impacts = {abs(state[1]):.8f} rad/s")

    standalone = axis is None
    if standalone:
        figure, axis = plt.subplots(figsize=(9, 8), constrained_layout=True)
    angle_grid, velocity_grid = np.meshgrid(
        np.linspace(lower_angle, upper_angle, 401), np.linspace(-2.2, 2.2, 401)
    )
    energy = calculate_stance_energy(np.stack([angle_grid, velocity_grid], axis=-1), params)
    upright_energy = params["hub_mass"] * params["gravity"] * params["spoke_length"]
    levels = np.unique(np.r_[np.linspace(energy.min(), energy.max(), 15), upright_energy])
    axis.contour(
        angle_grid, velocity_grid, energy, levels=levels,
        colors="0.7", linewidths=0.65, linestyles="dashed", zorder=1,
    )
    for orbit, after in transient:
        axis.plot(orbit[:, 0], orbit[:, 1], color="#2563eb", lw=1.4, zorder=3)
        axis.plot(
            [orbit[-1, 0], after[0]], [orbit[-1, 1], after[1]],
            color="#2563eb", ls="--", lw=0.9, alpha=0.65, zorder=2,
        )

    for initial, color, name, segments in resting_traces:
        for orbit, after in segments:
            axis.plot(orbit[:, 0], orbit[:, 1], color=color, lw=1.5, zorder=4)
            axis.plot([orbit[-1, 0], after[0]], [orbit[-1, 1], after[1]],
                      color=color, ls="--", lw=0.8, alpha=0.5, zorder=2)
        axis.scatter(*initial, marker="*", s=100, color=color, zorder=7)
        axis.annotate(name, initial, xytext=(7, 7), textcoords="offset points",
                      color=color, weight="bold", fontsize=14)
    axis.scatter([lower_angle, upper_angle], [0, 0], marker="s", s=40,
                 color="black", zorder=8)

    axis.plot(cycle[:, 0], cycle[:, 1], color="#c62828", lw=2.7, zorder=5)
    axis.plot(
        [cycle[-1, 0], reset[0]], [cycle[-1, 1], reset[1]],
        color="#c62828", ls="--", lw=2, zorder=5,
    )
    # Arrowheads distinguish forward stance flow from the backward coordinate jump.
    start, end = cycle[len(cycle) // 2], cycle[3 * len(cycle) // 5]
    axis.annotate("", xy=end, xytext=start,
                  arrowprops={"arrowstyle": "->", "color": "#c62828", "lw": 2}, zorder=6)
    jump = reset - cycle[-1]
    axis.annotate("", xy=cycle[-1] + 0.6 * jump, xytext=cycle[-1] + 0.4 * jump,
                  arrowprops={"arrowstyle": "->", "color": "#c62828", "lw": 2}, zorder=6)
    axis.scatter(lower_angle, INITIAL_POST_IMPACT_VELOCITY, marker="*", s=110,
                 color="#2563eb", clip_on=False, zorder=7)
    axis.annotate("R3", (lower_angle, INITIAL_POST_IMPACT_VELOCITY),
                  xytext=(7, 10), textcoords="offset points",
                  color="#2563eb", weight="bold", fontsize=14)
    for contact_angle in (lower_angle, upper_angle):
        axis.axvline(contact_angle, color="0.25", ls="--", lw=1)
    axis.axhline(0, color="0.8", lw=0.8)
    axis.axvline(0, color="0.8", lw=0.8)
    axis.set_xlim(lower_angle - 0.025, upper_angle + 0.025)
    axis.set_ylim(-2.2, 2.2)
    axis.set_xlabel(r"Stance angle $\theta$ (rad)")
    axis.set_ylabel(r"Angular velocity $\dot{\theta}$ (rad/s)")
    axis.set_title(
        "Rimless-wheel phase portrait\n"
        rf"$N={params['number_of_spokes']}$, $\gamma={np.rad2deg(gamma):g}^\circ$, "
        rf"$\dot{{\theta}}^+_*={fixed_velocity:.4f}$ rad/s"
    )
    handles = [
        Line2D([], [], color="0.7", ls="--", label="Constant stance energy"),
        Line2D([], [], color="#2563eb", label=f"R3: walking ({TRANSIENT_STEPS} steps)"),
        Line2D([], [], color="#15803d", label="R1: rocking toward rest"),
        Line2D([], [], color="#b45309", label="R2: rocking toward rest"),
        Line2D([], [], color="#c62828", lw=2.7, label="Walking limit cycle: stance"),
        Line2D([], [], color="0.35", ls="--", label="Colored dashes: impact resets"),
        Line2D([], [], color="0.35", marker="*", ls="none", markersize=10,
               label="Stars: initial states"),
        Line2D([], [], color="black", marker="s", ls="none",
               label="Rest (two contact coordinates)"),
        Line2D([], [], color="0.25", ls="--", label=r"Contact limits $\gamma\pm\alpha$"),
    ]
    if not standalone:
        return handles
    figure.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    output_path = OUTPUT_DIRECTORY / "phase_portrait.png"
    figure.savefig(output_path, dpi=200)
    all_segments = transient + [(cycle, reset)]
    for _, _, _, segments in resting_traces:
        all_segments.extend(segments)
    energy_drift = max(float(np.ptp(calculate_stance_energy(orbit, params)))
                       for orbit, _ in all_segments)
    print(f"Cycle closure residual: {closure_error:.3e}")
    print(f"Maximum within-stance energy drift: {energy_drift:.3e} J")
    print(f"Final transient post-impact velocity: {velocity:.8f} rad/s")
    print(f"Saved {output_path}")
    return output_path


def main():
    params = model.generate_params(NUMBER_OF_SPOKES)
    params["slope_angle"] = np.deg2rad(SLOPE_DEGREES)
    plot_phase_portrait(params)
    plt.show()


if __name__ == "__main__":
    main()
