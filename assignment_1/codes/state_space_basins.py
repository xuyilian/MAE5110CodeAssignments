"""Estimate rimless-wheel attraction basins for one fixed parameter set.

Run from the repository root: uv run python assignment_1/codes/state_space_basins.py
Edit the configuration below to change the grid or convergence tolerances.
Uses the RK4/contact-event and convergence routines in gamma_sweep.py.
"""

from pathlib import Path

import numpy as np
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from gamma_sweep import (
    REST,
    UNRESOLVED,
    WALKING,
    classify_initial_states,
    locate_impacts,
)
from integrators import rk4_step
from models import rimless_wheel as model

# Configuration: theta is measured from the upward vertical.
NUMBER_OF_SPOKES = 8
SLOPE_DEGREES = 5.0
ANGLE_SAMPLES = 401  # Include theta=0 explicitly if not already on this grid.
VELOCITY_SAMPLES = 401
VELOCITY_LIMITS = (-6.0, 3.0)  # rad/s; include backward and forward motion.
TIMESTEP = 0.002  # s; contact times are refined within each step.
TIME_LIMITS = (30.0, 60.0, 120.0)  # Retry unresolved states with longer horizons.
VELOCITY_ATOL = 1e-4  # rad/s
VELOCITY_RTOL = 1e-3
WALKING_COMPARISONS = 5
RESTING_COMPARISONS = 3
UNSTABLE_EQUILIBRIUM = 3  # A stationary solution, but not an attractor.
OUTPUT_DIRECTORY = Path(__file__).resolve().parents[1] / "figures"


def evaluate_basins(
    params,
    angle_samples=ANGLE_SAMPLES,
    velocity_samples=VELOCITY_SAMPLES,
    timestep=TIMESTEP,
    time_limits=TIME_LIMITS,
):
    """Simulate the grid until convergence, or report an unresolved timeout.

    Walking: successive forward post-impact velocities agree within
    atol + rtol * max(abs(omega_previous), abs(omega_current)) five times.
    Rest: three alternating-impact comparisons meet that tolerance, with
    decreasing post-impact speed <= atol (and <= settling_velocity).
    Exact two-contact rest and the unstable upright equilibrium are handled
    explicitly; all other grid points are integrated by the hybrid simulator.

    Each retry starts only unresolved trajectories again from their original
    initial conditions, using a longer simulation horizon.
    """
    alpha = params["half_spoke_angle"]
    gamma = params["slope_angle"]
    if not (6 <= params["number_of_spokes"] <= 12 and 0 < gamma < alpha):
        raise ValueError("This basin study supports N=6..12 and 0 < gamma < alpha")
    if angle_samples < 2 or velocity_samples < 2:
        raise ValueError("Each grid dimension must have at least two samples")
    if (
        not time_limits
        or any(t <= 0 for t in time_limits)
        or np.any(np.diff(time_limits) <= 0)
    ):
        raise ValueError("time_limits must be positive and strictly increasing")

    angle_values = np.unique(
        np.r_[np.linspace(gamma - alpha, gamma + alpha, angle_samples), 0.0]
    )
    velocity_values = np.unique(
        np.r_[np.linspace(*VELOCITY_LIMITS, velocity_samples), 0.0]
    )
    angles, velocities = np.meshgrid(angle_values, velocity_values)
    labels = np.full(angles.shape, UNRESOLVED, dtype=int)
    steady_velocities = np.full(angles.shape, np.nan)

    upright = (angles == 0.0) & (velocities == 0.0)
    labels[upright] = UNSTABLE_EQUILIBRIUM
    at_contact = (angles == angle_values[0]) | (angles == angle_values[-1])
    initially_resting = at_contact & (velocities == 0.0)
    labels[initially_resting] = REST
    steady_velocities[initially_resting] = 0.0

    for time_limit in time_limits:
        pending = labels == UNRESOLVED
        if not np.any(pending):
            break
        print(
            f"Simulating {np.count_nonzero(pending)} initial states, up to {time_limit:g} s...",
            flush=True,
        )
        result_labels, result_velocities = classify_initial_states(
            angles[pending],
            velocities[pending],
            params,
            simulation_time=time_limit,
            timestep=timestep,
            velocity_atol=VELOCITY_ATOL,
            velocity_rtol=VELOCITY_RTOL,
            convergence_steps=WALKING_COMPARISONS,
            resting_impacts=RESTING_COMPARISONS,
            return_velocities=True,
        )
        labels[pending] = result_labels
        steady_velocities[pending] = result_velocities
        print(
            f"Unresolved after this pass: {np.count_nonzero(labels == UNRESOLVED)}",
            flush=True,
        )

    return angles, velocities, labels, steady_velocities


def trace_walking_step(post_impact_velocity, params):
    """Numerically trace one step for the limit-cycle overlay."""
    state = np.array(
        [params["slope_angle"] - params["half_spoke_angle"], post_impact_velocity]
    )
    trajectory = [state.copy()]
    for step in range(int(TIME_LIMITS[-1] / TIMESTEP)):
        next_state = rk4_step(model.dynamics, step * TIMESTEP, state, TIMESTEP, params)
        if model.detect_contact(next_state, params):
            contact, _ = locate_impacts(
                state[:, None], np.array([TIMESTEP]), params, np.array([True])
            )
            trajectory.append(contact[:, 0])
            return np.asarray(trajectory), model.apply_impact_reset(
                contact[:, 0], params
            )
        if model.detect_backward_contact(next_state, params):
            raise RuntimeError(
                "The converged walking state failed to complete a forward step"
            )
        state = next_state
        trajectory.append(state.copy())
    raise RuntimeError("Timed out while tracing the walking limit cycle")


def plot_basins(angles, velocities, labels, steady_velocities, params, *, axis=None):
    """Plot categorical basins with the continuous part of the walking cycle."""
    palette = ["#c5cae9", "#80cbc4", "#ffcc80", "#ffffff"]
    names = [
        "Two-contact rest",
        "Periodic forward walking",
        "Unresolved",
        "Unstable equilibrium",
    ]
    standalone = axis is None
    if standalone:
        figure, axis = plt.subplots(figsize=(10, 7), constrained_layout=True)
    axis.pcolormesh(
        np.rad2deg(angles),
        velocities,
        labels,
        shading="nearest",
        cmap=colors.ListedColormap(palette),
        norm=colors.BoundaryNorm(np.arange(-0.5, 4.0, 1.0), 4),
    )
    handles = [
        Patch(facecolor=palette[i], label=names[i]) for i in (REST, WALKING)
    ]
    walking_velocities = steady_velocities[labels == WALKING]
    if walking_velocities.size:
        orbit, _ = trace_walking_step(float(np.median(walking_velocities)), params)
        (cycle,) = axis.plot(
            np.rad2deg(orbit[:, 0]),
            orbit[:, 1],
            color="#004d40",
            linewidth=2,
            label="Walking limit cycle (stance phase)",
        )
        handles.append(cycle)

    axis.set_xlim(np.rad2deg(angles.min()), np.rad2deg(angles.max()))
    axis.set_ylim(velocities.min(), velocities.max())
    axis.set_xlabel(r"Initial angle $\theta_0$ (deg)")
    axis.set_ylabel(r"Initial angular velocity $\dot{\theta}_0$ (rad/s)")
    axis.set_title(
        "Rimless-wheel regions of attraction\n"
        rf"$N={params['number_of_spokes']}$, "
        rf"$\alpha={np.rad2deg(params['half_spoke_angle']):g}^\circ$, "
        rf"$\gamma={np.rad2deg(params['slope_angle']):g}^\circ$"
    )
    if not standalone:
        return handles
    figure.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    output_path = OUTPUT_DIRECTORY / "state_space_basins.png"
    figure.savefig(output_path, dpi=200)
    return output_path


def main():
    params = model.generate_params(number_of_spokes=NUMBER_OF_SPOKES)
    params["slope_angle"] = np.deg2rad(SLOPE_DEGREES)
    angles, velocities, labels, steady_velocities = evaluate_basins(params)
    output_path = plot_basins(angles, velocities, labels, steady_velocities, params)
    np.savez_compressed(
        OUTPUT_DIRECTORY / "state_space_basins.npz",
        initial_angles=angles,
        initial_velocities=velocities,
        classifications=labels,
        steady_post_impact_velocities=steady_velocities,
        class_names=np.array(["rest", "walking", "unresolved", "unstable_equilibrium"]),
        timestep=TIMESTEP,
        time_limits=TIME_LIMITS,
        velocity_atol=VELOCITY_ATOL,
        velocity_rtol=VELOCITY_RTOL,
        walking_comparisons=WALKING_COMPARISONS,
        resting_comparisons=RESTING_COMPARISONS,
        **params,
    )
    for value, name in enumerate(
        ["Rest", "Walking", "Unresolved", "Unstable equilibrium"]
    ):
        print(f"{name}: {np.count_nonzero(labels == value)} / {labels.size}")
    print(f"Saved {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
