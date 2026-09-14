"""Run from the repository root: uv run python assignment_1/codes/gamma_sweep.py."""

from pathlib import Path

import numpy as np
from matplotlib import colors
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from integrators import rk4_step
from models import rimless_wheel as model

REST = 0
WALKING = 1
UNRESOLVED = 2


def locate_impacts(states, timesteps, params, forward_contact):
    """Locate the first contact within each RK4 step by bisection.

    Each column has its own elapsed time. Truncating a step at contact
    avoids applying the reset to an overshot state and adding energy.
    """
    direction = np.where(forward_contact, 1.0, -1.0)
    target_angles = (
        params["slope_angle"] + direction * params["half_spoke_angle"]
    )
    lower_times = np.zeros_like(timesteps)
    upper_times = timesteps.copy()
    for _ in range(30):
        midpoint_times = (lower_times + upper_times) / 2
        midpoint_states = rk4_step(
            model.dynamics, 0.0, states, midpoint_times, params
        )
        crossed = direction * (midpoint_states[0] - target_angles) >= 0
        upper_times = np.where(crossed, midpoint_times, upper_times)
        lower_times = np.where(crossed, lower_times, midpoint_times)

    contact_states = rk4_step(model.dynamics, 0.0, states, upper_times, params)
    contact_states[0] = target_angles
    return contact_states, upper_times


def classify_initial_states(
    initial_angles,
    initial_velocities,
    params,
    simulation_time=30.0,
    timestep=2e-3,
    velocity_atol=1e-4,
    velocity_rtol=1e-3,
    convergence_steps=5,
    resting_impacts=3,
    return_velocities=False,
):
    """Classify only trajectories that pass a convergence test.

    Walking requires `convergence_steps` consecutive comparisons satisfying
    abs(omega_next - omega_previous) <= atol + rtol * max(abs(omega)).
    Backward impacts reset this counter. Rest requires `resting_impacts`
    consecutive alternating-impact comparisons satisfying the same tolerance,
    decreasing speed, and convergence to zero (abs(omega) <= atol, also
    bounded by settling_velocity), in a geometry supporting two-contact rest.
    Timeouts remain UNRESOLVED.

    With return_velocities=True, also return the converged post-impact
    velocity grid: zero for rest and NaN for unresolved trajectories.

    Timesteps are independently truncated at contact for each trajectory;
    the model is autonomous, so trajectories can be integrated in a batch.
    """
    if simulation_time <= 0 or timestep <= 0:
        raise ValueError("simulation_time and timestep must be positive")
    if velocity_atol < 0 or velocity_rtol < 0:
        raise ValueError("Convergence tolerances must be nonnegative")
    if convergence_steps < 1 or resting_impacts < 2:
        raise ValueError("Require at least one walking comparison and two resting impacts")
    states = np.vstack((initial_angles.ravel(), initial_velocities.ravel())).astype(float)
    attractors = np.full(states.shape[1], UNRESOLVED, dtype=int)
    steady_velocities = np.full(states.shape[1], np.nan)
    elapsed_times = np.zeros(states.shape[1])
    previous_forward_velocities = np.full(states.shape[1], np.nan)
    stable_steps = np.zeros(states.shape[1], dtype=int)
    previous_impact_velocities = np.full(states.shape[1], np.nan)
    previous_directions = np.zeros(states.shape[1])
    quiet_impacts = np.zeros(states.shape[1], dtype=int)

    forward_contact_angle = (
        params["slope_angle"] + params["half_spoke_angle"]
    )
    backward_contact_angle = (
        params["slope_angle"] - params["half_spoke_angle"]
    )
    impact_velocity_scale = np.cos(2.0 * params["half_spoke_angle"])
    supports_rest = backward_contact_angle < 0 < forward_contact_angle

    while True:
        active = (attractors == UNRESOLVED) & (elapsed_times < simulation_time)
        if not np.any(active):
            break

        active_indices = np.flatnonzero(active)
        active_states = states[:, active]
        step_times = np.minimum(timestep, simulation_time - elapsed_times[active])
        next_states = rk4_step(
            model.dynamics, 0.0, active_states, step_times, params
        )

        angles = next_states[0]
        velocities = next_states[1]
        forward_contact = (
            (angles >= forward_contact_angle)
            & (velocities > 0.0)
        )
        backward_contact = (
            (angles <= backward_contact_angle)
            & (velocities < 0.0)
        )
        impacted = forward_contact | backward_contact
        if np.any(impacted):
            contact_states, contact_times = locate_impacts(
                active_states[:, impacted], step_times[impacted], params,
                forward_contact[impacted],
            )
            directions = np.where(forward_contact[impacted], 1.0, -1.0)
            contact_states[0] = (
                params["slope_angle"] - directions * params["half_spoke_angle"]
            )
            contact_states[1] *= impact_velocity_scale
            next_states[:, impacted] = contact_states
            step_times[impacted] = contact_times

            indices = active_indices[impacted]
            speeds = np.abs(contact_states[1])
            previous_impact = previous_impact_velocities[indices]
            impact_tolerance = velocity_atol + velocity_rtol * np.maximum(
                speeds, np.abs(previous_impact)
            )
            close_to_previous = (
                np.abs(contact_states[1] - previous_impact) <= impact_tolerance
            )
            converging_to_rest = (
                supports_rest
                & (speeds <= min(velocity_atol, params["settling_velocity"]))
                & (speeds <= np.abs(previous_impact))
                & close_to_previous
            )
            alternating = directions == -previous_directions[indices]
            quiet_impacts[indices] = np.where(
                converging_to_rest & alternating,
                quiet_impacts[indices] + 1,
                0,
            )
            previous_impact_velocities[indices] = contact_states[1]
            previous_directions[indices] = directions
            resting_indices = indices[quiet_impacts[indices] >= resting_impacts]
            attractors[resting_indices] = REST
            steady_velocities[resting_indices] = 0.0

        forward_indices = active_indices[forward_contact]
        current_velocities = next_states[1, forward_contact]
        previous_velocities = previous_forward_velocities[forward_indices]
        close = np.abs(current_velocities - previous_velocities) <= (
            velocity_atol
            + velocity_rtol * np.maximum(np.abs(current_velocities), np.abs(previous_velocities))
        )
        stable_steps[forward_indices] = np.where(
            close, stable_steps[forward_indices] + 1, 0
        )
        previous_forward_velocities[forward_indices] = current_velocities
        converged = stable_steps[forward_indices] >= convergence_steps
        attractors[forward_indices[converged]] = WALKING
        steady_velocities[forward_indices[converged]] = current_velocities[converged]

        backward_indices = active_indices[backward_contact]
        stable_steps[backward_indices] = 0
        previous_forward_velocities[backward_indices] = np.nan

        states[:, active] = next_states
        elapsed_times[active] += step_times

    classifications = attractors.reshape(initial_angles.shape)
    if return_velocities:
        return classifications, steady_velocities.reshape(initial_angles.shape)
    return classifications


def find_walking_intervals(angles_degrees, steady_velocities):
    """Find all walking intervals in one row at fixed initial velocity.

    Each endpoint is (angle, walking-side steady velocity, boundary kind).
    Rest/walking boundaries are estimated halfway between adjacent samples.
    Domain edges are separate from basin boundaries. An endpoint next to an
    unresolved sample is None, since its physical boundary is not known.
    """
    walking = np.isfinite(steady_velocities) & (steady_velocities > 0)
    transitions = np.diff(np.r_[False, walking, False].astype(int))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1) - 1
    intervals = []
    for start, end in zip(starts, ends, strict=True):
        endpoints = []
        for index, neighbor in ((start, start - 1), (end, end + 1)):
            if neighbor < 0 or neighbor >= len(walking):
                endpoints.append((angles_degrees[index], steady_velocities[index], "domain"))
            elif np.isfinite(steady_velocities[neighbor]):
                boundary = (angles_degrees[index] + angles_degrees[neighbor]) / 2
                endpoints.append((boundary, steady_velocities[index], "basin"))
            else:
                endpoints.append(None)
        intervals.append(endpoints)
    return intervals


def plot_3d_basins(results, slope_angles_degrees, number_of_spokes, normalization, color_map):
    """Trace walking-interval endpoints across slopes at fixed initial speed.

    Curves break at missing/unresolved endpoints and interval-count changes.
    When the preceding slope row is entirely at rest, close the top at the
    first sampled walking slope; this is a grid estimate of the transition.
    Color gives the converged post-impact velocity on the walking side.
    """
    figure = plt.figure(figsize=(11, 8))
    axis = figure.add_subplot(111, projection="3d")
    segments = {"basin": [], "domain": []}
    segment_velocities = {"basin": [], "domain": []}
    boundary_points = []
    boundary_velocities = []
    initial_velocities = results[0][1][:, 0]
    for row, initial_velocity in enumerate(initial_velocities):
        previous_intervals = []
        previous_slope = None
        previous_row_is_rest = False
        for slope, (angles, velocities, steady_velocities) in zip(
            slope_angles_degrees, results, strict=True
        ):
            if not np.allclose(velocities[row], initial_velocity):
                raise ValueError("Every slope must use the same initial-velocity grid")
            intervals = find_walking_intervals(np.rad2deg(angles[row]), steady_velocities[row])
            for interval in intervals:
                for endpoint in interval:
                    if endpoint is not None:
                        boundary_points.append([endpoint[0], initial_velocity, slope])
                        boundary_velocities.append(endpoint[1])

            if previous_row_is_rest:
                for left, right in intervals:
                    if left is None or right is None:
                        continue
                    segments["basin"].append([
                        [left[0], initial_velocity, slope],
                        [right[0], initial_velocity, slope],
                    ])
                    segment_velocities["basin"].append((left[1] + right[1]) / 2)

            if len(intervals) == len(previous_intervals):
                for current, previous in zip(intervals, previous_intervals, strict=True):
                    for endpoint, old_endpoint in zip(current, previous, strict=True):
                        if endpoint is None or old_endpoint is None:
                            continue
                        # Keep the curve continuous when it meets a stance limit.
                        kind = endpoint[2]
                        segments[kind].append([
                            [old_endpoint[0], initial_velocity, previous_slope],
                            [endpoint[0], initial_velocity, slope],
                        ])
                        segment_velocities[kind].append((old_endpoint[1] + endpoint[1]) / 2)
            previous_intervals = intervals
            previous_slope = slope
            previous_row_is_rest = bool(np.all(steady_velocities[row] == 0.0))

    for kind, style, width in (("basin", "solid", 0.7), ("domain", "solid", 0.7)):
        if segments[kind]:
            lines = Line3DCollection(
                segments[kind], cmap=color_map, norm=normalization,
                linewidths=width, linestyles=style,
            )
            lines.set_array(np.asarray(segment_velocities[kind]))
            axis.add_collection3d(lines)
    if boundary_points:
        points = np.asarray(boundary_points)
        axis.scatter(
            *points.T, c=boundary_velocities, cmap=color_map, norm=normalization,
            s=1, linewidths=0, depthshade=False,
        )

    axis.set_xlim(
        min(np.rad2deg(angles.min()) for angles, _, _ in results),
        max(np.rad2deg(angles.max()) for angles, _, _ in results),
    )
    axis.set_ylim(initial_velocities.min(), initial_velocities.max())
    # Larger slopes appear lower on the vertical axis.
    axis.set_zlim(max(slope_angles_degrees), min(slope_angles_degrees))

    axis.set_xlabel(r"Initial angle $\theta_0$ (deg)", labelpad=12)
    axis.set_ylabel(r"Initial $\dot{\theta}_0$ (rad/s)", labelpad=14)
    axis.set_zlabel(r"Slope $\gamma$ (deg)", labelpad=10)
    axis.set_zticks(np.linspace(slope_angles_degrees[0], slope_angles_degrees[-1], 5))
    axis.set_box_aspect((1.35, 1, 1.4))
    axis.view_init(elev=20, azim=-65)
    axis.set_title(
        rf"Walking-region boundaries: $N={number_of_spokes}$, "
        rf"$\alpha={180 / number_of_spokes:g}^\circ$",
        pad=20,
    )
    color_bar = figure.colorbar(
        plt.cm.ScalarMappable(norm=normalization, cmap=color_map),
        ax=axis, shrink=0.65, pad=0.12, aspect=25,
    )
    color_bar.set_label(r"Walking-side steady post-impact $\dot{\theta}^{+}_*$ (rad/s)")
    figure.legend(
        handles=[
            plt.Line2D([0], [0], color="0.3", linewidth=0.9, label=r"Walking-region boundary, including stance limits ($\gamma\pm\alpha$)"),
        ],
        loc="lower center", frameon=False, ncol=2,
    )
    figure.subplots_adjust(left=0.02, right=0.90, bottom=0.08, top=0.90)
    output_directory = Path(__file__).resolve().parents[1] / "figures"
    output_directory.mkdir(exist_ok=True)
    output_path = output_directory / "gamma_attraction_basins_3d.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    return output_path


def plot_gamma_sweep(number_of_spokes=6, view="3d"):
    """Color the basins by steady post-impact velocity; select '3d' or '2d'."""
    if view not in ("3d", "2d"):
        raise ValueError("view must be '3d' or '2d'")
    slope_angles_degrees = np.linspace(8.0, 30.0, 25)  # 0.5-degree spacing.
    velocity_values = np.linspace(0.05, 3.0, 60)  # Positive initial speeds (rad/s).

    results = []

    for slope_angle_degrees in slope_angles_degrees:
        params = model.generate_params(number_of_spokes=number_of_spokes)
        params["slope_angle"] = np.deg2rad(slope_angle_degrees)
        params["settling_velocity"] = 1e-2

        half_spoke_angle = params["half_spoke_angle"]
        angle_values = np.linspace(
            params["slope_angle"] - half_spoke_angle,
            params["slope_angle"] + half_spoke_angle,
            61,
        )
        initial_angles, initial_velocities = np.meshgrid(
            angle_values, velocity_values
        )
        attractors, steady_velocities = classify_initial_states(
            initial_angles, initial_velocities, params, return_velocities=True
        )
        results.append((initial_angles, initial_velocities, steady_velocities))
        print(
            f"gamma={slope_angle_degrees:g} deg: "
            f"rest={np.count_nonzero(attractors == REST)}, "
            f"walking={np.count_nonzero(attractors == WALKING)}, "
            f"unresolved={np.count_nonzero(attractors == UNRESOLVED)}"
        )

    # Use one physical velocity scale for every slope, including rest at zero.
    finite_velocities = np.concatenate([values[np.isfinite(values)] for _, _, values in results])
    maximum_velocity = max(float(finite_velocities.max(initial=0.0)), 1e-4)
    normalization = colors.Normalize(vmin=0.0, vmax=maximum_velocity)
    color_map = plt.get_cmap("viridis").copy()
    color_map.set_bad("lightgray")
    if view == "3d":
        return plot_3d_basins(
            results, slope_angles_degrees, number_of_spokes, normalization, color_map
        )
    number_of_columns = min(5, len(slope_angles_degrees))
    number_of_rows = int(np.ceil(len(slope_angles_degrees) / number_of_columns))
    figure, axes = plt.subplots(
        number_of_rows, number_of_columns,
        figsize=(15, 3.2 * number_of_rows + 1.3),
        sharey=True, constrained_layout=True, squeeze=False,
    )
    axes = axes.ravel()
    for unused_axis in axes[len(slope_angles_degrees):]:
        unused_axis.set_visible(False)
    axes = axes[:len(slope_angles_degrees)]

    for axis, slope_angle_degrees, result in zip(
        axes, slope_angles_degrees, results, strict=True
    ):
        initial_angles, initial_velocities, steady_velocities = result
        mesh = axis.pcolormesh(
            np.rad2deg(initial_angles),
            initial_velocities,
            np.ma.masked_invalid(steady_velocities),
            cmap=color_map,
            norm=normalization,
            shading="nearest",
        )
        axis.set_ylim(velocity_values[0], velocity_values[-1])
        axis.set_xlim(np.rad2deg(initial_angles[0, [0, -1]]))
        axis.set_title(rf"$\gamma={slope_angle_degrees:g}^\circ$")
        axis.set_xlabel(r"Initial angle $\theta_0$ (deg)")
        axis.grid(alpha=0.15)

    axes[0].set_ylabel(r"Initial angular velocity $\dot{\theta}_0$ (rad/s)")
    alpha_degrees = 180.0 / number_of_spokes
    figure.suptitle(
        rf"Converged post-impact angular velocity at fixed $\alpha={alpha_degrees:.1f}^\circ$ "
        rf"($N={number_of_spokes}$)"
    )

    color_bar = figure.colorbar(mesh, ax=axes, orientation="horizontal", fraction=0.08, pad=0.08, aspect=65)
    color_bar.set_label(r"Steady post-impact angular velocity $\dot{\theta}^{+}_*$ (rad/s); 0 = rest")
    figure.legend(
        handles=[plt.Line2D([0], [0], marker="s", color="none", markerfacecolor="lightgray", markersize=9, label="Unresolved (30 s limit)")],
        loc="outside lower center",
        frameon=False,
    )

    output_directory = Path(__file__).resolve().parents[1] / "figures"
    output_directory.mkdir(exist_ok=True)
    output_path = output_directory / "gamma_attraction_basins.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    return output_path


if __name__ == "__main__":
    saved_path = plot_gamma_sweep()
    print(f"Saved {saved_path}")
    plt.show()
