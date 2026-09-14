"""Numerically construct the signed impact-to-impact map and walking multiplier.

Run from the repository root: uv run python assignment_1/codes/poincare_map.py
Every impact is sampled, whether forward or backward.
"""

import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from gamma_sweep import REST, UNRESOLVED, WALKING, classify_initial_states
from models import rimless_wheel as model
from phase_portrait import calculate_walking_fixed_point, trace_next_impact

NUMBER_OF_SPOKES = 8
SLOPE_DEGREES = 5.0
TIMESTEP = 0.002
TIME_LIMIT = 30.0
MAP_SAMPLES = 121
MIN_VELOCITY = -3.0
MAX_VELOCITY = 3.0
PERTURBATIONS = (1e-2, 1e-3, 1e-4, 1e-5)
BASIN_SAMPLES = 1801
OUTPUT_DIRECTORY = Path(__file__).resolve().parents[1] / "figures"


def calculate_minimum_velocity(params):
    """Strict lower speed bound for crossing upright when 0 < gamma < alpha."""
    alpha, gamma = params["half_spoke_angle"], params["slope_angle"]
    if not (0 < gamma < alpha and 0 < np.cos(2 * alpha) < 1):
        raise ValueError("Require 0 < gamma < alpha and 0 < cos(2 alpha) < 1")
    return np.sqrt(2 * params["gravity"] / params["spoke_length"]
                   * (1 - np.cos(gamma - alpha)))


def calculate_backward_threshold(params):
    """Magnitude needed to cross upright after a backward impact."""
    calculate_minimum_velocity(params)  # Validate the supported geometry.
    return np.sqrt(2 * params["gravity"] / params["spoke_length"]
                   * (1 - np.cos(params["slope_angle"] + params["half_spoke_angle"])))


def evaluate_return_map(velocity, params, timestep=TIMESTEP):
    """Simulate exactly one impact from a signed post-impact velocity.

    Its sign selects the post-forward or post-backward stance coordinate.
    A timeout raises an error rather than being interpreted as a mapped speed.
    Either threshold approaches upright asymptotically; zero is resting contact.
    """
    minimum = calculate_minimum_velocity(params)
    backward = calculate_backward_threshold(params)
    if not np.isfinite(velocity) or velocity in (0, minimum, -backward):
        return np.nan
    initial = np.array([
        params["slope_angle"] - np.sign(velocity) * params["half_spoke_angle"], velocity
    ])
    _, after = trace_next_impact(initial, params, timestep, TIME_LIMIT)
    return float(after[1])


def evaluate_analytical_map(velocities, params):
    """Signed energy/impact prediction for the union of both contact sections."""
    values = np.asarray(velocities, dtype=float)
    alpha, gamma = params["half_spoke_angle"], params["slope_angle"]
    increment = 4 * params["gravity"] / params["spoke_length"] * np.sin(alpha) * np.sin(gamma)
    factor = np.cos(2 * alpha)
    minimum = calculate_minimum_velocity(params)
    backward = calculate_backward_threshold(params)
    result = np.full(values.shape, np.nan)
    forward_step = values > minimum
    backward_step = values < -backward
    reversing = (values > -backward) & (values < minimum) & (values != 0)
    result[forward_step] = factor * np.sqrt(values[forward_step]**2 + increment)
    result[backward_step] = -factor * np.sqrt(values[backward_step]**2 - increment)
    result[reversing] = -factor * values[reversing]
    return result


def find_numerical_fixed_point(params, lower, upper, timestep=TIMESTEP):
    """Find a walking fixed point using simulated, not analytical, values."""
    lower_residual = evaluate_return_map(lower, params, timestep) - lower
    upper_residual = evaluate_return_map(upper, params, timestep) - upper
    if not (np.isfinite(lower_residual) and np.isfinite(upper_residual)
            and lower_residual > 0 > upper_residual):
        raise ValueError("The chosen speeds do not bracket a stable walking fixed point")
    for _ in range(36):
        midpoint = (lower + upper) / 2
        residual = evaluate_return_map(midpoint, params, timestep) - midpoint
        if residual > 0:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2


def run_analysis(params):
    minimum = calculate_minimum_velocity(params)
    backward = calculate_backward_threshold(params)
    velocities = np.unique(np.r_[
        np.linspace(MIN_VELOCITY, -backward - 1e-4, MAP_SAMPLES), -backward,
        np.linspace(-backward + 1e-4, -1e-4, 61), 0,
        np.linspace(1e-4, minimum - 1e-4, 61), minimum,
        np.linspace(minimum + 1e-4, MAX_VELOCITY, MAP_SAMPLES),
    ])
    numerical = np.array([evaluate_return_map(speed, params) for speed in velocities])
    analytical = evaluate_analytical_map(velocities, params)
    fixed = find_numerical_fixed_point(params, minimum + 1e-4, MAX_VELOCITY)
    fixed_output = evaluate_return_map(fixed, params)
    expected_fixed = float(calculate_walking_fixed_point(params))
    multiplier = float(np.cos(2 * params["half_spoke_angle"])**2)
    estimates = []
    for epsilon in PERTURBATIONS:
        below = evaluate_return_map(fixed - epsilon, params)
        above = evaluate_return_map(fixed + epsilon, params)
        estimates.append({
            "perturbation": epsilon,
            "left_slope": (fixed_output - below) / epsilon,
            "right_slope": (above - fixed_output) / epsilon,
            "centered_slope": (above - below) / (2 * epsilon),
        })
    refined = find_numerical_fixed_point(params, minimum + 1e-4, MAX_VELOCITY, TIMESTEP / 2)
    epsilon = 1e-4
    refined_multiplier = (
        evaluate_return_map(refined + epsilon, params, TIMESTEP / 2)
        - evaluate_return_map(refined - epsilon, params, TIMESTEP / 2)
    ) / (2 * epsilon)
    metrics = {
        "parameters": params, "timestep": TIMESTEP, "map_samples": len(velocities),
        "valid_samples": int(np.isfinite(numerical).sum()),
        "velocity_range": [MIN_VELOCITY, MAX_VELOCITY],
        "backward_threshold": float(backward),
        "minimum_velocity": float(minimum), "numerical_fixed_point": float(fixed),
        "analytical_fixed_point": expected_fixed,
        "fixed_point_residual": float(abs(fixed_output - fixed)),
        "max_map_error": float(np.nanmax(abs(numerical - analytical))),
        "analytical_multiplier": multiplier, "slope_estimates": estimates,
        "refined_timestep": TIMESTEP / 2, "refined_fixed_point": float(refined),
        "refined_multiplier": float(refined_multiplier),
    }
    assert np.isnan(evaluate_return_map(minimum, params))
    assert np.isnan(evaluate_return_map(0, params))
    assert np.isnan(evaluate_return_map(-backward, params))
    np.testing.assert_allclose(evaluate_return_map(minimum / 2, params),
                               -np.sqrt(multiplier) * minimum / 2, atol=1e-8)
    np.testing.assert_allclose(evaluate_return_map(-backward / 2, params),
                               np.sqrt(multiplier) * backward / 2, atol=1e-8)
    assert np.array_equal(np.isnan(numerical), np.isnan(analytical))
    assert metrics["max_map_error"] < 1e-7
    assert abs(fixed - expected_fixed) < 1e-7
    assert abs(estimates[2]["centered_slope"] - multiplier) < 1e-6
    assert abs(refined_multiplier - multiplier) < 1e-6
    return velocities, numerical, analytical, metrics


def classify_post_impact_basins(params):
    """Classify the section states used for the map's background shading."""
    velocities = np.linspace(MIN_VELOCITY, MAX_VELOCITY, BASIN_SAMPLES)
    angles = params["slope_angle"] - np.sign(velocities) * params["half_spoke_angle"]
    labels = np.full(velocities.shape, UNRESOLVED, dtype=int)
    labels[velocities == 0] = REST
    # Exact separatrices approach upright, rather than either stable attractor.
    thresholds = (calculate_minimum_velocity(params), -calculate_backward_threshold(params))
    for threshold in thresholds:
        labels[np.isclose(velocities, threshold, atol=1e-12, rtol=0)] = 3
    for horizon in (30.0, 60.0, 120.0):
        pending = labels == UNRESOLVED
        if not np.any(pending):
            break
        labels[pending] = classify_initial_states(
            angles[pending], velocities[pending], params,
            simulation_time=horizon, timestep=TIMESTEP,
            velocity_atol=1e-6, velocity_rtol=1e-5,
            convergence_steps=5, resting_impacts=3,
        )
    print(f"Post-impact basin samples: walking={np.count_nonzero(labels == WALKING)}, "
          f"rest={np.count_nonzero(labels == REST)}, "
          f"unresolved={np.count_nonzero(labels == UNRESOLVED)}")
    return velocities, labels


def shade_post_impact_basins(axis, velocities, labels):
    """Draw vertical strips: their x coordinate selects the initial section state."""
    edges = np.r_[velocities[0], (velocities[:-1] + velocities[1:]) / 2, velocities[-1]]
    for category, color, name in [
        (WALKING, "#80cbc4", "Initial states converging to walking"),
        (UNRESOLVED, "#ffcc80", "Unresolved initial states"),
    ]:
        transitions = np.diff(np.r_[False, labels == category, False].astype(int))
        starts, ends = np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)
        for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
            axis.axvspan(edges[start], edges[end], color=color, alpha=0.28,
                         linewidth=0, zorder=-2, label=name if index == 0 else "_nolegend_")


def plot_return_map(velocities, numerical, analytical, metrics, params):
    fixed, minimum = metrics["numerical_fixed_point"], metrics["minimum_velocity"]
    figure, (axis, local) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    basin_velocities, basin_labels = classify_post_impact_basins(params)
    shade_post_impact_basins(axis, basin_velocities, basin_labels)
    axis.plot([MIN_VELOCITY, MAX_VELOCITY], [MIN_VELOCITY, MAX_VELOCITY], color="0.35", ls="--", label="Identity: next = current")
    axis.plot(velocities, analytical, color="#c62828", lw=1.6, label="Energy/impact prediction")
    axis.plot(velocities[::4], numerical[::4], "o", ms=3.5, markerfacecolor="none",
              color="#2563eb", label="Numerical next-impact map")
    axis.scatter(fixed, fixed, color="#c62828", s=55, zorder=6, label="Walking fixed point")
    axis.annotate(rf"$\dot{{\theta}}^*={fixed:.6f}$ rad/s", (fixed, fixed),
                  xytext=(0.5, 2.6), ha="center", fontsize=9,
                  arrowprops={"arrowstyle": "->"})
    axis.scatter(0, 0, s=45, facecolors="white", edgecolors="black", zorder=6,
                 label="Rest: limiting boundary state")
    axis.annotate("Reversal: velocity changes sign",
                  (minimum / 2, -np.cos(2 * params["half_spoke_angle"]) * minimum / 2),
                  xytext=(-0.2, -2.1), ha="center",
                  arrowprops={"arrowstyle": "->"}, fontsize=8)
    axis.axvline(minimum, color="0.6", ls=":", lw=1)
    axis.axvline(-metrics["backward_threshold"], color="0.6", ls=":", lw=1)
    axis.axhline(0, color="0.85", lw=0.8)
    axis.axvline(0, color="0.85", lw=0.8)
    axis.set(xlim=(MIN_VELOCITY, MAX_VELOCITY), ylim=(MIN_VELOCITY, MAX_VELOCITY),
             xlabel=r"Current post-impact velocity $\dot{\theta}_k$ (rad/s)",
             ylabel=r"Next post-impact velocity $\dot{\theta}_{k+1}$ (rad/s)",
             title="(a) Every-impact return map (both directions)")
    axis.legend(fontsize=8, loc="upper left")

    offsets = np.linspace(-0.08, 0.08, 17)
    mapped = np.array([evaluate_return_map(fixed + d, params) for d in offsets]) - fixed
    local.plot(offsets, offsets, "--", color="0.5", label="Unchanged perturbation")
    local.plot(offsets, metrics["analytical_multiplier"] * offsets,
               color="#c62828", label=rf"Linear prediction: $\delta\dot{{\theta}}_{{k+1}}={metrics['analytical_multiplier']:.3g}\delta\dot{{\theta}}_k$")
    local.plot(offsets, mapped, "o", ms=4, markerfacecolor="none", color="#2563eb",
               label="Numerical perturbations")
    local.axhline(0, color="0.85", lw=0.8)
    local.axvline(0, color="0.85", lw=0.8)
    local.set(xlabel=r"Input perturbation $\dot{\theta}_k-\dot{\theta}^*$ (rad/s)",
              ylabel=r"Returned perturbation $P(\dot{\theta}_k)-\dot{\theta}^*$ (rad/s)",
              title="(b) Local convergence near the walking cycle")
    local.legend(fontsize=8, loc="upper left")
    figure.suptitle(rf"Rimless wheel: $N={params['number_of_spokes']}$, "
                    rf"$\gamma={np.rad2deg(params['slope_angle']):g}^\circ$")
    path = OUTPUT_DIRECTORY / "poincare_map.png"
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    figure.savefig(path, dpi=220)
    np.savez_compressed(OUTPUT_DIRECTORY / "poincare_map_basins.npz",
                        initial_post_impact_velocities=basin_velocities,
                        classifications=basin_labels, timestep=TIMESTEP,
                        velocity_atol=1e-6, velocity_rtol=1e-5, **params)
    return path


def main():
    params = model.generate_params(NUMBER_OF_SPOKES)
    params["slope_angle"] = np.deg2rad(SLOPE_DEGREES)
    velocities, numerical, analytical, metrics = run_analysis(params)
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    output = plot_return_map(velocities, numerical, analytical, metrics, params)
    (OUTPUT_DIRECTORY / "poincare_map.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(OUTPUT_DIRECTORY / "poincare_map.npz", input_velocities=velocities,
                        numerical_outputs=numerical, analytical_outputs=analytical, **params)
    print(json.dumps(metrics, indent=2))
    print(f"Saved {output}")
    plt.show()


if __name__ == "__main__":
    main()
