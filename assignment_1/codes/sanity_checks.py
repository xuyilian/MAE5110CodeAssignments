"""Reproduce Section 3 of the assignment_1/assignment_1.md report.

Run from the repository root: uv run python assignment_1/codes/sanity_checks.py
"""

import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from gamma_sweep import (
    REST,
    UNRESOLVED,
    WALKING,
    classify_initial_states,
    locate_impacts,
)
from integrators import rk4_step
from models import rimless_wheel as model

OUTPUT_DIRECTORY = Path(__file__).resolve().parents[1] / "figures"


def simulate_next_contact(initial_state, params, timestep=0.002, time_limit=30.0):
    """Return accurately located pre/post states and direction of one impact."""
    state = initial_state.copy()
    elapsed = 0.0
    while elapsed < time_limit:
        step = min(timestep, time_limit - elapsed)
        trial = rk4_step(model.dynamics, elapsed, state, step, params)
        forward = model.detect_contact(trial, params)
        backward = model.detect_backward_contact(trial, params)
        if forward or backward:
            contact, interval = locate_impacts(
                state[:, None], np.array([step]), params, np.array([forward])
            )
            before = contact[:, 0]
            after = (
                model.apply_impact_reset
                if forward
                else model.apply_backward_impact_reset
            )(before, params)
            return before, after, elapsed + interval[0], forward
        state = trial
        elapsed += step
    raise RuntimeError("No contact before the sanity-check timeout")


def record_impacts(initial_state, params, count):
    """Record a fixed number of impacts without applying a rest threshold."""
    state = initial_state.copy()
    velocities = [state[1]]
    directions = []
    for _ in range(count):
        _, state, _, forward = simulate_next_contact(state, params)
        velocities.append(state[1])
        directions.append(1 if forward else -1)
    return np.array(velocities), np.array(directions)


def run_checks():
    params = model.generate_params(8)
    params["slope_angle"] = np.deg2rad(5.0)
    alpha, gamma = params["half_spoke_angle"], params["slope_angle"]
    scale = np.cos(2 * alpha)
    metrics = {"parameters": params}

    # Test 1: energy in a continuous stance, without any impact.
    step = 0.01
    times = np.linspace(0, 0.5, 51)
    states = [np.array([0.1, 0.2])]
    for time in times[:-1]:
        state = rk4_step(model.dynamics, time, states[-1], step, params)
        assert not model.detect_contact(state, params)
        assert not model.detect_backward_contact(state, params)
        states.append(state)
    states = np.array(states)
    energy = params["hub_mass"] * (
        0.5 * params["spoke_length"] ** 2 * states[:, 1] ** 2
        + params["gravity"] * params["spoke_length"] * np.cos(states[:, 0])
    )
    relative_error = np.abs((energy - energy[0]) / energy[0])
    metrics["energy_max_relative_error"] = float(relative_error.max())
    assert relative_error.max() < 1e-7

    # Test 2: guard direction, coordinate switching and momentum at contact.
    checks = 0
    reset_results = []
    for direction in (1, -1):
        guard = (
            model.detect_contact if direction == 1 else model.detect_backward_contact
        )
        reset = (
            model.apply_impact_reset
            if direction == 1
            else model.apply_backward_impact_reset
        )
        contact_angle = gamma + direction * alpha
        assert guard(np.array([contact_angle, direction * 2.0]), params)
        assert not guard(np.array([contact_angle, -direction * 2.0]), params)
        assert not guard(np.array([contact_angle, 0.0]), params)
        assert not guard(
            np.array([contact_angle - direction * 1e-6, direction * 2.0]), params
        )
        checks += 4
        before = np.array([contact_angle, direction * 2.0])
        original = before.copy()
        after = reset(before, params)
        assert np.array_equal(before, original)
        momentum_residual = (
            params["hub_mass"]
            * params["spoke_length"] ** 2
            * (after[1] - before[1] * scale)
        )
        assert abs(momentum_residual) < 1e-12
        assert np.isclose(after[0], gamma - direction * alpha)
        assert not guard(after, params)
        kinetic_ratio = (after[1] / before[1]) ** 2
        assert np.isclose(kinetic_ratio, 0.5)
        reset_results.append(
            {
                "direction": direction,
                "pre_angle_degrees": float(np.rad2deg(before[0])),
                "post_angle_degrees": float(np.rad2deg(after[0])),
                "pre_velocity": float(before[1]),
                "post_velocity": float(after[1]),
                "kinetic_energy_ratio": float(kinetic_ratio),
                "momentum_residual": float(momentum_residual),
            }
        )
    metrics["guard_cases_passed"] = checks
    metrics["resets"] = reset_results

    # Test 3: integrate a whole forward step and compare with energy balance.
    initial_walking_state = np.array([gamma - alpha, 2.0])
    expected_next_velocity = scale * np.sqrt(
        4
        + 4 * params["gravity"] / params["spoke_length"] * np.sin(alpha) * np.sin(gamma)
    )
    metrics["expected_first_post_impact_velocity"] = float(expected_next_velocity)
    refinement = []
    for timestep in (0.01, 0.005, 0.002, 0.001):
        before, after, contact_time, forward = simulate_next_contact(
            initial_walking_state, params, timestep
        )
        assert forward
        error = abs(after[1] - expected_next_velocity)
        assert error < 1e-6
        refinement.append(
            {
                "timestep": timestep,
                "contact_time": float(contact_time),
                "post_velocity": float(after[1]),
                "absolute_velocity_error": float(error),
            }
        )
    metrics["timestep_refinement"] = refinement

    # Test 4: actual impact sequences approach walking or rest.
    walking, walking_directions = record_impacts(initial_walking_state, params, 20)
    rocking, rocking_directions = record_impacts(
        np.array([gamma - alpha, 0.1]), params, 30
    )
    expected_fixed_velocity = scale * np.sqrt(
        4
        * params["gravity"]
        / params["spoke_length"]
        * np.sin(alpha)
        * np.sin(gamma)
        / (1 - scale**2)
    )
    assert np.all(walking_directions == 1)
    assert np.isclose(walking[-1], expected_fixed_velocity, rtol=2e-6)
    assert np.all(rocking_directions[1:] == -rocking_directions[:-1])
    assert np.all(np.diff(np.abs(rocking)) < 0)
    assert abs(rocking[-1]) < 1e-4
    metrics["walking_after_20_impacts"] = float(walking[-1])
    metrics["expected_fixed_velocity"] = float(expected_fixed_velocity)
    metrics["rocking_after_30_impacts"] = float(rocking[-1])

    # Test 5: verify the production convergence classifier, including timeout.
    angles = np.array([gamma - alpha, gamma - alpha])
    speeds = np.array([2.0, 0.1])
    labels, steady = classify_initial_states(
        angles, speeds, params, return_velocities=True
    )
    assert np.array_equal(labels, [WALKING, REST])
    assert np.isclose(steady[0], expected_fixed_velocity, rtol=1e-3)
    assert steady[1] == 0
    labels_short = classify_initial_states(
        angles, speeds, params, simulation_time=0.001
    )
    assert np.all(labels_short == UNRESOLVED)
    metrics["classified_post_impact_velocities"] = steady.tolist()
    metrics["timeout_labels"] = labels_short.tolist()

    # Check that convergence classifications survive timestep refinement.
    grid_angles, grid_speeds = np.meshgrid(
        np.linspace(gamma - alpha, gamma + alpha, 21), np.linspace(-3, 3, 20)
    )
    baseline = classify_initial_states(grid_angles, grid_speeds, params)
    refined = classify_initial_states(grid_angles, grid_speeds, params, timestep=0.001)
    agreement = np.count_nonzero(baseline == refined)
    assert agreement == baseline.size
    metrics["grid_refinement_agreement"] = int(agreement)
    metrics["grid_refinement_total"] = int(baseline.size)

    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    metrics_path = OUTPUT_DIRECTORY / "sanity_checks.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    axes[0].plot(times, relative_error, color="tab:blue")
    axes[0].set(
        xlabel="Time (s)",
        ylabel="Absolute relative energy error",
        title="Continuous stance: energy",
    )
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1].plot(
        np.arange(walking.size),
        walking,
        ".-",
        color="tab:green",
        label="Simulated impacts",
    )
    axes[1].axhline(
        expected_fixed_velocity,
        color="black",
        linestyle="--",
        label="Analytical fixed point",
    )
    axes[1].set(
        xlabel="Impact index (0 = initial state)",
        ylabel=r"Post-impact $\omega$ (rad/s)",
        title="Forward walking",
    )
    axes[1].legend(fontsize=8)
    axes[2].semilogy(np.arange(rocking.size), np.abs(rocking), ".-", color="tab:purple")
    axes[2].set(
        xlabel="Impact index (0 = initial state)",
        ylabel=r"Post-impact $|\omega|$ (rad/s)",
        title="Alternating rocking impacts",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.savefig(OUTPUT_DIRECTORY / "sanity_checks.png", dpi=200)
    plt.close(figure)
    print(json.dumps(metrics, indent=2))
    print("All sanity checks passed.")
    return metrics


if __name__ == "__main__":
    run_checks()
