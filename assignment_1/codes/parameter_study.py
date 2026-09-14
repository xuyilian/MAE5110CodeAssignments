"""Controlled slope/spoke sweeps for Section 6 of the assignment report.

Run from the repository root: uv run python -u assignment_1/codes/parameter_study.py
Redraw saved results: uv run python assignment_1/codes/parameter_study.py --plot-only
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from gamma_sweep import REST, UNRESOLVED, WALKING, classify_initial_states
from models import rimless_wheel as model
from phase_portrait import calculate_walking_fixed_point
from poincare_map import (
    calculate_minimum_velocity,
    evaluate_return_map,
    find_numerical_fixed_point,
)

ANGLE_SAMPLES = 61
VELOCITY_SAMPLES = 81
VELOCITY_LIMITS = (-3.0, 3.0)
TIMESTEP = 0.002
TIME_LIMITS = (30.0, 60.0, 120.0)
VELOCITY_ATOL = 1e-6
VELOCITY_RTOL = 1e-5
UPRIGHT = 3
OUTPUT_DIRECTORY = Path(__file__).resolve().parents[1] / "figures"


def calculate_critical_slope(number_of_spokes):
    """Infimum slope for a finite-period walking cycle, in radians."""
    alpha = np.pi / number_of_spokes
    return 2 * np.arctan(np.tan(alpha / 2) * np.tan(alpha)**2)


def make_params(number_of_spokes, slope_degrees):
    params = model.generate_params(number_of_spokes)
    params["slope_angle"] = np.deg2rad(slope_degrees)
    return params


def evaluate_grid(params, angle_samples=ANGLE_SAMPLES, velocity_samples=VELOCITY_SAMPLES,
                  timestep=TIMESTEP):
    """Use a common normalized grid and keep timeouts explicitly unresolved."""
    q, velocities = np.meshgrid(np.linspace(-1, 1, angle_samples),
                               np.linspace(*VELOCITY_LIMITS, velocity_samples))
    angles = params["slope_angle"] + params["half_spoke_angle"] * q
    # Recognize intended exact zeros despite degree/radian rounding.
    angles[np.isclose(angles, 0, atol=1e-14, rtol=0)] = 0
    labels = np.full(q.shape, UNRESOLVED, dtype=int)
    labels[(velocities == 0) & (np.abs(q) == 1)] = REST
    labels[(velocities == 0) & (angles == 0)] = UPRIGHT
    for horizon in TIME_LIMITS:
        pending = labels == UNRESOLVED
        if not pending.any():
            break
        labels[pending] = classify_initial_states(
            angles[pending], velocities[pending], params, timestep=timestep,
            simulation_time=horizon, velocity_atol=VELOCITY_ATOL,
            velocity_rtol=VELOCITY_RTOL, convergence_steps=5, resting_impacts=3,
        )
        print(f"  horizon {horizon:g} s: {np.count_nonzero(labels == UNRESOLVED)} unresolved",
              flush=True)
    return labels


def measure_walking_cycle(params):
    """Numerically locate and perturb the walking fixed point if it exists."""
    critical = calculate_critical_slope(params["number_of_spokes"])
    if params["slope_angle"] <= critical:
        return {"walking_exists": False, "walking_velocity": None,
                "analytical_velocity": None, "multiplier": None,
                "analytical_multiplier": None, "perturbation": None,
                "fixed_point_residual": None}
    prediction = float(calculate_walking_fixed_point(params))
    threshold = calculate_minimum_velocity(params)
    fixed = find_numerical_fixed_point(params, (threshold + prediction) / 2,
                                      1.5 * prediction + 0.1)
    epsilon = min(1e-4, 0.1 * (fixed - threshold))
    above = evaluate_return_map(fixed + epsilon, params)
    below = evaluate_return_map(fixed - epsilon, params)
    multiplier = (above - below) / (2 * epsilon)
    expected_multiplier = float(np.cos(2 * params["half_spoke_angle"])**2)
    residual = abs(evaluate_return_map(fixed, params) - fixed)
    if abs(fixed - prediction) > 1e-7 or abs(multiplier - expected_multiplier) > 1e-6:
        raise RuntimeError("Walking-cycle measurements failed the analytical cross-check")
    return {"walking_exists": True, "walking_velocity": float(fixed),
            "analytical_velocity": prediction, "multiplier": float(multiplier),
            "analytical_multiplier": expected_multiplier, "perturbation": float(epsilon),
            "fixed_point_residual": float(residual)}


def summarize_case(number_of_spokes, slope_degrees, labels):
    counts = {name: int(np.count_nonzero(labels == value)) for value, name in
              [(REST, "rest"), (WALKING, "walking"), (UNRESOLVED, "unresolved"),
               (UPRIGHT, "upright")]}
    record = {"number_of_spokes": number_of_spokes, "slope_degrees": float(slope_degrees),
              "critical_slope_degrees": float(np.rad2deg(calculate_critical_slope(number_of_spokes))),
              "grid_size": labels.size, "counts": counts,
              "walking_fraction": counts["walking"] / labels.size,
              "rest_fraction": counts["rest"] / labels.size,
              "unresolved_fraction": counts["unresolved"] / labels.size}
    record.update(measure_walking_cycle(make_params(number_of_spokes, slope_degrees)))
    if not record["walking_exists"] and counts["walking"]:
        raise RuntimeError("Convergence test reported walking where no cycle exists")
    return record


def run_study():
    critical = float(np.rad2deg(calculate_critical_slope(8)))
    slopes = np.array([1, 2, 3, 3.5, critical - 0.2, critical - 0.05,
                       critical + 0.05, critical + 0.2, 4.5, 5, 6, 8, 10, 12, 14])
    spokes = np.arange(6, 13)
    cache = {}
    records, maps = {"slope": [], "spokes": []}, {"slope": [], "spokes": []}
    for study, settings in [("slope", [(8, float(s)) for s in slopes]),
                            ("spokes", [(int(n), 5.0) for n in spokes])]:
        for number, slope in settings:
            key = (number, slope)
            if key not in cache:
                started = time.perf_counter()
                print(f"{study}: N={number}, gamma={slope:.6f} deg", flush=True)
                labels = evaluate_grid(make_params(number, slope))
                record = summarize_case(number, slope, labels)
                cache[key] = (record, labels)
                print(f"  counts={record['counts']}; {time.perf_counter() - started:.1f} s", flush=True)
            record, labels = cache[key]
            records[study].append(record)
            maps[study].append(labels)

    validations = []
    for number, slope, kind in [(8, 5.0, "timestep"),
                                (8, critical + 0.05, "grid"), (12, 5.0, "grid")]:
        print(f"Refinement: N={number}, gamma={slope:.6f}, {kind}", flush=True)
        original = cache[(number, slope)][1]
        if kind == "grid":
            refined = evaluate_grid(make_params(number, slope), 2 * ANGLE_SAMPLES - 1,
                                     2 * VELOCITY_SAMPLES - 1)
            common = refined[::2, ::2]
        else:
            refined = evaluate_grid(make_params(number, slope), timestep=TIMESTEP / 2)
            common = refined
        validations.append({
            "number_of_spokes": number, "slope_degrees": slope, "kind": kind,
            "refined_grid_size": refined.size,
            "common_grid_disagreements": int(np.count_nonzero(common != original)),
            "base_walking_fraction": float(np.mean(original == WALKING)),
            "refined_walking_fraction": float(np.mean(refined == WALKING)),
            "refined_unresolved": int(np.count_nonzero(refined == UNRESOLVED)),
        })
        print(f"  {validations[-1]}", flush=True)

    summary = {"configuration": {"angle_samples": ANGLE_SAMPLES,
                "velocity_samples": VELOCITY_SAMPLES, "velocity_limits": VELOCITY_LIMITS,
                "timestep": TIMESTEP, "time_limits": TIME_LIMITS,
                "velocity_atol": VELOCITY_ATOL, "velocity_rtol": VELOCITY_RTOL,
                "walking_comparisons": 5, "resting_comparisons": 3},
               **records, "validation": validations}
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    (OUTPUT_DIRECTORY / "parameter_study.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(OUTPUT_DIRECTORY / "parameter_study.npz",
                        slope_maps=np.array(maps["slope"]), spoke_maps=np.array(maps["spokes"]),
                        slopes=slopes, spokes=spokes, q=np.linspace(-1, 1, ANGLE_SAMPLES),
                        initial_velocities=np.linspace(*VELOCITY_LIMITS, VELOCITY_SAMPLES))
    return summary


def plot_study(summary):
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for row, study in enumerate(["slope", "spokes"]):
        records = summary[study]
        key = "slope_degrees" if study == "slope" else "number_of_spokes"
        x = np.array([r[key] for r in records])
        for name, color in [("walking", "#00897b"), ("rest", "#6554a4"), ("unresolved", "#e68a00")]:
            if name == "unresolved" and not any(r["counts"][name] for r in records):
                continue
            axes[row, 0].plot(x, [r[f"{name}_fraction"] for r in records], "o-",
                              ms=3, color=color, label=name.capitalize())
        for col, observed, predicted in [(1, "walking_velocity", "analytical_velocity"),
                                         (2, "multiplier", "analytical_multiplier")]:
            values = np.array([r[observed] if r[observed] is not None else np.nan for r in records])
            theory = np.array([r[predicted] if r[predicted] is not None else np.nan for r in records])
            axes[row, col].plot(x, theory, "-", color="0.4", label="Analytical")
            axes[row, col].plot(x, values, "o", mfc="none", color="#1565c0", label="Numerical")
        for col in range(3):
            axis = axes[row, col]
            axis.set_xlabel(r"Slope $\gamma$ (deg)" if row == 0 else "Number of spokes N")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
            if row == 0:
                axis.axvline(records[0]["critical_slope_degrees"], color="0.5", ls=":")
                if col > 0:
                    axis.axvspan(x.min(), records[0]["critical_slope_degrees"], color="0.93")
            else:
                axis.set_xticks(x)
                if col > 0:
                    axis.axvspan(6, 7.5, color="0.93")
        axes[row, 0].set_ylim(-0.02, 1.02)
        axes[row, 0].set_ylabel("Fraction of sampled initial states")
        axes[row, 1].set_ylabel(r"Walking post-impact $\dot{\theta}^{*}$ (rad/s)")
        axes[row, 2].set_ylabel(r"Walking Floquet multiplier $\mu$")
        axes[row, 2].set_ylim(0, 1)
        for col, name in enumerate(["Attraction basins", "Walking speed", "Local convergence"]):
            axes[row, col].set_title(name + (r" ($N=8$)" if row == 0 else r" ($\gamma=5^\circ$)"))
    figure.savefig(OUTPUT_DIRECTORY / "parameter_trends.png", dpi=220)

    with np.load(OUTPUT_DIRECTORY / "parameter_study.npz") as data:
        figure, axes = plt.subplots(2, 4, figsize=(13, 7), sharex=True, sharey=True,
                                    constrained_layout=True)
        palette = ["#c5cae9", "#80cbc4", "#ffcc80", "#ffffff"]
        selected_slopes = [1, 6, 9, 13]  # 2 deg, just above onset, 5 deg, 12 deg.
        selected_spokes = [0, 2, 4, 6]  # N=6, 8, 10, 12.
        for row, study, indices in [(0, "slope", selected_slopes), (1, "spokes", selected_spokes)]:
            for col, index in enumerate(indices):
                record = summary[study][index]
                labels = data["slope_maps" if row == 0 else "spoke_maps"][index]
                axis = axes[row, col]
                axis.pcolormesh(data["q"], data["initial_velocities"], labels, shading="nearest",
                                cmap=colors.ListedColormap(palette),
                                norm=colors.BoundaryNorm(np.arange(-0.5, 4, 1), 4))
                axis.set_title(rf"$N={record['number_of_spokes']}$, "
                               rf"$\gamma={record['slope_degrees']:.3g}^\circ$" + "\n"
                               + f"Walking: {100 * record['walking_fraction']:.1f}%")
                axis.set_xlim(-1, 1)
                axis.set_ylim(*VELOCITY_LIMITS)
                if row == 1:
                    axis.set_xlabel(r"$q_0=(\theta_0-\gamma)/\alpha$")
                if col == 0:
                    axis.set_ylabel(r"Initial $\dot{\theta}_0$ (rad/s)")
        handles = [Patch(color=palette[REST], label="Two-contact rest"),
                   Patch(color=palette[WALKING], label="Forward walking")]
        if any(r["counts"]["unresolved"] for study in ("slope", "spokes") for r in summary[study]):
            handles.append(Patch(color=palette[UNRESOLVED], label="Unresolved"))
        figure.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
        figure.savefig(OUTPUT_DIRECTORY / "parameter_basins.png", dpi=220)
    print(f"Saved {OUTPUT_DIRECTORY / 'parameter_trends.png'} and {OUTPUT_DIRECTORY / 'parameter_basins.png'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    summary = (json.loads((OUTPUT_DIRECTORY / "parameter_study.json").read_text())
               if args.plot_only else run_study())
    plot_study(summary)
    plt.show()


if __name__ == "__main__":
    main()
