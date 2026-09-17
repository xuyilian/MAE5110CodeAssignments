"""Numerical stance RoA sweep, boundary probes, and controlled examples.

Uses the implemented controller and model; no touchdown or stepping policy.
Run from the repository root: uv run python assignment_2/codes/validate_roa.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from assignment_2 import capture_bounds, standing_torque, model, rk4

OUT = Path(__file__).resolve().parents[1] / 'figures'
PARAMS = model.generate_params()
LEFT, RIGHT = PARAMS['incline'] - np.pi/8, PARAMS['incline'] + np.pi/7
TOL, HORIZON = 1e-6, 15.0


def inside(states):
    low, high = capture_bounds(states[:, 0], PARAMS)
    return (states[:, 1] > low) & (states[:, 1] < high)


def integrate(initial, dt, keep_history=False):
    """RK4 with stage-wise feedback; stop only trajectories leaving angle I.

    Success requires both tolerances throughout the FINAL half second.
    Boundary/invariance checks refer to discrete integration samples.
    """
    states = initial.copy()
    active = np.ones(len(states), dtype=bool)
    hold = np.zeros(len(states), dtype=int)
    onset = np.full(len(states), np.nan)
    min_margin = np.full(len(states), np.inf)
    torque_min, torque_max = np.inf, -np.inf
    history = [states.copy()] if keep_history else None
    steps = round(HORIZON/dt)
    for k in range(steps + 1):
        ids = np.flatnonzero(active)
        z = states[ids]
        low, high = capture_bounds(z[:, 0], PARAMS)
        min_margin[ids] = np.minimum(min_margin[ids], np.minimum(z[:, 1]-low, high-z[:, 1]))
        torque = standing_torque(z.T, PARAMS)
        if len(ids):
            torque_min = min(torque_min, float(torque.min()))
            torque_max = max(torque_max, float(torque.max()))
        small = np.all(np.abs(z) < TOL, axis=1)
        hold[ids] = np.where(small, hold[ids] + 1, 0)
        onset[ids[~small]] = np.nan
        fresh = small & np.isnan(onset[ids])
        onset[ids[fresh]] = k*dt
        if k == steps:
            break
        states[ids] = rk4(z.T, k*dt, dt, PARAMS, standing=True).T
        escaped = (~np.all(np.isfinite(states), axis=1)) | (states[:, 0] < LEFT-1e-10) | (states[:, 0] > RIGHT+1e-10)
        active &= ~escaped
        if keep_history:
            history.append(states.copy())
    success = active & (hold >= round(.5/dt)+1)
    return dict(success=success, escaped=~active, final=states, onset=onset,
                min_margin=min_margin, torque_range=[torque_min, torque_max],
                history=np.asarray(history) if keep_history else None)


def summarize(result, expected):
    success = result['success']
    return dict(total=len(expected), inside=int(expected.sum()),
                inside_converged=int(np.sum(expected & success)),
                outside_converged=int(np.sum(~expected & success)),
                escaped=int(result['escaped'].sum()),
                unresolved=int(np.sum(~success & ~result['escaped'])),
                classification_disagreement=int(np.sum(success != expected)),
                inside_boundary_violations=int(np.sum(expected & (result['min_margin'] < -1e-9))),
                inside_min_boundary_margin=float(result['min_margin'][expected].min()),
                inside_max_final_abs_state=np.max(np.abs(result['final'][expected]), axis=0).tolist(),
                latest_inside_onset=float(np.nanmax(result['onset'][expected])),
                torque_range=result['torque_range'])


def plots(grid, result, examples, history, dt):
    plt.rcParams.update({'font.size': 15, 'axes.titlesize': 19, 'axes.labelsize': 18,
                         'xtick.labelsize': 13, 'ytick.labelsize': 13})
    theta = np.linspace(LEFT, RIGHT, 1501)
    low, high = capture_bounds(theta, PARAMS)
    colors = plt.get_cmap('tab10')(np.arange(len(examples)))
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), layout='constrained')
    axes[0].scatter(*grid[~result['success']].T, s=3, c='#dddddd', rasterized=True, label='Does not settle in test')
    axes[0].scatter(*grid[result['success']].T, s=6, c='#237c71', rasterized=True, label='Numerically converged')
    axes[0].set_title('Uniform state grid: numerical RoA')
    axes[1].fill_between(theta, low, high, color='#d9eee8', label='Analytical interior')
    for ax in axes:
        ax.plot(theta, low, color='#404040', lw=1.8)
        ax.plot(theta, high, color='#404040', lw=1.8, label='Analytical boundaries')
        ax.set(xlabel=r'$\theta$ (rad)', ylabel=r'$\omega$ (rad/s)', xlim=(LEFT-.015, RIGHT+.015), ylim=(-1.8, 1.8))
        ax.grid(alpha=.2)
    for j, color in enumerate(colors):
        axes[1].plot(history[:, j, 0], history[:, j, 1], color=color, lw=2, label=f'T{j+1}')
        axes[1].scatter(*examples[j], color=color, s=55, marker='o', zorder=5)
        p, q = history[100, j], history[170, j]
        axes[1].annotate('', xy=q, xytext=p, arrowprops=dict(arrowstyle='->', color=color, lw=2))
    axes[1].scatter(0, 0, color='black', marker='*', s=130, zorder=6)
    axes[1].set_title('Six stance trajectories under control')
    axes[0].legend(loc='lower left', fontsize=12)
    axes[1].legend(loc='upper right', fontsize=11, ncol=2)
    fig.savefig(OUT/'numerical_roa.png', dpi=180)
    plt.close(fig)
    time = np.arange(len(history))*dt
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True, layout='constrained')
    for j, color in enumerate(colors):
        for ax, values in zip(axes, [history[:, j, 0], history[:, j, 1], standing_torque(history[:, j].T, PARAMS)]):
            ax.plot(time, values, color=color, lw=2, label=f'T{j+1}')
    scale = PARAMS['mass']*PARAMS['gravity']*PARAMS['length']
    for bound in [-.1*scale, .05*scale]:
        axes[2].axhline(bound, color='black', ls='--', lw=1.5)
    for ax, label in zip(axes, [r'$\theta$ (rad)', r'$\omega$ (rad/s)', r'$\tau$ (N m)']):
        ax.set_ylabel(label)
        ax.grid(alpha=.25)
        ax.set_xlim(0, 8)
    axes[0].legend(ncol=6, loc='upper right', fontsize=13)
    axes[0].set_title('Controlled stance: state convergence and bounded torque')
    axes[2].set_xlabel('Time (s)')
    fig.savefig(OUT/'roa_controlled_trajectories.png', dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    th, om = np.meshgrid(np.linspace(LEFT, RIGHT, 121), np.linspace(-1.8, 1.8, 181))
    grid = np.column_stack([th.ravel(), om.ravel()])
    # Offset probes inside AND outside each boundary, avoiding exact separatrices.
    theta = np.linspace(LEFT, RIGHT, 121)
    low, high = capture_bounds(theta, PARAMS)
    probes = np.concatenate([np.column_stack([theta, edge + sign*eps*(high-low)])
                             for eps in [1e-3, 1e-5] for edge in [low, high] for sign in [-1, 1]])
    initial = np.vstack([grid, probes])
    expected = inside(initial)
    runs = []
    summary = {}
    for dt in [.002, .001]:
        run = integrate(initial, dt)
        runs.append(run)
        summary[str(dt)] = {'grid': summarize({k: v[:len(grid)] if isinstance(v, np.ndarray) else v for k, v in run.items()}, expected[:len(grid)]),
                            'boundary_probes': summarize({k: v[len(grid):] if isinstance(v, np.ndarray) else v for k, v in run.items()}, expected[len(grid):])}
        print(dt, json.dumps(summary[str(dt)]), flush=True)
    summary['refinement'] = dict(max_inside_onset_difference=float(np.max(np.abs(runs[0]['onset'][expected]-runs[1]['onset'][expected]))), classification_changes=int(np.sum(runs[0]['success'] != runs[1]['success'])),
        max_inside_terminal_difference=np.max(np.abs(runs[0]['final'][expected]-runs[1]['final'][expected]), axis=0).tolist())
    theta = np.array([-.30, -.20, 0., 0., .30, .48])
    low, high = capture_bounds(theta, PARAMS)
    fractions = np.array([.5, .01, .99, .01, .99, .5])
    examples = np.column_stack([theta, low + fractions*(high-low)])
    traces = integrate(examples, .001, keep_history=True)
    summary['examples'] = [dict(label=f'T{j+1}', initial=z.tolist(), settled=bool(traces['success'][j]), onset=float(traces['onset'][j])) for j, z in enumerate(examples)]
    summary['settings'] = dict(horizon=HORIZON, tolerance=TOL, hold=.5, angle_domain=[LEFT, RIGHT], grid_shape=[121,181], boundary_offsets=[1e-3,1e-5])
    np.savez_compressed(OUT/'numerical_roa_data.npz', initial=initial, expected=expected,
                        coarse_success=runs[0]['success'], fine_success=runs[1]['success'],
                        fine_final=runs[1]['final'], examples=examples, example_history=traces['history'])
    (OUT/'numerical_roa_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    plots(grid, {k: v[:len(grid)] if isinstance(v, np.ndarray) else v for k,v in runs[1].items()}, examples, traces['history'], .001)
    print(json.dumps(summary['refinement']), json.dumps(summary['examples']), flush=True)
    for run in runs:
        assert np.all(run['success'][expected]), 'Some interior states did not settle.'
        assert not np.any(run['success'][~expected]), 'Unexpected exterior convergence.'
        assert np.all(run['min_margin'][expected] >= -1e-9), 'Interior boundary violation.'
    assert np.all(traces['success']), 'An example did not settle.'


if __name__ == '__main__':
    main()
