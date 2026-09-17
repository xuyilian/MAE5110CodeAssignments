"""Validate endpoint actions on a uniform nearest-node velocity grid."""
import json
import numpy as np
from two_interval_policy import (
    OUT, SWITCH, LARGE_LOWER, SMALL_UPPER, ONE_STEP_UPPER, TWO_STEP_UPPER,
    metrics,
)
from build_lookup_policy import CAPTURE, LIMIT, ALPHA_MIN, ALPHA_MAX, transition, rollout


def table(node_count):
    if node_count < 2:
        raise ValueError('At least two nodes are needed to span the velocity range')
    nodes = np.linspace(0, LIMIT, node_count)
    # Extend the low-speed walking action to zero. Actual RoA membership is
    # checked before lookup, so a rounded node never decides standing status.
    actions = np.full(node_count, ALPHA_MAX)
    actions[0] = ALPHA_MIN
    return nodes, actions


def choose_action(omega, node_count=3):
    """Check the RoA before choosing a nearest-node walking action."""
    omega = np.asarray(omega, dtype=float)
    if np.any(~np.isfinite(omega) | (omega < 0) | (omega > LIMIT)):
        raise ValueError('Velocity must lie in the designed section range')
    nodes, actions = table(node_count)
    index = np.searchsorted((nodes[:-1]+nodes[1:])/2, omega, side='right')
    return np.where(omega < CAPTURE-1e-10, np.nan, actions[index])


def execute(starts, node_count, max_steps=20):
    omega = np.asarray(starts, dtype=float).copy()
    counts = np.full(omega.shape, -1, dtype=int)
    counts[omega < CAPTURE - 1e-10] = 0
    active = counts < 0
    for step in range(1, max_steps + 1):
        ids = np.flatnonzero(active)
        if not ids.size:
            break
        captured, valid, nxt, _ = transition(omega[ids], choose_action(omega[ids], node_count))
        counts[ids[captured]] = step
        active[ids[captured | ~valid]] = False
        omega[ids[valid]] = nxt[valid]
        arrived = valid & (nxt < CAPTURE - 1e-10)
        counts[ids[arrived]] = step
        active[ids[arrived]] = False
    return counts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    centers = [CAPTURE, SWITCH, LARGE_LOWER, SMALL_UPPER,
               ONE_STEP_UPPER, TWO_STEP_UPPER, LIMIT/2, LIMIT/4]
    offsets = np.r_[0., -np.logspace(-6, -2, 5), np.logspace(-6, -2, 5)]
    starts = np.unique(np.r_[np.linspace(0, LIMIT, 100001),
                             *(c + offsets for c in centers)])
    references = []
    for filename in ['lookup_policy.npz', 'reference_4801_801.npz',
                     'reference_9601_801.npz']:
        with np.load(OUT / filename) as data:
            references.append(rollout(starts, data['omega'], data['policy'], max_steps=20))
    reference = references[-1]
    assert np.all(reference >= 0), 'Reference must recover every test state'
    rows = []
    for node_count in [2, 3]:
        nodes, actions = table(node_count)
        counts = execute(starts, node_count)
        wrong = np.flatnonzero((counts < 0) | (counts != reference))
        row = {'velocity_nodes': node_count, 'spacing': float(nodes[1]-nodes[0]),
               'nodes': nodes.tolist(), 'actions_rad': actions.tolist(),
               **metrics(counts, reference)}
        if wrong.size:
            j = wrong[0]
            row['first_counterexample'] = {'omega': float(starts[j]),
                                          'steps': int(counts[j]),
                                          'reference_steps': int(reference[j])}
        rows.append(row)
        np.savez(OUT / f'uniform_velocity_{node_count}_validation.npz',
                 omega=starts, steps=counts, reference_steps=reference)
    summary = {'test_count': len(starts), 'boundary_centers': centers,
               'reference_failures': int(np.sum(reference < 0)),
               'reference_refinement_disagreements':
                   [int(np.sum(r != reference)) for r in references],
               'results': rows,
               'scope': 'Uniform endpoint-inclusive nodes with low action at zero and high action elsewhere; '
                        'nearest-node actions, exact RoA guard, and unrounded dynamics.'}
    (OUT / 'uniform_velocity_validation.json').write_text(json.dumps(summary, indent=2)+'\n')
    nodes, actions = table(3)
    np.savetxt(OUT / 'uniform_velocity_policy.csv', np.column_stack([nodes, actions]),
               delimiter=',', header='omega_node,alpha_rad', comments='')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
