"""Physics and closed-loop checks independent of animation rendering."""

import numpy as np
import pytest

import assignment_2 as experiment
from models import inverted_pendulum_walker as model


def test_passive_energy_and_torque_power():
    p = model.generate_params()
    state = np.array([0.2, 1.3])
    p["ankle_torque"] = 0.2
    eps = 1e-6
    derivative = model.dynamics(0, state, p)
    power = (
        model.calculate_energy(state + eps * derivative, p)
        - model.calculate_energy(state - eps * derivative, p)
    ) / (2 * eps)
    assert power == pytest.approx(0.2 * state[1], abs=1e-8)
    p["ankle_torque"] = 0.0
    after = state.copy()
    for i in range(100):
        after = experiment.rk4(after, i * 0.001, 0.001, p)
    assert model.calculate_energy(after, p) == pytest.approx(
        model.calculate_energy(state, p), abs=1e-10
    )


@pytest.mark.parametrize("alpha", [np.pi / 8, np.pi / 7])
def test_impact_preserves_hub_position_and_dissipates_energy(alpha):
    p = model.generate_params()
    p["angle_of_attack"] = alpha
    gamma, length, mass = p["incline"], p["length"], p["mass"]
    before = np.array([gamma + alpha, 2.0])
    after = model.event_dynamics(before, p)
    foot = 2 * length * np.sin(alpha) * np.array([np.cos(gamma), -np.sin(gamma)])
    hub_before = length * np.array([np.sin(before[0]), np.cos(before[0])])
    hub_after = foot + length * np.array([np.sin(after[0]), np.cos(after[0])])
    np.testing.assert_allclose(hub_before, hub_after, atol=1e-14)
    lost = model.calculate_energy(before, p) - (
        model.calculate_energy(after, p) + mass * p["gravity"] * foot[1]
    )
    assert lost == pytest.approx(
        0.5 * mass * length**2 * before[1] ** 2 * np.sin(2 * alpha) ** 2
    )
    assert model.event_guard(before - [0.01, 0], before + [0.01, 0], p)
    assert not model.event_guard([before[0] + 0.01, -1], [before[0] - 0.01, -1], p)


@pytest.mark.parametrize(
    "state, steps",
    [
        ((0, 0), 0),
        ((0.03, 0), 0),
        ((0, -0.1), 0),
        ((-0.2, 1), 1),
        ((0.2, -0.5), 0),
        ((0.2, 1), 1),
        ((0, 0.4), 1),
        ((0, 1.2), 1),
        ((0, 2), 2),
        ((0, 4), 3),
    ],
)
def test_controlled_stopping(state, steps):
    result = experiment.simulate(state)
    assert result["summary"]["status"] == "standing"
    assert result["summary"]["footstrikes"] == steps
    assert np.max(np.abs(result["state"][-1])) < 1e-6
    assert result["torque"].min() >= -0.981 - 1e-12
    assert result["torque"].max() <= 0.4905 + 1e-12
    assert np.all(result["torque"][~result["standing"]] == 0)
    for event in result["summary"]["events"]:
        assert np.pi / 8 <= event["alpha"] <= np.pi / 7
        assert event["pre"][0] == pytest.approx(0.06 + event["alpha"], abs=1e-12)


def test_timestep_refinement():
    coarse = experiment.simulate((0, 4), dt=0.001)["summary"]
    fine = experiment.simulate((0, 4), dt=0.0005)["summary"]
    assert coarse["footstrikes"] == fine["footstrikes"] == 3
    assert abs(coarse["capture_time"] - fine["capture_time"]) < 1e-8
    np.testing.assert_allclose(
        coarse["events"][-1]["post"], fine["events"][-1]["post"], atol=1e-9
    )


def test_failure_and_timeout_are_not_reported_as_standing():
    assert experiment.simulate((-0.2, -0.1))["summary"]["status"] == "fallen"
    assert experiment.simulate((0.8, 0))["summary"]["status"] == "invalid_initial_pose"
    assert experiment.simulate((0, 4), duration=0.1)["summary"]["status"] == "timeout"
    with pytest.raises(ValueError):
        experiment.simulate((np.nan, 0))
