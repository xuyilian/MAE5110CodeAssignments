"""Simulate and animate passive stepping followed by torque-limited standing.

Examples:
    uv run python assignment_2/codes/assignment_2.py --theta 0 --omega 4
    uv run python assignment_2/codes/assignment_2.py --theta -0.2 --omega 1 --show
    uv run python assignment_2/codes/assignment_2.py --theta 0.03 --omega 0 --no-animation

Angles are radians. Arbitrary finite initial states are accepted, but recovery
is not guaranteed outside the designed RoA/section domain. Failed and timed-out
runs are saved with their actual status; no state is snapped to equilibrium.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from models import inverted_pendulum_walker as model

ALPHA_MIN, ALPHA_MAX = np.pi / 8, np.pi / 7
KP, KD = 240.0, 80.0
CAPTURE_MARGIN = 1e-10
STANDING_TOLERANCE = 1e-6


def capture_bounds(theta, params):
    """Analytical boundary velocities for normalized torque limits -.1, +.05."""
    theta = np.asarray(theta)
    q = params["gravity"] / params["length"]
    left, right = -np.arcsin(0.05), np.arcsin(0.1)
    upper_energy = 2 * q * (np.cos(right) - np.cos(theta) + 0.1 * (right - theta))
    lower_energy = 2 * q * (np.cos(left) - np.cos(theta) + 0.05 * (theta - left))
    upper = np.where(theta <= right, 1.0, -1.0) * np.sqrt(np.maximum(upper_energy, 0))
    lower = np.where(theta >= left, -1.0, 1.0) * np.sqrt(np.maximum(lower_energy, 0))
    return lower, upper


def in_capture(state, params):
    """Use only the angle domain covered by the standing-controller proof."""
    theta, omega = state
    lower, upper = capture_bounds(theta, params)
    return (
        (np.abs(theta) <= params["incline"] + ALPHA_MAX)
        & (omega > lower + CAPTURE_MARGIN)
        & (omega < upper - CAPTURE_MARGIN)
    )


def standing_torque(state, params):
    """Saturated torque for a state (2,) or a batch of states (2, N)."""
    theta, omega = state
    m, g, length = params["mass"], params["gravity"], params["length"]
    scale = m * g * length
    torque = np.clip(
        -scale * np.sin(theta) - m * length**2 * (KP * theta + KD * omega),
        -0.1 * scale,
        0.05 * scale,
    )
    return float(torque) if np.ndim(torque) == 0 else torque


def uniform_action(omega, params):
    """Nearest of nodes [0, Omega/2, Omega]; ties choose the higher node.

    Beyond Omega, continue with the largest angle as an explicit extension;
    the report's minimum-step validation applies only inside [0, Omega].
    """
    limit = np.sqrt(2 * params["gravity"] / params["length"])
    return np.where(np.asarray(omega) < limit / 4, ALPHA_MIN, ALPHA_MAX)


def post_impact(state, alpha, params):
    """Predict forward touchdown from conserved passive stance energy."""
    theta, omega = state
    q, gamma = params["gravity"] / params["length"], params["incline"]
    speed_squared = omega**2 + 2 * q * (np.cos(theta) - np.cos(gamma + alpha))
    post = np.array(
        [gamma - alpha, np.cos(2 * alpha) * np.sqrt(np.maximum(speed_squared, 0))]
    )
    # If the path crosses upright, it must have enough energy to pass it.
    reaches_upright = (theta >= 0) | (omega**2 + 2 * q * (np.cos(theta) - 1) > 0)
    valid = (gamma + alpha >= theta - 1e-12) & (speed_squared > 0) & reaches_upright
    return post, valid


def remaining_steps(velocities, params, max_steps=100):
    """Analytical rollout used only to rank the initial off-section actions."""
    omega = np.asarray(velocities).copy()
    result = np.full(omega.shape, np.inf)
    result[in_capture(np.array([np.zeros_like(omega), omega]), params)] = 0
    active = ~np.isfinite(result)
    q = params["gravity"] / params["length"]
    for step in range(1, max_steps + 1):
        ids = np.flatnonzero(active)
        if not ids.size:
            break
        alpha = uniform_action(omega[ids], params)
        post, valid = post_impact([0.0, omega[ids]], alpha, params)
        captured = valid & in_capture(post, params)
        result[ids[captured]] = step
        square = post[1] ** 2 + 2 * q * (np.cos(post[0]) - 1)
        returns = valid & ~captured & (square > 0)
        active[ids] = returns
        omega[ids[returns]] = np.sqrt(square[returns])
    return result


def initial_action(state, params):
    """Rank 401 feasible first impacts, then follow the uniform section policy.

    Negative initial velocity is propagated passively until capture, a forward
    section crossing, or a fall; no unanalysed backward stepping is introduced.
    """
    theta, omega = state
    if abs(theta) < 1e-12 and omega > 0:
        return float(uniform_action(omega, params))
    alpha = np.linspace(ALPHA_MIN, ALPHA_MAX, 401)
    if omega <= 0:
        return float(ALPHA_MAX)
    post, valid = post_impact(state, alpha, params)
    captured = valid & in_capture(post, params)
    cost = np.full(alpha.shape, np.inf)
    cost[captured] = 1
    q = params["gravity"] / params["length"]
    square = post[1] ** 2 + 2 * q * (np.cos(post[0]) - 1)
    returns = valid & ~captured & (square > 0)
    cost[returns] = 1 + remaining_steps(np.sqrt(square[returns]), params)
    if not np.any(np.isfinite(cost)):
        return float(ALPHA_MAX)
    best = np.flatnonzero(cost == np.min(cost))
    # Prefer capture margin among tied capture actions, otherwise lower return speed.
    if np.min(cost) == 1:
        lo, hi = capture_bounds(post[0, best], params)
        margin = np.minimum(post[1, best] - lo, hi - post[1, best])
        return float(alpha[best[np.argmax(margin)]])
    return float(alpha[best[np.argmin(square[best])]])


def rk4(state, time, dt, params, standing=False):
    """Re-evaluate feedback at every RK stage, not just once per timestep."""

    def derivative(t, y):
        controls = dict(
            params, ankle_torque=standing_torque(y, params) if standing else 0.0
        )
        return model.dynamics(t, y, controls)

    k1 = derivative(time, state)
    k2 = derivative(time + dt / 2, state + dt * k1 / 2)
    k3 = derivative(time + dt / 2, state + dt * k2 / 2)
    k4 = derivative(time + dt, state + dt * k3)
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def locate_event(state, time, dt, params, predicate):
    """Bisect a bracketed passive event to avoid stepping across impact resets."""
    lo, hi = 0.0, dt
    for _ in range(35):
        middle = (lo + hi) / 2
        if predicate(rk4(state, time, middle, params)):
            hi = middle
        else:
            lo = middle
    return hi, rk4(state, time, hi, params)


def simulate(initial_state=(0.0, 4.0), *, dt=0.001, duration=15.0, max_steps=100):
    """Return an event-resolved trajectory with world foot positions and status."""
    state = np.asarray(initial_state, dtype=float).copy()
    if state.shape != (2,) or not np.all(np.isfinite(state)):
        raise ValueError(
            "Initial state must contain two finite numbers: theta and omega"
        )
    if not np.isfinite(dt) or not 0 < dt <= 0.01:
        raise ValueError("dt must be in (0, 0.01] seconds for this controller")
    if not np.isfinite(duration) or duration <= 0 or max_steps < 1:
        raise ValueError("duration and max_steps must be positive")
    params = model.generate_params()
    standing = bool(in_capture(state, params))
    params["angle_of_attack"] = (
        initial_action(state, params) if not standing else ALPHA_MIN
    )
    time, foot, steps, held_since = 0.0, np.zeros(2), 0, None
    capture_time = 0.0 if standing else None
    events, records = [], []
    status, reason = (
        "timeout",
        "Standing tolerance was not reached within the time limit.",
    )

    def record():
        records.append(
            (
                time,
                *state,
                *foot,
                params["angle_of_attack"],
                standing_torque(state, params) if standing else 0.0,
                int(standing),
                steps,
            )
        )

    record()
    # A forward swing foot cannot be placed above ground beyond the last guard.
    if not standing and state[0] > params["incline"] + ALPHA_MAX + 1e-12:
        status, reason = (
            "invalid_initial_pose",
            "Initial angle is beyond every allowed forward touchdown guard.",
        )
    elif np.cos(state[0] - params["incline"]) <= 0 or abs(state[0]) > np.pi:
        status, reason = "fallen", "Initial point mass is at or below the ground."
    else:
        while time < duration - 1e-12:
            if standing:
                if np.max(np.abs(state)) < STANDING_TOLERANCE:
                    if held_since is None:
                        held_since = time
                    if time - held_since >= 0.5:
                        status, reason = (
                            "standing",
                            "Both state components stayed below 1e-6 for 0.5 seconds.",
                        )
                        break
                else:
                    held_since = None
            h = min(dt, duration - time)
            next_state = rk4(state, time, h, params, standing)
            kind = None
            if not standing:
                candidates = []
                if state[0] < 0 <= next_state[0] and next_state[1] > 0:
                    eh, ey = locate_event(state, time, h, params, lambda y: y[0] >= 0)
                    candidates.append((eh, "section", ey))
                if model.event_guard(state, next_state, params):
                    target = params["incline"] + params["angle_of_attack"]
                    eh, ey = locate_event(
                        state, time, h, params, lambda y, target=target: y[0] >= target
                    )
                    candidates.append((eh, "impact", ey))
                if in_capture(next_state, params):
                    eh, ey = locate_event(
                        state, time, h, params, lambda y: in_capture(y, params)
                    )
                    candidates.append((eh, "capture", ey))
                if candidates:
                    h, kind, next_state = min(candidates, key=lambda event: event[0])
            time += h
            state = next_state
            if kind == "section":
                state[0] = 0.0
                params["angle_of_attack"] = float(uniform_action(state[1], params))
            elif kind == "impact":
                state[0] = params["incline"] + params["angle_of_attack"]
                record()  # Retain the pre-impact state at the same event time.
                pre = state.copy()
                state = model.event_dynamics(state, params)
                foot += (
                    2
                    * params["length"]
                    * np.sin(params["angle_of_attack"])
                    * np.array([np.cos(params["incline"]), -np.sin(params["incline"])])
                )
                steps += 1
                events.append(
                    {
                        "time": time,
                        "alpha": params["angle_of_attack"],
                        "pre": pre.tolist(),
                        "post": state.tolist(),
                    }
                )
            if not standing and in_capture(state, params):
                standing, capture_time = True, time
            record()
            if np.cos(state[0] - params["incline"]) <= 0:
                status, reason = (
                    "fallen",
                    "Point mass reached the ground without capture.",
                )
                break
            if steps >= max_steps and not standing:
                status, reason = (
                    "step_limit",
                    "Maximum permitted footstrikes reached without capture.",
                )
                break
    data = np.asarray(records)
    summary = {
        "status": status,
        "reason": reason,
        "initial_state": list(map(float, initial_state)),
        "final_state": state.tolist(),
        "elapsed_time": time,
        "footstrikes": steps,
        "capture_time": capture_time,
        "standing_tolerance": STANDING_TOLERANCE,
        "dt": dt,
        "torque_min": float(data[:, 6].min()),
        "torque_max": float(data[:, 6].max()),
        "events": events,
        "policy_scope": "Uniform policy at forward section crossings; first off-section action uses analytical search. "
        "Speeds above sqrt(2g/l) use the largest angle as an unproved extension.",
    }
    return {
        "time": data[:, 0],
        "state": data[:, 1:3],
        "foot": data[:, 3:5],
        "alpha": data[:, 5],
        "torque": data[:, 6],
        "standing": data[:, 7].astype(bool),
        "steps": data[:, 8].astype(int),
        "summary": summary,
    }


def animate(result, output, fps=25, show=False, speed=0.5):
    """Fixed world camera and persistent artists for smooth, half-speed playback."""
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("Playback speed must be positive and finite")
    fig, (ax, phase) = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        gridspec_kw={"width_ratios": [2.5, 1]},
        layout="constrained",
    )
    params = model.generate_params()
    length = params["length"]
    state, foot = result["state"], result["foot"]
    hub = foot + length * np.column_stack((np.sin(state[:, 0]), np.cos(state[:, 0])))
    swing_angle = state[:, 0] - 2 * result["alpha"]
    swing = hub - length * np.column_stack((np.sin(swing_angle), np.cos(swing_angle)))
    walking = ~result["standing"]
    geometry = np.concatenate([foot, hub, swing[walking]], axis=0)
    x_min, y_min = geometry.min(axis=0) - [0.45, 0.3]
    x_max, y_max = geometry.max(axis=0) + [0.45, 0.4]
    # Keep short standing-only runs at a useful, consistent physical scale.
    if x_max - x_min < 3 * length:
        middle = (x_min + x_max) / 2
        x_min, x_max = middle - 1.5 * length, middle + 1.5 * length
    ax.set(
        xlim=(x_min, x_max),
        ylim=(y_min, y_max),
        xlabel="World x (m)",
        ylabel="World y (m)",
    )
    ax.set_aspect("equal", adjustable="box")
    ground_x = np.array([x_min, x_max])
    ground_y = -np.tan(params["incline"]) * ground_x
    ax.fill_between(ground_x, ground_y, y_min, color="#eee7dc")
    ax.plot(ground_x, ground_y, color="#7b6651", lw=2)
    # Ground markers make displacement visible without a moving coordinate frame.
    marks = np.arange(np.ceil(x_min / 0.5) * 0.5, x_max, 0.5)
    ax.plot(marks, -np.tan(params["incline"]) * marks, "|", color="#9d8c75", ms=7)
    ax.plot(hub[:, 0], hub[:, 1], color="#c7d9e4", lw=1.3, zorder=1)
    (stance_line,) = ax.plot([], [], color="#23699b", lw=4, zorder=4)
    (swing_line,) = ax.plot([], [], "--", color="#df8a25", lw=2.5, zorder=3)
    (stance_dot,) = ax.plot([], [], "s", color="#333333", ms=7, zorder=5)
    (swing_dot,) = ax.plot([], [], "o", color="#df8a25", ms=6, zorder=5)
    (mass_dot,) = ax.plot([], [], "o", color="#23699b", mec="white", ms=16, zorder=6)
    title = ax.set_title("")
    readout = fig.text(0.04, 0.055, "", fontsize=11, va="bottom")
    fig.suptitle(
        f"Controlled walker | fixed world view | {speed:g}× playback", fontsize=16
    )
    fig.get_layout_engine().set(rect=(0, 0.1, 1, 0.85))
    phase.plot(state[:, 0], state[:, 1], color="#ccd8df", lw=1)
    phase.plot(0, 0, "+", color="#385840", ms=10)
    (trail,) = phase.plot([], [], color="#477b98", lw=1.5)
    (marker,) = phase.plot([], [], "o", color="#d6752d", ms=7)
    phase.set(
        xlabel=r"$\theta$ (rad)",
        ylabel=r"$\dot\theta$ (rad/s)",
        title="State trajectory",
    )
    phase.grid(alpha=0.15)
    sample_times = np.arange(0, result["time"][-1] + speed / fps, speed / fps)
    indices = np.clip(
        np.searchsorted(result["time"], sample_times, side="right") - 1,
        0,
        len(state) - 1,
    ).tolist()
    indices += [len(state) - 1] * fps

    def point(artist, xy):
        artist.set_data([xy[0]], [xy[1]])

    def draw(frame):
        index = indices[frame]
        stance_line.set_data(
            [foot[index, 0], hub[index, 0]], [foot[index, 1], hub[index, 1]]
        )
        swing_line.set_data(
            [hub[index, 0], swing[index, 0]], [hub[index, 1], swing[index, 1]]
        )
        point(stance_dot, foot[index])
        point(swing_dot, swing[index])
        point(mass_dot, hub[index])
        swing_line.set_visible(bool(walking[index]))
        swing_dot.set_visible(bool(walking[index]))
        mode = "Standing control" if result["standing"][index] else "Passive stepping"
        if index == len(state) - 1:
            mode = result["summary"]["status"].replace("_", " ").capitalize()
        title.set_text(
            f"{mode} | t = {result['time'][index]:.2f} s | {result['steps'][index]} footstrikes"
        )
        readout.set_text(
            f"Angle: {state[index, 0]: .3f} rad     "
            f"Angular velocity: {state[index, 1]: .3f} rad/s     "
            f"Ankle torque: {result['torque'][index]: .3f} N m"
        )
        trail.set_data(state[: index + 1, 0], state[: index + 1, 1])
        marker.set_data([state[index, 0]], [state[index, 1]])
        return (
            stance_line,
            swing_line,
            stance_dot,
            swing_dot,
            mass_dot,
            trail,
            marker,
            title,
            readout,
        )

    draw(0)
    fig.canvas.draw()
    fig.set_layout_engine(None)
    animation = FuncAnimation(
        fig, draw, frames=range(len(indices)), interval=1000 / fps, repeat=False
    )
    animation.save(output, writer=PillowWriter(fps=fps), dpi=85)
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--theta", type=float, default=0.0, help="Initial angle in radians"
    )
    parser.add_argument(
        "--omega", type=float, default=4.0, help="Initial angular velocity in rad/s"
    )
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument(
        "--duration", type=float, default=15.0, help="Maximum simulated seconds"
    )
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument(
        "--speed",
        type=float,
        default=0.5,
        help="Playback speed: 0.5 for half speed, 1 for real time",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures" / "walker.gif",
    )
    parser.add_argument("--no-animation", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.fps < 1 or args.output.suffix.lower() != ".gif":
        parser.error("fps must be positive and output must end in .gif")
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error("speed must be positive and finite")
    result = simulate((args.theta, args.omega), dt=args.dt, duration=args.duration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(result["summary"], indent=2) + "\n"
    )
    np.savez(
        args.output.with_suffix(".npz"),
        **{k: v for k, v in result.items() if k != "summary"},
    )
    print(json.dumps(result["summary"], indent=2), flush=True)
    if not args.no_animation:
        animate(result, args.output, args.fps, args.show, args.speed)
        print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
