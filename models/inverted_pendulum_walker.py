"""Compatibility import for the organized Assignment 2 model."""

import runpy
from pathlib import Path

_model = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "assignment_2"
        / "codes"
        / "models"
        / "inverted_pendulum_walker.py"
    )
)
generate_params = _model["generate_params"]
dynamics = _model["dynamics"]
event_guard = _model["event_guard"]
event_dynamics = _model["event_dynamics"]
calculate_energy = _model["calculate_energy"]
visualize = _model["visualize"]
