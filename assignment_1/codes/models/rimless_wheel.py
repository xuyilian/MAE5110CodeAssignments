"""Rimless-wheel dynamics and forward/backward contact resets for Assignment 1."""

import numpy as np


def dynamics(t, state, params):

    del t  # The model is autonomous; keep t for compatibility with integrators.

    gravity = params["gravity"]
    spoke_length = params["spoke_length"]

    angle = state[0]
    angular_velocity = state[1]

    angular_acceleration = gravity / spoke_length * np.sin(angle)

    return np.array([angular_velocity, angular_acceleration])


def detect_contact(state, params):

    angle = state[0]
    angular_velocity = state[1]

    contact_angle = params["slope_angle"] + params["half_spoke_angle"]
    return bool(angle >= contact_angle and angular_velocity > 0.0)


def apply_impact_reset(state, params):

    angular_velocity_before_impact = state[1]
    half_spoke_angle = params["half_spoke_angle"]
    slope_angle = params["slope_angle"]

    angle_after_impact = slope_angle - half_spoke_angle
    angular_velocity_after_impact = angular_velocity_before_impact * np.cos(
        2.0 * half_spoke_angle
    )

    return np.array([angle_after_impact, angular_velocity_after_impact])


def detect_backward_contact(state, params):
    """Return whether the previous spoke contacts during backward rocking."""
    angle = state[0]
    angular_velocity = state[1]

    backward_contact_angle = (
        params["slope_angle"] - params["half_spoke_angle"]
    )
    return bool(
        angle <= backward_contact_angle and angular_velocity < 0.0
    )


def apply_backward_impact_reset(state, params):
    """Return the post-impact state after switching to the previous spoke."""
    angular_velocity_before_impact = state[1]
    half_spoke_angle = params["half_spoke_angle"]
    slope_angle = params["slope_angle"]

    angle_after_impact = slope_angle + half_spoke_angle
    angular_velocity_after_impact = angular_velocity_before_impact * np.cos(
        2.0 * half_spoke_angle
    )

    return np.array([angle_after_impact, angular_velocity_after_impact])


def generate_params(number_of_spokes=6):
    
    if not isinstance(number_of_spokes, (int, np.integer)):
        raise TypeError("number_of_spokes must be an integer")
    if number_of_spokes < 3:
        raise ValueError("number_of_spokes must be at least 3")

    return {
        "gravity": 9.81,  # Earth gravity (m/s^2)
        "spoke_length": 1.0,  # Distance from contact point to hub (m)
        "hub_mass": 1.0,  # Point mass at the hub (kg)
        "number_of_spokes": number_of_spokes,
        "half_spoke_angle": np.pi / number_of_spokes,  # alpha (rad)
        "slope_angle": np.deg2rad(10.0),  # gamma (rad)
        "settling_velocity": 1e-3,  # Two-contact rest threshold (rad/s)
    }
