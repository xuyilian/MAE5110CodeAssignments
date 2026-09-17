"""Evaluate conservative analytical gain bounds; no state grid or simulation."""
import math


def main():
    g, length = 9.81, 1.0
    a, b, gamma = .1, .05, .06
    theta_min, theta_max = gamma-math.pi/8, gamma+math.pi/7
    radius = max(abs(theta_min), abs(theta_max))
    q = g/length
    theta_l, theta_r = -math.asin(b), math.asin(a)
    slope_min, slope_max = math.sqrt(q*math.cos(radius)), math.sqrt(q)
    ratio, kd = 3.0, 80.0
    # Linear lower envelopes of ratio*theta + omega_upper and its lower counterpart.
    upper_margin = min(slope_min*theta_r-(ratio-slope_min)*radius,
                       ratio*theta_r,
                       slope_max*theta_r+(ratio-slope_max)*radius)
    lower_margin = min(-slope_max*theta_l-(slope_max-ratio)*radius,
                       -ratio*theta_l,
                       -slope_min*theta_l-(ratio-slope_min)*radius)
    upper_load, lower_load = q*(a+math.sin(radius)), q*(b+math.sin(radius))
    assert slope_min < ratio < slope_max
    assert upper_margin > 0 and lower_margin > 0
    assert kd*upper_margin > upper_load and kd*lower_margin > lower_load
    print(f'Angle interval: [{theta_min:.10f}, {theta_max:.10f}] rad')
    print(f'kp={ratio*kd:g} s^-2; kd={kd:g} s^-1')
    print(f'Boundary velocity margins: {upper_margin:.10f}, {lower_margin:.10f} rad/s')
    print(f'Sufficient kd threshold: {max(upper_load/upper_margin, lower_load/lower_margin):.10f} s^-1')
    print(f'Acceleration slacks: {kd*upper_margin-upper_load:.10f}, {kd*lower_margin-lower_load:.10f} rad/s^2')


if __name__ == '__main__':
    main()
