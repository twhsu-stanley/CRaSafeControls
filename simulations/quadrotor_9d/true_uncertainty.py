"""Physical uncertainty for the nine-state quadrotor simulation."""

import numpy as np


def noise_fcn(t, phase=0.4):
    """Return smooth acceleration and Euler-rate disturbances."""
    return np.array([
        0.0,
        0.0,
        0.0,
        0.008 * np.sin(0.71 * t + phase),
        0.008 * np.cos(0.53 * t + 0.3 + phase),
        0.006 * np.sin(0.37 * t + 0.2 + phase),
        0.003 * np.sin(0.43 * t + phase),
        0.003 * np.cos(0.39 * t + phase),
        0.0,
    ])


def true_uncertainty_fcn(x, t, schedule, payload=0.11, phase=0.4, residual_scale=1.0):
    """Return w(t,x): wind drag, a hover payload approximation and smooth residuals.

    The state velocities and scheduled wind are inertial. Drag coefficients
    include division by nominal mass; the final three entries are Euler rates.
    """
    x = np.asarray(x, dtype=float).reshape(9)
    wind = np.asarray(schedule, dtype=float).reshape(3)
    noise = noise_fcn(t, phase)
    if not np.all(np.isfinite(wind)) or not np.all(np.isfinite(noise)):
        raise ValueError("uncertainty schedules and noise must be finite")
    if residual_scale < 0:
        raise ValueError("residual_scale must be nonnegative")

    roll, pitch, yaw = x[6:9]
    cp, sp = np.cos(roll), np.sin(roll)
    ct, st = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    R = np.array([
        [cy * ct, cy * st * sp - sy * cp, cy * st * cp + sy * sp],
        [sy * ct, sy * st * sp + cy * cp, sy * st * cp - cy * sp],
        [-st, ct * sp, ct * cp],
    ])
    v_rel = R.T @ (x[3:6] - wind)
    beta = np.array([0.09, 0.23, 0.07])
    drag_body = -beta * v_rel * np.abs(v_rel)

    uncertainty = np.zeros(9)
    uncertainty[3:6] = R @ drag_body - 9.81 * payload * R[:, 2] + residual_scale * noise[3:6]
    uncertainty[6] = 0.04 * wind[0] + residual_scale * (noise[6] + 0.002 * np.sin(pitch))
    uncertainty[7] = -0.04 * wind[1] + residual_scale * (noise[7] + 0.002 * np.sin(roll))
    return uncertainty
