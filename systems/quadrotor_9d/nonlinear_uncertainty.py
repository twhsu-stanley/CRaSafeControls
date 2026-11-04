"""Independent physical disturbance for representation-learning experiments.

The plant never evaluates Psi, Theta or an estimated a. Velocities are inertial;
quadratic drag is evaluated in body coordinates relative to an ambient wind.

x = [p_x, p_y, p_z, v_x, v_y, v_z, roll, pitch, yaw]
u = [thrust_acceleration, omega_x, omega_y, omega_z]
"""

from dataclasses import dataclass, replace
import numpy as np


@dataclass(frozen=True)
class WindEnvironment:
    """Physical scenario: inertial wind [m/s], payload coefficient and gust phase."""

    wind_x: float = .08
    wind_y: float = -.06
    wind_z: float = .02
    payload: float = .11
    phase: float = .4
    residual_scale: float = 1.0


# One prescribed online cycle, independent of the Theta update period B.
ONLINE_WIND_PATTERN = np.array([
    [.08, -.06, .02], [.12, -.03, .04], [.05, -.10, -.02],
    [.10, -.08, .03], [.06, -.04, .00],
])
ONLINE_WIND_PATTERN.setflags(write=False)


def pretraining_wind_pattern(period):
    """Return B interval winds spanning one periodic excitation cycle [m/s]."""
    if period < 1 or int(period) != period:
        raise ValueError("Wind cycle length must be a positive integer")
    phase = 2*np.pi*np.arange(period)/period
    return np.column_stack((.15*np.sin(phase), .15*np.cos(phase),
                            .05*np.sin(phase+np.pi/4)))


@dataclass(frozen=True)
class WindSchedule:
    """Repeat prescribed wind vectors, holding each for one observation interval."""

    values: np.ndarray
    interval_duration: float
    environment: WindEnvironment = WindEnvironment()

    def __post_init__(self):
        values = np.array(self.values, dtype=float, copy=True)
        if (values.ndim != 2 or values.shape[1] != 3 or not len(values)
                or not np.isfinite(values).all()
                or not np.isfinite(self.interval_duration) or self.interval_duration <= 0):
            raise ValueError("Wind schedule needs finite Nx3 values and a positive interval duration")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)

    def wind(self, t):
        """Return the active inertial wind; switch at interval boundaries."""
        if not np.isfinite(t):
            raise ValueError("Wind evaluation time must be finite")
        index = int(np.floor(max(float(t), 0.)/self.interval_duration)) % len(self.values)
        return self.values[index].copy()

    def __call__(self, x, t):
        wind = self.wind(t)
        environment = replace(self.environment, wind_x=wind[0], wind_y=wind[1], wind_z=wind[2])
        return nonlinear_uncertainty(x, t, environment)

    def metadata(self):
        """Record the exact pattern and timing needed to reproduce the wind."""
        return dict(kind="periodic_piecewise_constant", values=self.values.tolist(),
                    interval_duration=self.interval_duration, cycle_intervals=len(self.values),
                    cycle_duration=len(self.values)*self.interval_duration)


def rotation_matrix(x):
    """Return the body-to-inertial rotation using ZYX Euler angles."""
    roll, pitch, yaw = np.asarray(x, dtype=float)[6:9]
    cp, sp = np.cos(roll), np.sin(roll)
    ct, st = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([[cy*ct, cy*st*sp-sy*cp, cy*st*cp+sy*sp],
                     [sy*ct, sy*st*sp+cy*cp, sy*st*cp-cy*sp],
                     [-st, ct*sp, ct*cp]])


def nonlinear_uncertainty(x, t, environment=WindEnvironment()):
    """Return a 9-vector of acceleration/rate disturbances, never a sensor error.

    beta contains drag coefficients divided by nominal mass, in 1/m.
    Payload retains the existing additive hover approximation. A small smooth
    gust and state/time-varying rate biases provide unmodelled residuals.
    """
    x = np.asarray(x, dtype=float).reshape(9)
    if environment.residual_scale < 0:
        raise ValueError("residual_scale must be nonnegative")
    R = rotation_matrix(x)
    wind = np.array([environment.wind_x, environment.wind_y, environment.wind_z])
    # Evaluate anisotropic quadratic drag on body-frame air-relative velocity.
    relative_velocity = R.T @ (x[3:6]-wind)
    beta = np.array([.09, .23, .07])
    drag_body = -beta*relative_velocity*np.abs(relative_velocity)
    phase = environment.phase
    gust = environment.residual_scale * np.array([
        .008*np.sin(.71*t+phase), .008*np.cos(.53*t+.3+phase),
        .006*np.sin(.37*t+.2+phase),
    ])
    result = np.zeros(9)
    # Transform drag back to inertial coordinates and add the hover payload term.
    result[3:6] = R @ drag_body - 9.81*environment.payload*R[:,2] + gust
    result[6] = (.04*environment.wind_x + environment.residual_scale *
                 (.003*np.sin(.43*t+phase) + .002*np.sin(x[7])))
    result[7] = (-.04*environment.wind_y + environment.residual_scale *
                 (.003*np.cos(.39*t+phase) + .002*np.sin(x[6])))
    return result
