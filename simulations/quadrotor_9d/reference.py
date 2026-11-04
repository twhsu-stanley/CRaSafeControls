"""Prepare position targets and the MPC that generates reference inputs.

run_example.py creates WaypointReference; simulation.py uses its guide and
make_planner(). The guide supplies desired positions to the MPC cost. The full
reference state x_d is integrated separately by simulation.py using estimated
dynamics (certainty-equivalent, or CE, dynamics).
"""

import json

import numpy as np
from scipy.interpolate import CubicSpline

from motion_planner import MotionPlanner


# Keep the original horizontal scale and 12-second lap when extending the mission.
SPATIAL_PERIOD = 12.  # Seconds per horizontal figure eight.
SPATIAL_AMPLITUDE = np.array([.45, .30])  # Horizontal half-widths [m].
SPATIAL_CLIMB_RATE = .03  # Vertical speed [m/s]; rise 0.36 m per lap.


def ascending_figure_eight(times):
    """Return relative position targets (N, 3) at a fixed lap and climb rate.

    Mission duration never rescales phase. This is a position cost guide, not
    the dynamically integrated reference x_d or a prescribed full-state path.
    """
    times = np.atleast_1d(np.asarray(times, dtype=float))
    phase = 2*np.pi*times/SPATIAL_PERIOD
    return np.column_stack((SPATIAL_AMPLITUDE[0]*np.sin(phase),
                            SPATIAL_AMPLITUDE[1]*np.sin(2*phase),
                            SPATIAL_CLIMB_RATE*times))


def validate_timing(config):
    """Align input holds, MPC replans and blockwise representation updates."""
    hold, replan = config.mpc_input_steps, config.mpc_replan_steps
    if (hold < 1 or replan < 1 or replan % hold
            or config.mpc_horizon_steps*hold < replan):
        raise ValueError("MPC replan steps must contain whole input holds and fit the horizon")
    if config.representation_period*config.interval_steps % replan:
        raise ValueError("Representation updates must align with the MPC replanning grid")


class WaypointReference:
    """Build a position-only cost guide alongside a separate full initial x_d."""

    def __init__(self, config):
        """Prepare a fixed-period spatial guide or a spline through user targets."""
        validate_timing(config)
        self.interval_duration = config.interval_steps*config.dt
        intervals = round(config.sim_time/self.interval_duration)
        if intervals < 1 or not np.isclose(intervals*self.interval_duration, config.sim_time):
            raise ValueError("Mission duration must contain complete waypoint/learning intervals")
        self.times = self.interval_duration*np.arange(intervals+1)
        # Initial x_d contains all nine states; the waypoint table contains only xyz.
        self.initial_state = np.asarray(config.initial_reference, dtype=float).reshape(9).copy()
        if config.waypoints is not None:
            self.positions = np.asarray(json.loads(config.waypoints.read_text()), dtype=float)
        elif config.trajectory == "point":
            self.positions = np.tile(np.asarray(config.goal, dtype=float), (intervals+1, 1))
            self.positions[0] = self.initial_state[:3]
        else:
            # Sample a time-based guide; longer missions add laps instead of slowing down.
            self.positions = ascending_figure_eight(self.times)+self.initial_state[:3]
        if (self.positions.shape != (intervals+1, 3) or not np.isfinite(self.positions).all()
                or not np.isfinite(self.initial_state).all()
                or not np.allclose(self.positions[0], self.initial_state[:3])):
            raise ValueError("Waypoints must be (K+1,3), finite, and start at initial-reference position")
        # Keep the default guide smooth across laps regardless of waypoint spacing.
        # Custom waypoints and point targets retain their natural cubic spline.
        self.spline = (None if config.waypoints is None and config.trajectory == "spatial"
                       else CubicSpline(self.times, self.positions, axis=0, bc_type="natural"))
        # Convert the simulation step counts into seconds for MPC configuration.
        self.mpc_dt = config.mpc_input_steps*config.dt
        self.horizon = config.mpc_horizon_steps*self.mpc_dt
        self.replan_interval = config.mpc_replan_steps*config.dt

    def guide(self, times):
        """Return position cost targets as a 3xN array, holding the final target."""
        times = np.clip(np.atleast_1d(times), self.times[0], self.times[-1])
        if self.spline is None:
            return (ascending_figure_eight(times)+self.initial_state[:3]).T
        return self.spline(times).T

    def metadata(self):
        """Return the waypoint and timing settings saved in the experiment report."""
        metadata = dict(
            positions=self.positions, times=self.times, initial_state=self.initial_state,
            guide="ascending_figure_eight" if self.spline is None else "natural_cubic_position_spline",
            interval_duration=self.interval_duration, mpc_dt=self.mpc_dt, horizon=self.horizon,
            replan_interval=self.replan_interval,
            inputs_per_replan=round(self.replan_interval/self.mpc_dt),
        )
        if self.spline is None:
            metadata.update(lap_duration=SPATIAL_PERIOD, horizontal_amplitude=SPATIAL_AMPLITUDE.copy(),
                            climb_rate=SPATIAL_CLIMB_RATE)
        return metadata


def make_planner(system, config):
    """Build MPC with position-error and hover-input costs.

    Only the first three states enter the tracking cost: C_p = [I_3, 0].
    Input bounds constrain the reference input u_d produced by MPC.
    """
    return MotionPlanner(
        system, config.dt, 8*np.eye(3), .03*np.eye(4), Q_terminal=60*np.eye(3),
        output_matrix=np.eye(3, 9), S=np.zeros((4, 4)),
        u_min=config.u_min, u_max=config.u_max,
    )
