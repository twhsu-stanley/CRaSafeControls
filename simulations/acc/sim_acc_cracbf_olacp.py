"""Algorithm 1 and CRaCBF simulation for adaptive cruise control
"""

from itertools import product
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import lsq_linear

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from olacp import OLACP
from systems.acc.acc import ACC


USE_CP = True
USE_ADAPTIVE = True

# Algorithm 1 defaults.  Twelve intervals give three complete representation
# blocks with the default period.
K = 12
B = 4
DT = 0.02
INTERVAL_DURATION = 2.0
N_CAL = 200

THETA_TRUE = 0.045
THETA_INIT = np.array([0.08])
THETA_LB = np.array([0.005])
THETA_UB = np.array([0.12])

# Prior physical ranges used to derive the seven-dimensional box A.  These
# ranges are broader than the deterministic environment schedule below.
PHYSICAL_PARAMETER_BOUNDS = {
    "b0": (160.0, 280.0),
    "b1": (3.5, 8.5),
    "b2": (0.25, 0.50),
    "b3": (0.15, 0.55),
    "lead_velocity": (21.5, 24.5),
}


def environment_coefficients(interval_index):
    """Return the piecewise-constant environment on interval ``I_k``.

    ``b4`` is deliberately fixed: it is the shared representation parameter
    learned by Algorithm 1.  The remaining drag coefficients and lead speed
    vary from interval to interval and form the interval-specific latent
    parameter.
    """

    phase = 2.0 * np.pi * int(interval_index) / 13.0
    return {
        "b0": 220.0 + 35.0 * np.sin(phase + 0.15),
        "b1": 6.0 + 1.1 * np.cos(0.8 * phase + 0.35),
        "b2": 0.375 + 0.065 * np.sin(1.3 * phase + 0.7),
        "b3": 0.35 + 0.09 * np.cos(1.1 * phase - 0.25),
        "b4": THETA_TRUE,
        "lead_velocity": 23.0 + 1.15 * np.sin(0.9 * phase + 0.55),
    }


def latent_parameter(environment, mass, wind_speed):
    r"""Map the physical environment into the paper's latent vector ``a``.

    Expanding ``-d_v/m`` in the feature order
    ``[1, v, v^2, exp(-theta*z), v exp(-theta*z),
    v^2 exp(-theta*z)]`` gives the first six entries.  The final entry is the
    lead-vehicle velocity because the third row of ``Y_theta`` is
    ``[0, ..., 0, 1]``.
    """

    b0 = float(environment["b0"])
    b1 = float(environment["b1"])
    b2 = float(environment["b2"])
    b3 = float(environment["b3"])
    lead_velocity = float(environment["lead_velocity"])
    mass = float(mass)
    wind_speed = float(wind_speed)

    return np.array(
        [
            -(b0 + b2 * wind_speed**2) / mass,
            (-b1 + 2.0 * b2 * wind_speed) / mass,
            -b2 / mass,
            b2 * b3 * wind_speed**2 / mass,
            -2.0 * b2 * b3 * wind_speed / mass,
            b2 * b3 / mass,
            lead_velocity,
        ]
    )


def latent_parameter_bounds(mass, wind_speed):
    """Derive coordinate-wise bounds on ``a`` from the physical prior."""

    keys = ("b0", "b1", "b2", "b3", "lead_velocity")
    corner_values = []
    for values in product(*(PHYSICAL_PARAMETER_BOUNDS[key] for key in keys)):
        environment = dict(zip(keys, values))
        corner_values.append(latent_parameter(environment, mass, wind_speed))
    corners = np.asarray(corner_values)
    return np.min(corners, axis=0), np.max(corners, axis=0)


def drag_force(x, environment, wind_speed):
    """Evaluate the true resistance force ``d_v(t, x)`` from Section V-B."""

    v = float(np.asarray(x, dtype=float).reshape(3)[1])
    z = float(np.asarray(x, dtype=float).reshape(3)[2])
    relative_air_speed = v - float(wind_speed)
    return (
        environment["b0"]
        + environment["b1"] * v
        + environment["b2"]
        * relative_air_speed**2
        * (1.0 - environment["b3"] * np.exp(-environment["b4"] * z))
    )


def make_true_uncertainty(interval_duration, mass, wind_speed):
    """Create the physical ``w(x, t)`` independently of learned ``theta``."""

    def true_uncertainty(x, t):
        interval_index = int(np.floor(max(float(t), 0.0) / interval_duration))
        environment = environment_coefficients(interval_index)
        return np.array(
            [
                0.0,
                -drag_force(x, environment, wind_speed) / mass,
                environment["lead_velocity"],
            ]
        )

    return true_uncertainty


def _initial_calibration_scores(
    system,
    true_uncertainty,
    theta_init,
    a_lb,
    a_ub,
    *,
    n_cal,
    interval_length,
    interval_duration,
    rng,
):
    """Build the historical calibration set using Algorithm 1 lines 8--11."""

    scores = np.empty(n_cal)
    for calibration_index in range(n_cal):
        # Historical ACC states span the intended operating envelope.  The
        # environmental schedule is deterministic, while sampling is seeded.
        positions = rng.uniform(0.0, 500.0, interval_length)
        velocities = rng.uniform(19.0, 28.0, interval_length)
        distances = rng.uniform(12.0, 65.0, interval_length)
        states = np.column_stack((positions, velocities, distances))
        sample_times = (
            calibration_index * interval_duration
            + (np.arange(interval_length) + 0.5)
            * interval_duration
            / interval_length
        )

        Y_samples = [system.Y_theta(x, theta_init) for x in states]
        w_samples = [
            true_uncertainty(x, t) for x, t in zip(states, sample_times)
        ]
        fit = lsq_linear(
            np.vstack(Y_samples),
            np.hstack(w_samples),
            bounds=(a_lb, a_ub),
        )
        scores[calibration_index] = max(
            np.linalg.norm(Yx @ fit.x - w, ord=2)
            for Yx, w in zip(Y_samples, w_samples)
        )
    return scores


def _plot_results(results):
    """Create the diagnostic plots for the Section V-B experiment."""

    import matplotlib.pyplot as plt

    tt = results["time"]
    x_hist = results["x"]
    params = results["params"]
    figures = []

    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(8, 9))
    figures.append(fig)
    axs[0].plot(tt, x_hist[:, 1], label="ego velocity")
    axs[0].plot(tt, results["lead_velocity"], "--", label="lead velocity")
    axs[0].axhline(params["vd"], color="k", linestyle=":", label="desired")
    axs[0].set_ylabel("v (m/s)")
    axs[0].legend(ncol=3)
    axs[1].plot(tt, x_hist[:, 2])
    axs[1].axhline(params["z_min"], color="r", linestyle="--")
    axs[1].set_ylabel("z (m)")
    axs[2].plot(tt, results["u"])
    axs[2].axhline(params["u_max"], color="k", linestyle=":")
    axs[2].axhline(params["u_min"], color="k", linestyle=":")
    axs[2].set_ylabel("u (N)")
    axs[3].plot(tt, results["h"], label="h(x, a_hat)")
    axs[3].plot(tt, results["physical_safety_margin"], label="z - z_min")
    axs[3].plot(
        tt,
        results["tightened_cbf_margin"],
        ":",
        label="tightened h margin",
    )
    axs[3].plot(
        tt,
        results["augmented_barrier"],
        "--",
        label="fitted-parameter barrier",
    )
    axs[3].axhline(0.0, color="r", linestyle="--")
    axs[3].set_ylabel("safety margin")
    axs[3].set_xlabel("Time (s)")
    axs[3].legend()
    for ax in axs:
        ax.grid(True)
    fig.suptitle("Adaptive cruise control with a CRaCBF-QP")

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
    figures.append(fig)
    axs[0].step(results["interval_time"], results["score"], where="post", label="s_k")
    axs[0].step(tt, results["Q"], where="post", label="Q_k")
    axs[0].set_ylabel("score")
    axs[0].legend()
    axs[1].step(results["interval_time"], results["delta"], where="post")
    axs[1].set_ylabel("delta_k")
    axs[2].step(results["interval_time"], results["miscoverage"], where="post")
    axs[2].set_ylabel("e_k")
    axs[2].set_xlabel("Time (s)")
    for ax in axs:
        ax.grid(True)
    fig.suptitle("Adaptive conformal prediction")

    fig, axs = plt.subplots(7, 1, sharex=True, figsize=(8, 12))
    figures.append(fig)
    for index, ax in enumerate(axs):
        ax.plot(tt, results["a_hat"][:, index], label="a_hat")
        ax.plot(tt, results["a_interval"][:, index], "--", label="a_k")
        ax.plot(tt, results["a_true"][:, index], ":", label="physical a")
        ax.set_ylabel(f"a{index + 1}")
        ax.grid(True)
    axs[0].legend(ncol=3)
    axs[-1].set_xlabel("Time (s)")
    fig.suptitle("Adaptive, fitted, and physical latent parameters")

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
    figures.append(fig)
    axs[0].plot(tt, results["theta"], label="theta_hat")
    axs[0].axhline(results["theta_true"], color="k", linestyle="--", label="theta true")
    axs[0].set_ylabel("theta")
    axs[0].legend()
    axs[1].plot(tt, results["nu"])
    axs[1].set_ylabel("nu(rho)")
    axs[2].plot(tt, results["rho"])
    axs[2].set_ylabel("rho")
    axs[2].set_xlabel("Time (s)")
    for ax in axs:
        ax.grid(True)
    fig.suptitle("Learned representation and CRaCBF scaling")

    for fig in figures:
        fig.tight_layout()
    plt.show()
    return figures


def run_simulation(
    *,
    plot=True,
    k_intervals=K,
    representation_period=B,
    dt=DT,
    interval_duration=INTERVAL_DURATION,
    n_cal=N_CAL,
    seed=7,
    use_cp=USE_CP,
    use_adaptive=USE_ADAPTIVE,
    verbose=True,
):
    """Run the ACC Algorithm 1/CRaCBF example and return all histories.

    Setting ``plot=False`` avoids importing pyplot and is intended for tests
    and batch runs.  The horizon and sample period are keyword arguments so a
    short, deterministic smoke test does not need to modify module globals.
    """

    if int(k_intervals) != k_intervals or k_intervals < 1:
        raise ValueError("k_intervals must be a positive integer")
    if int(representation_period) != representation_period or representation_period < 1:
        raise ValueError("representation_period must be a positive integer")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if not np.isfinite(interval_duration) or interval_duration <= 0.0:
        raise ValueError("interval_duration must be finite and positive")
    if int(n_cal) != n_cal or n_cal < 100:
        raise ValueError("n_cal must be an integer of at least 100")

    k_intervals = int(k_intervals)
    representation_period = int(representation_period)
    n_cal = int(n_cal)
    interval_length = int(round(interval_duration / dt))
    if interval_length < 10 or not np.isclose(interval_length * dt, interval_duration):
        raise ValueError(
            "interval_duration must be an integer multiple of dt and contain "
            "at least 10 samples"
        )
    sample_count = k_intervals * interval_length
    tt = np.arange(sample_count, dtype=float) * dt

    mass = 1650.0
    wind_speed = 5.0
    gravity = 9.81
    acceleration_fraction = 0.3
    deceleration_fraction = 0.3
    a_lb, a_ub = latent_parameter_bounds(mass, wind_speed)
    a_center = 0.5 * (a_lb + a_ub)
    box_radius = 0.5 * np.linalg.norm(a_ub - a_lb, ord=2)
    # The estimator set A_hat need not equal the broader environmental set A.
    # A small projection ball is important here because the raw v^2 features
    # multiply coefficients whose physical scale is much smaller than a_7.
    a_hat_norm_max = 5e-3
    true_uncertainty = make_true_uncertainty(interval_duration, mass, wind_speed)

    # A uniform positive gain keeps the worst-case Gamma^{-1} tightening from
    # Proposition 1 compatible with the chosen initial following distance.
    # The resulting adaptive dynamics are stiff, so integration below uses
    # BDF while the QP input remains zero-order held.
    Gamma_cbf = 0.1 * np.eye(7)
    params = {
        "xdim": 3,
        "udim": 1,
        "adim": 7,
        "Theta_init": THETA_INIT.copy(),
        "true_uncertainty": true_uncertainty,
        "m": mass,
        "grav": gravity,
        "vd": 30.0,
        "Kp": 800.0,
        "z_min": 10.0,
        "T_h": 1.8,
        "cbf_smoothing_epsilon": 0.1,
        "use_adaptive": bool(use_adaptive),
        "use_cp": bool(use_cp),
        "Gamma_cbf": Gamma_cbf,
        "a_true": latent_parameter(environment_coefficients(0), mass, wind_speed),
        "a_ub": a_ub,
        "a_lb": a_lb,
        "a_hat_norm_max": a_hat_norm_max,
        "epsilon": 0.1 * a_hat_norm_max,
        "eta_cbf": 5.0,
        "cbf": {"rate": 0.5},
        "u_max": acceleration_fraction * mass * gravity,
        "u_min": -deceleration_fraction * mass * gravity,
        "dt": dt,
    }
    system = ACC(params)

    rng = np.random.default_rng(seed)
    calibration_scores = _initial_calibration_scores(
        system,
        true_uncertainty,
        THETA_INIT,
        a_lb,
        a_ub,
        n_cal=n_cal,
        interval_length=interval_length,
        interval_duration=interval_duration,
        rng=rng,
    )

    olacp = OLACP(
        calibration_scores,
        N_cal=n_cal,
        acp_lr=0.02,
        delta_target=0.1,
        delta_init=0.1,
        buffer_maxlen=interval_length,
        theta_init=THETA_INIT,
        representation_period=representation_period,
        # The latent drag coefficients contain the physical 1/m scaling, so
        # the scalar representation gradient is correspondingly small.
        representation_lr=lambda update_index: 1e5 / update_index,
        theta_lb=THETA_LB,
        theta_ub=THETA_UB,
        Y_theta=system.Y_theta,
        representation_loss_gradient=system.representation_loss_gradient,
    )
    system.set_representation(olacp.Theta)
    system.cp_quantile = olacp.Q_k

    # A moderately close initial headway makes the wake-effect representation
    # observable while remaining comfortably inside the certificate set.
    x = np.array([0.0, 24.0, 24.0])
    a_hat = a_center.copy()
    rho = 0.0
    x_ext = np.hstack((x, a_hat, rho))

    x_hist = np.zeros((sample_count, system.xdim))
    u_hist = np.zeros(sample_count)
    h_hist = np.zeros(sample_count)
    physical_safety_hist = np.zeros(sample_count)
    augmented_barrier_hist = np.zeros(sample_count)
    augmented_barrier_hist.fill(np.nan)
    true_parameter_barrier_hist = np.zeros(sample_count)
    tightened_cbf_margin_hist = np.zeros(sample_count)
    a_hat_hist = np.zeros((sample_count, system.adim))
    a_interval_hist = np.full((sample_count, system.adim), np.nan)
    a_true_hist = np.zeros((sample_count, system.adim))
    lead_velocity_hist = np.zeros(sample_count)
    rho_hist = np.zeros(sample_count)
    nu_hist = np.zeros(sample_count)
    Q_hist = np.zeros(sample_count)
    theta_hist = np.zeros(sample_count)

    interval_times = []
    score_hist = []
    delta_hist = []
    miscoverage_hist = []
    quantile_next_hist = []
    representation_update_hist = []
    Gamma_cbf_inv = np.linalg.inv(Gamma_cbf)

    for sample_index, t in enumerate(tt):
        interval_index = sample_index // interval_length
        environment = environment_coefficients(interval_index)
        a_true = latent_parameter(environment, mass, wind_speed)

        x_hist[sample_index] = x
        a_hat_hist[sample_index] = a_hat
        a_true_hist[sample_index] = a_true
        lead_velocity_hist[sample_index] = environment["lead_velocity"]
        rho_hist[sample_index] = rho
        nu_hist[sample_index] = system.nu_cbf(rho)
        Q_hist[sample_index] = system.cp_quantile
        theta_hist[sample_index] = system.Theta_hat.item()
        h_hist[sample_index] = float(np.asarray(system.cbf(x, a_hat)).item())
        physical_safety_hist[sample_index] = x[2] - params["z_min"]
        parameter_error = a_hat - a_true
        true_parameter_barrier_hist[sample_index] = (
            nu_hist[sample_index] * h_hist[sample_index]
            - 0.5 * parameter_error @ Gamma_cbf_inv @ parameter_error
        )
        tightened_cbf_margin_hist[sample_index] = (
            h_hist[sample_index]
            - 0.5
            / nu_hist[sample_index]
            * system.safe_set_tightening
        )
        if sample_index == 0 and h_hist[sample_index] < 0.0:
            raise ValueError(f"Initial condition is outside the CBF set: h={h_hist[0]:.3f}")
        if (
            sample_index % interval_length == 0
            and use_adaptive
            and tightened_cbf_margin_hist[sample_index] < 0.0
        ):
            raise ValueError(
                "Interval-start state violates equation (37): "
                f"interval={interval_index + 1}, "
                f"margin={tightened_cbf_margin_hist[sample_index]:.3f}"
            )

        u_ref = system.ctrl_nominal(x)
        try:
            u = system.ctrl_cracbf(x, a_hat, u_ref, rho)
        except ValueError as error:
            raise RuntimeError(
                "CRaCBF-QP failed at "
                f"t={t:.3f}, h={h_hist[sample_index]:.6g}, "
                f"rho={rho:.6g}, x={x.tolist()}, a_hat={a_hat.tolist()}"
            ) from error
        u_hist[sample_index] = float(np.asarray(u).item())

        # Line 6 of Algorithm 1: use exact simulated derivatives in place of
        # a finite-difference state-derivative observer.
        olacp.add_data_to_buffers(
            x,
            system.dynamics_nominal(x, u),
            xdot=system.dynamics(x, u, t),
        )

        # Zero-order hold on u.  The last sample of the final interval needs no
        # propagation, but it is still included in that interval's score.
        if sample_index < sample_count - 1:
            solution = solve_ivp(
                lambda tau, state: system.dynamics_extended(state, u, tau),
                (t, t + dt),
                x_ext,
                method="BDF",
                rtol=1e-7,
                atol=1e-9,
                t_eval=[t + dt],
            )
            if not solution.success:
                raise RuntimeError(solution.message)
            x_ext = solution.y[:, -1]
            if not np.all(np.isfinite(x_ext)):
                raise RuntimeError("The extended ACC state became non-finite")
            x = x_ext[: system.xdim]
            a_hat = x_ext[system.xdim : system.xdim + system.adim]
            rho = float(x_ext[system.xdim + system.adim])

        # Lines 7--23 of Algorithm 1, executed after the final sample in I_k.
        if (sample_index + 1) % interval_length == 0:
            olacp.estimate_uncertainty(dt)
            score = float(olacp.compute_score(system.a_ub, system.a_lb))
            miscoverage = int(olacp.update_delta(score))
            olacp.append_score(score)
            representation_update = olacp.update_representation()

            interval_start = sample_index - interval_length + 1
            a_interval_hist[interval_start : sample_index + 1] = olacp.a_k
            interval_parameter_error = (
                a_hat_hist[interval_start : sample_index + 1] - olacp.a_k
            )
            augmented_barrier_hist[interval_start : sample_index + 1] = (
                nu_hist[interval_start : sample_index + 1]
                * h_hist[interval_start : sample_index + 1]
                - 0.5
                * np.einsum(
                    "bi,ij,bj->b",
                    interval_parameter_error,
                    Gamma_cbf_inv,
                    interval_parameter_error,
                )
            )
            interval_times.append((interval_index + 1) * interval_duration)
            score_hist.append(score)
            delta_hist.append(olacp.delta)
            miscoverage_hist.append(miscoverage)

            if representation_update is not None:
                system.set_representation(representation_update["Theta"])
                representation_update_hist.append(
                    {
                        "interval": interval_index + 1,
                        **representation_update,
                    }
                )

            system.cp_quantile = olacp.compute_quantile()
            quantile_next_hist.append(system.cp_quantile)
            olacp.clear_buffers()

            if verbose:
                print(
                    f"interval={interval_index + 1:02d}, "
                    f"score={score:.5f}, Q_used={Q_hist[sample_index]:.5f}, "
                    f"Q_next={system.cp_quantile:.5f}, "
                    f"delta_next={olacp.delta:.3f}, "
                    f"miscoverage={miscoverage}, "
                    f"theta={system.Theta_hat.item():.5f}"
                )

    results = {
        "time": tt,
        "x": x_hist,
        "u": u_hist,
        "h": h_hist,
        "physical_safety_margin": physical_safety_hist,
        "tightened_cbf_margin": tightened_cbf_margin_hist,
        "augmented_barrier": augmented_barrier_hist,
        "true_parameter_barrier": true_parameter_barrier_hist,
        "a_hat": a_hat_hist,
        "a_interval": a_interval_hist,
        "a_true": a_true_hist,
        "lead_velocity": lead_velocity_hist,
        "rho": rho_hist,
        "nu": nu_hist,
        "Q": Q_hist,
        "theta": theta_hist,
        "theta_true": THETA_TRUE,
        "interval_time": np.asarray(interval_times),
        "score": np.asarray(score_hist),
        "delta": np.asarray(delta_hist),
        "miscoverage": np.asarray(miscoverage_hist),
        "quantile_next": np.asarray(quantile_next_hist),
        "representation_updates": representation_update_hist,
        "calibration_scores": calibration_scores,
        "params": params,
        "system": system,
        "olacp": olacp,
    }
    if plot:
        results["figures"] = _plot_results(results)
    return results


if __name__ == "__main__":
    run_simulation(plot=True)
