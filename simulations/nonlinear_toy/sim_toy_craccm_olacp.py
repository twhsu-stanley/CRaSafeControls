"""Algorithm 1 and CRaCCM for the nonlinear toy system."""

import copy
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from geodesic_solver import GeodesicSolver
from motion_planner import MotionPlanner
from olacp import OLACP
from systems.nonlinear_toy.nonlinear_toy import NONLINEAR_TOY


def pretraining_control(x, t, config):
    """Return a bounded stabilizing and persistently exciting nominal input."""
    x = np.asarray(x, dtype=float).reshape(3)
    feedback = -np.tanh(x[1]) - 2.0 * x[0] - 2.0 * x[2]
    excitation = config["pretrain_excitation_amplitude"] * np.sin(
        2.0 * np.pi * config["pretrain_excitation_frequency"] * t
    )
    return np.array([feedback + excitation])


def run_pretraining(system, olacp, config, plot=True):
    """Fill the calibration window and train the shared representation."""
    K_pre = config["K_pre"]
    I_length = config["I_length"]
    dt = config["dt"]
    t_pre = np.arange(K_pre * I_length, dtype=float) * dt

    system.set_representation(olacp.Theta)
    x = config["pretrain_initial_state"].copy()
    x_hist = np.zeros((len(t_pre), system.xdim))
    u_hist = np.zeros(len(t_pre))
    a_k_hist = np.full((len(t_pre), system.adim), np.nan)
    theta_hist = np.zeros((K_pre + 1,) + olacp.Theta.shape)
    theta_hist[0] = olacp.Theta
    score_hist = np.zeros(K_pre)
    true_uncertainty_hist = np.full((len(t_pre), system.xdim), np.nan)
    fitted_uncertainty_hist = np.full((len(t_pre), system.xdim), np.nan)

    for interval_index in range(K_pre):
        for interval_sample_index in range(I_length):
            sample_index = interval_index * I_length + interval_sample_index
            t = t_pre[sample_index]
            u = pretraining_control(x, t, config)
            u = np.clip(u, system.params["u_min"], system.params["u_max"])

            x_hist[sample_index] = x
            u_hist[sample_index] = float(u.item())
            xdot = system.dynamics(x, u, t)
            olacp.add_data_to_buffers(x, system.dynamics_nominal(x, u), xdot=xdot)

            if sample_index < len(t_pre) - 1:
                solution = solve_ivp(
                    lambda tau, state: system.dynamics(state, u, tau),
                    (t, t + dt),
                    x,
                    method="RK45",
                    rtol=1e-7,
                    atol=1e-9,
                    t_eval=[t + dt],
                )
                if not solution.success:
                    raise RuntimeError(solution.message)
                x = solution.y[:, -1]

        olacp.estimate_uncertainty(dt)
        score = float(olacp.compute_score(system.a_ub, system.a_lb))
        interval_true = np.asarray(olacp._w_buffer, dtype=float)
        interval_fitted = np.asarray([Y_t @ olacp.a_k for Y_t in olacp._Y_buffer])
        olacp.append_score(score)
        representation_update = olacp.update_representation()
        if representation_update is not None:
            system.set_representation(representation_update["Theta"])

        interval_start = interval_index * I_length
        interval_slice = slice(interval_start, interval_start + I_length)
        a_k_hist[interval_slice] = olacp.a_k
        true_uncertainty_hist[interval_slice] = interval_true
        fitted_uncertainty_hist[interval_slice] = interval_fitted
        score_hist[interval_index] = score
        theta_hist[interval_index + 1] = olacp.Theta
        olacp.clear_buffers()

    if len(olacp.S_cal) != config["N_cal"]:
        raise RuntimeError("Pretraining did not fill the calibration window")
    if olacp._representation_intervals:
        raise RuntimeError("Pretraining left an incomplete representation block")
    if not np.allclose(system.Theta_hat, olacp.Theta):
        raise RuntimeError("The trained representation was not installed in the toy system")

    quantile = olacp.compute_quantile()
    history = {
        "t": t_pre,
        "x_hist": x_hist,
        "u_hist": u_hist,
        "a_k_hist": a_k_hist,
        "theta_hist": theta_hist,
        "score_hist": score_hist,
        "true_uncertainty_hist": true_uncertainty_hist,
        "fitted_uncertainty_hist": fitted_uncertainty_hist,
        "quantile": quantile,
    }

    print(
        f"Pretraining complete: Q_0={quantile:.3e}, a_last={olacp.a_k}, "
        f"theta_change={np.linalg.norm(theta_hist[-1] - theta_hist[0]):.3e}, "
        f"max_state_norm={np.max(np.linalg.norm(x_hist, axis=1)):.3e}"
    )

    if plot:
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
        for state_index, ax in enumerate(axs):
            ax.plot(t_pre, x_hist[:, state_index])
            ax.set_ylabel(rf"$x_{state_index + 1}$")
            ax.grid(True)
        axs[-1].set_xlabel("time (s)")
        fig.suptitle("Nonlinear-toy pretraining states")

        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
        axs[0].plot(t_pre, u_hist)
        axs[0].set_ylabel("input")
        axs[1].plot(np.arange(K_pre), score_hist, marker="o")
        axs[1].set_ylabel("score")
        axs[1].set_xlabel("interval")
        for ax in axs:
            ax.grid(True)
        fig.suptitle("Nonlinear-toy pretraining diagnostics")

    return olacp, history


def plan_nominal_trajectory(system, config):
    """Plan a nominal trajectory satisfying the uncertainty-free dynamics."""
    horizon_steps = config["K"] * config["I_length"]
    planner = MotionPlanner(
        system=system,
        dt=config["dt"],
        Q=np.eye(system.xdim),
        R=0.1 * np.eye(system.udim),
        Q_f=10.0 * np.eye(system.xdim),
        u_min=np.array([system.params["u_min"]]),
        u_max=np.array([system.params["u_max"]]),
    )
    alpha = np.linspace(0.0, 1.0, horizon_steps + 1)
    x_guess = config["nominal_initial_state"][:, None]
    x_guess = x_guess + (
        config["nominal_goal_state"] - config["nominal_initial_state"]
    )[:, None] * alpha
    u_guess = np.zeros((system.udim, horizon_steps))
    x_d, u_d = planner.plan(
        config["nominal_initial_state"],
        config["nominal_goal_state"],
        horizon_steps,
        x_guess,
        u_guess,
    )
    t_x = config["dt"] * np.arange(horizon_steps + 1)
    t_u = t_x[:-1]
    return {
        "t_x": t_x,
        "t_u": t_u,
        "x_d": x_d,
        "u_d": u_d,
        "interp_x_d": interp1d(t_x, x_d, axis=1, bounds_error=False, fill_value="extrapolate"),
        "interp_u_d": interp1d(t_u, u_d, axis=1, bounds_error=False, fill_value="extrapolate"),
    }


def run_craccm_simulation(
    system,
    online_olacp,
    trajectory,
    config,
    use_cp,
    use_adaptive,
    label=None,
):
    """Run one CRaCCM experiment from an independent pretrained snapshot."""
    if bool(system.use_cp) != bool(use_cp) or bool(system.use_adaptive) != bool(use_adaptive):
        raise ValueError("The toy system must be constructed with the requested controller flags")

    online_olacp.clear_buffers()
    if online_olacp._representation_intervals:
        raise RuntimeError("The pretrained OLACP contains an incomplete representation block")
    online_olacp.Y_Theta = system.Y_Theta
    online_olacp.representation_loss_gradient = system.representation_loss_gradient
    system.set_representation(online_olacp.Theta)
    system.cp_quantile = online_olacp.compute_quantile()
    if np.shares_memory(system.Theta_hat, online_olacp.Theta):
        raise RuntimeError("The toy system and OLACP must own independent Theta arrays")

    I_length = config["I_length"]
    dt = config["dt"]
    t_full = np.arange(config["K"] * I_length, dtype=float) * dt
    run_label = label or f"CP={bool(use_cp)}, adaptive={bool(use_adaptive)}"

    x = trajectory["x_d"][:, 0] + config["tracking_initial_offset"]
    a_hat = system.a_center.copy()
    rho = 0.0
    x_ext = np.hstack((x, a_hat, rho))

    geodesic_solver = GeodesicSolver(
        system.xdim,
        config["geodesic_degree"],
        config["geodesic_nodes"],
        system.W_fcn,
        system.dW_dxi_fcn,
        system.dW_dai_fcn,
    )

    x_hist = np.zeros((len(t_full), system.xdim))
    x_d_hist = np.zeros_like(x_hist)
    u_hist = np.full(len(t_full), np.nan)
    u_d_hist = np.full(len(t_full), np.nan)
    energy_hist = np.full(len(t_full), np.nan)
    slack_hist = np.full(len(t_full), np.nan)
    a_hat_hist = np.zeros((len(t_full), system.adim))
    a_k_hist = np.full((len(t_full), system.adim), np.nan)
    rho_hist = np.zeros(len(t_full))
    nu_hist = np.zeros(len(t_full))
    quantile_hist = np.zeros(len(t_full))
    theta_hist = np.zeros((len(t_full),) + online_olacp.Theta.shape)
    true_uncertainty_hist = np.full((len(t_full), system.xdim), np.nan)
    fitted_uncertainty_hist = np.full((len(t_full), system.xdim), np.nan)
    interval_times = []
    score_hist = []
    delta_hist = []
    miscoverage_hist = []
    status = "completed"
    last_sample_index = -1

    for sample_index, t in enumerate(t_full):
        last_sample_index = sample_index
        x_d = np.asarray(trajectory["interp_x_d"](t), dtype=float).reshape(system.xdim)
        u_d = np.asarray(trajectory["interp_u_d"](t), dtype=float).reshape(system.udim)

        x_hist[sample_index] = x
        x_d_hist[sample_index] = x_d
        a_hat_hist[sample_index] = a_hat
        rho_hist[sample_index] = rho
        nu_hist[sample_index] = system.nu_ccm(rho)
        quantile_hist[sample_index] = system.cp_quantile
        theta_hist[sample_index] = system.Theta_hat

        u, slack = system.ctrl_craccm(
            x,
            a_hat,
            x_d,
            u_d,
            geodesic_solver,
            use_qpsolvers=config["use_qpsolvers"],
            use_slack=True,
            verify_geodesic=config["verify_geodesic"],
        )
        u_hist[sample_index] = float(u.item())
        u_d_hist[sample_index] = float(u_d.item())
        energy_hist[sample_index] = system.Erem
        slack_hist[sample_index] = float(slack)

        xdot = system.dynamics(x, u, t)
        online_olacp.add_data_to_buffers(x, system.dynamics_nominal(x, u), xdot=xdot)

        if sample_index < len(t_full) - 1:
            solution = solve_ivp(
                lambda tau, state: system.dynamics_extended(
                    state, x_d, u, geodesic_solver, tau
                ),
                (t_full[sample_index], t_full[sample_index + 1]),
                x_ext,
                method="RK45",
                rtol=1e-7,
                atol=1e-9,
                t_eval=[t_full[sample_index + 1]],
            )
            if not solution.success:
                raise RuntimeError(
                    f"{run_label}: extended dynamics failed at t={t:.3f}: {solution.message}"
                )
            x_ext = solution.y[:, -1]
            x = x_ext[: system.xdim]
            a_hat = x_ext[system.xdim : system.xdim + system.adim]
            rho = float(x_ext[-1])

        if not np.all(np.isfinite(x_ext)) or np.linalg.norm(x) > config["divergence_norm"]:
            status = "diverged"
            break

        if (sample_index + 1) % I_length == 0:
            online_olacp.estimate_uncertainty(dt)
            score = float(online_olacp.compute_score(system.a_ub, system.a_lb))
            interval_true = np.asarray(online_olacp._w_buffer, dtype=float)
            interval_fitted = np.asarray(
                [Y_t @ online_olacp.a_k for Y_t in online_olacp._Y_buffer]
            )
            miscoverage = online_olacp.update_delta(score)
            online_olacp.append_score(score)
            representation_update = online_olacp.update_representation()

            interval_start = sample_index - I_length + 1
            interval_slice = slice(interval_start, sample_index + 1)
            a_k_hist[interval_slice] = online_olacp.a_k
            true_uncertainty_hist[interval_slice] = interval_true
            fitted_uncertainty_hist[interval_slice] = interval_fitted
            interval_times.append(t)
            score_hist.append(score)
            delta_hist.append(online_olacp.delta)
            miscoverage_hist.append(miscoverage)

            if representation_update is not None:
                system.set_representation(representation_update["Theta"])
            system.cp_quantile = online_olacp.compute_quantile()
            online_olacp.clear_buffers()

    used_slice = slice(0, last_sample_index + 1)
    t = t_full[used_slice]
    x_hist = x_hist[used_slice]
    x_d_hist = x_d_hist[used_slice]
    u_hist = u_hist[used_slice]
    u_d_hist = u_d_hist[used_slice]
    energy_hist = energy_hist[used_slice]
    slack_hist = slack_hist[used_slice]
    a_hat_hist = a_hat_hist[used_slice]
    a_k_hist = a_k_hist[used_slice]
    rho_hist = rho_hist[used_slice]
    nu_hist = nu_hist[used_slice]
    quantile_hist = quantile_hist[used_slice]
    theta_hist = theta_hist[used_slice]
    true_uncertainty_hist = true_uncertainty_hist[used_slice]
    fitted_uncertainty_hist = fitted_uncertainty_hist[used_slice]

    tracking_error = np.linalg.norm(x_hist - x_d_hist, axis=1)
    tail_start = max(int(0.8 * len(tracking_error)), 0)
    metrics = {
        "final_error": float(tracking_error[-1]),
        "tail_rms": float(np.sqrt(np.mean(tracking_error[tail_start:] ** 2))),
        "max_error": float(np.max(tracking_error)),
        "max_control": float(np.nanmax(np.abs(u_hist))),
        "max_slack": float(np.nanmax(slack_hist)),
        "a_hat_change": float(np.linalg.norm(a_hat_hist[-1] - a_hat_hist[0])),
        "rho_range": float(np.ptp(rho_hist)),
    }
    result = {
        "label": run_label,
        "use_cp": bool(use_cp),
        "use_adaptive": bool(use_adaptive),
        "t": t,
        "x_hist": x_hist,
        "x_d_hist": x_d_hist,
        "tracking_error": tracking_error,
        "u_hist": u_hist,
        "u_d_hist": u_d_hist,
        "energy_hist": energy_hist,
        "slack_hist": slack_hist,
        "a_hat_hist": a_hat_hist,
        "a_k_hist": a_k_hist,
        "rho_hist": rho_hist,
        "nu_hist": nu_hist,
        "quantile_hist": quantile_hist,
        "theta_hist": theta_hist,
        "true_uncertainty_hist": true_uncertainty_hist,
        "fitted_uncertainty_hist": fitted_uncertainty_hist,
        "interval_times": np.asarray(interval_times),
        "score_hist": np.asarray(score_hist),
        "delta_hist": np.asarray(delta_hist),
        "miscoverage_hist": np.asarray(miscoverage_hist),
        "status": status,
        "metrics": metrics,
        "final_theta": online_olacp.Theta.copy(),
    }

    print(
        f"{run_label}: status={status}, final_error={metrics['final_error']:.3e}, "
        f"tail_rms={metrics['tail_rms']:.3e}, max_error={metrics['max_error']:.3e}, "
        f"max|u|={metrics['max_control']:.3e}, max_slack={metrics['max_slack']:.3e}"
    )
    return result


def plot_craccm_results(results, trajectory):
    """Plot the nominal plan and the three controller configurations."""
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(9, 8))
    for state_index, ax in enumerate(axs):
        ax.plot(
            trajectory["t_x"],
            trajectory["x_d"][state_index],
            "k--",
            label="nominal" if state_index == 0 else None,
        )
        for label, result in results.items():
            ax.plot(result["t"], result["x_hist"][:, state_index], label=label)
        ax.set_ylabel(rf"$x_{state_index + 1}$")
        ax.grid(True)
    axs[0].legend()
    axs[-1].set_xlabel("time (s)")
    fig.suptitle("CRaCCM state tracking")

    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(9, 9))
    for label, result in results.items():
        axs[0].semilogy(result["t"], np.maximum(result["tracking_error"], 1e-12), label=label)
        axs[1].plot(result["t"], result["u_hist"], label=label)
        axs[2].semilogy(result["t"], np.maximum(result["energy_hist"], 1e-12), label=label)
        axs[3].plot(result["t"], result["slack_hist"], label=label)
    axs[0].set_ylabel(r"$\|x-x_d\|_2$")
    axs[1].set_ylabel("input")
    axs[2].set_ylabel("energy")
    axs[3].set_ylabel("slack")
    axs[3].set_xlabel("time (s)")
    for ax in axs:
        ax.grid(True)
    axs[0].legend()
    fig.suptitle("CRaCCM tracking diagnostics")

    fig, axs = plt.subplots(NONLINEAR_TOY.adim + 2, 1, sharex=True, figsize=(9, 8))
    for label, result in results.items():
        for parameter_index in range(NONLINEAR_TOY.adim):
            axs[parameter_index].plot(
                result["t"], result["a_hat_hist"][:, parameter_index], label=label
            )
        axs[-2].plot(result["t"], result["rho_hist"], label=label)
        axs[-1].plot(result["t"], result["quantile_hist"], label=label)
    for parameter_index in range(NONLINEAR_TOY.adim):
        axs[parameter_index].set_ylabel(rf"$\hat a_{parameter_index + 1}$")
        axs[parameter_index].grid(True)
    axs[-2].set_ylabel(r"$\rho$")
    axs[-1].set_ylabel(r"$Q_k$")
    axs[-1].set_xlabel("time (s)")
    axs[-2].grid(True)
    axs[-1].grid(True)
    axs[0].legend()
    fig.suptitle("CRaCCM adaptation and conformal quantile")


def main():
    """Pretrain Algorithm 1, plan once, and compare three controller settings."""
    K_pre = 30
    N_cal = 30
    K = 3
    B = 5
    dt = 0.01
    interval_duration = 2.0
    I_length = int(round(interval_duration / dt))
    if K_pre < N_cal or K_pre % B != 0:
        raise ValueError("K_pre must fill N_cal and contain complete representation blocks")
    if I_length < 10 or not np.isclose(I_length * dt, interval_duration):
        raise ValueError("interval_duration must be an integer multiple of dt")

    theta_init = np.array([[0.8, -0.3], [0.2, 0.7], [-0.4, 0.5]])
    theta_lb = -2.0 * np.ones(NONLINEAR_TOY.theta_shape)
    theta_ub = 2.0 * np.ones(NONLINEAR_TOY.theta_shape)
    a_lb = -1.5 * np.ones(NONLINEAR_TOY.adim)
    a_ub = 1.5 * np.ones(NONLINEAR_TOY.adim)
    projection_epsilon = 0.01
    a_hat_norm_max = 0.5 * np.linalg.norm(a_ub - a_lb) + projection_epsilon

    config = {
        "K_pre": K_pre,
        "N_cal": N_cal,
        "K": K,
        "B": B,
        "dt": dt,
        "interval_duration": interval_duration,
        "I_length": I_length,
        "pretrain_excitation_amplitude": 0.5,
        "pretrain_excitation_frequency": 0.25,
        "pretrain_initial_state": np.array([0.5, -0.3, 0.0]),
        "nominal_initial_state": np.array([1.0, -1.8, -1.2]),
        "nominal_goal_state": np.array([2.0, 1.0, 1.5]),
        "tracking_initial_offset": np.array([0.25, -0.20, 0.15]),
        "divergence_norm": 20.0,
        "geodesic_degree": 2,
        "geodesic_nodes": 8,
        "use_qpsolvers": True,
        "verify_geodesic": False,
    }
    base_params = {
        "Theta_init": theta_init.copy(),
        "Gamma_ccm": 2.0 * np.eye(NONLINEAR_TOY.adim),
        "a_ub": a_ub,
        "a_lb": a_lb,
        "a_hat_norm_max": a_hat_norm_max,
        "epsilon": projection_epsilon,
        "eta_ccm": 5.0,
        "ccm_rate": 0.8,
        "weight_slack": 1000.0,
        "u_min": -20.0,
        "u_max": 20.0,
        "dt": dt,
    }

    pretrain_params = dict(base_params)
    pretrain_params.update({"use_cp": False, "use_adaptive": False})
    pretrain_system = NONLINEAR_TOY(pretrain_params)

    def representation_rate(update_index):
        return 1e-2 / np.sqrt(update_index)

    olacp = OLACP(
        [],
        N_cal=N_cal,
        acp_lr=0.02,
        delta_target=0.1,
        delta_init=0.1,
        buffer_maxlen=I_length,
        Theta_init=theta_init,
        representation_period=B,
        representation_lr=representation_rate,
        Theta_lb=theta_lb,
        Theta_ub=theta_ub,
        Y_Theta=pretrain_system.Y_Theta,
        representation_loss_gradient=pretrain_system.representation_loss_gradient,
    )
    trained_olacp, pretraining_history = run_pretraining(
        pretrain_system, olacp, config, plot=True
    )
    trajectory = plan_nominal_trajectory(pretrain_system, config)

    canonical_theta = trained_olacp.Theta.copy()
    canonical_a_k = trained_olacp.a_k.copy()
    canonical_scores = np.asarray(trained_olacp.S_cal).copy()
    canonical_delta = float(trained_olacp.delta)

    settings = (
        ("CP + adaptive", True, True),
        ("No CP + adaptive", False, True),
        ("No CP + nonadaptive", False, False),
    )
    results = {}
    for label, use_cp, use_adaptive in settings:
        run_params = dict(base_params)
        run_params["Theta_init"] = canonical_theta.copy()
        run_params["use_cp"] = use_cp
        run_params["use_adaptive"] = use_adaptive
        run_system = NONLINEAR_TOY(run_params)
        run_olacp = copy.deepcopy(trained_olacp)
        results[label] = run_craccm_simulation(
            run_system,
            run_olacp,
            trajectory,
            config,
            use_cp,
            use_adaptive,
            label=label,
        )

    if not np.array_equal(trained_olacp.Theta, canonical_theta):
        raise RuntimeError("A main simulation mutated the canonical pretrained Theta")
    if not np.array_equal(trained_olacp.a_k, canonical_a_k):
        raise RuntimeError("A main simulation mutated the canonical pretrained a_k")
    if not np.array_equal(np.asarray(trained_olacp.S_cal), canonical_scores):
        raise RuntimeError("A main simulation mutated the canonical calibration set")
    if trained_olacp.delta != canonical_delta:
        raise RuntimeError("A main simulation mutated the canonical adaptive delta")

    plot_craccm_results(results, trajectory)
    plt.tight_layout()
    if os.environ.get("CRASAFE_NO_PLOTS", "0") == "1":
        plt.close("all")
    else:
        plt.show()
    return pretraining_history, trajectory, results


if __name__ == "__main__":
    main()
