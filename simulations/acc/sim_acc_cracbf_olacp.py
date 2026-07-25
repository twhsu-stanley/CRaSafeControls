"""Algorithm 1 and CRaCBF simulation for adaptive cruise control"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from olacp import OLACP
from systems.acc.acc import ACC


USE_CP = True#False#
USE_ADAPTIVE = True#False#

# Algorithm 1 setup.
K_pre = 120
N_cal = 100
K = 12 # total number of time intervals I_k
B = 4   # update the representation every B intervals
#if K_pre < N_cal:
#    raise ValueError("K_pre must be at least as large as N_cal")
#if K_pre % B != 0:
#    raise ValueError("K_pre must be an integer multiple of B")
#
# Time setup for each interval I_k.
dt = 0.01
interval_duration = 2.0
I_length = int(round(interval_duration / dt))
if I_length < 10 or not np.isclose(I_length * dt, interval_duration):
    raise ValueError(
        "interval_duration must be an integer multiple of dt and contain "
        "at least 10 samples"
    )
sim_T = K * interval_duration
tt = np.arange(0.0, sim_T, dt)
if len(tt) != K * I_length:
    raise ValueError("K, interval_duration, and dt define inconsistent sample counts")

# System and environment parameters.
mass = 1000.0
gravity = 9.81
beta_1 = -10.0
beta_2 = -0.75
beta_3 = 0.2
beta_4 = 0.02
desired_velocity = 26.0

# Psi contains raw polynomial features with very different magnitudes. Scale
# the representation bounds row-wise so every basis term has a comparable
# effect near the reference ACC operating point.
v_reference = desired_velocity
z_reference = 30.0
psi_reference = np.kron(
    np.array([1.0, v_reference, v_reference**2]),
    np.array([1.0, z_reference, z_reference**2]),
)
Theta_scale = 1 / psi_reference
Theta_lb = -5.0 * Theta_scale[:, None] * np.ones((1, 3))
Theta_ub = 5.0 * Theta_scale[:, None] * np.ones((1, 3))
theta_rng = np.random.default_rng(11)
Theta_init = theta_rng.uniform(
    -Theta_scale[:, None],
    Theta_scale[:, None],
    size=ACC.theta_shape,
)

# The first three coordinates are latent drag parameters.
# The fourth is the scaled lead velocity
a_lb = np.array([-1.0, -1.0, -1.0, -8.0])
a_ub = np.array([1.0, 1.0, 1.0, 8.0])
a_center = 0.5 * (a_lb + a_ub)

projection_epsilon = 0.01
a_hat_norm_max = 0.5 * np.linalg.norm(a_ub - a_lb, ord=2) + projection_epsilon
Gamma_cbf = np.diag([10.0, 10.0, 10.0, 5.0])
Gamma_cbf_inv = np.linalg.inv(Gamma_cbf)

# Use a separate piecewise-constant environment during pretraining. The lead
# vehicle remains strictly faster than the desired ego speed.
pretrain_environment_phase = 2.0 * np.pi * np.arange(K_pre) / K_pre * 2
pretrain_d0_schedule = -50.0 + 150.0 * np.sin(0.5 * pretrain_environment_phase)
pretrain_wind_velocity_schedule = 0 - 25.0 * np.sin(1.0 * pretrain_environment_phase)
pretrain_delta_lead_velocity_schedule = 1.0 * np.sin(pretrain_environment_phase)

def pretrain_true_uncertainty(x, t):
    """Evaluate w(x, t) in the separate pretraining environment."""
    interval_index = min(int(np.floor(max(float(t), 0.0) / interval_duration)), K_pre - 1)
    x = np.asarray(x, dtype=float).reshape(3)
    v, z = x[1], x[2]
    wind_velocity = pretrain_wind_velocity_schedule[interval_index]
    drag_force =  (
        pretrain_d0_schedule[interval_index]
        + beta_1 * v
        + beta_2 * (v - wind_velocity)**2 * (1.0 - beta_3 * np.exp(-beta_4 * z))
    )
    return np.array(
        [
            0.0,
            drag_force / mass,
            pretrain_delta_lead_velocity_schedule[interval_index],
        ]
    )


# Construct the ACC system
params = {
    "Theta_init": Theta_init.copy(),
    "true_uncertainty": pretrain_true_uncertainty,
    "m": mass,
    "grav": gravity,
    "vd": desired_velocity,
    "Kp": 1000.0,
    "z_min": 5.0,
    "T_h": 0.5,
    "use_adaptive": USE_ADAPTIVE,
    "use_cp": USE_CP,
    "Gamma_cbf": Gamma_cbf,
    "a_ub": a_ub,
    "a_lb": a_lb,
    "a_hat_norm_max": a_hat_norm_max,
    "epsilon": projection_epsilon,
    #"projection_geometry": "box",
    "eta_cbf": 1.0,
    "cbf_rate": 1.0,
    "u_max": 0.3 * mass * gravity,
    "u_min": -0.3 * mass * gravity,
    "dt": dt,
}
system = ACC(params)

# Initialize one OLACP object with an empty calibration set. This same object
# is used for pretraining and the subsequent online CRaCBF simulation.
olacp = OLACP(
    [],
    N_cal=N_cal,
    acp_lr=0.02,
    delta_target=0.05,
    delta_init=0.05,
    buffer_maxlen=I_length,
    Theta_init=Theta_init,
    representation_period=B,
    # Normalize each row's step by its reference feature magnitude and by the
    # number of samples accumulated in one representation-learning block.
    representation_lr=lambda j: (1/ (psi_reference[:, None])/ j),
    Theta_lb=Theta_lb,
    Theta_ub=Theta_ub,
    Y_Theta=system.Y_Theta,
    representation_loss_gradient=system.representation_loss_gradient,
)
system.set_representation(olacp.Theta)

#############################################################################
# Pretrain the representation and collect calibration scores. 
# OLACP retains only the latest N_cal scores and continues its interval/update 
# counters when the online simulation starts.
tt_pre = np.arange(K_pre * I_length, dtype=float) * dt

x_pre = np.array([0.0, 20.0, 30.0])

x_pre_hist = np.zeros((len(tt_pre), system.xdim))
Y_pre_hist = np.zeros((len(tt_pre), system.xdim, system.adim))
a_pre_hist = np.zeros((len(tt_pre), system.adim))
Theta_pre_hist = np.zeros((len(tt_pre),)+ Theta_init.shape)
pretrain_prediction_error_hist = np.zeros(len(tt_pre))


for i_pre, t_pre in enumerate(tt_pre):
    pretrain_interval_index = i_pre // I_length

    x_pre_hist[i_pre] = x_pre

    Y_pre_hist[i_pre] = system.Y(x_pre)
    Theta_pre_hist[i_pre] = system.Theta_hat
    
    u_pre = system.ctrl_nominal(x_pre)
    
    # Store the data required by line 6 of Algorithm 1.
    olacp.add_data_to_buffers(
        x_pre,
        system.dynamics_nominal(x_pre, u_pre),
        xdot=system.dynamics(x_pre, u_pre, t_pre),
    )

    # Propagate with zero-order hold on the CRaCBF control input.
    if i_pre < len(tt_pre) - 1:
        sol = solve_ivp(
            lambda tau, y: system.dynamics(y, u_pre, tau),
            (tt_pre[i_pre], tt_pre[i_pre + 1]),
            x_pre,
            method="BDF",
            rtol=1e-7,
            atol=1e-9,
            t_eval=[tt_pre[i_pre + 1]],
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        x_pre = sol.y[:, -1]

    # Complete the pretraining version of lines 7--23 of Algorithm 1.
    if (i_pre + 1) % I_length == 0:
        olacp.estimate_uncertainty(dt)
        s_pre = olacp.compute_score(system.a_ub, system.a_lb)
        
        interval_prediction_error = np.array(
            [
                np.linalg.norm(Y_t @ olacp.a_k - w_t, ord=2) ** 2
                for Y_t, w_t in zip(olacp._Y_buffer, olacp._w_buffer)
            ]
        )

        olacp.append_score(s_pre)
        representation_update = olacp.update_representation()

        interval_start = i_pre - I_length + 1
        a_pre_hist[interval_start : i_pre + 1] = olacp.a_k
        pretrain_prediction_error_hist[interval_start : i_pre + 1] = interval_prediction_error
        
        theta_step_norm = 0.0
        if representation_update is not None:
            theta_before = system.Theta_hat.copy()
            system.set_representation(representation_update["Theta"])
            theta_step_norm = np.max(np.abs(system.Theta_hat - theta_before))

        olacp.clear_buffers()

        print(
            f"pretrain_interval={pretrain_interval_index + 1:03d}, "
            f"score={s_pre:.3e}, "
            f"theta_step={theta_step_norm:.3e}"
        )

#if len(olacp.S_cal) != N_cal:
#    raise RuntimeError("Pretraining did not fill the requested calibration window")


# Plot states and input.
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 11))
axs[0].plot(tt_pre, x_pre_hist[:, 1], label="ego velocity")
axs[0].axhline(params["vd"], color="k", linestyle=":", label="desired")
axs[0].set_ylabel("v (m/s)")
axs[0].legend(ncol=3)
axs[1].plot(tt_pre, x_pre_hist[:, 2], label=r"$z(t)$")
axs[1].axhline(params["z_min"], color="r", linestyle="--", label=r"$z_{\min}$")
axs[1].set_ylabel("z (m)")
axs[1].legend()
for ax in axs:
    ax.grid(True)
fig.suptitle("State During Pre-training")

fig, axs = plt.subplots(1, 3)
axs[0].plot(tt_pre, Y_pre_hist[:,1,0])
axs[0].set_ylabel("Y_21")
axs[1].plot(tt_pre, Y_pre_hist[:,1,1])
axs[1].set_ylabel("Y_22")
axs[2].plot(tt_pre, Y_pre_hist[:,1,2])
axs[2].set_ylabel("Y_23")
for ax in axs:
    ax.grid(True)

fig, axs = plt.subplots(*Theta_init.shape, sharex=True)
for row in range(Theta_init.shape[0]):
    for column in range(Theta_init.shape[1]):
        axs[row, column].plot(tt_pre, Theta_pre_hist[:, row, column])
        axs[row, column].set_ylabel(
            rf"$\Theta_{{{row + 1},{column + 1}}}$"
        )
        axs[row, column].grid(True)
for ax in axs[-1, :]:
    ax.set_xlabel("Time (s)")
fig.suptitle("Pre-trianed Theta")

fig, axs = plt.subplots(system.adim, 1, sharex=True, figsize=(8, 12))
for i in range(system.adim):
    axs[i].plot(tt_pre, a_pre_hist[:, i], "--", label="a_k")
    axs[i].set_ylabel(f"ak{i + 1}")
    axs[i].grid(True)
axs[0].legend(ncol=3)
axs[-1].set_xlabel("Time (s)")
for ax in axs:
    ax.grid(True)
    for interval_index in range(1, K):
        ax.axvline(
            interval_index * interval_duration,
            color="0.7",
            linestyle=":",
            linewidth=1.0,
        )

fig, ax = plt.subplots(1, 1, figsize=(8, 7))
ax.semilogy(
    tt_pre,
    np.maximum(pretrain_prediction_error_hist, 1e-16),
    label=r"$\|Y_{\Theta_j}(x_t)a_k-w_t\|_2^2$",
)
for update_index in range(B, K_pre, B):
    ax.axvline(
        update_index * interval_duration,
        color="k",
        linestyle=":",
        alpha=0.4,
        label=(
            "new representation active"
            if update_index == B
            else None
        ),
    )
ax.set_ylabel("squared prediction error")
ax.grid(True)
ax.legend()
ax.set_title("Pretraining")
ax.set_xlabel("Time (s)")
fig.suptitle("Uncertainty-prediction error")

plt.show()

#########################################################################
# The actual CRaCBF simulation begins here
system.set_representation(olacp.Theta)
system.cp_quantile = olacp.compute_quantile()

# Generate a piecewise-constant environment on the Algorithm 1 intervals.
# beta_1,...,beta_4 are fixed; d_0, wind speed, and lead velocity vary.
environment_phase = 2.0 * np.pi * np.arange(K) / 20.0
d0_schedule = -220.0 - 440.0 * np.sin(environment_phase + 0.15)
wind_velocity_schedule = 2.0 + 4.0 * np.sin(0.7 * environment_phase - 0.2)
delta_lead_velocity_schedule = -6.0 * np.sin(0.9 * environment_phase)

def true_uncertainty(x, t):
    """Evaluate the unknown physical uncertainty w(x, t)"""
    interval_index = min(int(np.floor(max(float(t), 0.0) / interval_duration)), K - 1)
    x = np.asarray(x, dtype=float).reshape(3)
    v, z = x[1], x[2]
    wind_velocity = wind_velocity_schedule[interval_index]
    drag_force = (
        d0_schedule[interval_index]
        + beta_1 * v
        + beta_2
        * (v - wind_velocity) ** 2
        * (1.0 - beta_3 * np.exp(-beta_4 * z))
    )
    return np.array(
        [
            0.0,
            drag_force / mass,
            delta_lead_velocity_schedule[interval_index],
        ]
    )
system.true_uncertainty_fcn = true_uncertainty

# Simulation initialization.
x = np.array([0.0, 20.0, 40.0])
a_hat_cbf = a_center.copy()
rho_cbf = 0.0
x_ext = np.hstack((x, a_hat_cbf, rho_cbf))

x_hist = np.zeros((len(tt), system.xdim))
u_hist = np.zeros(len(tt))
h_hist = np.zeros(len(tt))
physical_safety_hist = np.zeros(len(tt))
tightened_cbf_margin_hist = np.zeros(len(tt))
z_b_hist = np.zeros(len(tt))
z_b_exponential_bound_hist = np.zeros(len(tt))
a_hat_cbf_hist = np.zeros((len(tt), system.adim))
a_k_hist = np.full((len(tt), system.adim), np.nan)
lead_velocity_hist = np.zeros(len(tt))
rho_cbf_hist = np.zeros(len(tt))
nu_cbf_hist = np.zeros(len(tt))
Q_k_hist = np.zeros(len(tt))
Theta_hist = np.zeros((len(tt),) + Theta_init.shape)
prediction_error_hist = np.full(len(tt), np.nan)

interval_times = []
s_k_hist = []
delta_k_hist = []
e_k_hist = []

# Main simulation loop
for i, t in enumerate(tt):
    interval_index = i // I_length

    x_hist[i] = x
    a_hat_cbf_hist[i] = a_hat_cbf
    lead_velocity_hist[i] = delta_lead_velocity_schedule[interval_index] + system.nominal_lead_velocity
    rho_cbf_hist[i] = rho_cbf
    nu_cbf_hist[i] = system.nu_cbf(rho_cbf)
    Q_k_hist[i] = system.cp_quantile
    Theta_hist[i] = system.Theta_hat
    h_hist[i] = system.cbf(x, a_hat_cbf)
    physical_safety_hist[i] = x[2] - params["z_min"]
    tightened_cbf_margin_hist[i] = h_hist[i] - 0.5 / nu_cbf_hist[i] * system.safe_set_tightening
    if i == 0 and h_hist[i] < 0.0:
        raise ValueError(
            f"Initial condition is outside the CBF set: h={h_hist[i]:.3f}"
        )
    if (
        i % I_length == 0
        and USE_ADAPTIVE
        and tightened_cbf_margin_hist[i] < 0.0
    ):
        raise ValueError(
            "Interval-start state violates equation (37): "
            f"interval={interval_index + 1}, "
            f"margin={tightened_cbf_margin_hist[i]:.3f}"
        )

    u_ref = system.ctrl_nominal(x)
    u = system.ctrl_cracbf(x, a_hat_cbf, u_ref, rho_cbf)
    u_hist[i] = u.item()

    # Store the data required by line 6 of Algorithm 1.
    olacp.add_data_to_buffers(
        x,
        system.dynamics_nominal(x, u),
        xdot=system.dynamics(x, u, t),
    )

    # Propagate with zero-order hold on the control input
    if i < len(tt) - 1:
        sol = solve_ivp(
            lambda tau, y: system.dynamics_extended(y, u, tau),
            (tt[i], tt[i + 1]),
            x_ext,
            method="BDF",
            rtol=1e-7,
            atol=1e-9,
            t_eval=[tt[i + 1]],
        )
        if not sol.success:
            raise RuntimeError(sol.message)

        x_ext = sol.y[:, -1]
        if not np.all(np.isfinite(x_ext)):
            raise RuntimeError("The extended ACC state became non-finite")
        x = x_ext[: system.xdim]
        a_hat_cbf = x_ext[system.xdim : system.xdim + system.adim]
        rho_cbf = float(x_ext[system.xdim + system.adim])

    # Complete lines 7--23 of Algorithm 1 at the interval boundary
    if (i + 1) % I_length == 0:
        olacp.estimate_uncertainty(dt)
        s_k = olacp.compute_score(system.a_ub, system.a_lb)
        interval_prediction_error = np.array(
            [
                np.linalg.norm(Y_t @ olacp.a_k - w_t, ord=2) ** 2
                for Y_t, w_t in zip(olacp._Y_buffer, olacp._w_buffer)
            ]
        )
        e_k = int(olacp.update_delta(s_k))
        olacp.append_score(s_k)
        representation_update = olacp.update_representation()

        interval_start = i - I_length + 1
        # Algorithm 1 obtains a_k only after I_k is complete. Treat it
        # retrospectively as the fictitious true parameter for that same
        # interval; never carry it forward as the plant parameter of I_{k+1}.
        system.a_true = olacp.a_k.copy()
        a_k_hist[interval_start : i + 1] = olacp.a_k
        prediction_error_hist[interval_start : i + 1] = interval_prediction_error
        interval_z_b = np.empty(i - interval_start + 1)
        for interval_sample_index, history_index in enumerate(
            range(interval_start, i + 1)
        ):
            a_tilde = a_hat_cbf_hist[history_index] - system.a_true
            interval_z_b[interval_sample_index] = (
                nu_cbf_hist[history_index] * h_hist[history_index]
                - 0.5 * a_tilde @ Gamma_cbf_inv @ a_tilde
            )
        z_b_hist[interval_start : i + 1] = interval_z_b
        interval_time = tt[interval_start : i + 1]
        z_b_exponential_bound_hist[interval_start : i + 1] = (
            interval_z_b[0] * np.exp(-system.cbf_rate * (interval_time - interval_time[0]))
        )
        interval_times.append(t)
        s_k_hist.append(s_k)
        delta_k_hist.append(olacp.delta)
        e_k_hist.append(e_k)

        if representation_update is not None:
            system.set_representation(representation_update["Theta"])

        system.cp_quantile = olacp.compute_quantile()
        olacp.clear_buffers()

        print(
            f"interval={interval_index + 1:02d}, "
            f"score={s_k:.3e}, Q={Q_k_hist[i]:.3e}, "
            f"delta_next={olacp.delta:.3f}, miscoverage={e_k}, "
            f"||Theta||_F={np.linalg.norm(system.Theta_hat):.3e}"
        )

interval_times = np.asarray(interval_times)
s_k_hist = np.asarray(s_k_hist)
delta_k_hist = np.asarray(delta_k_hist)
e_k_hist = np.asarray(e_k_hist)

# Plot states and input.
fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 11))
axs[0].plot(tt, x_hist[:, 1], label="ego velocity")
axs[0].plot(tt, lead_velocity_hist, "--", label="lead velocity")
axs[0].axhline(params["vd"], color="k", linestyle=":", label="desired")
axs[0].set_ylabel("v (m/s)")
axs[0].legend(ncol=3)
axs[1].plot(tt, x_hist[:, 2], label=r"$z(t)$")
axs[1].axhline(params["z_min"], color="r", linestyle="--", label=r"$z_{\min}$")
axs[1].set_ylabel("z (m)")
axs[1].legend()
axs[2].plot(tt, u_hist, label="u")
#axs[2].axhline(params["u_max"], color="k", linestyle=":", label="input bounds")
#axs[2].axhline(params["u_min"], color="k", linestyle=":")
axs[2].set_ylabel("u (N)")
axs[2].legend()
for ax in axs:
    ax.grid(True)
fig.suptitle("Adaptive cruise control with a CRaCBF-QP")

# Plot safety margins.
fig, axs = plt.subplots(3, 1, sharex=True)
axs[0].plot(tt, h_hist, label=r"$h(x,\hat{a})$")
axs[0].plot(tt, tightened_cbf_margin_hist, ":", label="tightened h margin")
axs[0].axhline(0.0, color="r", linestyle="--", label=r"$h=0$")
axs[0].set_ylabel(r"$h(x,\hat{a})$")
axs[0].legend()
axs[1].plot(tt, physical_safety_hist, label=r"$z-z_{\min}$")
axs[1].axhline(0.0, color="r", linestyle="--", label="safety boundary")
axs[1].set_ylabel("safety margin")
axs[1].legend()
# Verify the within-interval comparison result used in Theorem 3:
# z_b(t) >= z_b(tau_k) exp(-alpha_b (t - tau_k)), t in I_k.
z_b_bound_gap_hist = z_b_hist - z_b_exponential_bound_hist
axs[2].plot(tt, z_b_hist, label=r"$z_b(t)$")
axs[2].plot(tt, z_b_exponential_bound_hist, "--", label=r"$z_b(\tau_k)e^{-\alpha_b(t-\tau_k)}$")
axs[2].set_ylabel(r"$z_b$")
axs[2].set_xlabel("Time (s)")
axs[2].legend()
for ax in axs:
    ax.grid(True)
    for interval_index in range(1, K):
        ax.axvline(
            interval_index * interval_duration,
            color="0.7",
            linestyle=":",
            linewidth=1.0,
        )
fig.suptitle("Safety margins")

# Plot OLACP variables.
fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
axs[0].step(interval_times, s_k_hist, where="post", label="s_k")
axs[0].step(tt, Q_k_hist, where="post", label="Q_k")
axs[0].set_ylabel("score")
axs[0].legend()
axs[1].step(interval_times, delta_k_hist, where="post")
axs[1].set_ylabel("delta_k")
axs[2].step(interval_times, e_k_hist, where="post")
axs[2].set_ylabel("e_k")
axs[2].set_xlabel("Time (s)")
for ax in axs:
    ax.grid(True)
    for interval_index in range(1, K):
        ax.axvline(
            interval_index * interval_duration,
            color="0.7",
            linestyle=":",
            linewidth=1.0,
        )
fig.suptitle("Adaptive conformal prediction")

# Plot the pretraining and online losses minimized by Algorithm 1. Each
# interval uses the representation installed while its data were sampled.
fig, axs = plt.subplots(2, 1, figsize=(8, 7))
axs[0].semilogy(
    tt_pre,
    np.maximum(pretrain_prediction_error_hist, 1e-16),
    label=r"$\|Y_{\Theta_j}(x_t)a_k-w_t\|_2^2$",
)
axs[1].semilogy(
    tt,
    np.maximum(prediction_error_hist, 1e-16),
    label=r"$\|Y_{\Theta_j}(x_t)a_k-w_t\|_2^2$",
)
for ax, interval_count in zip(axs, (K_pre, K)):
    for update_index in range(B, interval_count, B):
        ax.axvline(
            update_index * interval_duration,
            color="k",
            linestyle=":",
            alpha=0.4,
            label=(
                "new representation active"
                if update_index == B
                else None
            ),
        )
    ax.set_ylabel("squared prediction error")
    ax.grid(True)
    ax.legend()
axs[0].set_title("Pretraining")
axs[1].set_title("Online implementation")
axs[1].set_xlabel("Time (s)")
fig.suptitle("Uncertainty-prediction error")

# Plot the learned representation. Multiplication by the reference feature
# magnitude puts all nine rows on a comparable scale.
fig, axs = plt.subplots(*Theta_init.shape, sharex=True)
for row in range(Theta_init.shape[0]):
    for column in range(Theta_init.shape[1]):
        axs[row, column].plot(tt, Theta_hist[:, row, column])
        axs[row, column].set_ylabel(
            rf"$\Theta_{{{row + 1},{column + 1}}}$"
        )
        axs[row, column].grid(True)
for ax in axs[-1, :]:
    ax.set_xlabel("Time (s)")
fig.suptitle("Learned representation")

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(9, 9))
for column_index in range(3):
    normalized_column = (
        Theta_hist[:, :, column_index] * psi_reference[None, :]
    )
    for basis_index in range(9):
        axs[column_index].plot(
            tt,
            normalized_column[:, basis_index],
            label=rf"$\psi_{basis_index + 1}\theta_{{{basis_index + 1},{column_index + 1}}}$",
        )
    axs[column_index].axhline(
        Theta_ub[0, column_index] * psi_reference[0],
        color="0.5",
        linestyle=":",
    )
    axs[column_index].axhline(
        Theta_lb[0, column_index] * psi_reference[0],
        color="0.5",
        linestyle=":",
    )
    axs[column_index].set_ylabel(rf"$\Theta a_{{{column_index + 1}}}$ scale")
    axs[column_index].legend(ncol=3, fontsize=8)
axs[-1].set_xlabel("Time (s)")
for ax in axs:
    ax.grid(True)
    for interval_index in range(1, K):
        ax.axvline(
            interval_index * interval_duration,
            color="0.7",
            linestyle=":",
            linewidth=1.0,
        )
fig.suptitle(r"Learned Representation $\Theta\in\mathbb{R}^{9\times3}$")

# Plot adaptive, interval-fitted, and physical latent parameters.
fig, axs = plt.subplots(system.adim, 1, sharex=True, figsize=(8, 12))
for i in range(system.adim):
    axs[i].plot(tt, a_hat_cbf_hist[:, i], label="a_hat")
    axs[i].plot(tt, a_k_hist[:, i], "--", label="a_k")
    axs[i].set_ylabel(f"a{i + 1}")
    axs[i].grid(True)
axs[0].legend(ncol=3)
axs[-1].set_xlabel("Time (s)")
for ax in axs:
    ax.grid(True)
    for interval_index in range(1, K):
        ax.axvline(
            interval_index * interval_duration,
            color="0.7",
            linestyle=":",
            linewidth=1.0,
        )
fig.suptitle("Adaptive and interval-fitted latent parameters")

# Plot the CRaCBF scaling variables.
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 10))
axs[0].plot(tt, nu_cbf_hist)
axs[0].set_ylabel("nu(rho)")
axs[1].plot(tt, rho_cbf_hist)
axs[1].set_ylabel("rho")
axs[1].set_xlabel("Time (s)")
for ax in axs:
    ax.grid(True)
fig.suptitle("Learned representation and CRaCBF scaling")

plt.tight_layout()
plt.show()
