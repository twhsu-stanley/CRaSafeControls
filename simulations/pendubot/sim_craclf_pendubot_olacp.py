import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from olacp import OLACP
from systems.pendubot.pendubot import Pendubot


USE_CP = True#False #
USE_ADAPTIVE = True#False #

# Algorithm 1 setup
K_pretrain = 100
N_cal = 90
K = 16
B = 5
if K_pretrain < 1:
    raise ValueError("K_pretrain must be at least 1")
if K_pretrain % B != 0:
    raise ValueError("K_pretrain must be an integer multiple of B")
if K_pretrain < N_cal:
    raise ValueError("K_pretrain must be at least as large as N_cal")

# Time setup for each piecewise-constant environment interval I_k
dt = 0.01
interval_duration = 2.0
I_length = int(round(interval_duration / dt))
if I_length < 2 or not np.isclose(I_length * dt, interval_duration):
    raise ValueError("interval_duration must be an integer multiple of dt")
sim_T = K * interval_duration
tt = np.arange(0.0, sim_T, dt)
if len(tt) != K * I_length:
    raise ValueError("K, interval_duration, and dt define inconsistent samples")

def wind_velocity(t):
    """Piecewise-constant wind, so each interval has a fixed trim input."""
    interval_index = int(np.floor(t / interval_duration))
    phase = 2.0 * np.pi * (interval_index % B) / B
    return np.array(
        [0.5 * np.cos(phase), -1.8 * np.sin(phase + 0.25)]
    )

# Known parameters
m1 = 1.0
m2 = 2.0
L1 = 1.0
L2 = 1.5
r1 = 0.5
r2 = 0.5
I1 = m1 * L1**2 / 12.0
I2 = m2 * L2**2 / 12.0

theta_ub = np.ones((13, 5)) * 0.2
theta_lb = -np.ones((13, 5)) * 0.2
theta_rng = np.random.default_rng(11)
Theta_init = theta_rng.uniform(theta_lb, theta_ub)

a_lb = -np.ones(5) * 1.0
a_ub = np.ones(5) * 1.0
a_center = 0.5 * (a_lb + a_ub)
projection_epsilon = 0.01
a_radius = 0.5 * np.linalg.norm(a_ub - a_lb, ord=2) + projection_epsilon

params = {
    "m1": m1,
    "m2": m2,
    "L1": L1,
    "L2": L2,
    "r1": r1,
    "r2": r2,
    "I1": I1,
    "I2": I2,
    "grav": 9.30,
    "true_grav": 10.0,
    "damping": np.array([0.15, 0.18]),
    "true_damping": np.array([0.01, 0.01]),
    "L_w": 0.5,
    "c_w": 1.0,
    "wind_velocity": wind_velocity,
    "Theta_init": Theta_init,
    "lqr_Q": np.diag([10.0, 10.0, 1.0, 1.0]),
    "lqr_R": 0.5,
    "use_adaptive": USE_ADAPTIVE,
    "use_cp": USE_CP,
    "Gamma_clf": 0.02 * np.eye(5),
    "a_ub": a_ub,
    "a_lb": a_lb,
    "a_hat_norm_max": a_radius,
    #"projection_geometry": "box",
    "epsilon": projection_epsilon,
    "eta_clf": 10.0,
    # This is deliberately below the nominal linear CARE decay rate. The
    # QP uses the exact nonlinear Lie derivatives of the local certificate.
    "clf_rate": 0.01,
    "weight_slack": 1e4,
    "u_min": -25.0,
    "u_max": 25.0,
}
system = Pendubot(params)

# Initialize OLACP with an empty calibration set
olacp = OLACP(
    [],
    N_cal=N_cal,
    acp_lr=0.02,
    delta_target=0.1,
    delta_init=0.1,
    buffer_maxlen=I_length,
    theta_init=Theta_init,
    representation_period=B,
    # OLACP sums B * I_length sample gradients. Scale the update accordingly.
    representation_lr=lambda j: 5e-2,
    theta_lb=theta_lb,
    theta_ub=theta_ub,
    Y_theta=system.Y_theta,
    representation_loss_gradient=system.representation_loss_gradient,
)
system.set_representation(olacp.Theta)

# Pre-training
# The state is reset before the CRaCLF simulation, while OLACP retains the
# learned representation and calibration scores
pretrain_sample_count = K_pretrain * I_length
tt_pretrain = np.arange(pretrain_sample_count, dtype=float) * dt
x_pretrain = np.array([0.22, -0.16, 0.0, 0.0])
Theta_pretrain_hist = np.zeros((K_pretrain + 1, Theta_init.size))
Theta_pretrain_hist[0] = olacp.Theta.reshape(-1)
s_pretrain_hist = np.zeros(K_pretrain)
pretrain_prediction_error_hist = np.full(pretrain_sample_count, np.nan)

for pretrain_interval_index in range(K_pretrain):
    for interval_sample_index in range(I_length):
        pretrain_sample_index = (
            pretrain_interval_index * I_length + interval_sample_index
        )
        t_pretrain = pretrain_sample_index * dt
        u_pretrain = system.local_lqr_control(x_pretrain, system.a_center) + 0.1 * np.sin(2.0 * np.pi * 1 * t_pretrain)

        olacp.add_data_to_buffers(
            x_pretrain,
            system.dynamics_nominal(x_pretrain, u_pretrain),
            xdot=system.dynamics(x_pretrain, u_pretrain, t_pretrain),
        )

        if pretrain_sample_index < pretrain_sample_count - 1:
            sol = solve_ivp(
                lambda tau, y: system.dynamics(y, u_pretrain, tau),
                (t_pretrain, t_pretrain + dt),
                x_pretrain,
                method="RK45",
                rtol=1e-7,
                atol=1e-9,
                t_eval=[t_pretrain + dt],
            )
            if not sol.success:
                raise RuntimeError(sol.message)
            x_pretrain = sol.y[:, -1]

    olacp.estimate_uncertainty(dt)
    s_pretrain = olacp.compute_score(system.a_ub, system.a_lb)
    interval_prediction_error = np.array(
        [
            np.linalg.norm(Y_t @ olacp.a_k - w_t, ord=2) ** 2
            for Y_t, w_t in zip(olacp._Y_buffer, olacp._w_buffer)
        ]
    )
    olacp.append_score(s_pretrain)
    representation_update = olacp.update_representation()

    theta_step_norm = 0.0
    if representation_update is not None:
        theta_before = system.Theta_hat.copy()
        olacp.Theta = representation_update["Theta"]
        system.set_representation(olacp.Theta)
        theta_step_norm = np.max(np.abs(olacp.Theta - theta_before))

    s_pretrain_hist[pretrain_interval_index] = s_pretrain
    interval_start = pretrain_interval_index * I_length
    pretrain_prediction_error_hist[
        interval_start : interval_start + I_length
    ] = interval_prediction_error
    Theta_pretrain_hist[pretrain_interval_index + 1] = olacp.Theta.reshape(-1)
    olacp.clear_buffers()

    print(
        f"pretrain_interval={pretrain_interval_index + 1:03d}, "
        f"score={s_pretrain:.3e}, theta_step={theta_step_norm:.3e}"
    )

system.cp_quantile = olacp.compute_quantile()

# CRaCLF simulation initialization
x = np.array([0.18, -0.12, 0.0, 0.0])
a_hat_clf = olacp.a_k.copy() if USE_ADAPTIVE else np.zeros(system.adim)
rho_clf = 0.0
x_ext = np.hstack((x, a_hat_clf, rho_clf))

x_hist = np.zeros((len(tt), system.xdim))
u_hist = np.zeros(len(tt))
#u_ref_hist = np.zeros(len(tt))
true_trim_hist = np.zeros(len(tt))
estimated_trim_hist = np.zeros(len(tt))
slack_hist = np.zeros(len(tt))
V_hist = np.zeros(len(tt))
a_hat_clf_hist = np.zeros((len(tt), system.adim))
a_k_hist = np.full((len(tt), system.adim), np.nan)
rho_clf_hist = np.zeros(len(tt))
nu_clf_hist = np.zeros(len(tt))
Q_k_hist = np.zeros(len(tt))
Theta_hist = np.zeros((len(tt), Theta_init.size))
prediction_error_hist = np.full(len(tt), np.nan)

interval_times = []
s_k_hist = []
delta_k_hist = []
e_k_hist = []

for i, t in enumerate(tt):
    x_hist[i] = x
    a_hat_clf_hist[i] = a_hat_clf
    rho_clf_hist[i] = rho_clf
    nu_clf_hist[i] = system.nu_clf(rho_clf)
    Q_k_hist[i] = system.cp_quantile
    Theta_hist[i] = system.Theta_hat.reshape(-1)
    V_hist[i] = float(np.asarray(system.clf(x, a_hat_clf)).item())
    true_trim_hist[i] = system.true_trim_input(t)
    estimated_trim_hist[i] = system.estimated_trim_input(a_hat_clf)

    #u_ref = system.local_lqr_control(x, a_hat_clf)
    u_ref = system.ctrl_nominal(x)
    u, slack = system.ctrl_craclf(x, a_hat_clf, u_ref, use_slack=True)
    u_hist[i] = u.item()
    slack_hist[i] = slack
    olacp.add_data_to_buffers(
        x,
        system.dynamics_nominal(x, u),
        xdot=system.dynamics(x, u, t),
    )

    if i < len(tt) - 1:
        sol = solve_ivp(
            lambda tau, y: system.dynamics_extended(y, u, tau),
            (tt[i], tt[i + 1]),
            x_ext,
            method="RK45",
            rtol=1e-7,
            atol=1e-9,
            t_eval=[tt[i + 1]],
        )
        if not sol.success:
            raise RuntimeError(sol.message)

        x_ext = sol.y[:, -1]
        x = x_ext[: system.xdim]
        a_hat_clf = x_ext[system.xdim : system.xdim + system.adim]
        rho_clf = x_ext[system.xdim + system.adim]

    if (i + 1) % I_length == 0:
        interval_index = (i + 1) // I_length - 1
        olacp.estimate_uncertainty(dt)
        s_k = olacp.compute_score(system.a_ub, system.a_lb)
        interval_prediction_error = np.array(
            [
                np.linalg.norm(Y_t @ olacp.a_k - w_t, ord=2) ** 2
                for Y_t, w_t in zip(olacp._Y_buffer, olacp._w_buffer)
            ]
        )
        e_k = olacp.update_delta(s_k)
        olacp.append_score(s_k)
        representation_update = olacp.update_representation()

        interval_start = i - I_length + 1
        a_k_hist[interval_start : i + 1] = olacp.a_k
        prediction_error_hist[interval_start : i + 1] = (
            interval_prediction_error
        )
        interval_times.append(t)
        s_k_hist.append(s_k)
        delta_k_hist.append(olacp.delta)
        e_k_hist.append(e_k)

        if representation_update is not None:
            olacp.Theta = representation_update["Theta"].copy()
            system.set_representation(olacp.Theta)

        Q_used = Q_k_hist[i]
        system.cp_quantile = olacp.compute_quantile()
        olacp.clear_buffers()

        print(
            f"interval={interval_index + 1:02d}, "
            f"score={s_k:.3e}, Q={Q_used:.3e}, "
            f"delta_next={olacp.delta:.3f}, miscoverage={e_k}"
        )

interval_times = np.asarray(interval_times)
s_k_hist = np.asarray(s_k_hist)
delta_k_hist = np.asarray(delta_k_hist)
e_k_hist = np.asarray(e_k_hist)

# Pendubot angles and angular velocities
fig, axs = plt.subplots(4, 1, sharex=True)
state_labels = (
    r"$q_1$ (deg)",
    r"$q_2$ (deg)",
    r"$\dot q_1$ (deg/s)",
    r"$\dot q_2$ (deg/s)",
)
for index, ax in enumerate(axs):
    ax.plot(tt, np.rad2deg(x_hist[:, index]), linewidth=1.4)
    ax.set_ylabel(state_labels[index])
    ax.grid(True)
axs[-1].set_xlabel("Time (s)")
fig.suptitle("Pendubot states")

# Piecewise-constant wind disturbance
wind_hist = np.asarray([wind_velocity(t) for t in tt])
fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].step(tt, wind_hist[:, 0], where="post")
axs[0].set_ylabel(r"$w_x$ (m/s)")
axs[1].step(tt, wind_hist[:, 1], where="post")
axs[1].set_ylabel(r"$w_z$ (m/s)")
axs[1].set_xlabel("Time (s)")
for ax in axs:
    ax.grid(True)
fig.suptitle("Wind disturbance")

# Control, trim estimates, local CRaCLF, and QP relaxation
fig, axs = plt.subplots(3, 1, sharex=True, figsize=(7, 7))
axs[0].plot(tt, u_hist, label=r"$u$")
#axs[0].plot(tt, u_ref_hist, "--", label=r"$u_{\mathrm{LQR}}$")
axs[0].plot(tt, true_trim_hist, ":", label=r"$u^\star$")
axs[0].plot(tt, estimated_trim_hist, "-.", label=r"$\hat u^\star$")
axs[0].set_ylabel("torque (N m)")
axs[0].legend(ncol=2)
axs[1].plot(tt, V_hist)
axs[1].set_ylabel(r"$V_r$")
axs[2].plot(tt, slack_hist)
axs[2].set_ylabel("QP slack")
axs[2].set_xlabel("Time (s)")
for ax in axs:
    ax.grid(True)
fig.suptitle("Local CRaCLF-QP")

# Adaptive conformal prediction variables
fig, axs = plt.subplots(3, 1, sharex=True)
axs[0].step(interval_times, s_k_hist, where="post", label=r"$s_k$")
axs[0].step(tt, Q_k_hist, where="post", label=r"$Q_k$")
axs[0].set_ylabel("score")
axs[0].legend()
axs[1].step(interval_times, delta_k_hist, where="post")
axs[1].set_ylabel(r"$\delta_k$")
axs[2].step(interval_times, e_k_hist, where="post")
axs[2].set_ylabel(r"$e_k$")
axs[2].set_xlabel("Time (s)")
for ax in axs:
    ax.grid(True)
    for interval_index in range(1, K):
        ax.axvline(
            interval_index * interval_duration,
            color="0.7",
            linestyle=":",
            linewidth=0.8,
        )
fig.suptitle("Adaptive conformal prediction")

# Residual loss
fig, axs = plt.subplots(2, 1, figsize=(7, 7))
axs[0].semilogy(
    tt_pretrain,
    pretrain_prediction_error_hist,
    label=r"$\|Y_{\Theta_j}(x_t)a_k-w_t\|_2^2$",
)
axs[1].semilogy(
    tt,
    prediction_error_hist,
    label=r"$\|Y_{\Theta_j}(x_t)a_k-w_t\|_2^2$",
)
for ax, interval_count in zip(axs, (K_pretrain, K)):
    for update_index in range(B, interval_count, B):
        ax.axvline(
            update_index * interval_duration,
            color="k",
            linestyle=":",
            alpha=0.6,
            label="new representation active" if update_index == B else None,
        )
    ax.set_ylabel("squared prediction error")
    ax.grid(True)
    ax.legend()
axs[0].set_title("Pretraining")
axs[1].set_title("Online implementation")
axs[1].set_xlabel("Time (s)")
fig.suptitle("Uncertainty-prediction error")

# Adaptive and interval-fitted environmental parameters
fig, axs = plt.subplots(system.adim, 1, sharex=True, figsize=(7, 9))
for index in range(system.adim):
    axs[index].plot(tt, a_hat_clf_hist[:, index], label=r"$\hat a$")
    axs[index].plot(tt, a_k_hist[:, index], "--", label=r"$a_k$")
    axs[index].set_ylabel(f"a{index + 1}")
    axs[index].grid(True)
    axs[index].legend()
axs[-1].set_xlabel("Time (s)")
fig.suptitle("Adaptive and interval-fitted parameters")

# Learned 13-by-5 representation
"""
fig, axs = plt.subplots(
    *Theta_init.shape, sharex=True, figsize=(13, 18), squeeze=False
)
for row in range(Theta_init.shape[0]):
    for column in range(Theta_init.shape[1]):
        theta_index = row * Theta_init.shape[1] + column
        axs[row, column].plot(tt, Theta_hist[:, theta_index])
        axs[row, column].set_ylabel(
            rf"$\Theta_{{{row + 1},{column + 1}}}$"
        )
        axs[row, column].grid(True)
for ax in axs[-1, :]:
    ax.set_xlabel("Time (s)")
fig.suptitle("Learned representation")
"""

# CRaCLF scaling state and scaling function
fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].plot(tt, nu_clf_hist)
axs[0].set_ylabel(r"$\nu_{\mathrm{clf}}$")
axs[1].plot(tt, rho_clf_hist)
axs[1].set_ylabel(r"$\rho_{\mathrm{clf}}$")
axs[1].set_xlabel("Time (s)")
for ax in axs:
    ax.grid(True)

plt.tight_layout()
plt.show()
