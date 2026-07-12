import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import lsq_linear

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from acp import ACP
from systems.strict_feedback.strict_feedback import StrictFeedbackSystem


USE_CP = True
USE_ADAPTIVE = True

# Algorithm 1 setup
K = 10  # total number of time intervals I_k
B = 4   # update the representation every B intervals

# Time setup for each interval I_k
dt = 0.01
interval_duration = 2.5
I_length = int(round(interval_duration / dt))
if I_length < 2 or not np.isclose(I_length * dt, interval_duration):
    raise ValueError("interval_duration must be an integer multiple of dt")
sim_T = K * interval_duration
tt = np.arange(0.0, sim_T, dt)
if len(tt) != K * I_length:
    raise ValueError("K, interval_duration, and dt define inconsistent sample counts")

# True uncertainty w(x,t), unknown to the controller. Its environmental
# coefficients change at the start of each time interval I_k.
true_uncertainty = lambda x, t: np.array(
    [
        #-(0.01 - 0.1 * np.floor(t / interval_duration)) * np.sin(x[0])
        #-(0.06 - 0.05 * np.floor(t / interval_duration)) * x[0] ** 2,
        -0.01 * np.sin(2*np.pi*1*x[0]) - 0.05 * x[0] ** 2,
        0.0,
        0.0,
    ]
)

# The first column starts from a linear approximation of -sin(x1); the
# initially missing cubic term gives representation learning useful work.
Theta_init = np.array(
    [
        [1.0, 0.001],
        [0.001, 1.0],
        [-1/6, 0.001],
    ]
)
a_lb = np.array([-0.1, -0.1])
a_ub = np.array([0.1, 0.1])
a_center = 0.5 * (a_lb + a_ub)
a_radius = 0.5 * np.linalg.norm(a_ub - a_lb, ord=2) + 0.2

params = {
    "Theta_init": Theta_init,
    "true_uncertainty": true_uncertainty,
    "use_adaptive": USE_ADAPTIVE,
    "use_cp": USE_CP,
    "Gamma_clf": np.diag([0.2, 0.2]),
    "a_true": np.zeros(2),
    "a_ub": a_ub,
    "a_lb": a_lb,
    "a_hat_norm_max": a_radius,
    "epsilon": 0.05,
    "eta_clf": 10.0,
    # The backstepping construction establishes rate 4. A smaller QP rate
    # leaves control authority for the conformal tightening.
    "clf": {"rate": 2.0},
    "weight_slack": 1e4,
}

# Construct the strict-feedback system.
system = StrictFeedbackSystem(params)

# Initial calibration set. Each historical interval follows lines 8 and 11 of
# Algorithm 1: fit a bounded a_k and retain the largest residual norm.
rng = np.random.default_rng(7)
N_cal = 200
S_cal_init = []
for k in range(N_cal):
    x1_cal = rng.uniform(-1.25, 1.25, I_length)
    Y_cal = []
    w_cal = []
    for x1 in x1_cal:
        Psi_x = np.array(
            [
                [x1, x1**2, x1**3],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        Y_cal.append(Psi_x @ Theta_init)
        w_cal.append(
            true_uncertainty(
                np.array([x1, 0.0, 0.0]), k * interval_duration
            )
        )

    Y_stack = np.vstack(Y_cal)
    w_stack = np.hstack(w_cal)
    a_cal = lsq_linear(Y_stack, w_stack, bounds=(a_lb, a_ub)).x
    S_cal_init.append(
        max(
            np.linalg.norm(Y_t @ a_cal - w_t, ord=2)
            for Y_t, w_t in zip(Y_cal, w_cal)
        )
    )
S_cal_init = np.asarray(S_cal_init)

# Adaptive conformal prediction and Algorithm 1 representation learning.
acp = ACP(
    S_cal_init,
    N_cal=N_cal,
    lr=0.02,
    delta_target=0.1,
    delta_init=0.1,
    buffer_maxlen=I_length,
    theta_init=Theta_init,
    representation_period=B,
    representation_learning_rate=lambda j: 2e-4 / j,
    theta_lb=-2.0,
    theta_ub=2.0,
)
system.cp_quantile = acp.Q_k

# Simulation initialization.
x = np.array([0.8, 0.6, 0.5])
a_hat_clf = a_center.copy()
rho_clf = 0.0
x_ext = np.hstack((x, a_hat_clf, rho_clf))

x_hist = np.zeros((len(tt), system.xdim))
u_hist = np.zeros(len(tt))
V_hist = np.zeros(len(tt))
a_hat_clf_hist = np.zeros((len(tt), system.adim))
a_k_hist = np.full((len(tt), system.adim), np.nan)
rho_clf_hist = np.zeros(len(tt))
nu_clf_hist = np.zeros(len(tt))
Q_k_hist = np.zeros(len(tt))
Theta_hist = np.zeros((len(tt), Theta_init.size))

interval_times = []
s_k_hist = []
delta_k_hist = []
e_k_hist = []

# Main simulation loop.
for i, t in enumerate(tt):
    x_hist[i] = x
    a_hat_clf_hist[i] = a_hat_clf
    rho_clf_hist[i] = rho_clf
    nu_clf_hist[i] = system.nu_clf(rho_clf)
    Q_k_hist[i] = system.cp_quantile
    Theta_hist[i] = system.Theta_hat.reshape(-1)
    V_hist[i] = float(np.asarray(system.clf(x, a_hat_clf)).item())

    u_ref = system.ctrl_nominal(x)
    u, _ = system.ctrl_craclf(x, a_hat_clf, u_ref, use_slack=False)
    u_hist[i] = u.item()

    # Store the data required by line 6 of Algorithm 1.
    acp.add_data_to_buffers(
        x,
        system.dynamics_nominal(x, u),
        system.Y(x),
        Psi_x=system.psi(x),
        xdot=system.dynamics(x, u, t),
    )

    # Propagate with zero-order hold on the control input.
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

    # Complete lines 7--23 of Algorithm 1 at the interval boundary.
    if (i + 1) % I_length == 0:
        interval_index = (i + 1) // I_length - 1
        acp.estimate_uncertainty(dt)
        s_k = acp.compute_score(system.a_ub, system.a_lb)
        e_k = acp.update_delta(s_k)
        acp.append_score(s_k)
        representation_update = acp.update_representation(acp.a_k)

        interval_start = i - I_length + 1
        a_k_hist[interval_start : i + 1] = acp.a_k
        interval_times.append(t)
        s_k_hist.append(s_k)
        delta_k_hist.append(acp.delta)
        e_k_hist.append(e_k)

        if representation_update is not None:
            system.set_representation(representation_update["Theta"])

        system.cp_quantile = acp.compute_quantile()
        acp.clear_buffers()

        print(
            f"interval={interval_index + 1:02d}, "
            f"score={s_k:.4f}, Q={Q_k_hist[i]:.4f}, "
            f"delta_next={acp.delta:.3f}, miscoverage={e_k}"
        )

interval_times = np.asarray(interval_times)
s_k_hist = np.asarray(s_k_hist)
delta_k_hist = np.asarray(delta_k_hist)
e_k_hist = np.asarray(e_k_hist)

# Plot states.
fig, axs = plt.subplots(3, 1, sharex=True)
for i in range(system.xdim):
    axs[i].plot(tt, x_hist[:, i], linewidth=1.4)
    axs[i].set_ylabel(f"x{i + 1}")
    axs[i].grid(True)
axs[-1].set_xlabel("Time (s)")
fig.suptitle("Strict-feedback states")

# Plot the control input and CRaCLF.
fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].plot(tt, u_hist, linewidth=1.2)
axs[0].set_ylabel("u")
axs[1].semilogy(tt, np.maximum(V_hist, 1e-12), linewidth=1.2)
axs[1].set_ylabel("V_r")
axs[1].set_xlabel("Time (s)")
for ax in axs:
    ax.grid(True)
fig.suptitle("CRaCLF-QP")

# Plot ACP variables.
fig, axs = plt.subplots(3, 1, sharex=True)
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
fig.suptitle("Adaptive conformal prediction")

# Plot adaptive and interval-fitted parameters.
fig, axs = plt.subplots(system.adim, 1, sharex=True)
for i in range(system.adim):
    axs[i].plot(tt, a_hat_clf_hist[:, i], label="a_hat")
    axs[i].plot(tt, a_k_hist[:, i], "--", label="a_k")
    axs[i].set_ylabel(f"a{i + 1}")
    axs[i].grid(True)
    axs[i].legend()
axs[-1].set_xlabel("Time (s)")
fig.suptitle("Adaptive and interval-fitted parameters")

# Plot the learned representation.
fig, axs = plt.subplots(Theta_init.size, 1, sharex=True, figsize=(7, 10))
for i in range(Theta_init.size):
    axs[i].plot(tt, Theta_hist[:, i])
    axs[i].set_ylabel(f"theta{i + 1}")
    axs[i].grid(True)
axs[-1].set_xlabel("Time (s)")
fig.suptitle("Learned representation")

# Plot scaling state and scaling function.
fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].plot(tt, nu_clf_hist)
axs[0].set_ylabel("nu_clf")
axs[1].plot(tt, rho_clf_hist)
axs[1].set_ylabel("rho_clf")
axs[1].set_xlabel("Time (s)")
for ax in axs:
    ax.grid(True)

plt.tight_layout()
plt.show()
