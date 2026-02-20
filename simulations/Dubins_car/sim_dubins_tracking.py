import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from dynsys.Dubins_car.Dubins_uncertain import DUBINS_UNCERTAIN
from dynsys.Dubins_car.Dubins_uncertain_sindy import DUBINS_UNCERTAIN_SINDY
from dynsys.utils import wrapToPi
import pickle

USE_CP = 1 # 1 or 0: whether to use conformal prediction
USE_ADAPTIVE = 1 # 1 or 0: whether to use adaptive control

# Load the SINDy model ##########################################################################
with open('sindy_models/model_dubins_car_sindy', 'rb') as file:
    model = pickle.load(file)

feature_names = model["feature_names"]
n_features = len(feature_names)
for i in range(n_features):
    feature_names[i] = feature_names[i].replace(" ", "*")
    feature_names[i] = feature_names[i].replace("^", "**")
    feature_names[i] = feature_names[i].replace("sin", "np.sin")
    feature_names[i] = feature_names[i].replace("cos", "np.cos")

coefficients = model["coefficients"]

idx_x = [] # Indices for f(x)
idx_u = [] # Indices for g(x)*u
for i in range(len(feature_names)):
    if 'u0' in feature_names[i]:
        idx_u.append(i)
    else:
        idx_x.append(i)
        
cp_quantile = model["model_error"]['quantile']
print("cp_quantile = ", cp_quantile)

cp_quantile = cp_quantile * USE_CP # setting cp_quantile = 0 is equivalent to using the regular cbf

# Time setup
dt = 0.002
sim_T = 10 # Simulation time
tt = np.arange(0, sim_T, dt)

# System parameters
params = {
    "v": 1.0,
    "target": {"x": 0, "y": 0},
    "obstacle": {"x": 5, "y": 4, "r": 2},
    "cbf": {"rate": 1, "gamma": 15},
    "weight": {"input": 10000},
    "K": 10,  # P gain for the nominal controller
}
params["a_true"] = np.array([
    [0.05],
    [0.05], 
    [0.05]
]) # true a(Theta)
params["K_track"] = np.array([[10, 0, 0],
                               [0, 20, 0],
                               [0, 0, 10]])

# True system
Dubins_true = DUBINS_UNCERTAIN(params)

params["feature_names"] = feature_names
params["coefficients"] = coefficients
params["idx_x"] = idx_x
params["idx_u"] = idx_u
params["use_adaptive"] = USE_ADAPTIVE
params["Gamma_t"] = np.eye(3) * 1 # adaptive gain matrix for CRa-tracking
#params["Gamma_b"] = np.eye(3) * 0.01 # adaptive gain matrix for CRaCBF
#params["Gamma_L"] = np.eye(3) * 0.01 # adaptive gain matrix for CRaCLF
params["a_hat_norm_max"] = np.linalg.norm(np.array([[0.1], [0.1], [0.1]]), 2) # max norm of a_hat
params["a_0"] = np.array([[0.0], [0.0], [0.0]]) # initial guess for a_hat
params["epsilon"] = 1e-2 # small value for numerical stability of projection operator

# Learned mode
Dubins_learned = DUBINS_UNCERTAIN_SINDY(params)

# Generate xd by simulation
xd0 = np.array([-10, 1, -3.0])
xd = np.zeros((len(tt), 3))
ud = np.sin(2*np.pi*0.2*tt)
x = np.copy(xd0)
for k in range(len(tt)):
    x[2] = wrapToPi(x[2])
    xd[k, :] = x

    # Propagate dynamics
    dx = Dubins_true.dynamics(x, ud[k].reshape(-1, 1))
    x = x + dx.reshape(-1) * dt
xd_dot = np.diff(xd, axis=0) / dt
xd_dot = np.vstack((xd_dot, xd_dot[-1,:]))
xd_dot[:, 2] = wrapToPi(xd_dot[:, 2])

# History arrays
x_hist = np.zeros((len(tt), 3))
u_hist = np.zeros((len(tt)))
a_hat = np.zeros((len(tt), 3))
V_hist = np.zeros((len(tt)))
V_dot_hist = np.zeros((len(tt)))

x0 = np.array([-11, 2, 0.0])
x = np.copy(x0)

for k in range(len(tt)):
    x[2] = wrapToPi(x[2])
    x_hist[k, :] = x

    # Control
    a_hat[k, :] = Dubins_learned.a_t_hat[:,0]
    e = x - xd[k,:]
    e[2] = wrapToPi(e[2]) # Wrap the angle error to [-pi, pi]
    u, V, V_dot = Dubins_learned.ctrl_cra_tracking(x, e.reshape(-1,1), xd_dot[k,:], cp_quantile, dt)

    u_hist[k] = u.item()
    V_hist[k] = V
    V_dot_hist[k] = V_dot

    # Propagate dynamics
    dx = Dubins_true.dynamics(x, u, param_uncertainty=True)
    x = x + dx.reshape(-1) * dt

# === Plots ===
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(tt, x_hist[:, 0].T)
plt.plot(tt, xd[:, 0], 'r--')
plt.ylabel("p_x")
plt.grid(True)
plt.subplot(3, 1, 2)
plt.plot(tt, x_hist[:, 1].T)
plt.plot(tt, xd[:, 1], 'r--')
plt.ylabel("p_y")
plt.grid(True)
plt.subplot(3, 1, 3)
plt.plot(tt, x_hist[:, 2].T)
plt.plot(tt, xd[:, 2], 'r--')
plt.ylabel("theta")
plt.grid(True)

plt.figure()
plt.plot(x0[0], x0[1], 'ro', markersize=10)
plt.plot(xd[:,0], xd[:,1], 'r')
plt.plot(x_hist[:, 0], x_hist[:, 1])
plt.axis('equal')
plt.grid(True)

plt.figure()
plt.plot(tt, u_hist.T)
plt.ylabel("Control Input: u (N)")
plt.grid(True)

plt.figure()
plt.plot(tt, V_hist.T)
plt.ylabel("V")
plt.grid(True)

plt.figure()
plt.plot(tt, V_dot_hist.T, 'r--')
#plt.plot(tt[:-1], np.diff(V_hist) / dt)
plt.ylabel("V_dot Upper Bound")
plt.grid(True)

plt.figure()
plt.subplot(4, 2, 1)
plt.plot(tt, a_hat[:, 0].T)
plt.axhline(Dubins_learned.a_err_max[0,0], color='r', linewidth=2)
plt.axhline(-Dubins_learned.a_err_max[0,0], color='r', linewidth=2)
plt.ylabel("a0_hat")
plt.grid(True)
plt.subplot(4, 2, 3)
plt.plot(tt, a_hat[:, 1].T)
plt.axhline(Dubins_learned.a_err_max[1,0], color='r', linewidth=2)
plt.axhline(-Dubins_learned.a_err_max[1,0], color='r', linewidth=2)
plt.ylabel("a1_hat")
plt.grid(True)
plt.subplot(4, 2, 5)
plt.plot(tt, a_hat[:, 2].T)
plt.axhline(Dubins_learned.a_err_max[2,0], color='r', linewidth=2)
plt.axhline(-Dubins_learned.a_err_max[2,0], color='r', linewidth=2)
plt.ylabel("a2_hat")
plt.grid(True)
plt.subplot(4, 2, 7)
plt.plot(tt, np.linalg.norm(a_hat[:, :], axis=1, ord=2).T)
plt.axhline(Dubins_learned.a_hat_norm_max + Dubins_learned.epsilon, color='r', linewidth=2)
plt.ylabel("a_hat_norm")
plt.grid(True)

#plt.figure()
plt.subplot(4, 2, 2)
plt.plot(tt, a_hat[:, 0].T - params["a_true"][0,0])
plt.axhline(Dubins_learned.a_err_max[0,0], color='r', linewidth=2)
plt.axhline(-Dubins_learned.a_err_max[0,0], color='r', linewidth=2)
plt.ylabel("a0 error")
plt.grid(True)
plt.subplot(4, 2, 4)
plt.plot(tt, a_hat[:, 1].T - params["a_true"][1,0])
plt.axhline(Dubins_learned.a_err_max[1,0], color='r', linewidth=2)
plt.axhline(-Dubins_learned.a_err_max[1,0], color='r', linewidth=2)
plt.ylabel("a1 error")
plt.grid(True)
plt.subplot(4, 2, 6)
plt.plot(tt, a_hat[:, 2].T - params["a_true"][2,0])
plt.axhline(Dubins_learned.a_err_max[2,0], color='r', linewidth=2)
plt.axhline(-Dubins_learned.a_err_max[2,0], color='r', linewidth=2)
plt.ylabel("a2 error")
plt.grid(True)

plt.show()