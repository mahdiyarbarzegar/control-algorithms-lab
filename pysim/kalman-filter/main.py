import numpy as np
import matplotlib.pyplot as plt
from lib import kalman_filter as kf

# ============================================================
# System parameters
# ============================================================

m = 1.0  # mass [kg]
c = 0.8  # damping [N.s/m]
k = 4.0  # spring constant [N/m]
Ts = 0.01  # sampling time [s]

sim_time = 10.0
N = int(sim_time / Ts)
time = np.zeros(N)

# ============================================================
# Continuous-time model
# ============================================================

Ac = np.array([
    [0.0, 1.0],
    [-k / m, -c / m]
])

Bc = np.array([
    [0.0],
    [1.0 / m]
])

C = np.array([
    [1.0, 0.0],
    [0.0, 1.0]
])

# ============================================================
# Discrete-time model using forward Euler
# x[k] = A x[k-1] + B u[k]
# ============================================================

A = np.eye(2) + Ts * Ac
B = Ts * Bc
H = C

# ============================================================
# Noise covariance matrices
# ============================================================

Q = np.array([
    [1e-5, 0.0],
    [0.0, 1e-3]
])

R = np.array([
    [0.03 ** 2, 0.0],
    [0.0, 0.10 ** 2]
])

kalman = kf.KalmanFilter(A, B, H, R, Q)

x_true_history = np.zeros((A.shape[0], N))
x_meas_history = np.zeros((A.shape[0], N))
x_prd_history = np.zeros((A.shape[0], N))
u_history = np.zeros(N)

x_true = np.zeros((A.shape[0], 1))
x_prd = np.zeros((A.shape[0], 1))
z_meas = np.zeros((H.shape[0], 1))

for i in range(N):
    t = i * Ts
    time[i] = t

    if t >= 1.0:
        u = 1.0
    else:
        u = 0.0

    u = np.array([[u]])
    u_history[i] = u[0, 0]

    # -----------------------------------
    # True System simulation
    # -----------------------------------
    process_noise = np.random.multivariate_normal(
        mean=np.zeros(x_true.shape[0]), cov=Q
    ).reshape(x_true.shape[0], 1)

    x_true = A @ x_true + B @ u + process_noise

    # -----------------------------------
    # Measurement simulation
    # -----------------------------------
    measurement_noise = np.random.multivariate_normal(
        mean=np.zeros(x_true.shape[0]), cov=R
    ).reshape(z_meas.shape[0], 1)

    z_meas = H @ x_true + measurement_noise

    x_prd = kalman.estimate(z_meas, u)

    # -----------------------------------
    # Store Results
    # -----------------------------------
    x_true_history[:, i] = x_true.flatten()
    x_meas_history[:, i] = z_meas.flatten()
    x_prd_history[:, i] = x_prd.flatten()

# -----------------------------------
# Plot Results
# -----------------------------------
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(time, x_true_history[0, :], label="True Position")
plt.plot(time, x_meas_history[0, :], '.', markersize=2, alpha=0.4, label="Measured Position")
plt.plot(time, x_prd_history[0, :], label="Estimated Position")
plt.ylabel("Position (m)")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time, x_true_history[1, :], label="True Velocity")
plt.plot(time, x_meas_history[1, :], '.', markersize=2, alpha=0.4, label="Measured Velocity")
plt.plot(time, x_prd_history[1, :], label="Estimated Velocity")
plt.ylabel("Velocity (m/s)")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(time, u_history[:], label="Input Force")
plt.xlabel("Time (s)")
plt.ylabel("Input (N)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
