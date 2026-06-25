import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import place_poles
from lib.eimpc import EiMpcLuenbergerObs

# -----------------------------
# Model
# -----------------------------
ns = 4
nu = 4
nd = 2
no = 4

beta = 0.12

alpha_out = np.array([0.030, 0.025, 0.025, 0.030])
alpha_ground = np.array([0.010, 0.015, 0.015, 0.010])

c12 = 0.015
c23 = 0.015
c34 = 0.015

Am = np.array([
    [1 - beta - alpha_out[0] - alpha_ground[0] - c12, c12, 0.0, 0.0],
    [c12, 1 - beta - alpha_out[1] - alpha_ground[1] - c12 - c23, c23, 0.0],
    [0.0, c23, 1 - beta - alpha_out[2] - alpha_ground[2] - c23 - c34, c34],
    [0.0, 0.0, c34, 1 - beta - alpha_out[3] - alpha_ground[3] - c34],
])

Bm = beta * np.eye(4)
Cm = np.eye(4)

Em = np.column_stack([
    alpha_out,
    alpha_ground
])

# -----------------------------
# MPC
# -----------------------------
Np = 10
Nc = 10

# ----------------------------
# Kalman Filter
# ----------------------------
process_noise_std = 0.05  # °C per sample
measurement_noise_std = 0.1  # °C sensor noise
Q_true = process_noise_std ** 2 * np.eye(ns)
R_true = measurement_noise_std ** 2 * np.eye(no)

# ----------------------------
# Luenberger Observer
# ----------------------------
# Analyze Plant Poles (Open Loop)
plant_poles = np.linalg.eigvals(Am)
print(f"Discrete Plant Poles: {plant_poles}")
print(f"Magnitudes: {np.abs(plant_poles)}")

# Design Observer Gain L
desired_poles = np.array([0.4, 0.5, 0.45, 0.55])

# Duality: L is designed by placing poles of (A - LC)
# Equivalent to placing poles of (A.T - C.T * L.T)
placement = place_poles(Am.T, Cm.T, desired_poles)
L = placement.gain_matrix.T
print(f"Calculated L:\n{L}")

mpc = EiMpcLuenbergerObs(Am, Bm, Cm, Em, Np, Nc, L)

mpc.set_weights(
    r=0.1,
    q=1.0
)

rs = np.array([[22.0],
               [22.0],
               [22.0],
               [22.0]])

mpc.set_setpoint(rs)


# -----------------------------
# Helper functions
# -----------------------------
def outdoor_temperature(k, season):
    if season == "summer":
        return 32.0 + 4.0 * np.sin(2.0 * np.pi * k / 48.0)
    elif season == "winter":
        return 4.0 + 3.0 * np.sin(2.0 * np.pi * k / 48.0)
    else:
        raise ValueError("Invalid season")


def ground_temperature(k, season):
    if season == "summer":
        return 18.0 + 1.0 * np.sin(2.0 * np.pi * k / 96.0)
    elif season == "winter":
        return 10.0 + 1.0 * np.sin(2.0 * np.pi * k / 96.0)
    else:
        raise ValueError("Invalid season")


def build_disturbance_prediction(k, season, Np):
    d_pred = []

    for i in range(Np):
        kk = k + i

        tout = outdoor_temperature(kk, season)
        tground = ground_temperature(kk, season)

        d_pred.extend([tout, tground])

    return np.array(d_pred).reshape(-1, 1)


def update_hvac_constraints(mpc, y, season):
    if season == "summer":
        u_min = 10 * np.ones((4, 1))
        u_max = y.copy()

        # protect against impossible bound
        u_max = np.maximum(u_max, u_min)
    elif season == "winter":
        u_min = y.copy()
        u_max = 50.0 * np.ones((4, 1))

        # protect against impossible bound
        u_min = np.minimum(u_min, u_max)
    else:
        raise ValueError("season must be 'summer' or 'winter'")

    delta_u_max = 1.0 * np.ones((4, 1))
    delta_u_min = -1.0 * np.ones((4, 1))

    mpc.set_constraints(
        u_max=u_max,
        u_min=u_min,
        delta_u_max=delta_u_max,
        delta_u_min=delta_u_min
    )


def simulate_plant_state(x, u, d):
    # model a random process noise
    w = np.random.multivariate_normal(
        mean=np.zeros(ns),
        cov=Q_true
    ).reshape(4, 1)

    return Am @ x + Bm @ u + Em @ d + w


def measure_output_plant(x):
    # model a random measurment noise
    v = np.random.multivariate_normal(
        mean=np.zeros(no),
        cov=R_true
    ).reshape(4, 1)

    return Cm @ x + v


# -----------------------------
# Simulation
# -----------------------------
n_steps = 120
# season = "summer"
season = "winter"

if season == "summer":
    x = np.array([[28.0],
                  [29.0],
                  [27.0],
                  [28.5]])
elif season == "winter":
    x = np.array([[16.0],
                  [17.0],
                  [15.5],
                  [16.5]])
else:
    raise ValueError("Invalid season")

mpc.set_init_u(x)

x_hist = []
y_hist = []
u_hist = []
d_hist = []

for k in range(n_steps):
    if k == 60:
        if season == "summer":
            rs = rs - 6
        else:
            rs = rs + 6
        mpc.set_setpoint(rs)

    y = measure_output_plant(x)

    # Update MPC with current measurement
    mpc.observe(y)

    # Update measured disturbance prediction
    d_pred = build_disturbance_prediction(k, season, Np)
    mpc.set_disturbance(d_pred)
    d = d_pred[:nd]

    # Update dynamic actuator constraints
    update_hvac_constraints(mpc, y, season)

    # Solve MPC
    mpc.calc_u()
    u = mpc.get_u()

    # Simulate plant
    x = simulate_plant_state(x, u, d)

    x_hist.append(x.flatten())
    y_hist.append(y.flatten())
    u_hist.append(u.flatten())
    d_hist.append(d.flatten())

x_hist = np.array(x_hist)
y_hist = np.array(y_hist)
u_hist = np.array(u_hist)
d_hist = np.array(d_hist)

# -----------------------------
# Plot
# -----------------------------
t = np.arange(n_steps)

plt.figure(figsize=(12, 9))

plt.subplot(3, 1, 1)
for i in range(4):
    plt.plot(t, y_hist[:, i], label=f"Room {i + 1}")
plt.axhline(22.0, color="k", linestyle="--", label="Setpoint")
plt.ylabel("Room temperature [°C]")
plt.title(f"4-Room HVAC MPC Simulation - {season}")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
for i in range(4):
    plt.plot(t, u_hist[:, i], label=f"HVAC setpoint {i + 1}")
plt.ylabel("HVAC setpoint [°C]")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, d_hist[:, 0], label="Outdoor temperature")
plt.plot(t, d_hist[:, 1], label="Ground temperature")
plt.ylabel("Disturbance [°C]")
plt.xlabel("Time step")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
