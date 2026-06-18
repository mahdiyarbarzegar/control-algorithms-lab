import numpy as np
import matplotlib.pyplot as plt
from lib.eimpc import EiMpc

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

mpc = EiMpc(Am, Bm, Cm, Em, Np, Nc)

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


# -----------------------------
# Simulation
# -----------------------------
n_steps = 120
season = "summer"
# season = "winter"

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

mpc.u = x #np.tile(x, (Nc, 1))

x_hist = []
y_hist = []
u_hist = []
d_hist = []

for k in range(n_steps):
    y = Cm @ x

    # Update MPC with current measurement
    mpc.observe(y)

    # Update measured disturbance prediction
    d_pred = build_disturbance_prediction(k, season, Np)
    mpc.set_disturbance(d_pred)

    # Update dynamic actuator constraints
    update_hvac_constraints(mpc, y, season)

    # Solve MPC
    mpc.calc_u()
    u = mpc.get_u()

    # Current disturbance
    tout = outdoor_temperature(k, season)
    tground = ground_temperature(k, season)
    d_now = np.array([[tout],
                      [tground]])

    # Simulate plant
    x = Am @ x + Bm @ u + Em @ d_now

    x_hist.append(x.flatten())
    y_hist.append(y.flatten())
    u_hist.append(u.flatten())
    d_hist.append(d_now.flatten())

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
