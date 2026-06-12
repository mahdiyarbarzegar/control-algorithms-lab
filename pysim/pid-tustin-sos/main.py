import numpy as np
import matplotlib.pyplot as plt
from lib import pid_tustin_method as pid, van_loan_zoh as vloan

# Continuous State-Space Model
cssm_A = np.array([[0, 1], [-5, -2]])
cssm_B = np.array([[0], [1]])
ccsm_C = np.array([[1, 0]])

# sampling period
Ts = 0.01

# simulation time [s]
SimTime = 10

# descritizing the continuous state-space model
dssm_A, dssm_B = vloan.discretize_vanloan_zoh_method(cssm_A, cssm_B, Ts)
dssm_C = ccsm_C

# PID coefficients
pid_coeff = {
    'Kp': 2,
    'Ki': 9.5,
    'Kd': 1.6,
    'Ka': 0.1,
    'N': 5
}

fixed_point = {
    'BitLength': 32,
    'FracLength': 15,
}

# setpoint
r = 1

# simulation params
N = int(SimTime / Ts)
y = np.zeros(N)
y_fp = np.zeros(N)
e = np.zeros(N)
e_fp = np.zeros(N)
u = np.zeros(N)
u_fp = np.zeros(N)
x = np.zeros((dssm_A.shape[0], 1))
x_fp = np.zeros((dssm_A.shape[0], 1))

pid_ctrl = pid.PidCalcTustin(
    Ts,
    pid_coeff['Kp'],
    pid_coeff['Ki'],
    pid_coeff['Kd'],
    pid_coeff['Ka'],
    pid_coeff['N'],
    fixed_point['BitLength'],
    fixed_point['FracLength'],
    -10,
    10
)

for k in range(1, N):
    e[k] = r - y[k - 1]
    e_fp[k] = r - y_fp[k - 1]
    u[k] = pid_ctrl.calc_u(e[k])
    u_fp[k] = pid_ctrl.calc_u_fp(e_fp[k])
    x = dssm_A @ x + dssm_B * u[k]
    x_fp = dssm_A @ x + dssm_B * u_fp[k]
    y[k] = (dssm_C @ x).item()
    y_fp[k] = (dssm_C @ x_fp).item()

t = np.arange(N) * Ts
_, ax = plt.subplots(2, 1)

ax[0].plot(t, y, label='Output')
# ax[0].plot(t, u, label='U Plant')
ax[0].plot(t, np.ones(N) * r, label='Setpoint')
ax[0].legend()
ax[0].grid(True)

ax[1].plot(t, y_fp, label='Output')
# ax[1].plot(t, u_fp, label='U Plant')
ax[1].plot(t, np.ones(N) * r, label='Setpoint')
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()
