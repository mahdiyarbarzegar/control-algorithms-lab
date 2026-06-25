import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.signal import place_poles
from lib.luenberger_observer import LuenbergerObserver


def simulate_system():
    # Plant parameters (Mass-Spring-Damper)
    m, k, c = 1.0, 5.0, 0.5
    Ts = 0.1
    # 1. Define Continuous Dynamics (Mass-Spring-Damper)
    # m*q'' + c*q' + k*q = u
    Ac = np.array([[0, 1], [-k / m, -c / m]])
    Bc = np.array([[0], [1 / m]])
    Cc = np.array([[1, 0]])  # Measure position
    Dc = np.array([[0]])
    Ec = np.array([[0.1], [0.1]])  # Disturbance matrix

    ns = 2
    nu = 1
    nd = 1
    no = 1

    # 2. Discretization
    dt = Ts
    ns = Ac.shape[0]
    # Simple discretization: Ad = exp(Ac*dt), Bd = integral(exp(Ac*tau) dtau)*Bc
    Ad = expm(Ac * dt)
    # Approximation for Bd:
    Bd = np.linalg.solve(Ac, (Ad - np.eye(ns)) @ Bc)
    Cd = Cc
    Dd = Dc
    Ed = np.linalg.solve(Ac, (Ad - np.eye(ns)) @ Ec)

    # Noise setup
    meas_noise_std = 0.01  # 1cm measurement noise

    # 3. Analyze Plant Poles (Open Loop)
    plant_poles = np.linalg.eigvals(Ad)
    print(f"Discrete Plant Poles: {plant_poles}")
    print(f"Magnitudes: {np.abs(plant_poles)}")

    # 4. Design Observer Gain L
    # We want observer poles to be faster than plant poles (closer to origin)
    # Plant poles are around 0.95 +/- 0.21j. Let's pick [0.5, 0.6]
    desired_poles = np.array([0.4, 0.5])

    # Duality: L is designed by placing poles of (A - LC)
    # Equivalent to placing poles of (A.T - C.T * L.T)
    placement = place_poles(Ad.T, Cd.T, desired_poles)
    L = placement.gain_matrix.T
    print(f"Calculated L:\n{L}")

    # 5. Initialize Observer
    obs = LuenbergerObserver(Ad, Bd, Cd, Dd, Ed, L)

    # 6. Simulation Loop
    steps = 100
    x = np.array([[1.0], [0.0]])  # True initial state (displaced)

    u = np.ones((1, 1)) * 0.5  # Constant input
    d = np.zeros((1, 1))  # No disturbance for this test

    history_x = []
    history_x_hat = []

    for _ in range(steps):
        # model a random measurment noise
        v = np.random.normal(0, meas_noise_std)
        # Measure from plant
        y_meas = Cd @ x + Dd @ u + v

        # Update Observer
        x_hat = obs.estimate(y_meas, u, d)

        # Save history
        history_x.append(x.flatten())
        history_x_hat.append(x_hat.flatten())

        # Update Plant (Physics)
        x = Ad @ x + Bd @ u + Ed @ d

    history_x = np.array(history_x)
    history_x_hat = np.array(history_x_hat)
    error = history_x - history_x_hat

    # 7. Visualization
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(history_x[:, 0], 'b-', label='True Position (x1)')
    plt.plot(history_x_hat[:, 0], 'r--', label='Est Position (x1_hat)')
    plt.plot(history_x[:, 1], 'g-', label='True Velocity (x2)')
    plt.plot(history_x_hat[:, 1], 'm--', label='Est Velocity (x2_hat)')
    plt.title("Luenberger Observer: State Tracking")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(error[:, 0], label='Error x1')
    plt.plot(error[:, 1], label='Error x2')
    plt.title("Estimation Error (x - x_hat)")
    plt.xlabel("Steps")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    simulate_system()
