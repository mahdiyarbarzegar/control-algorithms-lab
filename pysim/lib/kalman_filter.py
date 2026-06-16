import numpy as np


class KalmanFilter:
    def __init__(self, A, B, H, R, Q):
        self.transition_size = A.shape[0]
        self.input_size = B.shape[1]
        self.observation_size = H.shape[0]

        if A.shape[0] != A.shape[1]:
            raise ValueError("State transition matrix A is not square")

        if B.shape[0] != A.shape[0]:
            raise ValueError("Row size of the input matrix B in not equal to the transition size")

        if H.shape[1] != A.shape[0]:
            raise ValueError("Column size of the observation matrix H is not equal to the transition size")

        self.A = A
        self.B = B
        self.H = H
        self.P = np.zeros_like(self.A)
        self.K = np.zeros((self.transition_size, self.observation_size))
        self.I = np.eye(self.transition_size)
        self.set_q(Q)
        self.set_r(R)
        self.x_estimate = np.zeros((self.transition_size, 1))

    def set_q(self, Q):
        if Q.shape[0] != self.transition_size or Q.shape[1] != self.transition_size:
            raise ValueError("Q matrix size should be the same as the transition matrix A")
        self.Q = Q

    def set_r(self, R):
        if R.shape[0] != self.observation_size or R.shape[1] != self.observation_size:
            raise ValueError("R matrix size should be the same as the observation matrix H")
        self.R = R

    def estimate(self, z, u, R=None, Q=None):
        if R is not None:
            self.set_r(R)
        if Q is not None:
            self.set_q(Q)

        z = z.reshape(self.observation_size, 1)
        u = u.reshape(self.input_size, 1)

        # -----------------------------------
        # Kalman Filter Prediction
        # -----------------------------------
        x_prediction = self.A @ self.x_estimate + self.B @ u
        p_prediction = self.A @ self.P @ self.A.T + self.Q

        # -----------------------------------
        # Kalman Filter Correction
        # -----------------------------------
        s = self.H @ p_prediction @ self.H.T + self.R
        self.K = p_prediction @ self.H.T @ np.linalg.solve(s, np.eye(self.observation_size))

        residual = z - self.H @ x_prediction
        self.x_estimate = x_prediction + self.K @ residual

        i_kh = self.I - self.K @ self.H
        self.P = i_kh @ p_prediction @ i_kh.T + self.K @ self.R @ self.K.T

        return self.x_estimate
