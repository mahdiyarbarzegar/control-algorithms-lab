import numpy as np
from lib import fixed_point as fp


class KalmanFilter:
    def __init__(self, A, B, H, E, R, Q, fp_bl=32, fp_q=24):
        self.transition_size = A.shape[0]
        self.input_size = B.shape[1]
        self.observation_size = H.shape[0]
        self.measurable_disturbance_size = E.shape[1]

        if A.shape[0] != A.shape[1]:
            raise ValueError("State transition matrix A is not square")

        if B.shape[0] != A.shape[0]:
            raise ValueError("Row size of the input matrix B in not equal to the transition size")

        if H.shape[1] != A.shape[0]:
            raise ValueError("Column size of the observation matrix H is not equal to the transition size")

        if E.shape[0] != A.shape[0]:
            raise ValueError("Row size of the input matrix E in not equal to the transition size")

        self.fp_wrap_style = 'saturate'
        self.fp_bl = fp_bl
        self.fp_q = fp_q

        self.A = A
        self.A_fp = fp.FixedPointMat.from_float(A, self.fp_bl, self.fp_q, self.fp_wrap_style)

        self.B = B
        self.B_fp = fp.FixedPointMat.from_float(B, self.fp_bl, self.fp_q, self.fp_wrap_style)

        self.E = E
        self.E_fp = fp.FixedPointMat.from_float(E, self.fp_bl, self.fp_q, self.fp_wrap_style)

        self.H = H
        self.H_fp = fp.FixedPointMat.from_float(H, self.fp_bl, self.fp_q, self.fp_wrap_style)

        self.P = np.eye(self.transition_size)
        self.P_fp = fp.FixedPointMat.eye(self.transition_size, self.fp_bl, self.fp_q, self.fp_wrap_style)

        self.K = np.zeros((self.transition_size, self.observation_size))
        self.K_fp = fp.FixedPointMat.zeros(self.transition_size, self.observation_size, self.fp_bl, self.fp_q,
                                           self.fp_wrap_style)

        self.I = np.eye(self.transition_size)
        self.I_fp = fp.FixedPointMat.eye(self.transition_size, self.fp_bl, self.fp_q, self.fp_wrap_style)

        self.set_q(Q)
        self.set_r(R)

        self.x_estimate = np.zeros((self.transition_size, 1))
        self.x_estimate_fp = fp.FixedPointMat.zeros(self.transition_size, 1, self.fp_bl, self.fp_q, self.fp_wrap_style)

    def set_q(self, Q):
        if Q.shape[0] != self.transition_size or Q.shape[1] != self.transition_size:
            raise ValueError("Q matrix size should be the same as the transition matrix A")
        self.Q = Q
        self.Q_fp = fp.FixedPointMat.from_float(Q, self.fp_bl, self.fp_q, self.fp_wrap_style)

    def set_r(self, R):
        if R.shape[0] != self.observation_size or R.shape[1] != self.observation_size:
            raise ValueError("R matrix size should be the same as the observation matrix H")
        self.R = R
        self.R_fp = fp.FixedPointMat.from_float(R, self.fp_bl, self.fp_q, self.fp_wrap_style)

    def estimate(self, z, u, d, R=None, Q=None):
        if R is not None:
            self.set_r(R)
        if Q is not None:
            self.set_q(Q)

        z = z.reshape(self.observation_size, 1)
        u = u.reshape(self.input_size, 1)
        d = d.reshape(self.measurable_disturbance_size, 1)

        # -----------------------------------
        # Kalman Filter Prediction
        # -----------------------------------
        x_prediction = self.A @ self.x_estimate + self.B @ u + self.E @ d
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

    def estimate_fp(self, z, u, d, R=None, Q=None):
        if R is not None:
            self.set_r(R)
        if Q is not None:
            self.set_q(Q)

        z_fp = fp.FixedPointMat.from_float(z.reshape(self.observation_size, 1), self.fp_bl, self.fp_q,
                                           self.fp_wrap_style)
        u_fp = fp.FixedPointMat.from_float(u.reshape(self.input_size, 1), self.fp_bl, self.fp_q, self.fp_wrap_style)
        d_fp = fp.FixedPointMat.from_float(d.reshape(self.measurable_disturbance_size, 1), self.fp_bl, self.fp_q,
                                           self.fp_wrap_style)

        # -----------------------------------
        # Kalman Filter Prediction
        # -----------------------------------
        x_prediction_fp = self.A_fp @ self.x_estimate_fp + self.B_fp @ u_fp + self.E_fp @ d_fp
        p_prediction_fp = self.A_fp @ self.P_fp @ self.A_fp.T + self.Q_fp

        # -----------------------------------
        # Kalman Filter Correction
        # -----------------------------------
        s_fp = self.H_fp @ p_prediction_fp @ self.H_fp.T + self.R_fp

        s_inv_fp = s_fp.inv()

        self.K_fp = p_prediction_fp @ self.H_fp.T @ s_inv_fp

        residual_fp = z_fp - self.H_fp @ x_prediction_fp
        self.x_estimate_fp = x_prediction_fp + self.K_fp @ residual_fp

        i_kh_fp = self.I_fp - self.K_fp @ self.H_fp
        self.P_fp = i_kh_fp @ p_prediction_fp @ i_kh_fp.T + self.K_fp @ self.R_fp @ self.K_fp.T

        return self.x_estimate_fp.to_float()
