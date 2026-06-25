import numpy as np
from numpy.linalg import matrix_power
import osqp
from scipy import sparse


class EiMpc:
    def __init__(self, Am, Bm, Cm, Em, Np, Nc):
        self.ns = Am.shape[0]
        self.nu = Bm.shape[1]
        self.nd = Em.shape[1]
        self.no = Cm.shape[0]

        if Am.shape[0] != Am.shape[1]:
            raise ValueError("Am must be square")

        if Bm.shape[0] != self.ns:
            raise ValueError("Bm matrix rows should be equal to number of states")

        if Em.shape[0] != self.ns:
            raise ValueError("Em matrix rows should be equal to number of states")

        if Cm.shape[1] != self.ns:
            raise ValueError("Cm matrix columns should be equal to number of states")

        if Nc > Np:
            raise ValueError("The Control Horizon (Nc) should be less than or equal to the Prediction Horizon (Np)")

        self.Am = Am.copy()
        self.Bm = Bm.copy()
        self.Cm = Cm.copy()
        self.Em = Em.copy()

        self.Np = Np
        self.Nc = Nc

        self.y = np.zeros((self.no, 1))
        self.xm = np.zeros((self.ns, 1))
        self.delta_xm = np.zeros((self.ns, 1))
        self.u = np.zeros((self.nu, 1))
        self.d = np.zeros((self.Np * self.nd, 1))
        self.delta_d = np.zeros((self.Np * self.nd, 1))

        self.A = np.zeros((self.ns + self.no, self.ns + self.no))
        self.B = np.zeros((self.ns + self.no, self.nu))
        self.C = np.zeros((self.no, self.ns + self.no))
        self.E = np.zeros((self.ns + self.no, self.nd))

        self.F = np.zeros((self.Np * self.no, self.ns + self.no))
        self.Phi = np.zeros((self.Np * self.no, self.Nc * self.nu))
        self.Z = np.zeros((self.Np * self.no, self.Np * self.nd))

        self.R = np.zeros((self.Nc * self.nu, self.Nc * self.nu))
        self.Q = np.zeros((self.Np * self.no, self.Np * self.no))

        self.Rs = np.zeros((self.Np * self.no, 1))

        self.cumulative_constr = np.zeros((self.Nc * self.nu, self.Nc * self.nu))
        self.u_max = np.zeros((self.nu, 1))
        self.delta_u_max = np.zeros((self.nu, 1))
        self.u_min = np.zeros((self.nu, 1))
        self.delta_u_min = np.zeros((self.nu, 1))

        self._calc_augmented_model()
        self._calc_compact_matrices_form()
        if np.linalg.matrix_rank(self.__observability_matrix()) < self.ns:
            raise ValueError("The Observability Matrix rank should be equal or grater than the number of states")

    def __observability_matrix(self):
        return np.vstack([self.Cm @ np.linalg.matrix_power(self.Am, i) for i in range(self.ns)])

    def __calc_augmented_model(self):
        self.A[0:self.ns, 0:self.ns] = self.Am
        self.A[0:self.ns, self.ns:] = np.zeros((self.ns, self.no))
        self.A[self.ns:, 0:self.ns] = self.Cm @ self.Am
        self.A[self.ns:, self.ns:] = np.eye(self.no)

        self.B[0:self.ns, :] = self.Bm
        self.B[self.ns:, :] = self.Cm @ self.Bm

        self.C[:, 0:self.ns] = np.zeros((self.no, self.ns))
        self.C[:, self.ns:] = np.eye(self.no)

        self.E[0:self.ns, :] = self.Em
        self.E[self.ns:, :] = self.Cm @ self.Em

    def _calc_compact_matrices_form(self):
        for i in range(self.Np):
            self.F[i * self.no:(i + 1) * self.no, :] = self.C @ matrix_power(self.A, i + 1)

        for i in range(self.Np):
            for j in range(self.Nc):
                if i >= j:
                    self.Phi[i * self.no:(i + 1) * self.no, j * self.nu:(j + 1) * self.nu] \
                        = self.C @ matrix_power(self.A, i - j) @ self.B

            for k in range(self.Np):
                if i >= k:
                    self.Z[i * self.no:(i + 1) * self.no, k * self.nd:(k + 1) * self.nd] \
                        = self.C @ matrix_power(self.A, i - k) @ self.E

        for i in range(self.Nc):
            for j in range(self.nu):
                for k in range(i + 1):
                    self.cumulative_constr[i * self.nu + j, k * self.nu + j] = 1

        self.A_constr = sparse.vstack([
            sparse.eye(self.Nc * self.nu),
            self.cumulative_constr
        ])

    def set_weights(self, r, q):
        self.R = np.diag(np.diag(float(r) * np.ones(self.R.shape)))
        self.Q = np.diag(np.diag(float(q) * np.ones(self.Q.shape)))

    def set_constraints(self, u_max, u_min, delta_u_max, delta_u_min):
        u_max = np.asarray(u_max, dtype=float)
        u_min = np.asarray(u_min, dtype=float)
        delta_u_max = np.asarray(delta_u_max, dtype=float)
        delta_u_min = np.asarray(delta_u_min, dtype=float)

        if u_max.shape == ():
            u_max = u_max * np.ones((self.nu, 1))
        if u_min.shape == ():
            u_min = u_min * np.ones((self.nu, 1))
        if delta_u_max.shape == ():
            delta_u_max = delta_u_max * np.ones((self.nu, 1))
        if delta_u_min.shape == ():
            delta_u_min = delta_u_min * np.ones((self.nu, 1))

        if u_max.shape != (self.nu, 1):
            raise ValueError("u_max must be scalar or vector of size nu")

        if u_min.shape != (self.nu, 1):
            raise ValueError("u_min must be scalar or vector of size nu")

        if delta_u_max.shape != (self.nu, 1):
            raise ValueError("delta_u_max must be scalar or vector of size nu")

        if delta_u_min.shape != (self.nu, 1):
            raise ValueError("delta_u_min must be scalar or vector of size nu")

        if np.any(u_min > u_max):
            raise ValueError("u_min must be less than or equal to u_max")

        if np.any(delta_u_min > delta_u_max):
            raise ValueError("delta_u_min must be less than or equal to delta_u_max")

        self.u_max = u_max
        self.u_min = u_min
        self.delta_u_max = delta_u_max
        self.delta_u_min = delta_u_min

    def set_setpoint(self, rs):
        if rs.shape != (self.no, 1):
            raise ValueError("Rs must be a vector of size no")

        for i in range(self.Np):
            self.Rs[i * self.no:(i + 1) * self.no, :] = rs

    def set_disturbance(self, d):
        if d.shape != self.d.shape:
            raise ValueError("disturbance vector must have a size of Np*nd")

        d_last = self.d.copy()
        self.d = d.copy()
        self.delta_d = self.d - d_last

    def observe(self, yk):
        if yk.shape != self.y.shape:
            raise ValueError("The input yk vector has difference with dynamic model")

        self.y = yk.copy()
        xm_last = self.xm.copy()
        self.xm = np.linalg.solve(self.Cm, self.y)
        self.delta_xm = self.xm - xm_last

    def calc_u(self):
        xk = np.concatenate((self.delta_xm, self.y), axis=0)

        p = 2.0 * (self.Phi.T @ self.Q @ self.Phi + self.R)
        p = sparse.csc_matrix((p + p.T) / 2.0)

        c = self.Phi.T @ (self.Q + self.Q.T) @ ((self.F @ xk) + (self.Z @ self.delta_d) - self.Rs)

        du_max = np.tile(self.delta_u_max, (self.Nc, 1))
        du_min = np.tile(self.delta_u_min, (self.Nc, 1))
        u_min_h = np.tile(self.u_min, (self.Nc, 1))
        u_max_h = np.tile(self.u_max, (self.Nc, 1))
        u0_h = np.tile(self.u, (self.Nc, 1))

        l_abs = u_min_h - u0_h
        u_abs = u_max_h - u0_h

        l = np.vstack([
            du_min,
            l_abs
        ])

        u = np.vstack([
            du_max,
            u_abs
        ])

        prob = osqp.OSQP()
        prob.setup(p, c.ravel(), self.A_constr, l, u, verbose=False)
        res = prob.solve()

        if res.info.status != "solved":
            raise RuntimeError(res.info.status)

        du = res.x.reshape(-1, 1)
        du0 = du[:self.nu]
        self.u += du0

    def get_u(self):
        return self.u
