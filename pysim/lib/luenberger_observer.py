import numpy as np


class LuenbergerObserver:
    def __init__(self, A, B, C, D, E, L):
        self.ns = A.shape[0]
        self.nu = B.shape[1]
        self.nd = E.shape[1]
        self.no = C.shape[0]

        if A.shape[0] != A.shape[1]:
            raise ValueError("A matrix should be square")

        if B.shape != (self.ns, self.nu):
            raise ValueError(f"B matrix should have a shape {self.ns} x {self.nu}")

        if C.shape != (self.no, self.ns):
            raise ValueError(f"C matrix should have a shape {self.no} x {self.ns}")

        if D.shape != (self.no, self.nu):
            raise ValueError(f"D matrix should have a shape {self.no} x {self.nu}")

        if E.shape != (self.ns, self.nd):
            raise ValueError(f"E matrix should have a shape {self.ns} x {self.nd}")

        if L.shape != (self.ns, self.no):
            raise ValueError(f"L matrix should have a shape {self.ns} x {self.no}")

        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self.L = L

        if not self.__observability_check():
            raise ValueError("The system dynamic is not observable")

        self.x_hat = np.zeros((self.ns, 1))

    def __observability_check(self):
        observer_matrix = np.vstack([self.C @ np.linalg.matrix_power(self.A, i) for i in range(self.ns)])
        if np.linalg.matrix_rank(observer_matrix) < self.ns:
            return False
        else:
            return True

    def estimate(self, y_meas, u, d):
        if y_meas.shape != (self.no, 1):
            raise ValueError(f"y_meas matrix should have a shape {self.no} x 1")

        if u.shape != (self.nu, 1):
            raise ValueError(f"u matrix should have a shape {self.nu} x 1")

        if d.shape != (self.nd, 1):
            raise ValueError(f"d matrix should have a shape {self.nd} x 1")

        y_hat = self.C @ self.x_hat + self.D @ u
        innovation = y_meas - y_hat
        self.x_hat = self.A @ self.x_hat + self.B @ u + self.E @ d + self.L @ innovation

        return self.x_hat
