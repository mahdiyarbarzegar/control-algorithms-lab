import numpy as np
from scipy.linalg import expm

"""
The Van Loan method is the standard numerical technique to compute the
exact ZOH discretization
Given A,B,Ts

A is n*n
B is n*m

M = [ A B
      0 0 ]
E = e^M*Ts
Ad = E[:n, :n]
Bd = E[:n, :n]
"""


def discretize_vanloan_zoh_method(A, B, Ts):
    n = A.shape[0]
    m = B.shape[1]

    M = np.zeros(((n + m), (n + m)))
    M[:n, :n] = A
    M[:n, n:] = B

    E = expm(M * Ts)

    Ad = E[:n, :n]
    Bd = E[:n, n:]

    return Ad, Bd
