import numpy as np
import numpy.typing as npt
from numba import njit, prange


class Solver1D:
    def __init__(
            self,
            T: npt.ArrayLike[float],
            k: float | npt.ArrayLike[float],
            rho: float | npt.ArrayLike[float],
            cp: float | npt.ArrayLike[float],
            dx: float
    ):
        self.T = np.asarray(T)
        self.k = np.full(T.shape, k) if isinstance(k, float) else np.asarray(k)
        self.rho = np.full(T.shape, rho) if isinstance(rho, float) else np.asarray(rho)
        self.cp = np.full(T.shape, cp) if isinstance(cp, float) else np.asarray(cp)
        self.dx = dx

        if not (self.T.shape == self.k.shape == self.rho.shape == self.k.shape):
            raise ValueError("Arrays must be the same length")

    @njit
    def step(self, dt):
        T_n = np.copy(self.T)
        for i in prange(1, self.T.size - 1):
            self.T[i] = T_n[i] + dt / (self.rho[i] * self.cp[i] * self.dx**2) * (
                self.k[i] * (T_n[i-1] - 2*T_n[i] + T_n[i+1]) +
                0.25 * (self.k[i+1] - self.k[i-1]) * (T_n[i+1] - T_n[i-1])  # Non-linear conduction term
            )

