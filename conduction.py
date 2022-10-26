import numpy as np
import numpy.typing as npt
from numba import njit, prange


class Solver1D:
    def __init__(
            self,
            T: npt.ArrayLike,
            k: float | npt.ArrayLike,
            rho: float | npt.ArrayLike,
            cp: float | npt.ArrayLike,
            dx: float
    ):
        self.T = np.asarray(T)
        self.k = np.full(T.shape, k) if isinstance(k, (float, int)) else np.asarray(k)
        self.rho = np.full(T.shape, rho) if isinstance(rho, (float, int)) else np.asarray(rho)
        self.cp = np.full(T.shape, cp) if isinstance(cp, (float, int)) else np.asarray(cp)
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


class Solver2D:
    def __init__(
            self,
            T: npt.ArrayLike,
            k: float | npt.ArrayLike,
            rho: float | npt.ArrayLike,
            cp: float | npt.ArrayLike,
            dx: float,
            dy: float = None
    ):
        self.T = np.asarray(T)
        self.k = np.full(T.shape, k) if isinstance(k, (float, int)) else np.asarray(k)
        self.rho = np.full(T.shape, rho) if isinstance(rho, (float, int)) else np.asarray(rho)
        self.cp = np.full(T.shape, cp) if isinstance(cp, (float, int)) else np.asarray(cp)
        self.dx = dx
        self.dy = dy if dy is not None else dx

        if not (self.T.shape == self.k.shape == self.rho.shape == self.k.shape):
            raise ValueError("Arrays must have the same shape")

    @njit
    def step(self, dt):
        T_n = np.copy(self.T)
        for j in prange(1, self.T.shape[0]):
            for i in prange(1, self.T.shape[1]):
                self.T[i, j] = T_n[i, j] + dt / (self.rho[i, j] * self.cp[i, j]) * (
                    self.k[i, j] * ((T_n[i-1, j] - 2 * T_n[i, j] + T_n[i+1, j]) / self.dx**2 +
                                    (T_n[i, j-1] - 2 * T_n[i, j] + T_n[i, j+1]) / self.dy**2) +
                    # Non-linear conduction term
                    0.25 * ((self.k[i+1, j] - self.k[i-1, j]) * (T_n[i+1, j] - T_n[i-1, j]) / self.dx**2 +
                            (self.k[i, j+1] - self.k[i, j-1]) * (T_n[i, j+1] - T_n[i, j-1]) / self.dy**2)
                )
