from warnings import warn

import numpy as np
import numpy.typing as npt
from numba import njit, prange


class ConvergenceError(Exception):
    pass


class Solver1D:
    stability_criterion = 0.9

    def __init__(
            self,
            T: npt.ArrayLike,
            k: float | npt.ArrayLike,
            rho: float | npt.ArrayLike,
            cp: float | npt.ArrayLike,
            dx: float
    ):
        self.T = np.asarray(T)
        if not np.issubdtype(T.dtype, np.float):
            warn(f"Temperature array dtype ({self.T.dtype}) changed to float64 "
                 f"to avoid possible truncation.")
            self.T = self.T.astype(np.float64)

        self.k = np.full(T.shape, k) if isinstance(k, (float, int)) else np.asarray(k)
        self.rho = np.full(T.shape, rho) if isinstance(rho, (float, int)) else np.asarray(rho)
        self.cp = np.full(T.shape, cp) if isinstance(cp, (float, int)) else np.asarray(cp)
        self.dx = dx

        if not (self.T.shape == self.k.shape == self.rho.shape == self.k.shape):
            raise ValueError("Arrays must be the same length")

    @staticmethod
    @njit
    def step(T_n, k, rho, cp, dx, dt):
        T = np.copy(T_n)
        for i in prange(1, T.size - 1):
            T[i] = T_n[i] + dt / (rho[i] * cp[i] * dx**2) * (
                k[i] * (T_n[i-1] - 2*T_n[i] + T_n[i+1]) +
                0.25 * (k[i+1] - k[i-1]) * (T_n[i+1] - T_n[i-1])  # Non-linear conduction term
            )
        return T

    @property
    def max_time_step(self):
        return self.dx**2 / (2 * np.max(self.k / (self.rho * self.cp)))

    def advance_steady_state(self, *, dt: float = None, max_iter=1e6, rtol: float = 1e-6):
        for i in range(int(max_iter)):
            T = self.step(self.T, self.k, self.rho, self.cp, self.dx,
                          dt if dt is not None else self.stability_criterion * self.max_time_step)
            eps = np.max(np.abs(self.T - T) / self.T)
            self.T = T

            if eps <= rtol:
                return i + 1, eps

        raise ConvergenceError(f"Maximum number of iterations reached without relative error ({eps:.3e}) "
                               f"meeting convergence criterion ({rtol:.3e}).")


class Solver2D:
    stability_criterion = 0.9

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

    @staticmethod
    @njit
    def step(T_n, k, rho, cp, dx, dy, dt):
        T = np.copy(T_n)
        for j in prange(1, T.shape[0]):
            for i in prange(1, T.shape[1]):
                T[i, j] = T_n[i, j] + dt / (rho[i, j] * cp[i, j]) * (
                    k[i, j] * ((T_n[i-1, j] - 2 * T_n[i, j] + T_n[i+1, j]) / dx**2 +
                               (T_n[i, j-1] - 2 * T_n[i, j] + T_n[i, j+1]) / dy**2) +
                    # Non-linear conduction term
                    0.25 * ((k[i+1, j] - k[i-1, j]) * (T_n[i+1, j] - T_n[i-1, j]) / dx**2 +
                            (k[i, j+1] - k[i, j-1]) * (T_n[i, j+1] - T_n[i, j-1]) / dy**2)
                )
        return T

    def advance_steady_state(self, *, dt: float = None, max_iter=1e6, rtol: float = 1e-6):
        for i in range(int(max_iter)):
            T = self.step(self.T, self.k, self.rho, self.cp, self.dx,
                          dt if dt is not None else self.stability_criterion * self.max_time_step)
            eps = np.max(np.abs(self.T - T) / self.T)
            self.T = T

            if eps <= rtol:
                return i + 1, eps

        raise ConvergenceError(f"Maximum number of iterations reached without relative error ({eps:.3e}) "
                               f"meeting convergence criterion ({rtol:.3e}).")

    @property
    def max_time_step(self):
        return 1 / (2 * np.max(self.k / (self.rho * self.cp)) * (self.dx**-2 + self.dy**-2))
