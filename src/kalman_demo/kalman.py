import numpy as np
from numpy.typing import NDArray


class KalmanFilter:
    """
    Linear Kalman filter implementation.

    State equation:
        x_k = F x_{k-1} + w
    Measurement equation:
        z_k = H x_k + v
    """

    def __init__(
        self,
        F: NDArray[np.float64],
        H: NDArray[np.float64],
        Q: NDArray[np.float64],
        R: NDArray[np.float64],
        x0: NDArray[np.float64],
        P0: NDArray[np.float64],
        B: NDArray[np.float64] | None = None,
    ) -> None:
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.B = B

        self.x = x0
        self.P = P0

    def predict(self, u=None) -> None:
        """Prediction step."""
        if self.B is not None and u is not None:
            self.x = self.F @ self.x + self.B * u
        else:
            self.x = self.F @ self.x

        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: NDArray[np.float64]) -> None:
        """Measurement update."""

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y

        Identity = np.eye(self.P.shape[0], dtype=self.P.dtype)
        self.P = (Identity - K @ self.H) @ self.P
