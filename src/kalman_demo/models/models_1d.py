import numpy as np
from numpy.typing import NDArray


def linear_kinematic_model(
    dt: float, process_accel_std: float, with_control: bool = False
) -> tuple[NDArray, NDArray, NDArray, NDArray | None]:
    """
        Returns matrices for a 1D Kalman filter with constant velocity.

        State: x = [position, velocity]

        Parameters
        ----------
        dt : float
            Time step
        process_accel_std : float
    cx        Standard deviation of acceleration noise - external assumption, tuned from
            system specs or experimentally
        with_control : bool
            If True, include control matrix B for commanded acceleration

        Returns
        -------
        F : (2,2) state transition
        H : (1,2) measurement matrix
        Q : (2,2) process noise covariance
        B : (2,1) control matrix if with_control, else None
    """

    F = np.array([[1.0, dt], [0.0, 1.0]])

    # we measure only the position
    H = np.array([[1.0, 0.0]])

    # Acceleration process noise
    q = process_accel_std**2
    Q = q * np.array([[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]])

    B = np.array([[0.5 * dt**2], [dt]]) if with_control else None

    return F, H, Q, B
