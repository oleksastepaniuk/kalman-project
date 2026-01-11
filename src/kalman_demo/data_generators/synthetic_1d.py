import numpy as np
from numpy.typing import NDArray


def generate_1d_data(
    n_steps: int,
    dt: float,
    pos0: float,
    vel0: float,
    accel_cmds: NDArray,
    accel_std: float,
    meas_std: float,
) -> tuple[NDArray, NDArray]:
    """
    Generate 1D motion with with commanded acceleration per step + noise.

    State:
        x = [position, velocity]

    Control:
        accel_cmds[k] = commanded acceleration at step k

    Acceleration noise:
        a_k ~ N(0, accel_std^2)
    """
    if len(accel_cmds) != n_steps:
        raise ValueError("Length of accel_cmds must equal n_steps")

    x = np.array([pos0, vel0], dtype=float)

    states: list[NDArray] = []
    measurements: list[float] = []

    for k in range(n_steps):
        a_noise = np.random.randn() * accel_std
        a_total = accel_cmds[k] + a_noise

        pos_prev = x[0]
        vel_prev = x[1]

        # state update
        pos_current = pos_prev + vel_prev * dt + 0.5 * a_total * dt**2
        vel_current = vel_prev + a_total * dt

        # measurement (position only)
        m_current = pos_current + np.random.randn() * meas_std

        x = np.array([pos_current, vel_current])
        states.append(x.copy())
        measurements.append(m_current)

    return np.asarray(states), np.asarray(measurements)
