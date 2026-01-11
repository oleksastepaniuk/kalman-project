import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def plot_1d(
    n_steps: int,
    true_positions: NDArray,
    measurements: NDArray,
    est_positions: NDArray,
    true_velocities: NDArray,
    est_velocities: NDArray,
    accel_cmds: NDArray,
    est_accelerations: NDArray,
):
    """
    Visualize 1D Kalman filter results: position, velocity, acceleration
    """
    t = np.arange(n_steps)

    plt.figure(figsize=(12, 5))

    # --- Position ---
    plt.subplot(1, 3, 1)
    plt.scatter(t, true_positions, label="True Position", s=15, alpha=0.5)
    plt.plot(t, measurements, label="Measurements", alpha=0.5)
    plt.plot(t, est_positions, label="Estimated Position", linestyle="--")
    plt.xlabel("Time step")
    plt.ylabel("Position")
    plt.legend()
    plt.grid(True)

    # --- Velocity ---
    plt.subplot(1, 3, 2)
    plt.scatter(t, true_velocities, label="True Velocity", s=15, alpha=0.5)
    plt.plot(t, est_velocities, label="Estimated Velocity", linestyle="--")
    plt.xlabel("Time step")
    plt.ylabel("Velocity")
    plt.legend()
    plt.grid(True)

    # --- Acceleration ---
    plt.subplot(1, 3, 3)
    plt.scatter(t, accel_cmds, label="Commanded Acceleration", s=15, alpha=0.5)
    plt.plot(t, est_accelerations, label="Estimated Acceleration", linestyle="--")
    plt.xlabel("Time step")
    plt.ylabel("Acceleration")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
