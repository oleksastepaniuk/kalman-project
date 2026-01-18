import numpy as np
from kalman_demo.data_generators.synthetic_1d import generate_1d_data
from kalman_demo.kalman import KalmanFilter
from kalman_demo.models.models_1d import linear_kinematic_model
from kalman_demo.visualizers.viz_1d import plot_1d
from kalman_demo.metrics import rmse


def main() -> None:
    np.random.seed(42)

    # Experiment with these parameters
    # ------------------
    n_steps = 40
    accel_std_true = 0.05  # true process accel noise (used in generator)
    meas_std = 3.0

    # Time-varying commanded acceleration
    accel_cmds = np.concatenate([np.ones(20) * 0.2, np.ones(20) * -0.1])
    # ------------------

    dt = 1.0
    pos0, vel0 = 0.0, 0.0

    # Ground truth + measurements
    states, measurements = generate_1d_data(
        n_steps=n_steps,
        dt=dt,
        pos0=pos0,
        vel0=vel0,
        accel_cmds=accel_cmds,
        accel_std=accel_std_true,
        meas_std=meas_std,
    )
    true_position = states[:, 0]
    true_velocity = states[:, 1]

    # Kalman model
    F, H, Q, B = linear_kinematic_model(
        dt=dt, process_accel_std=accel_std_true, with_control=True
    )
    R = np.array([[meas_std**2]], dtype=float)

    # initial state guess + covariance
    x0 = np.array([pos0, vel0], dtype=float)
    P0 = np.eye(2, dtype=float) * 10.0

    kf = KalmanFilter(F, H, Q, R, x0, P0, B=B)

    # --- Run filtering ---
    est_positions, est_velocities, est_accels = [], [], []
    prev_vel = None

    for k in range(n_steps):
        u_k = accel_cmds[k]
        z_k = measurements[k]

        kf.predict(float(u_k))
        kf.update(np.array([[z_k]], dtype=float))

        est_positions.append(kf.x[0, 0])
        est_velocities.append(kf.x[1, 0])

        if prev_vel is None:
            est_accels.append(np.nan)
        else:
            est_accels.append((kf.x[1, 0] - prev_vel) / dt)
        prev_vel = kf.x[1, 0]

    est_positions = np.array(est_positions)
    est_velocities = np.array(est_velocities)
    est_accels = np.array(est_accels)

    # --- Metrics ---
    rmse_meas = rmse(true_position, measurements)
    rmse_kf = rmse(true_position, est_positions)
    impr = rmse_meas / rmse_kf if rmse_kf > 0 else np.inf

    print(f"RMSE(measurements vs true position): {rmse_meas:.3f}")
    print(f"RMSE(KF estimate vs true position): {rmse_kf:.3f}")
    print(f"Improvement factor (meas / kf): {impr:.2f}x")

    plot_1d(
        n_steps=n_steps,
        true_positions=true_position,
        measurements=measurements,
        est_positions=est_positions,
        true_velocities=true_velocity,
        est_velocities=est_velocities,
        accel_cmds=accel_cmds,
        est_accelerations=est_accels,
    )


if __name__ == "__main__":
    main()
