import numpy as np
from kalman_demo.data_generators.synthetic_1d import generate_1d_data
from kalman_demo.kalman import KalmanFilter
from kalman_demo.models.models_1d import linear_kinematic_model
from kalman_demo.visualizers.viz_1d import plot_1d


def main() -> None:
    np.random.seed(42)

    # Experiment with these parameters
    # ------------------
    n_steps = 40
    accel_std = 0.05
    meas_std = 3.0

    # Time-varying commanded acceleration
    accel_cmds = np.concatenate([np.ones(20) * 0.2, np.ones(20) * -0.1])
    # ------------------

    dt = 1.0
    pos0, vel0 = 0.0, 0.0

    # Ground truth + measurements
    true_states, measurements = generate_1d_data(
        n_steps=n_steps,
        dt=dt,
        pos0=pos0,
        vel0=vel0,
        accel_cmds=accel_cmds,
        accel_std=accel_std,
        meas_std=meas_std,
    )

    # Kalman model
    F, H, Q, B = linear_kinematic_model(dt, accel_std, with_control=True)
    R = np.array([[meas_std**2]])

    x0 = np.array([pos0, vel0])
    P0 = np.eye(2) * 10.0

    kf = KalmanFilter(F, H, Q, R, x0, P0, B=B)

    est_positions, est_velocities, est_accels = [], [], []
    prev_vel = None

    for k in range(n_steps):
        u_k = accel_cmds[k]
        z_k = measurements[k]

        kf.predict(u_k)
        kf.update(z_k)

        est_positions.append(kf.x[0, 0])
        est_velocities.append(kf.x[1, 0])

        if prev_vel is None:
            est_accels.append(u_k)
        else:
            est_accels.append((kf.x[1, 0] - prev_vel) / dt)
        prev_vel = kf.x[1, 0]

    est_positions = np.array(est_positions)
    est_velocities = np.array(est_velocities)
    est_accels = np.array(est_accels)

    plot_1d(
        n_steps=n_steps,
        true_positions=true_states[:, 0],
        measurements=measurements,
        est_positions=est_positions,
        true_velocities=true_states[:, 1],
        est_velocities=est_velocities,
        accel_cmds=accel_cmds,
        est_accelerations=est_accels,
    )


if __name__ == "__main__":
    main()
