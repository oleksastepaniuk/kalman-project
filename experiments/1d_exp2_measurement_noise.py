import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from kalman_demo.data_generators.synthetic_1d import generate_1d_data
from kalman_demo.kalman import KalmanFilter
from kalman_demo.metrics import rmse
from kalman_demo.models.models_1d import linear_kinematic_model


def run_one_trial(
    *,
    seed: int,
    n_steps: int,
    dt: float,
    pos0: float,
    vel0: float,
    accel_cmds: NDArray[np.float64],
    accel_std_true: float,
    meas_std: float,
) -> tuple[float, float]:
    """
    Run one simulation + filtering pass and return:
    (RMSE(measurements vs true position), RMSE(KF position vs true position)).
    """
    np.random.seed(seed)

    states, measurements = generate_1d_data(
        n_steps=n_steps,
        dt=dt,
        pos0=pos0,
        vel0=vel0,
        accel_cmds=accel_cmds,
        accel_std=accel_std_true,
        meas_std=meas_std,
    )
    true_pos = states[:, 0]

    F, H, Q, B = linear_kinematic_model(
        dt=dt,
        process_accel_std=accel_std_true,  # fixed process noise
        with_control=True,
    )
    R = np.array([[meas_std**2]], dtype=float)

    x0 = np.array([pos0, vel0], dtype=float)
    P0 = np.eye(2, dtype=float) * 10.0

    kf = KalmanFilter(F, H, Q, R, x0, P0, B=B)

    est_pos = np.empty(n_steps, dtype=float)
    for k in range(n_steps):
        u_k = float(accel_cmds[k])
        z_k = float(measurements[k])

        kf.predict(u_k)
        # update() expects an array compatible with (H @ x) which is (1, 1)
        kf.update(np.array([[z_k]], dtype=float))

        est_pos[k] = float(kf.x[0, 0])

    return rmse(true_pos, measurements), rmse(true_pos, est_pos)


def main() -> None:
    # ------------------
    # Fixed process noise
    accel_std_true = 0.05

    # Sweep measurement noise
    meas_stds = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0], dtype=float)

    # Monte-Carlo trials per setting
    n_trials = 50
    base_seed = 42

    # Motion config (keep consistent with Experiment 1)
    n_steps = 40
    dt = 1.0
    pos0, vel0 = 0.0, 0.0
    accel_cmds = np.concatenate([np.ones(20) * 0.2, np.ones(20) * -0.1]).astype(float)
    # ------------------

    rmse_meas_mean = np.zeros_like(meas_stds, dtype=float)
    rmse_kf_mean = np.zeros_like(meas_stds, dtype=float)

    rmse_meas_std = np.zeros_like(meas_stds, dtype=float)
    rmse_kf_std = np.zeros_like(meas_stds, dtype=float)

    print("1D. Experiment 2: measurement noise sweep")
    print("-------------------------------------")

    for j, meas_std in enumerate(meas_stds):
        vals_meas = np.empty(n_trials, dtype=float)
        vals_kf = np.empty(n_trials, dtype=float)

        for i in range(n_trials):
            seed = base_seed + i
            a, b = run_one_trial(
                seed=seed,
                n_steps=n_steps,
                dt=dt,
                pos0=pos0,
                vel0=vel0,
                accel_cmds=accel_cmds,
                accel_std_true=accel_std_true,
                meas_std=float(meas_std),
            )

            vals_meas[i] = a
            vals_kf[i] = b

        rmse_meas_mean[j] = float(vals_meas.mean())
        rmse_kf_mean[j] = float(vals_kf.mean())
        rmse_meas_std[j] = float(vals_meas.std(ddof=1))
        rmse_kf_std[j] = float(vals_kf.std(ddof=1))

        impr = rmse_meas_mean[j] / rmse_kf_mean[j] if rmse_kf_mean[j] > 0 else np.inf
        print(
            f"meas_std={meas_std:>5.1f} |\t"
            f"RMSE_meas={rmse_meas_mean[j]:.3f} ± {rmse_meas_std[j]:.3f} |\t"
            f"RMSE_KF={rmse_kf_mean[j]:.3f} ± {rmse_kf_std[j]:.3f} |\t"
            f"impr={impr:.2f}x"
        )

    # One plot: two RMSE curves vs meas_std
    plt.figure(figsize=(9, 4))
    plt.plot(meas_stds, rmse_meas_mean, marker="o", label="RMSE(measurements)")
    plt.plot(meas_stds, rmse_kf_mean, marker="o", label="RMSE(KF estimate)")

    # Optional variability bands
    plt.fill_between(
        meas_stds,
        rmse_meas_mean - rmse_meas_std,
        rmse_meas_mean + rmse_meas_std,
        alpha=0.15,
    )
    plt.fill_between(
        meas_stds,
        rmse_kf_mean - rmse_kf_std,
        rmse_kf_mean + rmse_kf_std,
        alpha=0.15,
    )

    plt.xlabel("Measurement noise std (meas_std)")
    plt.ylabel("RMSE vs true position")
    plt.title("Experiment 2: measurement noise sweep")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
