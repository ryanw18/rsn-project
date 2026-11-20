from bag_reader import read_bag
import json
from pathlib import Path
import numpy as np


def fit_ellipsoid(mag: np.ndarray):
    """
    Fits an ellipsoid to 3D magnetometer points (N x 3) and returns:
      b: (3,) hard-iron offset vector
      A: (3, 3) soft-iron calibration matrix such that m_cal = A @ (m_raw - b)
    """
    x, y, z = mag[:, 0], mag[:, 1], mag[:, 2]

    D = np.column_stack([
        x**2, y**2, z**2,
        2 * x * y, 2 * x * z, 2 * y * z,
        2 * x, 2 * y, 2 * z
    ])
    v = np.linalg.lstsq(D, np.ones_like(x), rcond=None)[0]

    # Algebraic form matrix
    M = np.array([
        [v[0], v[3], v[4], v[6]],
        [v[3], v[1], v[5], v[7]],
        [v[4], v[5], v[2], v[8]],
        [v[6], v[7], v[8], -1.0]
    ])

    # Hard-iron center b
    Q = M[:3, :3]
    b = -np.linalg.solve(Q, M[:3, 3])

    # Translate to center and normalize radius
    T = np.eye(4)
    T[3, :3] = b
    M_centered = T @ M @ T.T
    Q_norm = M_centered[:3, :3] / (-M_centered[3, 3])

    # Soft-iron matrix A = sqrtm(Q_norm), scaled to nominal Earth field (~50 uT)
    eigvals, eigvecs = np.linalg.eigh(Q_norm)
    if np.any(eigvals <= 0):
        raise ValueError(
            f"Degenerate ellipsoid fit (non-positive eigenvalues: {eigvals}). "
            "Check that the calibration bag covers all 3D orientations."
        )

    target_field = 50e-6  # 50 uT in Tesla
    A = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T * target_field

    return b, A


if __name__ == "__main__":

    data = read_bag("data/imu_calibration")
    mag_raw = np.column_stack([data["mag_x"], data["mag_y"], data["mag_z"]])

    # calibration
    b, A = fit_ellipsoid(mag_raw)

    # save b and A
    Path("config").mkdir(exist_ok=True)
    with open("config/mag_cal.json", "w") as f:
        json.dump({"b": b.tolist(), "A": A.tolist()}, f, indent=2)
    print("Saved config/mag_cal.json")
