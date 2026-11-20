import json
import numpy as np
from vqf import VQF
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from bag_reader import read_bag


def get_rotations(bag_path: str, cal_path: str = "config/mag_cal.json"):
    
    data = read_bag(bag_path)
    t = data["time"]

    gyr = np.column_stack([data["gyro_x"], data["gyro_y"], data["gyro_z"]])
    acc = np.column_stack([data["acc_x"], data["acc_y"], data["acc_z"]])
    mag_raw = np.column_stack([data["mag_x"], data["mag_y"], data["mag_z"]])

    with open(cal_path, "r") as f:
        cal = json.load(f)
        
    b = np.array(cal["b"])
    A = np.array(cal["A"])
    mag_cal = (mag_raw - b) @ A.T

    # Compute exact sample period (dt) and run VQF
    dt = float(np.median(np.diff(t)))
    print(f"Loaded {len(t)} samples. Duration: {t[-1]:.1f} s")

    # VQF:
    # Loops through the entire recording, combines the three sensors at every timestamp:
    #     uses gyroscope to track arm swings smoothly
    #     uses accelerometer to keep "down" locked, so estimated arm height never drifts
    #     uses calibrated magnetometer to keep horizontal heading locked, so forward/backward drift stays accurate
    # For every timestamp, gives a rotation quaternion (relative to earth)

    vqf = VQF(dt)
    out = vqf.updateBatch(gyr, acc, mag_cal)
    quat = out["quat9D"]

    print("VQF output shape:", quat.shape)
    return t, acc, gyr, mag_cal, quat


def find_initial_rest(t: np.ndarray, gyr: np.ndarray, threshold: float = 0.15):
    """
    Finds the first continuous stationary block at the start of the recording
    (ignoring the first 1.0s in case you were still hitting 'record' or settling).
    """
    gyro_speed = np.linalg.norm(gyr, axis=1)
    is_still = (gyro_speed < threshold) & (t >= 2.0)

    # Walk through samples to find the first continuous stretch >= 5s
    dt = float(np.median(np.diff(t)))
    min_samples = int(5 / dt)

    start_idx = None
    for i, still in enumerate(is_still):
        if still:
            if start_idx is None:
                start_idx = i
        else:
            if start_idx is not None and (i - start_idx) >= min_samples:
                pad = int(0.5 / dt)
                mask = np.zeros(len(t), dtype=bool)
                mask[start_idx + pad : i - pad] = True
                print(f"Auto-detected rest window: {t[start_idx + pad]:.1f}s to {t[i - pad]:.1f}s")
                return mask
            start_idx = None

    raise RuntimeError("Could not find an initial stationary rest period of 5+ seconds")


def calculate_elevation_angle(
    t: np.ndarray,
    acc: np.ndarray,
    quat_wxyz: np.ndarray,
    rest_mask: np.ndarray
):
    # Convert VQF quaternions to SciPy format [x, y, z, w]
    quat_xyzw = np.column_stack([quat_wxyz[:, 1:], quat_wxyz[:, 0]])
    rotations = Rotation.from_quat(quat_xyzw)

    # Average world frame arm vector during rest
    acc_world_rest = rotations[rest_mask].apply(acc[rest_mask]).mean(axis=0)
    up_world = acc_world_rest / np.linalg.norm(acc_world_rest)

    # Transform that up along arm direction back into the sensor's local frame
    r_rest_mean = rotations[rest_mask].mean()
    u_arm_local = r_rest_mean.inv().apply(up_world)

    # Rotate u_arm_local into the world frame at every timestamp t
    v_arm_world = rotations.apply(u_arm_local)

    # Elevation angle: angle between current arm vector and resting arm vector
    cos_elev = np.clip(v_arm_world @ up_world, -1.0, 1.0)
    elevation_deg = np.degrees(np.arccos(cos_elev))

    return elevation_deg, v_arm_world, up_world


def calculate_plane_drift(
    elevation_deg: np.ndarray,
    v_arm_world: np.ndarray,
    up_world: np.ndarray,
    reference_plane_dir: np.ndarray | None = None,
):
    """
    - Pass reference_plane_dir=None on a good-rep bag to get ideal plane.
    - Pass that into bad-rep bags (if facing the same direction).
    """
    # Remove vertical component to get horizontal floor projection of the arm
    v_vert = (v_arm_world @ up_world)[:, None] * up_world[None, :]
    v_horiz = v_arm_world - v_vert
    horiz_norm = np.linalg.norm(v_horiz, axis=1, keepdims=True)

    v_horiz_unit = np.zeros_like(v_horiz)
    valid = horiz_norm[:, 0] > 1e-6
    v_horiz_unit[valid] = v_horiz[valid] / horiz_norm[valid]

    # If no reference plane given, compute average raise direction when arm > 25°
    if reference_plane_dir is None:
        raised_mask = elevation_deg > 25.0
        mean_dir = v_horiz_unit[raised_mask].mean(axis=0)
        reference_plane_dir = mean_dir / np.linalg.norm(mean_dir)

    # Signed horizontal angle between current arm direction and reference plane
    cos_plane = np.clip(v_horiz_unit @ reference_plane_dir, -1.0, 1.0)
    sin_plane = np.cross(reference_plane_dir, v_horiz_unit) @ up_world
    plane_angle_deg = np.degrees(np.arctan2(sin_plane, cos_plane))

    # Hide angles when arm is hanging down (< 20°), since horizontal direction is just noise
    plane_angle_deg = np.where(elevation_deg >= 20.0, plane_angle_deg, np.nan)

    return plane_angle_deg, reference_plane_dir


if __name__ == "__main__":

    bags = {"good_form_1": 0.15, "too_high": 0.15, "too_low": 0.25}

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=True)

    for ax, (name, thr) in zip(axes, bags.items()):
        t, acc, gyr, _, quat = get_rotations(f"data/{name}")
        rest = find_initial_rest(t, gyr, threshold=thr)
        elev, _, _ = calculate_elevation_angle(t, acc, quat, rest)

        ax.plot(t, elev, label=name)
        ax.fill_between(t, 0, 1, where=rest, transform=ax.get_xaxis_transform(),
                        color="gray", alpha=0.25, label="rest window")
        ax.axhline(90, color="k", linestyle="--", alpha=0.5)
        ax.set_ylabel("Elevation (deg)")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

    # # Calibrate ideal lateral plane on a good bag
    # t_good, acc_good, gyr_good, _, quat_good = get_rotations("data/good_form_1")
    # rest_good = find_initial_rest(t_good, gyr_good)
    # elev_good, v_arm_good, up_good = calculate_elevation_angle(t_good, acc_good, quat_good, rest_good)
    # plane_good, ref_dir = calculate_plane_drift(elev_good, v_arm_good, up_good)

    # # Evaluate a bad bag using that same ref_dir
    # t_bad, acc_bad, gyr_bad, _, quat_bad = get_rotations("data/rotated_or_wrong_plane")
    # rest_bad = find_initial_rest(t_bad, gyr_bad)
    # elev_bad, v_arm_bad, up_bad = calculate_elevation_angle(t_bad, acc_bad, quat_bad, rest_bad)
    # plane_bad, _ = calculate_plane_drift(elev_bad, v_arm_bad, up_bad, reference_plane_dir=ref_dir)

    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    # ax1.plot(t_bad, elev_bad, label="Elevation (deg)", color="tab:blue")
    # ax1.axhline(90, color="g", linestyle="--", alpha=0.5)
    # ax1.set_ylabel("Elevation (deg)")
    # ax1.legend()

    # ax2.plot(t_bad, plane_bad, label="Plane Drift (deg)", color="tab:orange")
    # ax2.axhline(0, color="k", linestyle="--", alpha=0.4)
    # ax2.set_ylabel("Plane Drift (deg)")
    # ax2.set_xlabel("Time (s)")
    # ax2.legend()

    # plt.tight_layout()
    # plt.show()
