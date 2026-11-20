import glob
import os

import numpy as np
from mcap_ros2.reader import read_ros2_messages


def quat_to_euler(x, y, z, w):
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    return yaw, pitch, roll


def mcap_files(path):
    """Accept a rosbag2 directory (with *.mcap + metadata.yaml) or a single .mcap"""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, '*.mcap')))
        if not files:
            raise FileNotFoundError(f'No .mcap files in {path}')
        return files
    return [path]


def read_bag(path, topic='/imu', storage_id=None):
    """Read an IMUmsg rosbag into a dict of numpy arrays, using mcap.
       - time starts at 0
    """
    keys = ['sec', 'nsec', 'yaw', 'pitch', 'roll',
            'mag_x', 'mag_y', 'mag_z',
            'acc_x', 'acc_y', 'acc_z',
            'gyro_x', 'gyro_y', 'gyro_z']
    cols = {k: [] for k in keys}

    for f in mcap_files(path):
        for item in read_ros2_messages(f, topics=[topic]):
            msg = item.ros_msg
            o = msg.imu.orientation
            yaw, pitch, roll = quat_to_euler(o.x, o.y, o.z, o.w)
            m = msg.mag_field.magnetic_field
            a = msg.imu.linear_acceleration
            g = msg.imu.angular_velocity
            row = [msg.header.stamp.sec, msg.header.stamp.nanosec,
                   yaw, pitch, roll,
                   m.x, m.y, m.z, a.x, a.y, a.z, g.x, g.y, g.z]
            for k, v in zip(keys, row):
                cols[k].append(v)

    if not cols['sec']:
        raise ValueError(f'No messages on topic {topic} in {path}')

    out = {k: np.array(v) for k, v in cols.items()}
    sec, nsec = out.pop('sec'), out.pop('nsec')
    out['time'] = (sec - sec[0]) + (nsec - nsec[0]) * 1e-9
    return out
