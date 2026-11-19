from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    port_param = DeclareLaunchArgument('port')

    port_arg = LaunchConfiguration('port')

    return LaunchDescription([
        port_param,
        Node(
            package='imu_driver',
            executable='imu_driver',
            parameters=[
                {'port': port_arg}
            ]
        )
    ])
