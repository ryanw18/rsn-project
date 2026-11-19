import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
import serial
from imu_msg.msg import IMUmsg
import math


def extract_data(line):

    orientation = [math.radians(float(i)) for i in line[1:4]]
    mag = [float(i) * 0.0001 for i in line[4:7]]
    acc = [float(i) for i in line[7:10]]
    line[12] = line[12].split("*")[0]
    gyro = [float(i) for i in line[10:13]]

    return orientation, mag, acc, gyro


def quaternion(ypr):

    cy = math.cos(ypr[0] / 2)
    cp = math.cos(ypr[1] / 2)
    cr = math.cos(ypr[2] / 2)
    sy = math.sin(ypr[0] / 2)
    sp = math.sin(ypr[1] / 2)
    sr = math.sin(ypr[2] / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return [x, y, z, w]


class ImuDriver(Node):
    def __init__(self):
        super().__init__('imu_driver')

        self.declare_parameter('port', Parameter.Type.STRING)

        serial_port = self.get_parameter('port').value

        self.port = serial.Serial(serial_port, 115200, timeout=1.0)
        self.get_logger().debug("Using IMU on port " + serial_port)

        self.imu_pub = self.create_publisher(IMUmsg, '/imu', 5)
        self.get_logger().debug("Initialization complete")
        self.get_logger().info("Publishing IMU data.")

        self.imu_msg = IMUmsg()
        self.imu_msg.header.frame_id = "IMU1_Frame"
        self.imu_msg.imu.header.frame_id = "IMU1_Frame"
        self.imu_msg.mag_field.header.frame_id = "IMU1_Frame"

        self.port.write(b'$VNWRG,06,14*59\r\n')
        self.port.write(b'$VNWRG,07,40*59\r\n')
        
        self.port.reset_input_buffer()

        self.timer = self.create_timer(0.005, self.timer_callback)

    def timer_callback(self):
        r_line = self.port.readline().decode(errors='ignore').strip()

        if r_line == '':
            self.get_logger().warn("IMU: No data")
        else:
            if r_line.startswith('$VNYMR'):
                try:
                    line = [field.strip() for field in r_line.split(',')]
                    orientation, mag, acc, gyro = extract_data(line)
                except Exception as e:
                    self.get_logger().warn("Data exception: " + str(e))
                    return

                quat_xyzw = quaternion(orientation)
                timestamp = self.get_clock().now().to_msg()

                self.imu_msg.header.stamp = timestamp
                self.imu_msg.imu.header.stamp = timestamp
                self.imu_msg.mag_field.header.stamp = timestamp

                self.imu_msg.imu.orientation.x = quat_xyzw[0]
                self.imu_msg.imu.orientation.y = quat_xyzw[1]
                self.imu_msg.imu.orientation.z = quat_xyzw[2]
                self.imu_msg.imu.orientation.w = quat_xyzw[3]

                self.imu_msg.mag_field.magnetic_field.x = mag[0]
                self.imu_msg.mag_field.magnetic_field.y = mag[1]
                self.imu_msg.mag_field.magnetic_field.z = mag[2]

                self.imu_msg.imu.linear_acceleration.x = acc[0]
                self.imu_msg.imu.linear_acceleration.y = acc[1]
                self.imu_msg.imu.linear_acceleration.z = acc[2]

                self.imu_msg.imu.angular_velocity.x = gyro[0]
                self.imu_msg.imu.angular_velocity.y = gyro[1]
                self.imu_msg.imu.angular_velocity.z = gyro[2]

                self.imu_msg.raw_line = r_line

                self.imu_pub.publish(self.imu_msg)


def main(args=None):
    rclpy.init(args=args)
    imu_driver = ImuDriver()
    rclpy.spin(imu_driver)

    imu_driver.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
