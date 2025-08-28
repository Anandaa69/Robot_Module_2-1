import time
import robomaster
from robomaster import robot


def sub_imu_info_handler(imu_info):
    acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z = imu_info
    print("chassis imu: acc_x:{0}, acc_y:{1}, acc_z:{2}, gyro_x:{3}, gyro_y:{4}, gyro_z:{5}".format(
        acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z))


if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_chassis = ep_robot.chassis

    ep_chassis.sub_imu(freq=10, callback=sub_imu_info_handler)
    time.sleep(3)
    ep_chassis.unsub_imu()

    ep_robot.close()

