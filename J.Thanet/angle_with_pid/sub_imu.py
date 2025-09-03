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
    ep_chassis.sub_imu(freq=5, callback=sub_imu_info_handler)
    ep_chassis.move(x=1.2, y=0, z=0, xy_speed=0.7).wait_for_completed()
    # time.sleep(0.3)
    # ep_chassis.move(x=0, y=-0.3, z=0, xy_speed=0.7).wait_for_completed()
    # time.sleep(0.3)
    # ep_chassis.move(x=0, y=0.3, z=0, xy_speed=0.7).wait_for_completed()
    # time.sleep(0.3)
    # ep_chassis.move(x=-0.6, y=0, z=0, xy_speed=0.7).wait_for_completed()9*/
    

