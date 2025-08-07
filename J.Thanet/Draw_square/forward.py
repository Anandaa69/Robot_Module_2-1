from robomaster import robot
import time

def sub_position_handler(position_info):
    x, y, z = position_info
    print("chassis position: x:{0}, y:{1}, z:{2}".format(x, y, z))

if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_chassis = ep_robot.chassis

    X_VAL = -1.8       # เดินไปข้างหน้า
    WALK_SPEED = 0.8    # ความเร็วในการเดิน   

    ep_chassis.sub_position(freq=10, callback=sub_position_handler)
    ep_chassis.move(x=X_VAL, y=0, z=0, xy_speed=WALK_SPEED).wait_for_completed(timeout=3)
    ep_chassis.stop()  # หยุดล้อ
    time.sleep(1)
    ep_chassis.unsub_position()

    ep_robot.close()