from robomaster import robot
import time

if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_chassis = ep_robot.chassis

    X_VAL = 1.8       # เดินไปข้างหน้า
    WALK_SPEED = 0.8    # ความเร็วในการเดิน   

    ep_chassis.move(x=X_VAL, y=0, z=0, xy_speed=WALK_SPEED).wait_for_completed(timeout=3)
    ep_chassis.stop()  # หยุดล้อ
    time.sleep(1)

    ep_robot.close()