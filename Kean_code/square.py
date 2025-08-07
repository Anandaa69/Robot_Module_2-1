from robomaster import robot
import time

if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_chassis = ep_robot.chassis

    x_val = 0.6       # เดินไปข้างหน้า 0.6 เมตร
    z_val = 90        # หมุนขวา 88 องศา
    rounds = 1        # จำนวนรอบของสี่เหลี่ยมจัตุรัส
    walk_speed = 0.7    # ความเร็วในการเดิน   
    turn_speed = 45       # ความเร็วในการหมุน

    for i in range(rounds):
        print(f"Starting round {i + 1}")
        for j in range(4):
            # เดินหน้า
            ep_chassis.move(x=x_val, y=0, z=0, xy_speed=walk_speed).wait_for_completed(timeout=3)
            ep_chassis.stop()  # หยุดล้อ
            time.sleep(1)    # หน่วง 0.5 วินาที

            # หมุนขวา
            ep_chassis.move(x=0, y=0, z=-z_val, z_speed=turn_speed).wait_for_completed(timeout=3)
            ep_chassis.stop()
            time.sleep(1)

    ep_robot.close()