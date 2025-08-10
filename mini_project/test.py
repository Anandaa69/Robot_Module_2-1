from robomaster import robot
import time
import math

def main():
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    gimbal = ep_robot.gimbal
    chassis = ep_robot.chassis

    try:
        while True:
            best_angle = None
            max_distance = 0

            # สแกน ToF รอบ 0 - 180 องศาทีละ 15 องศา (หันซ้ายไปขวา)
            for angle in range(0, 181, 15):
                gimbal.rotate(angle)
                time.sleep(0.2)  # รอให้ gimbal หมุนถึงตำแหน่ง
                dist = gimbal.get_tof_distance()  # อ่านระยะ (mm)
                dist_m = dist / 1000.0  # แปลงเป็นเมตร
                print(f"Angle: {angle}°, Distance: {dist_m:.2f} m")

                # หาองศาที่ระยะไกลสุดและเกิน 0.3 เมตร (ทางโล่ง)
                if dist_m > max_distance and dist_m > 0.3:
                    max_distance = dist_m
                    best_angle = angle

            if best_angle is None:
                print("ไม่มีทางเดินต่อ หยุดทำงาน")
                break

            # หมุนหุ่นยนต์ chassis ไปทาง best_angle
            turn_angle = best_angle - 90  # ปรับมุม (gimbal 0 = ซ้ายสุด, chassis 0 = หน้า)
            print(f"หมุนหุ่นยนต์ {turn_angle} องศา แล้วเดินหน้า {max_distance:.2f} เมตร")

            chassis.rotate(turn_angle)
            time.sleep(1)  # รอหมุนเสร็จ

            # เดินหน้า 0.3 เมตร (หรือระยะสั้นสุด 0.3)
            step_dist = min(0.3, max_distance)
            chassis.drive_speed(distance=step_dist, speed=0.3)
            time.sleep(1)  # รอเดินเสร็จ

    except KeyboardInterrupt:
        print("หยุดโปรแกรมโดยผู้ใช้")

    ep_robot.close()

if __name__ == "__main__":
    main()
