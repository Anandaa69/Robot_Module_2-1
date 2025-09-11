import time
from robomaster import robot

def main():
    # เริ่มต้นเชื่อมต่อหุ่น
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="sta")  # หรือ "ap" ถ้าเชื่อมแบบ AP

    # ดึงโมดูล servo
    servo_ctrl = ep_robot.servo

    # === ทดลองทีละ servo ===
    for servo_id in range(1, 5):  # servo_id: 1 ถึง 4
        print(f"\n=== ทดสอบ Servo {servo_id} ===")

        # 1) อ่านมุมปัจจุบัน
        angle_now = servo_ctrl.get_angle(servo_id)
        print(f"มุมปัจจุบัน Servo {servo_id}: {angle_now/10:.1f}°")

        # 2) หมุนไป +90°
        print(f"หมุน Servo {servo_id} ไป +90° (clockwise)")
        servo_ctrl.set_angle(servo_id, 900, wait_for_complete=True)
        time.sleep(1)

        # 3) หมุนไป -90°
        print(f"หมุน Servo {servo_id} ไป -90° (counterclockwise)")
        servo_ctrl.set_angle(servo_id, -900, wait_for_complete=True)
        time.sleep(1)

        # 4) กลับสู่ตำแหน่งปกติ
        print(f"รีเซ็ต Servo {servo_id} กลับตำแหน่งปกติ")
        servo_ctrl.recenter(servo_id, wait_for_complete=True)
        time.sleep(1)

    print("\n✅ ทดสอบเสร็จสิ้น!")

    # ปิดการเชื่อมต่อ
    ep_robot.close()

if __name__ == "__main__":
    main()
