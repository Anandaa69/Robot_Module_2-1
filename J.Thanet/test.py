from robomaster import robot
import time

# --- ตั้งค่าตรงนี้ ---
# ใส่ ID ของ Servo ที่คุณเสียบสายไว้ (1 สำหรับพอร์ต S1, 2 สำหรับพอร์ต S2)
SERVO_ID_TO_TEST = 4

if __name__ == '__main__':
    ep_robot = robot.Robot()
    print("กำลังเชื่อมต่อกับหุ่นยนต์...")
    ep_robot.initialize(conn_type="ap")
    print("เชื่อมต่อสำเร็จ!")

    ep_servo = ep_robot.servo

    while True :
        # 1. อ่านค่ามุมเริ่มต้น
        initial_angle = ep_servo.get_angle(index=1)
        print(f"--- ผลการตรวจสอบ Servo ID: {1} ---")
        print(f"[ขั้นตอนที่ 1] มุมเริ่มต้นที่อ่านได้คือ: {initial_angle} องศา")


        initial_angle = ep_servo.get_angle(index=3)
        print(f"--- ผลการตรวจสอบ Servo ID: {3} ---")
        print(f"[ขั้นตอนที่ 1] มุมเริ่มต้นที่อ่านได้คือ: {initial_angle} องศา")


        time.sleep(2)
