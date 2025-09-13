from robomaster import robot
import time

# --- เราจะทดสอบแค่พอร์ต S1 เท่านั้น ---
SERVO_PORT_TO_TEST = 1
# -----------------------------------

if __name__ == '__main__':
    ep_robot = robot.Robot()
    try:
        print("กำลังเชื่อมต่อ...")
        ep_robot.initialize(conn_type="ap")
        print("เชื่อมต่อสำเร็จ!")

        ep_servo = ep_robot.servo

        print(f"\n--- เริ่มการทดสอบ Servo ที่พอร์ต S{SERVO_PORT_TO_TEST} ---")
        
        # 1. อ่านค่ามุมเริ่มต้น
        initial_angle = ep_servo.get_angle(index=SERVO_PORT_TO_TEST)
        print(f"มุมเริ่มต้น: {initial_angle} องศา")
        time.sleep(1)

        # 2. สั่งให้หมุนไปตำแหน่งแรก (90 องศา) เพื่อดูว่าขยับหรือไม่
        print("กำลังสั่งให้หมุนไปที่ 90 องศา...")
        ep_servo.moveto(index=SERVO_PORT_TO_TEST, angle=90).wait_for_completed()
        time.sleep(3) # รอ 3 วินาทีเพื่อให้แน่ใจว่ามันมีเวลาขยับ
        current_angle = ep_servo.get_angle(index=SERVO_PORT_TO_TEST)
        print(f"มุมปัจจุบัน (เป้าหมาย 90): {current_angle} องศา")

        # 3. สั่งให้หมุนไปตำแหน่งที่สอง (180 องศา)
        print("\nกำลังสั่งให้หมุนไปที่ 180 องศา...")
        ep_servo.moveto(index=SERVO_PORT_TO_TEST, angle=200).wait_for_completed()
        time.sleep(3) # รอ 3 วินาที
        current_angle = ep_servo.get_angle(index=SERVO_PORT_TO_TEST)
        print(f"มุมปัจจุบัน (เป้าหมาย 180): {current_angle} องศา")
        
        print("\n--- สิ้นสุดการทดสอบ ---")

    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {e}")
    finally:
        # กลับสู่ตำแหน่งเริ่มต้นก่อนปิด
        if 'initial_angle' in locals():
            print("กำลังกลับสู่ตำแหน่งเริ่มต้น...")
            ep_servo.moveto(index=SERVO_PORT_TO_TEST, angle=initial_angle).wait_for_completed()
        ep_robot.close()
        print("ปิดการเชื่อมต่อแล้ว")