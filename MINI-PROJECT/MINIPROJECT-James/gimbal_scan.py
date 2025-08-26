import time
import robomaster
from robomaster import robot

def simple_movement_sequence(gimbal, chassis):
    """
    ลำดับการเคลื่อนไหวแบบง่ายสำหรับ Gimbal โดยจะทำการล็อคล้อของหุ่นยนต์ก่อนเริ่ม
    """
    print("\n=== เริ่มลำดับการเคลื่อนไหวแบบง่าย ===")

    # ล็อคล้อทั้ง 4 ล้อเพื่อป้องกันการเคลื่อนที่ของตัวหุ่นยนต์
    print("🔒 ล็อคล้อทั้ง 4 ล้อ...")
    chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
    time.sleep(0.5)

    # ความเร็วในการเคลื่อนที่ของ Gimbal (องศา/วินาที)
    speed = 540

    # 1. หมุนไปทางซ้าย 90 องศา
    print("1. หมุนไปซ้าย 90°...")
    gimbal.moveto(pitch=0, yaw=90, pitch_speed=speed, yaw_speed=speed).wait_for_completed()
    time.sleep(1) # รอให้การเคลื่อนที่เสร็จสมบูรณ์และนิ่ง

    # 2. หมุนกลับไปทางขวา -90 องศา
    print("2. หมุนไปขวา -90°...")
    gimbal.moveto(pitch=0, yaw=-90, pitch_speed=speed, yaw_speed=speed).wait_for_completed()
    time.sleep(1) # รอให้การเคลื่อนที่เสร็จสมบูรณ์และนิ่ง

    # 3. กลับไปยังตำแหน่งเริ่มต้น (0, 0)
    print("3. กลับสู่ตำแหน่งเริ่มต้น...")
    gimbal.moveto(pitch=0, yaw=0, pitch_speed=speed, yaw_speed=speed).wait_for_completed()
    time.sleep(1) # รอให้การเคลื่อนที่เสร็จสมบูรณ์และนิ่ง

    # ปลดล็อคล้อ
    print("🔓 ปลดล็อคล้อ...")
    chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0, timeout=0.1)
    time.sleep(0.5)

    print("✓ การทำงานเสร็จสมบูรณ์!")


if __name__ == '__main__':
    print("กำลังเชื่อมต่อหุ่นยนต์...")
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    
    ep_gimbal = ep_robot.gimbal
    ep_chassis = ep_robot.chassis
    
    try:
        # รีเซ็ตตำแหน่ง Gimbal ไปที่ (0, 0) ก่อนเริ่ม
        print("รีเซ็ตตำแหน่งเริ่มต้น...")
        ep_gimbal.moveto(pitch=0, yaw=0, pitch_speed=50, yaw_speed=50).wait_for_completed()
        time.sleep(1)
        
        # วนลูปการเคลื่อนไหว 20 รอบ
        for i in range(20):
            print(f"\n--- เริ่มรอบที่ {i+1} จาก 20 ---")
            simple_movement_sequence(ep_gimbal, ep_chassis)
            
    except KeyboardInterrupt:
        print("\nหยุดการทำงานโดยผู้ใช้")
    except Exception as e:
        print(f"\nเกิดข้อผิดพลาด: {e}")
    finally:
        # ปิดการเชื่อมต่อเมื่อจบการทำงานหรือเกิดข้อผิดพลาด
        ep_robot.close()
        print("ปิดการเชื่อมต่อเรียบร้อย")