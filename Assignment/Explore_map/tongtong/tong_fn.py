# -*-coding:utf-8-*-

import robomaster
from robomaster import robot
import time
import math # เพิ่ม math เข้ามาเพื่อใช้คำนวณระยะทาง

# =======================================================================
# === คลาสสำหรับควบคุมการเคลื่อนที่ (มีการแก้ไขและเพิ่มฟังก์ชันใหม่) ===
# =======================================================================
class PositionController:
    def __init__(self, ep_robot):
        self.robot = ep_robot
        self.chassis = ep_robot.chassis
        
        # --- ตัวแปรสำหรับเก็บค่าจากเซ็นเซอร์ ---
        self.current_yaw = 0.0
        self.current_x = 0.0
        self.current_y = 0.0
        self.target_yaw = 0.0
        
        # --- เริ่มรับข้อมูลจากเซ็นเซอร์ ---
        self.chassis.sub_attitude(freq=20, callback=self._attitude_callback)
        self.chassis.sub_position(freq=20, callback=self._position_callback)
        
        # รอสักครู่เพื่อให้ค่าเริ่มต้นจากเซ็นเซอร์ถูกส่งเข้ามา
        time.sleep(1) 
        
        # ตั้งค่า Yaw เป้าหมายเริ่มต้นให้เป็นค่าปัจจุบัน
        self.target_yaw = self.current_yaw 
        print(f"Controller initialized. Initial Yaw: {self.current_yaw:.2f}, Position (x,y): ({self.current_x:.2f}, {self.current_y:.2f})")

    # --- Callback Functions: ทำงานเบื้องหลังเพื่ออัปเดตค่าจากเซ็นเซอร์ตลอดเวลา ---
    def _attitude_callback(self, attitude_info):
        self.current_yaw, _, _ = attitude_info

    def _position_callback(self, position_info):
        self.current_x, self.current_y, _ = position_info

    # --- ฟังก์ชันคำนวณการหมุนชดเชย (ไม่เปลี่ยนแปลง) ---
    def _calculate_yaw_correction(self):
        KP_YAW = 0.8
        MAX_YAW_SPEED = 20
        yaw_error = self.target_yaw - self.current_yaw
        if yaw_error > 180: yaw_error -= 360
        elif yaw_error < -180: yaw_error += 360
        rotation_speed_z = KP_YAW * yaw_error
        return max(min(rotation_speed_z, MAX_YAW_SPEED), -MAX_YAW_SPEED)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++ ฟังก์ชันใหม่: เดินตรงตามระยะทาง โดยใช้ Yaw Lock +++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def move_straight_for_distance(self, target_distance, forward_speed=0.2):
        """
        เคลื่อนที่เป็นเส้นตรงตามระยะทางที่กำหนด โดยใช้ Yaw Lock
        :param target_distance: ระยะทางเป้าหมาย (เมตร)
        :param forward_speed: ความเร็ว (m/s). > 0 คือเดินหน้า, < 0 คือถอยหลัง
        """
        direction = "Forward" if forward_speed > 0 else "Backward"
        print(f"\n--- Moving {direction} for {target_distance:.2f} meters (with Yaw Lock) ---")

        # 1. ล็อคทิศทางและตำแหน่งเริ่มต้น
        self.target_yaw = self.current_yaw
        start_x, start_y = self.current_x, self.current_y
        print(f"Locking Yaw at: {self.target_yaw:.2f}°. Start Pos: ({start_x:.2f}, {start_y:.2f})")

        # 2. เริ่ม Loop การเคลื่อนที่
        start_time = time.time()
        MAX_EXECUTION_TIME = 15 # Timeout 15 วินาที เผื่อหุ่นยนต์ติดขัด

        while True:
            # คำนวณระยะทางที่เคลื่อนที่ได้จริง
            distance_traveled = math.sqrt((self.current_x - start_x)**2 + (self.current_y - start_y)**2)

            # 3. ตรวจสอบเงื่อนไขการหยุด
            if distance_traveled >= target_distance:
                print(f"\n✅ Target distance reached! Traveled: {distance_traveled:.3f} m")
                break
            
            if time.time() - start_time > MAX_EXECUTION_TIME:
                print(f"\n⚠️ Movement timed out! Traveled: {distance_traveled:.3f} m")
                break

            # 4. คำนวณการชดเชยและสั่งเคลื่อนที่
            rotation_speed_z = self._calculate_yaw_correction()
            self.chassis.drive_speed(x=forward_speed, y=0, z=rotation_speed_z)

            print(f"Moving... Dist: {distance_traveled:.3f}/{target_distance:.2f} m | YawErr: {self.target_yaw - self.current_yaw:5.2f}°", end='\r')
            time.sleep(0.02)
        
        # 5. หยุดหุ่นยนต์ แต่ยังคงใช้ yaw correction เพื่อให้หยุดนิ่งจริงๆ
        self.chassis.drive_speed(x=0, y=0, z=self._calculate_yaw_correction())
        time.sleep(0.1)
        self.chassis.drive_speed(x=0, y=0, z=0) # หยุดสนิท

    # --- ฟังก์ชันสำหรับ Cleanup ---
    def cleanup(self):
        """ยกเลิกการ Subscribe ข้อมูลจากเซ็นเซอร์"""
        print("Unsubscribing from sensors...")
        try:
            self.chassis.unsub_attitude()
            self.chassis.unsub_position()
        except Exception as e:
            print(f"Error during unsubscribing: {e}")


# =======================================================================
# === ฟังก์ชัน main (แก้ไขเพื่อรอรับ Input จากผู้ใช้) ===
# =======================================================================
def main():
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    controller = PositionController(ep_robot)

    try:
        print("\n" + "="*50)
        print("✅ ROBOT IS READY FOR MANUAL CONTROL")
        print("   - Type 'yes' and press Enter to move forward 0.6m.")
        print("   - Type anything else (e.g., 'exit') to quit.")
        print("="*50 + "\n")

        # Loop หลักเพื่อรอรับคำสั่ง
        while True:
            command = input(">> Enter command: ").lower().strip()

            if command == 'yes':
                # เรียกใช้ฟังก์ชันเดินตรงตามระยะทาง
                controller.move_straight_for_distance(
                    target_distance=0.6, 
                    forward_speed=0.25 # สามารถปรับความเร็วได้ตามต้องการ
                )
                print("\nReady for the next command.")
            else:
                print("🛑 Exiting control loop.")
                break

    except KeyboardInterrupt:
        print("\n\nProgram stopped by user.")

    finally:
        print("\nStopping robot movement...")
        ep_robot.chassis.drive_speed(x=0, y=0, z=0) 
        time.sleep(0.5)
        
        # เรียกใช้ cleanup เพื่อ unsub เซ็นเซอร์
        controller.cleanup()
        
        print("Closing robot connection.")
        ep_robot.close()
        print("Connection closed.")

if __name__ == '__main__':
    main()