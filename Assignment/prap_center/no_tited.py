# -*-coding:utf-8-*-

import robomaster
from robomaster import robot
import time
import statistics
import threading # <-- 1. Import threading

# --- ฟังก์ชันพื้นฐาน (ไม่เปลี่ยนแปลง) ---
def convert_adc_to_cm(adc_value):
    if adc_value <= 0: return float('inf')
    return 30263 * (adc_value ** -1.352)

def get_stable_distance(sensor_adaptor, sensor_id, port, num_samples=10):
    readings = []
    print(f"Averaging sensor readings from port {port}...", end='')
    for _ in range(num_samples):
        adc = sensor_adaptor.get_adc(id=sensor_id, port=port)
        readings.append(convert_adc_to_cm(adc))
        time.sleep(0.02)
    print(" Done.")
    return statistics.median(readings)

# =======================================================================
# === คลาสสำหรับควบคุมการเคลื่อนที่ (ปรับปรุงใหม่ทั้งหมด) ===
# =======================================================================
class PositionController:
    def __init__(self, ep_robot):
        self.robot = ep_robot
        self.chassis = ep_robot.chassis
        self.current_yaw = 0.0
        self.target_yaw = 0.0
        
        # --- 2. ตัวแปรสำหรับควบคุม Thread ---
        self._is_running = True
        self._yaw_hold_enabled = False
        self._stabilizer_thread = None

        self.chassis.sub_attitude(freq=20, callback=self._attitude_callback)
        time.sleep(0.5) 
        self.target_yaw = self.current_yaw 
        
        # --- 3. สร้างและเริ่ม Thread ---
        self._stabilizer_thread = threading.Thread(target=self._yaw_stabilizer_task, daemon=True)
        self._stabilizer_thread.start()
        
        print(f"Controller initialized. Initial target Yaw: {self.target_yaw:.2f} degrees")

    def _attitude_callback(self, attitude_info):
        yaw, pitch, roll = attitude_info
        self.current_yaw = yaw

    def _calculate_yaw_correction(self):
        KP_YAW = 1.6
        MAX_YAW_SPEED = 20
        YAW_DEADBAND_DEG = 0.5 # เพิ่ม Deadband เพื่อความนิ่ง
        
        yaw_error = self.target_yaw - self.current_yaw
        if yaw_error > 180: yaw_error -= 360
        elif yaw_error < -180: yaw_error += 360
        
        if abs(yaw_error) < YAW_DEADBAND_DEG:
            return 0.0

        rotation_speed_z = KP_YAW * yaw_error
        return max(min(rotation_speed_z, MAX_YAW_SPEED), -MAX_YAW_SPEED)

    # --- 4. เมธอดสำหรับ Thread โดยเฉพาะ ---
    def _yaw_stabilizer_task(self):
        """
        ทำงานใน background thread เพื่อรักษามุม Yaw ของหุ่นยนต์ให้คงที่
        เมื่อ self._yaw_hold_enabled เป็น True
        """
        print("Yaw stabilizer thread started.")
        while self._is_running:
            if self._yaw_hold_enabled:
                rotation_speed_z = self._calculate_yaw_correction()
                # ส่งคำสั่งขับเคลื่อนเมื่อจำเป็นเท่านั้น เพื่อลดภาระ
                if abs(rotation_speed_z) > 0:
                    self.chassis.drive_speed(x=0, y=0, z=rotation_speed_z)
            
            time.sleep(0.02) # ลดการทำงานของ CPU
        print("Yaw stabilizer thread stopped.")

    # --- 5. เมธอดสำหรับควบคุมการทำงานของ Thread จากภายนอก ---
    def start_yaw_hold(self):
        """เปิดใช้งานการรักษามุม Yaw อัตโนมัติ"""
        print("\n[Stabilizer] Starting autonomous Yaw Hold.")
        self.target_yaw = self.current_yaw # อัปเดตมุมเป้าหมายทุกครั้งที่เริ่ม Hold
        self._yaw_hold_enabled = True

    def stop_yaw_hold(self):
        """ปิดใช้งานการรักษามุม Yaw (เพื่อให้ฟังก์ชันอื่นควบคุมแทน)"""
        print("\n[Stabilizer] Stopping autonomous Yaw Hold.")
        self._yaw_hold_enabled = False
        # หยุดหุ่นยนต์ก่อนที่ฟังก์ชันอื่นจะเข้ามาควบคุม
        self.chassis.drive_speed(x=0, y=0, z=0)
        time.sleep(0.1)

    def close(self):
        """หยุดการทำงานของ Thread และยกเลิกการ Subscribe"""
        print("Closing controller...")
        self._is_running = False
        if self._stabilizer_thread:
            self._stabilizer_thread.join(timeout=1) # รอ Thread จบการทำงาน
        self.chassis.unsub_attitude()


    def check_for_wall(self, sensor_adaptor, sensor_id, port, side_name):
        # ... (โค้ดส่วนนี้ไม่เปลี่ยนแปลง) ...
        print(f"\n[{side_name}] Performing wall detection check...")
        CHECK_DURATION_S = 1.65
        MAX_STD_DEV_THRESHOLD = 0.8
        MAX_AVG_DISTANCE_THRESHOLD = 50
        readings = []
        start_time = time.time()
        while time.time() - start_time < CHECK_DURATION_S:
            adc = sensor_adaptor.get_adc(id=sensor_id, port=port)
            readings.append(convert_adc_to_cm(adc))
            time.sleep(0.02)
        if len(readings) < 2:
            print(f"[{side_name}] Wall Check Error: Not enough sensor data collected.")
            return False
        avg_distance = statistics.mean(readings)
        std_dev = statistics.stdev(readings)
        print(f"[{side_name}] Wall Check Stats -> Avg Dist: {avg_distance:.2f} cm, Std Dev: {std_dev:.2f}")
        if std_dev > MAX_STD_DEV_THRESHOLD:
            print(f"[{side_name}] Wall NOT Detected: Sensor readings are too unstable (Std Dev > {MAX_STD_DEV_THRESHOLD}).")
            return False
        if avg_distance > MAX_AVG_DISTANCE_THRESHOLD:
            print(f"[{side_name}] Wall NOT Detected: Average distance is too far (Avg Dist > {MAX_AVG_DISTANCE_THRESHOLD} cm).")
            return False
        print(f"[{side_name}] Wall detected. Proceeding with adjustment.")
        return True

    def adjust_position(self, sensor_adaptor, sensor_id, sensor_port, 
                        target_distance_cm, side_name, direction_multiplier):
        
        print(f"\n--- Adjusting {side_name} Side (with Yaw Lock) ---")
        
        if not self.check_for_wall(sensor_adaptor, sensor_id, sensor_port, side_name):
            return False

        # --- 6. หยุดการทำงานของ Thread ก่อนเข้าควบคุมเอง ---
        self.stop_yaw_hold()
        self.target_yaw = self.current_yaw # ตั้งค่ามุมเป้าหมายใหม่ล่าสุด
        
        print(f"[{side_name}] Locking target Yaw at: {self.target_yaw:.2f} degrees")
        
        TOLERANCE_CM = 0.5
        MAX_EXECUTION_TIME = 10 
        KP_SLIDE = 0.035
        MAX_SLIDE_SPEED = 0.15

        initial_distance = get_stable_distance(sensor_adaptor, sensor_id, sensor_port)
        print(f"[{side_name}] Initial Distance: {initial_distance:.2f} cm, Target: {target_distance_cm:.2f} cm")
        
        if abs(initial_distance - target_distance_cm) <= TOLERANCE_CM:
            print(f"[{side_name}] Already at the target position.")
            self.start_yaw_hold() # <-- 7. กลับมาเปิดใช้ Thread
            return True

        print(f"[{side_name}] Executing movement with Yaw Lock...")
        start_time = time.time()

        while True:
            current_distance = convert_adc_to_cm(sensor_adaptor.get_adc(id=sensor_id, port=sensor_port))
            
            distance_error = target_distance_cm - current_distance
            rotation_speed_z = self._calculate_yaw_correction()

            if abs(distance_error) <= TOLERANCE_CM or time.time() - start_time > MAX_EXECUTION_TIME:
                if abs(distance_error) <= TOLERANCE_CM:
                    print(f"\n[{side_name}] Target distance reached!")
                else:
                    print(f"\n[{side_name}] Movement timed out!")
                break

            slide_speed_y = direction_multiplier * KP_SLIDE * distance_error
            slide_speed_y = max(min(slide_speed_y, MAX_SLIDE_SPEED), -MAX_SLIDE_SPEED)
            
            self.chassis.drive_speed(x=0, y=slide_speed_y, z=rotation_speed_z)
            
            print(f"Adjusting {side_name}... DistErr: {distance_error:5.2f}cm | YawErr: {self.target_yaw - self.current_yaw:5.2f}° | Y_Spd: {slide_speed_y:4.2f} | Z_Spd: {rotation_speed_z:5.1f}", end='\r')
            time.sleep(0.02)
        
        # หยุดนิ่งแล้วค่อยคืนการควบคุมให้ Thread
        self.chassis.drive_speed(x=0, y=0, z=self._calculate_yaw_correction())
        time.sleep(0.1)
        
        self.start_yaw_hold() # <-- 7. คืนการควบคุมให้ Thread หลัก
        return True

# =======================================================================
# === ฟังก์ชัน main (ปรับปรุงให้ใช้ Controller ใหม่) ===
# =======================================================================
def main():
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    sensor_adaptor = ep_robot.sensor_adaptor
    
    SENSOR_ADAPTOR_ID = 1
    LEFT_SENSOR_PORT = 1
    LEFT_TARGET_CM = 18

    RIGHT_ADAPTOR_ID = 2
    RIGHT_SENSOR_PORT = 1
    RIGHT_TARGET_CM = 16.0

    controller = PositionController(ep_robot)
    controller.start_yaw_hold() # <-- 8. เริ่มการรักษามุม Yaw ทันที

    left_wall_found = False
    right_wall_found = False

    try:
        # ลำดับที่ 1: ปรับตำแหน่งด้านซ้าย
        left_wall_found = controller.adjust_position(sensor_adaptor, SENSOR_ADAPTOR_ID,
                                                     LEFT_SENSOR_PORT, LEFT_TARGET_CM, 
                                                     "Left", direction_multiplier=1)
        
        if left_wall_found:
            print("\nLeft side adjustment complete. Pausing for 2 seconds...")
            time.sleep(0.55) # Thread จะรักษามุมให้เองระหว่างรอ
        else:
            print("\nSkipping pause because left wall was not found.")

        # ลำดับที่ 2: ปรับตำแหน่งด้านขวา
        right_wall_found = controller.adjust_position(sensor_adaptor, RIGHT_ADAPTOR_ID,
                                                      RIGHT_SENSOR_PORT, RIGHT_TARGET_CM, 
                                                      "Right", direction_multiplier=-1)
        
        if right_wall_found:
            print("\nRight side adjustment complete.")
        
        print("\n--- Sequence Finished ---")
        if not left_wall_found and not right_wall_found:
            print("\nCritical Failure: No walls detected on either side.")
        else:
            print("\nTask complete. At least one side was successfully adjusted.")

    except KeyboardInterrupt:
        print("\n\nProgram stopped by user.")

    finally:
        print("Stopping robot and cleaning up...")
        controller.close() # <-- 9. สั่งให้ Controller ปิด Thread อย่างถูกต้อง
        ep_robot.chassis.drive_speed(x=0, y=0, z=0) 
        time.sleep(0.5)
        ep_robot.close()
        print("Connection closed.")

if __name__ == '__main__':
    main()