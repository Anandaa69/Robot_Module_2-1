# -*-coding:utf-8-*-

import robomaster
from robomaster import robot
import time
import statistics

# --- ฟังก์ชันพื้นฐาน (ไม่เปลี่ยนแปลง) ---

def convert_adc_to_cm(adc_value):
    """แปลงค่า ADC เป็นระยะทาง (cm)"""
    if adc_value <= 0:
        return float('inf')
    A = 30263
    B = -1.352
    distance_cm = A * (adc_value ** B)
    return distance_cm

def get_stable_distance(sensor_adaptor, sensor_id, port, num_samples=10):
    """ฟังก์ชันสำหรับอ่านค่าจากเซ็นเซอร์หลายๆ ครั้งแล้วหาค่าเฉลี่ย (median)"""
    readings = []
    print(f"Averaging sensor readings from port {port}...", end='')
    for _ in range(num_samples):
        adc = sensor_adaptor.get_adc(id=sensor_id, port=port)
        readings.append(convert_adc_to_cm(adc))
        time.sleep(0.02)
    print(" Done.")
    return statistics.median(readings)

# =======================================================================
# === คลาสสำหรับควบคุมการเคลื่อนที่พร้อม Yaw Lock (วิธีที่ถูกต้องและต่อเนื่อง) ===
# =======================================================================
class PositionController:
    def __init__(self, ep_robot):
        self.robot = ep_robot
        self.chassis = ep_robot.chassis
        self.current_yaw = 0.0
        self.target_yaw = 0.0 # ใช้ตัวแปรนี้เพื่อเก็บมุม Yaw เป้าหมายปัจจุบัน

        # 1. Subscribe ข้อมูล Attitude แค่ครั้งเดียวตอนเริ่มต้น
        self.chassis.sub_attitude(freq=20, callback=self._attitude_callback)
        time.sleep(1) # รอเล็กน้อยเพื่อให้แน่ใจว่าได้รับข้อมูลอย่างน้อย 1 ครั้ง

        # ตั้งค่า target_yaw เริ่มต้นให้เป็นมุมปัจจุบันทันทีที่เริ่มทำงาน
        self.target_yaw = self.current_yaw 
        print(f"Controller initialized. Initial target Yaw: {self.target_yaw:.2f} degrees")

    def _attitude_callback(self, attitude_info):
        """
        ฟังก์ชัน Callback ที่จะถูก SDK เรียกใช้ทุกครั้งที่ได้รับข้อมูล Yaw ใหม่
        """
        yaw, pitch, roll = attitude_info
        self.current_yaw = yaw

    def _calculate_yaw_correction(self):
        """
        คำนวณความเร็วในการหมุนเพื่อรักษามุม Yaw ให้คงที่
        """
        KP_YAW = 0.8  # ค่าความแรงในการแก้แกนหมุน (อาจต้องปรับจูน)
        MAX_YAW_SPEED = 20 # ความเร็วการหมุนสูงสุด (องศา/วินาที)

        yaw_error = self.target_yaw - self.current_yaw

        if yaw_error > 180: yaw_error -= 360
        elif yaw_error < -180: yaw_error += 360

        rotation_speed_z = KP_YAW * yaw_error
        return max(min(rotation_speed_z, MAX_YAW_SPEED), -MAX_YAW_SPEED)

    def adjust_position(self, sensor_adaptor, sensor_id, sensor_port, 
                        target_distance_cm, side_name):
        """
        ฟังก์ชันหลักที่ใช้ในการปรับตำแหน่งด้านข้างพร้อมล็อคแกนหมุน
        """
        print(f"\n--- Adjusting {side_name} Side (with Yaw Lock) ---")
        
        # --- พารามิเตอร์ควบคุม ---
        TOLERANCE_CM = 0.5
        MAX_EXECUTION_TIME = 10 

        KP_SLIDE = 0.035
        MAX_SLIDE_SPEED = 0.15
        
        # อัปเดต target_yaw ให้เป็นมุมปัจจุบัน ณ จุดเริ่มต้นของการปรับนี้
        self.target_yaw = self.current_yaw
        print(f"[{side_name}] Locking target Yaw at: {self.target_yaw:.2f} degrees")
        
        initial_distance = get_stable_distance(sensor_adaptor, sensor_id, sensor_port)
        print(f"[{side_name}] Initial Distance: {initial_distance:.2f} cm, Target: {target_distance_cm:.2f} cm")
        
        distance_error = initial_distance - target_distance_cm
        
        if abs(distance_error) > TOLERANCE_CM:
            print(f"[{side_name}] Executing movement with Yaw Lock...")
            start_time = time.time()

            while True:
                current_distance = convert_adc_to_cm(sensor_adaptor.get_adc(id=sensor_id, port=sensor_port))
                
                distance_error = target_distance_cm - current_distance
                rotation_speed_z = self._calculate_yaw_correction() # ใช้ฟังก์ชันคำนวณ Yaw Correction

                if abs(distance_error) <= TOLERANCE_CM:
                    print(f"\n[{side_name}] Target distance reached!")
                    break
                
                if time.time() - start_time > MAX_EXECUTION_TIME:
                    print(f"\n[{side_name}] Movement timed out!")
                    break

                slide_speed_y = KP_SLIDE * distance_error
                slide_speed_y = max(min(slide_speed_y, MAX_SLIDE_SPEED), -MAX_SLIDE_SPEED)
                
                # ส่งคำสั่งเคลื่อนที่ โดยบังคับให้ x=0 และเพิ่มการแก้แกน z เข้าไป
                self.chassis.drive_speed(x=0, y=slide_speed_y, z=rotation_speed_z)
                
                print(f"Adjusting {side_name}... DistErr: {distance_error:5.2f}cm | YawErr: {self.target_yaw - self.current_yaw:5.2f}° | Y_Spd: {slide_speed_y:4.2f} | Z_Spd: {rotation_speed_z:5.1f}", end='\r')
                time.sleep(0.02)
        else:
            print(f"[{side_name}] Already at the target position.")
        
        # หลังจากปรับเสร็จ ให้สั่งหยุดการเคลื่อนที่ แต่ยังคง Yaw Lock ไว้
        self.chassis.drive_speed(x=0, y=0, z=self._calculate_yaw_correction())
        time.sleep(0.1) # ให้หุ่นยนต์มีเวลาปรับเล็กน้อยก่อนไปขั้นตอนถัดไป

    def hold_yaw_during_pause(self, duration_s):
        """
        ฟังก์ชันที่ทำให้หุ่นยนต์คุมมุม Yaw ไว้เฉยๆ เป็นระยะเวลาหนึ่ง
        """
        print(f"\n--- Holding Yaw during pause for {duration_s} seconds ---")
        start_time = time.time()
        
        while time.time() - start_time < duration_s:
            rotation_speed_z = self._calculate_yaw_correction()
            self.chassis.drive_speed(x=0, y=0, z=rotation_speed_z) # คุมแค่ Z
            print(f"Holding Yaw... YawErr: {self.target_yaw - self.current_yaw:5.2f}° | Z_Spd: {rotation_speed_z:5.1f}", end='\r')
            time.sleep(0.02)
        
        print("\nYaw hold complete.")
        # หลังจากจบการ hold แล้ว ยังคงต้องสั่ง chassis.drive_speed เพื่อให้มันยังคงคุม yaw ต่อไปจนกว่าจะมีการสั่งใหม่
        # หรือหยุดทั้งหมดเมื่อ main สิ้นสุด
        self.chassis.drive_speed(x=0, y=0, z=self._calculate_yaw_correction())


def main():
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    sensor_adaptor = ep_robot.sensor_adaptor
    
    SENSOR_ADAPTOR_ID = 1
    LEFT_SENSOR_PORT = 2
    LEFT_TARGET_CM = 22.5
    RIGHT_SENSOR_PORT = 1
    RIGHT_TARGET_CM = 26.0

    # สร้าง instance ของ Controller
    controller = PositionController(ep_robot)

    try:
        # --- ลำดับที่ 1: ปรับตำแหน่งด้านซ้าย ---
        controller.adjust_position(sensor_adaptor, SENSOR_ADAPTOR_ID,
                                   LEFT_SENSOR_PORT, LEFT_TARGET_CM, "Left")

        print("\nLeft side adjustment complete.")
        # --- ใช้ฟังก์ชัน hold_yaw_during_pause แทน time.sleep ธรรมดา ---
        controller.hold_yaw_during_pause(2.0)

        # --- ลำดับที่ 2: ปรับตำแหน่งด้านขวา ---
        controller.adjust_position(sensor_adaptor, SENSOR_ADAPTOR_ID,
                                   RIGHT_SENSOR_PORT, RIGHT_TARGET_CM, "Right")
        
        print("\nSequence complete.")

    except KeyboardInterrupt:
        print("\n\nProgram stopped by user.")

    finally:
        print("Stopping robot movement...")
        ep_robot.chassis.drive_speed(x=0, y=0, z=0) 
        time.sleep(0.5)
        print("Closing robot connection.")
        # 2. Unsubscribe ข้อมูล Attitude แค่ครั้งเดียวตอนจบ
        ep_robot.chassis.unsub_attitude()
        ep_robot.close()
        print("Connection closed.")

if __name__ == '__main__':
    main()