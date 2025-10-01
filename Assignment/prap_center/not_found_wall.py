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
        self.target_yaw = 0.0

        self.chassis.sub_attitude(freq=20, callback=self._attitude_callback)
        time.sleep(1) 

        self.target_yaw = self.current_yaw 
        print(f"Controller initialized. Initial target Yaw: {self.target_yaw:.2f} degrees")

    def _attitude_callback(self, attitude_info):
        yaw, pitch, roll = attitude_info
        self.current_yaw = yaw

    def _calculate_yaw_correction(self):
        KP_YAW = 0.8
        MAX_YAW_SPEED = 20

        yaw_error = self.target_yaw - self.current_yaw
        if yaw_error > 180: yaw_error -= 360
        elif yaw_error < -180: yaw_error += 360

        rotation_speed_z = KP_YAW * yaw_error
        return max(min(rotation_speed_z, MAX_YAW_SPEED), -MAX_YAW_SPEED)

    ### เพิ่มเข้ามาใหม่: ฟังก์ชันสำหรับตรวจสอบการมีอยู่ของกำแพง ###
    def check_for_wall(self, sensor_adaptor, sensor_id, port, side_name):
        """
        ตรวจสอบว่ามีกำแพงอยู่ด้านที่กำหนดหรือไม่ โดยเช็คจากความเสถียรของค่าเซ็นเซอร์
        - คืนค่า True หากพบกำแพง
        - คืนค่า False หากไม่พบ (ค่าแกว่งเกินไป หรือไกลเกินไป)
        """
        print(f"\n[{side_name}] Performing wall detection check...")

        # --- พารามิเตอร์สำหรับตรวจสอบกำแพง (สามารถปรับจูนได้) ---
        CHECK_DURATION_S = 1.0  # ระยะเวลาที่ใช้ในการเก็บข้อมูลเพื่อตรวจสอบ (วินาที)
        # ค่าความเบี่ยงเบนมาตรฐานสูงสุดที่ยอมรับได้ (ค่ายิ่งน้อยแปลว่าค่ายิ่งต้องนิ่ง)
        MAX_STD_DEV_THRESHOLD = 4.0
        # ระยะทางเฉลี่ยสูงสุดที่ยอมรับได้ว่าเป็นกำแพง
        MAX_AVG_DISTANCE_THRESHOLD = 80.0

        readings = []
        start_time = time.time()
        while time.time() - start_time < CHECK_DURATION_S:
            adc = sensor_adaptor.get_adc(id=sensor_id, port=port)
            readings.append(convert_adc_to_cm(adc))
            time.sleep(0.02)

        # ต้องการข้อมูลอย่างน้อย 2 ค่าเพื่อคำนวณ std dev
        if len(readings) < 2:
            print(f"[{side_name}] Wall Check Error: Not enough sensor data collected.")
            return False

        avg_distance = statistics.mean(readings)
        std_dev = statistics.stdev(readings)

        print(f"[{side_name}] Wall Check Stats -> Avg Dist: {avg_distance:.2f} cm, Std Dev: {std_dev:.2f}")

        # เงื่อนไขในการตัดสินว่าไม่มีกำแพง
        if std_dev > MAX_STD_DEV_THRESHOLD:
            print(f"[{side_name}] Wall NOT Detected: Sensor readings are too unstable (Std Dev > {MAX_STD_DEV_THRESHOLD}).")
            return False
        
        if avg_distance > MAX_AVG_DISTANCE_THRESHOLD:
            print(f"[{side_name}] Wall NOT Detected: Average distance is too far (Avg Dist > {MAX_AVG_DISTANCE_THRESHOLD} cm).")
            return False

        print(f"[{side_name}] Wall detected. Proceeding with adjustment.")
        return True

    def adjust_position(self, sensor_adaptor, sensor_id, sensor_port, 
                        target_distance_cm, side_name):
        """
        ฟังก์ชันหลักที่ใช้ในการปรับตำแหน่งด้านข้างพร้อมล็อคแกนหมุน
        (ปรับปรุงให้คืนค่า True เมื่อสำเร็จ และ False เมื่อล้มเหลว)
        """
        print(f"\n--- Adjusting {side_name} Side (with Yaw Lock) ---")
        
        ### เพิ่มเข้ามาใหม่: เรียกใช้ฟังก์ชันตรวจสอบกำแพงก่อนเริ่มทำงาน ###
        if not self.check_for_wall(sensor_adaptor, sensor_id, sensor_port, side_name):
            return False # คืนค่า False เพื่อบอกว่าไม่พบกำแพง และหยุดการทำงานส่วนนี้

        # --- พารามิเตอร์ควบคุม ---
        TOLERANCE_CM = 0.5
        MAX_EXECUTION_TIME = 10 
        KP_SLIDE = 0.035
        MAX_SLIDE_SPEED = 0.15
        
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
                rotation_speed_z = self._calculate_yaw_correction()

                if abs(distance_error) <= TOLERANCE_CM:
                    print(f"\n[{side_name}] Target distance reached!")
                    break
                
                if time.time() - start_time > MAX_EXECUTION_TIME:
                    print(f"\n[{side_name}] Movement timed out!")
                    break

                slide_speed_y = KP_SLIDE * distance_error
                slide_speed_y = max(min(slide_speed_y, MAX_SLIDE_SPEED), -MAX_SLIDE_SPEED)
                
                self.chassis.drive_speed(x=0, y=slide_speed_y, z=rotation_speed_z)
                
                print(f"Adjusting {side_name}... DistErr: {distance_error:5.2f}cm | YawErr: {self.target_yaw - self.current_yaw:5.2f}° | Y_Spd: {slide_speed_y:4.2f} | Z_Spd: {rotation_speed_z:5.1f}", end='\r')
                time.sleep(0.02)
        else:
            print(f"[{side_name}] Already at the target position.")
        
        self.chassis.drive_speed(x=0, y=0, z=self._calculate_yaw_correction())
        time.sleep(0.1)
        
        return True # ### เพิ่มเข้ามาใหม่: คืนค่า True เพื่อบอกว่าทำงานสำเร็จ ###


    def hold_yaw_during_pause(self, duration_s):
        print(f"\n--- Holding Yaw during pause for {duration_s} seconds ---")
        start_time = time.time()
        
        while time.time() - start_time < duration_s:
            rotation_speed_z = self._calculate_yaw_correction()
            self.chassis.drive_speed(x=0, y=0, z=rotation_speed_z)
            print(f"Holding Yaw... YawErr: {self.target_yaw - self.current_yaw:5.2f}° | Z_Spd: {rotation_speed_z:5.1f}", end='\r')
            time.sleep(0.02)
        
        print("\nYaw hold complete.")
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

    controller = PositionController(ep_robot)

    try:
        # --- ลำดับที่ 1: ปรับตำแหน่งด้านซ้าย ---
        # ### ปรับปรุง: ตรวจสอบผลลัพธ์จากฟังก์ชัน ###
        is_successful = controller.adjust_position(sensor_adaptor, SENSOR_ADAPTOR_ID,
                                                   LEFT_SENSOR_PORT, LEFT_TARGET_CM, "Left")
        
        # หากการปรับด้านซ้ายล้มเหลว (ไม่เจอผนัง) ให้จบโปรแกรมทันที
        if not is_successful:
            print("\nStopping program because a wall was not detected.")
            return # ออกจากฟังก์ชัน main เพื่อไปที่ finally

        print("\nLeft side adjustment complete.")
        controller.hold_yaw_during_pause(2.0)

        # --- ลำดับที่ 2: ปรับตำแหน่งด้านขวา ---
        # ### ปรับปรุง: ตรวจสอบผลลัพธ์จากฟังก์ชัน ###
        is_successful = controller.adjust_position(sensor_adaptor, SENSOR_ADAPTOR_ID,
                                                   RIGHT_SENSOR_PORT, RIGHT_TARGET_CM, "Right")

        # หากการปรับด้านขวาล้มเหลว (ไม่เจอผนัง) ให้จบโปรแกรมทันที
        if not is_successful:
            print("\nStopping program because a wall was not detected.")
            return # ออกจากฟังก์ชัน main เพื่อไปที่ finally
        
        print("\nSequence complete.")

    except KeyboardInterrupt:
        print("\n\nProgram stopped by user.")

    finally:
        print("Stopping robot movement...")
        ep_robot.chassis.drive_speed(x=0, y=0, z=0) 
        time.sleep(0.5)
        print("Closing robot connection.")
        ep_robot.chassis.unsub_attitude()
        ep_robot.close()
        print("Connection closed.")

if __name__ == '__main__':
    main()