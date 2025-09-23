# -*-coding:utf-8-*-

import robomaster
from robomaster import robot
import time
import statistics

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
# === คลาสสำหรับควบคุมการเคลื่อนที่ (มีการแก้ไข) ===
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

    def check_for_wall(self, sensor_adaptor, sensor_id, port, side_name):
        print(f"\n[{side_name}] Performing wall detection check...")
        CHECK_DURATION_S = 1.0
        MAX_STD_DEV_THRESHOLD = 4.0
        MAX_AVG_DISTANCE_THRESHOLD = 80.0
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

    # ### แก้ไข: เพิ่มพารามิเตอร์ direction_multiplier เข้ามา ###
    def adjust_position(self, sensor_adaptor, sensor_id, sensor_port, 
                        target_distance_cm, side_name, direction_multiplier):
        
        print(f"\n--- Adjusting {side_name} Side (with Yaw Lock) ---")
        
        if not self.check_for_wall(sensor_adaptor, sensor_id, sensor_port, side_name):
            return False

        TOLERANCE_CM = 0.5
        MAX_EXECUTION_TIME = 10 
        KP_SLIDE = 0.035
        MAX_SLIDE_SPEED = 0.15
        
        self.target_yaw = self.current_yaw
        print(f"[{side_name}] Locking target Yaw at: {self.target_yaw:.2f} degrees")
        
        initial_distance = get_stable_distance(sensor_adaptor, sensor_id, sensor_port)
        print(f"[{side_name}] Initial Distance: {initial_distance:.2f} cm, Target: {target_distance_cm:.2f} cm")
        
        if abs(initial_distance - target_distance_cm) > TOLERANCE_CM:
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

                # ### แก้ไข: นำ direction_multiplier มาคูณเพื่อควบคุมทิศทาง ###
                slide_speed_y = direction_multiplier * KP_SLIDE * distance_error
                slide_speed_y = max(min(slide_speed_y, MAX_SLIDE_SPEED), -MAX_SLIDE_SPEED)
                
                self.chassis.drive_speed(x=0, y=slide_speed_y, z=rotation_speed_z)
                
                print(f"Adjusting {side_name}... DistErr: {distance_error:5.2f}cm | YawErr: {self.target_yaw - self.current_yaw:5.2f}° | Y_Spd: {slide_speed_y:4.2f} | Z_Spd: {rotation_speed_z:5.1f}", end='\r')
                time.sleep(0.02)
        else:
            print(f"[{side_name}] Already at the target position.")
        
        self.chassis.drive_speed(x=0, y=0, z=self._calculate_yaw_correction())
        time.sleep(0.1)
        
        return True

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

# =======================================================================
# === ฟังก์ชัน main (มีการแก้ไข) ===
# =======================================================================
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

    left_wall_found = False
    right_wall_found = False

    try:
        # --- ลำดับที่ 1: ตรวจสอบและปรับตำแหน่งด้านซ้าย ---
        # ### แก้ไข: เพิ่ม direction_multiplier=1 (สำหรับสไลด์ซ้าย) ###
        left_wall_found = controller.adjust_position(sensor_adaptor, SENSOR_ADAPTOR_ID,
                                                     LEFT_SENSOR_PORT, LEFT_TARGET_CM, 
                                                     "Left", direction_multiplier=1)
        
        if left_wall_found:
            print("\nLeft side adjustment complete.")
            controller.hold_yaw_during_pause(2.0)
        else:
            print("\nSkipping pause because left wall was not found.")

        # --- ลำดับที่ 2: ตรวจสอบและปรับตำแหน่งด้านขวา ---
        # ### แก้ไข: เพิ่ม direction_multiplier=-1 (สำหรับสไลด์ขวา) ###
        right_wall_found = controller.adjust_position(sensor_adaptor, SENSOR_ADAPTOR_ID,
                                                      RIGHT_SENSOR_PORT, RIGHT_TARGET_CM, 
                                                      "Right", direction_multiplier=-1)
        
        if right_wall_found:
            print("\nRight side adjustment complete.")
        
        print("\n--- Sequence Finished ---")

        if not left_wall_found and not right_wall_found:
            print("\nCritical Failure: No walls detected on either side. Program will now stop.")
        else:
            print("\nTask complete. At least one side was successfully adjusted.")

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