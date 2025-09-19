# -*-coding:utf-8-*-

import robomaster
from robomaster import robot
import time

# =======================================================================
# --- ค่าคงที่และตัวแปรสำหรับการปรับจูน (เวอร์ชันแก้ไขทิศทางที่ถูกต้อง) ---
# =======================================================================

# --- การตั้งค่าเซ็นเซอร์ IR ด้านซ้าย ---
IR_FRONT_LEFT_ID = 3
IR_FRONT_LEFT_PORT = 1
IR_REAR_LEFT_ID = 1
IR_REAR_LEFT_PORT = 1
SENSOR_OFFSET_CM = 1.0 

# --- การตั้งค่าเป้าหมายการเคลื่อนที่ ---
TARGET_DISTANCE_CM = 5.0
FORWARD_SPEED = 0.18
STOP_DISTANCE_CM = 32.0
STOP_DISTANCE_MM = STOP_DISTANCE_CM * 10

# --- PD Controller สำหรับการหมุน (แกน z) ---
KP_DISTANCE = 2
KP_ANGLE = 2.5
KD = 8.0
MAX_ROTATE_SPEED = 30.0

# --- ตัวแปร Global ---
last_error = 0.0
tof_distances = {}
imu_data = {} # <<<<<<<<<<<< [เพิ่ม!] สำหรับเก็บค่าจาก IMU
telemetry_data = {} # <<<<<<<<<<<< [เพิ่ม!] สำหรับเก็บข้อมูลการเคลื่อนที่

# =======================================================================
# --- ฟังก์ชันต่างๆ ---
# =======================================================================
def convert_adc_to_cm(adc_value):
    if adc_value is None or adc_value <= 0: return float('inf')
    A = 30263
    B = -1.352
    return A * (adc_value ** B)

def sub_distance_handler(sub_info):
    global tof_distances
    if sub_info and sub_info[0] is not None:
        tof_distances['front'] = sub_info[0]

# <<<<<<<<<<<< [เพิ่ม!] Callback function สำหรับ IMU >>>>>>>>>>>>
def sub_imu_handler(sub_info):
    global imu_data
    if sub_info and sub_info[2] is not None:
        imu_data['yaw'] = sub_info[2]

# =======================================================================
# --- Main Program ---
# =======================================================================
def main():
    global last_error, telemetry_data
    
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_robot.set_robot_mode(mode=robomaster.robot.CHASSIS_LEAD)
    print("Robot mode set to CHASSIS_LEAD.")

    sensor_adaptor = ep_robot.sensor_adaptor
    chassis = ep_robot.chassis
    sensor = ep_robot.sensor
    ep_gimbal = ep_robot.gimbal

    ep_gimbal.recenter().wait_for_completed()
    print("Gimbal recentered.")

    sensor.sub_distance(freq=20, callback=sub_distance_handler)
    print("Subscribed to ToF sensor data.")
    
    # <<<<<<<<<<<< [เพิ่ม!] เปิดรับข้อมูลจาก IMU >>>>>>>>>>>>
    chassis.sub_imu(freq=50, callback=sub_imu_handler)
    print("Subscribed to IMU sensor data.")
    
    print("\n=============================================")
    print("=== Corrected Direction Wall Following ===")
    print(f"Target Distance: {TARGET_DISTANCE_CM} cm")
    print("Control Scheme: +z = Clockwise (Right), -z = Counter-Clockwise (Left)")
    print("=============================================")
    time.sleep(3) # รอให้เซ็นเซอร์เริ่มทำงานและส่งข้อมูลที่เสถียร

    # --- [เพิ่ม!] ส่วนเก็บข้อมูลเริ่มต้น ---
    try:
        # อ่านค่าเริ่มต้น
        start_dist_left = convert_adc_to_cm(sensor_adaptor.get_adc(id=IR_FRONT_LEFT_ID, port=IR_FRONT_LEFT_PORT))
        start_yaw = imu_data.get('yaw', 0.0)
        start_time = time.time()
        
        # บันทึกค่า
        telemetry_data['start_time'] = start_time
        telemetry_data['start_yaw'] = start_yaw
        telemetry_data['start_dist_left'] = start_dist_left
        print("--- Initial state recorded ---")
    except Exception as e:
        print(f"Error recording initial state: {e}")
        return

    # --- ประกาศตัวแปรที่จะใช้เก็บค่าสุดท้าย ---
    final_dist_left = 0
    final_yaw = 0
    final_stop_distance_mm = float('inf')

    try:
        while True:
            # --- 1. อ่านค่าจากเซ็นเซอร์ ---
            dist_front = convert_adc_to_cm(sensor_adaptor.get_adc(id=IR_FRONT_LEFT_ID, port=IR_FRONT_LEFT_PORT))
            dist_rear = convert_adc_to_cm(sensor_adaptor.get_adc(id=IR_REAR_LEFT_ID, port=IR_REAR_LEFT_PORT))
            front_tof_mm = tof_distances.get('front', float('inf'))

            # --- 2. ตรวจสอบเงื่อนไขการหยุด ---
            if front_tof_mm <= STOP_DISTANCE_MM:
                print(f"\nObstacle detected. Stopping.")
                final_stop_distance_mm = front_tof_mm # <<<<<<<<<<<< [เพิ่ม!] บันทึกระยะที่หยุด
                chassis.drive_speed(x=0, y=0, z=0)
                break

            # ... (ส่วนคำนวณ Error และการควบคุมเหมือนเดิมทุกประการ) ...
            error_distance = TARGET_DISTANCE_CM - dist_front
            error_angle = (dist_rear + SENSOR_OFFSET_CM) - dist_front
            total_error = (KP_DISTANCE * error_distance) + (KP_ANGLE * error_angle)
            derivative = total_error - last_error
            z_speed = (total_error + (KD * derivative))
            last_error = total_error
            z_speed = max(-MAX_ROTATE_SPEED, min(MAX_ROTATE_SPEED, z_speed))
            chassis.drive_speed(x=FORWARD_SPEED, y=0, z=z_speed)
            
            print(f"Dist F:{dist_front:5.1f} R:{dist_rear:5.1f} | Err D:{error_distance:5.1f} A:{error_angle:5.1f} | Z_Speed:{z_speed:5.1f}", end='\r')
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\nProgram stopped by user.")

    finally:
        # --- [เพิ่ม!] ส่วนเก็บข้อมูลสิ้นสุดและแสดงผล ---
        end_time = time.time()
        final_dist_left = convert_adc_to_cm(sensor_adaptor.get_adc(id=IR_FRONT_LEFT_ID, port=IR_FRONT_LEFT_PORT))
        final_yaw = imu_data.get('yaw', 0.0)

        telemetry_data['total_travel_time'] = end_time - telemetry_data.get('start_time', end_time)
        telemetry_data['end_yaw'] = final_yaw
        telemetry_data['end_dist_left'] = final_dist_left
        if final_stop_distance_mm != float('inf'):
            telemetry_data['stop_distance_cm'] = final_stop_distance_mm / 10.0

        print("\n\n=============================================")
        print("=========== TELEMETRY SUMMARY ===========")
        print(f"  Total Travel Time : {telemetry_data.get('total_travel_time', 0):.2f} seconds")
        print(f"  Start Yaw Angle   : {telemetry_data.get('start_yaw', 0):.2f} degrees")
        print(f"  End Yaw Angle     : {telemetry_data.get('end_yaw', 0):.2f} degrees")
        print(f"  Start Left Distance : {telemetry_data.get('start_dist_left', 0):.2f} cm")
        print(f"  End Left Distance   : {telemetry_data.get('end_dist_left', 0):.2f} cm")
        if 'stop_distance_cm' in telemetry_data:
            print(f"  Final Stop Distance : {telemetry_data.get('stop_distance_cm'):.2f} cm (from ToF)")
        print("=============================================\n")
        
        print("Stopping robot...")
        if 'chassis' in locals() and chassis:
            chassis.drive_speed(x=0, y=0, z=0)
        if 'sensor' in locals() and sensor:
            sensor.unsub_distance()
        if 'chassis' in locals() and chassis: # <<<<<<<<<<<< [เพิ่ม!] หยุดรับข้อมูลจาก IMU
            chassis.unsub_imu()
        if ep_robot:
            ep_robot.set_robot_mode(mode=robomaster.robot.FREE) 
            ep_robot.close()
            print("Connection closed.")

if __name__ == '__main__':
    main()