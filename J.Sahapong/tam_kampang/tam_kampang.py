# -*-coding:utf--*-

import robomaster
from robomaster import robot
import time

# =======================================================================
# --- ค่าคงที่และตัวแปรสำหรับการปรับจูน ---
# =======================================================================

# --- การตั้งค่าเซ็นเซอร์ IR ด้านซ้าย ---
IR_FRONT_LEFT_ID = 3
IR_FRONT_LEFT_PORT = 1
IR_REAR_LEFT_ID = 1
IR_REAR_LEFT_PORT = 1
SENSOR_OFFSET_CM = 0.5

# --- การตั้งค่าเป้าหมายการเคลื่อนที่ ---
TARGET_DISTANCE_CM = 16.0 # ระยะห่างจากผนัง
FORWARD_SPEED = 0.25 
STOP_DISTANCE_CM = 23.5 
STOP_DISTANCE_MM = STOP_DISTANCE_CM * 10

# --- PD Controller สำหรับการหมุน (แกน z) ---
KP_DISTANCE = 4.5
KP_ANGLE = 8.75
KD = 6.0
MAX_ROTATE_SPEED = 40.0

# --- Deadband สำหรับความเร็วการหมุน ---
Z_SPEED_DEADBAND = 1.0

# --- ค่าคงที่สำหรับตรรกะการหาผนังที่มุม ---
SEARCH_ROTATE_SPEED = 7.5
SEARCH_TOLERANCE_CM = 5.0

# --- ตัวแปรสำหรับนับการเลี้ยว ---
MAX_CONSECUTIVE_CORNERS = 2
corner_count = 0

# --- ตัวแปร Global ---
last_error = 0.0
tof_distances = {}
imu_data = {}  # [เพิ่ม!] สำหรับเก็บค่าจาก IMU
telemetry_data = {}  # [เพิ่ม!] สำหรับเก็บข้อมูลการเคลื่อนที่

# =======================================================================
# --- ฟังก์ชันต่างๆ ---
# =======================================================================
def convert_adc_to_cm(adc_value):
    # ป้องกันกรณีค่าที่อ่านได้เป็น None หรือ 0
    if adc_value is None or adc_value <= 0: return float('inf')
    A = 30263
    B = -1.352
    return A * (adc_value ** B)

def sub_distance_handler(sub_info):
    global tof_distances
    # ป้องกันกรณี sub_info ไม่มีข้อมูล
    if sub_info and sub_info[0] is not None:
        tof_distances['front'] = sub_info[0]

# [เพิ่ม!] Callback function สำหรับ IMU
def sub_imu_handler(sub_info):
    global imu_data
    # sub_info[2] คือค่า yaw angle
    if sub_info and sub_info[2] is not None:
        imu_data['yaw'] = sub_info[2]

# =======================================================================
# --- Main Program ---
# =======================================================================
def main():
    global last_error
    global corner_count
    global telemetry_data

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

    # [เพิ่ม!] เปิดรับข้อมูลจาก IMU
    chassis.sub_imu(freq=50, callback=sub_imu_handler)
    print("Subscribed to IMU sensor data.")

    print("\n=============================================")
    print("=== Smart Cornering Wall Following (Smoother) ===")
    print(f"Target Distance: {TARGET_DISTANCE_CM} cm")
    print(f"Stop Distance: {STOP_DISTANCE_CM} cm")
    print(f"Program will terminate after {MAX_CONSECUTIVE_CORNERS} consecutive corners.")
    print("=============================================")
    time.sleep(3) # รอให้เซ็นเซอร์เริ่มทำงาน

    # [เพิ่ม!] ส่วนเก็บข้อมูลเริ่มต้น
    try:
        # รอเพื่อให้แน่ใจว่าได้รับข้อมูลเริ่มต้นจาก callback แล้ว
        time.sleep(0.5)
        start_dist_left = convert_adc_to_cm(sensor_adaptor.get_adc(id=IR_FRONT_LEFT_ID, port=IR_FRONT_LEFT_PORT))
        start_yaw = imu_data.get('yaw', 0.0)
        start_time = time.time()

        telemetry_data['start_time'] = start_time
        telemetry_data['start_yaw'] = start_yaw
        telemetry_data['start_dist_left'] = start_dist_left
        print("--- Initial state recorded ---")
    except Exception as e:
        print(f"Error recording initial state: {e}")
        ep_robot.close()
        return

    # [เพิ่ม!] ประกาศตัวแปรที่จะใช้เก็บค่าสุดท้าย
    final_stop_distance_mm = float('inf')

    try:
        while True:
            dist_front = convert_adc_to_cm(sensor_adaptor.get_adc(id=IR_FRONT_LEFT_ID, port=IR_FRONT_LEFT_PORT))
            dist_rear = convert_adc_to_cm(sensor_adaptor.get_adc(id=IR_REAR_LEFT_ID, port=IR_REAR_LEFT_PORT))
            front_tof_mm = tof_distances.get('front', float('inf'))

            # --- ตรรกะการจัดการมุมอับ (Corner Handling Logic) ---
            if front_tof_mm <= STOP_DISTANCE_MM:
                corner_count += 1
                print(f"\nObstacle detected! Corner #{corner_count} maneuver started.")
                
                # [เพิ่ม!] บันทึกระยะที่ทำให้หยุดในครั้งสุดท้ายที่เจอ
                if corner_count >= MAX_CONSECUTIVE_CORNERS:
                    final_stop_distance_mm = front_tof_mm

                if corner_count >= MAX_CONSECUTIVE_CORNERS:
                    print(f"\nReached {MAX_CONSECUTIVE_CORNERS} consecutive corners. Terminating program.")
                    break

                chassis.drive_speed(x=0, y=0, z=0)
                time.sleep(0.5)

                print("Step 1: Turning right to clear obstacle...")
                chassis.move(z=-90, z_speed=45).wait_for_completed()
                time.sleep(0.5)

                print("Step 2: Searching for the new wall by rotating left...")
                search_timeout = time.time() + 2

                while time.time() < search_timeout:
                    current_dist = convert_adc_to_cm(sensor_adaptor.get_adc(id=IR_FRONT_LEFT_ID, port=IR_FRONT_LEFT_PORT))
                    print(f"Searching... Current distance: {current_dist:.1f} cm", end='\r')

                    if abs(current_dist - TARGET_DISTANCE_CM) <= SEARCH_TOLERANCE_CM:
                        print(f"\nWall found at {current_dist:.1f} cm. Stopping search.")
                        chassis.drive_speed(x=0, y=0, z=0)
                        break

                    chassis.drive_speed(x=0, y=0, z=-SEARCH_ROTATE_SPEED)
                    time.sleep(0.05)
                else:
                    print("\nSearch timed out. Stopping.")
                    chassis.drive_speed(x=0, y=0, z=0)

                print("Resuming wall following.")
                last_error = 0.0
                time.sleep(1.0)
                continue

            # --- ส่วนของ PD Controller (ถ้าไม่เจอสิ่งกีดขวาง) ---

            if corner_count > 0:
                print("\nBack to normal wall following. Resetting corner count.")
                corner_count = 0

            error_distance = TARGET_DISTANCE_CM - dist_front
            error_angle = (dist_rear + SENSOR_OFFSET_CM) - dist_front

            total_error = (KP_DISTANCE * error_distance) + (KP_ANGLE * error_angle)
            derivative = total_error - last_error
            z_speed = (total_error + (KD * derivative))
            last_error = total_error

            if abs(z_speed) < Z_SPEED_DEADBAND:
                z_speed = 0

            z_speed = max(-MAX_ROTATE_SPEED, min(MAX_ROTATE_SPEED, z_speed))

            chassis.drive_speed(x=FORWARD_SPEED, y=0, z=z_speed)

            print(f"Dist F:{dist_front:5.1f} R:{dist_rear:5.1f} | Err D:{error_distance:5.1f} A:{error_angle:5.1f} | Z_Speed:{z_speed:5.1f}", end='\r')
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nProgram stopped by user.")
    finally:
        # [เพิ่ม!] ส่วนเก็บข้อมูลสิ้นสุดและแสดงผล
        print("\nStopping robot...")
        if 'chassis' in locals() and chassis:
            chassis.drive_speed(x=0, y=0, z=0)
        
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

        # ส่วนของการปิดการเชื่อมต่อ
        if 'sensor' in locals() and sensor:
            sensor.unsub_distance()
        if 'chassis' in locals() and chassis:
            chassis.unsub_imu()
        if ep_robot:
            ep_robot.set_robot_mode(mode=robomaster.robot.FREE)
            ep_robot.close()
            print("Connection closed.")

if __name__ == '__main__':
    main()