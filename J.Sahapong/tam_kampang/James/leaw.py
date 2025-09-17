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
TARGET_DISTANCE_CM = 14.0
FORWARD_SPEED = 0.30
STOP_DISTANCE_CM = 35
STOP_DISTANCE_MM = STOP_DISTANCE_CM * 10

# --- PD Controller สำหรับการหมุน (แกน z) ---
KP_DISTANCE = 6
KP_ANGLE = 6
KD = 6.0
MAX_ROTATE_SPEED = 45.0

# --- ค่าคงที่สำหรับตรรกะการหาผนังที่มุม ---
SEARCH_ROTATE_SPEED = 7.5  # ความเร็วในการหมุนตัวเพื่อหาผนัง (องศา/วินาที)
SEARCH_TOLERANCE_CM = 5.0   # ค่าความคลาดเคลื่อนที่ยอมรับได้ในการหาระยะเป้าหมาย

# --- [ใหม่!] ตัวแปรสำหรับนับการเลี้ยว ---
MAX_CONSECUTIVE_CORNERS = 2 # จำนวนการเลี้ยวสูงสุดที่อนุญาต
corner_count = 0

# --- ตัวแปร Global ---
last_error = 0.0
tof_distances = {}

# =======================================================================
# --- ฟังก์ชันต่างๆ (เหมือนเดิม) ---
# =======================================================================
def convert_adc_to_cm(adc_value):
    if adc_value <= 0: return float('inf')
    A = 30263
    B = -1.352
    return A * (adc_value ** B)

def sub_distance_handler(sub_info):
    global tof_distances
    tof_distances['front'] = sub_info[0]

# =======================================================================
# --- Main Program ---
# =======================================================================
def main():
    global last_error
    global corner_count

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

    print("\n=============================================")
    print("=== Smart Cornering Wall Following ===")
    print(f"Target Distance: {TARGET_DISTANCE_CM} cm")
    print(f"Stop Distance: {STOP_DISTANCE_CM} cm")
    print(f"Program will terminate after {MAX_CONSECUTIVE_CORNERS} consecutive corners.")
    print("=============================================")
    time.sleep(3)

    try:
        while True:
            dist_front = convert_adc_to_cm(sensor_adaptor.get_adc(id=IR_FRONT_LEFT_ID, port=IR_FRONT_LEFT_PORT))
            dist_rear = convert_adc_to_cm(sensor_adaptor.get_adc(id=IR_REAR_LEFT_ID, port=IR_REAR_LEFT_PORT))
            front_tof_mm = tof_distances.get('front', float('inf'))

            # --- [แก้ไข!] ตรรกะการจัดการมุมอับ (Corner Handling Logic) ---
            if front_tof_mm <= STOP_DISTANCE_MM:
                corner_count += 1
                print(f"\nObstacle detected! Corner #{corner_count} maneuver started.")

                # --- [ใหม่!] ตรวจสอบเงื่อนไขการจบโปรแกรม ---
                if corner_count >= MAX_CONSECUTIVE_CORNERS:
                    print(f"\nReached {MAX_CONSECUTIVE_CORNERS} consecutive corners. Terminating program.")
                    break # ออกจาก Loop while

                # 1. หยุดการเคลื่อนที่ปัจจุบัน
                chassis.drive_speed(x=0, y=0, z=0)
                time.sleep(0.5)

                # 2. หมุนขวา 90 องศาเพื่อหลบกำแพง
                print("Step 1: Turning right to clear obstacle...")
                chassis.move(z=-90, z_speed=45).wait_for_completed()
                time.sleep(0.5)
                
                # 3. ค้นหาผนังใหม่โดยการหมุนซ้ายช้าๆ
                print("Step 2: Searching for the new wall by rotating left...")
                search_timeout = time.time() + 10 
                
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
                
                # 4. รีเซ็ตค่า error และกลับไปทำงานในลูปหลัก
                print("Resuming wall following.")
                last_error = 0.0
                time.sleep(1.0) 
                continue 

            # --- ส่วนของ PD Controller (ถ้าไม่เจอสิ่งกีดขวาง) ---
            
            # [ใหม่!] รีเซ็ตตัวนับการเลี้ยวเมื่อหุ่นยนต์เดินตามกำแพงปกติ
            if corner_count > 0:
                print("\nBack to normal wall following. Resetting corner count.")
                corner_count = 0

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
        print("\nStopping robot...")
        if ep_robot:
            chassis.drive_speed(x=0, y=0, z=0)
            sensor.unsub_distance()
            ep_robot.set_robot_mode(mode=robomaster.robot.FREE)
            ep_robot.close()
            print("Connection closed.")

if __name__ == '__main__':
    main()