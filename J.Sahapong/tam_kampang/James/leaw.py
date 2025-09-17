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
SENSOR_OFFSET_CM = 1.0

# --- การตั้งค่าเป้าหมายการเคลื่อนที่ ---
TARGET_DISTANCE_CM = 15.0
FORWARD_SPEED = 0.18
STOP_DISTANCE_CM = 32.0
STOP_DISTANCE_MM = STOP_DISTANCE_CM * 10

# --- PD Controller สำหรับการหมุน (แกน z) ---
KP_DISTANCE = 3
KP_ANGLE = 3
KD = 8.0
MAX_ROTATE_SPEED = 20.0

# --- [ใหม่!] ค่าคงที่สำหรับตรรกะการหาผนังที่มุม ---
SEARCH_ROTATE_SPEED = 15.0  # ความเร็วในการหมุนตัวเพื่อหาผนัง (องศา/วินาที)
SEARCH_TOLERANCE_CM = 5.0   # ค่าความคลาดเคลื่อนที่ยอมรับได้ในการหาระยะเป้าหมาย

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
    print("=============================================")
    time.sleep(3)

    try:
        while True:
            dist_front = convert_adc_to_cm(sensor_adaptor.get_adc(id=IR_FRONT_LEFT_ID, port=IR_FRONT_LEFT_PORT))
            dist_rear = convert_adc_to_cm(sensor_adaptor.get_adc(id=IR_REAR_LEFT_ID, port=IR_REAR_LEFT_PORT))
            front_tof_mm = tof_distances.get('front', float('inf'))

            # --- [แก้ไข!] ตรรกะการจัดการมุมอับ (Corner Handling Logic) ---
            if front_tof_mm <= STOP_DISTANCE_MM:
                print(f"\nObstacle detected at {front_tof_mm}mm. Starting cornering maneuver.")
                
                # 1. หยุดการเคลื่อนที่ปัจจุบัน
                chassis.drive_speed(x=0, y=0, z=0)
                time.sleep(0.5)

                # 2. หมุนขวา 90 องศาเพื่อหลบกำแพง (z>0 คือตามเข็มนาฬิกา/ขวา)
                print("Step 1: Turning right to clear obstacle...")
                chassis.move(z=-90, z_speed=45).wait_for_completed()
                time.sleep(0.5)
                
                # 3. ค้นหาผนังใหม่โดยการหมุนซ้ายช้าๆ
                print("Step 2: Searching for the new wall by rotating left...")
                search_timeout = time.time() + 10 # ตั้งเวลาหมดเวลา 10 วินาที ป้องกันการหมุนไม่สิ้นสุด
                
                while time.time() < search_timeout:
                    # อ่านค่าเซ็นเซอร์ IR ด้านหน้าซ้าย
                    current_dist = convert_adc_to_cm(sensor_adaptor.get_adc(id=IR_FRONT_LEFT_ID, port=IR_FRONT_LEFT_PORT))
                    print(f"Searching... Current distance: {current_dist:.1f} cm", end='\r')

                    # ถ้าเจอระยะที่เหมาะสม (เป้าหมาย +/- ค่าคลาดเคลื่อน)
                    if abs(current_dist - TARGET_DISTANCE_CM) <= SEARCH_TOLERANCE_CM:
                        print(f"\nWall found at {current_dist:.1f} cm. Stopping search.")
                        chassis.drive_speed(x=0, y=0, z=0) # หยุดหมุน
                        break
                    
                    # สั่งให้หมุนซ้ายช้าๆ (z<0 คือทวนเข็มนาฬิกา/ซ้าย)
                    chassis.drive_speed(x=0, y=0, z=-SEARCH_ROTATE_SPEED)
                    time.sleep(0.05)
                else: # กรณีที่ Loop หมุนจนหมดเวลา (Timeout)
                    print("\nSearch timed out. Stopping.")
                    chassis.drive_speed(x=0, y=0, z=0)
                
                # 4. รีเซ็ตค่า error และกลับไปทำงานในลูปหลัก
                print("Resuming wall following.")
                last_error = 0.0
                time.sleep(1.0) # หยุดนิ่งเล็กน้อยก่อนเริ่มเดิน
                continue # ข้ามไปเริ่มต้น Loop while ใหม่

            # --- ส่วนของ PD Controller (เหมือนเดิม) ---
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