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
TARGET_DISTANCE_CM = 5.0
FORWARD_SPEED = 0.18
STOP_DISTANCE_CM = 32.0
STOP_DISTANCE_MM = STOP_DISTANCE_CM * 10
CHECK_RIGHT_WALL_DISTANCE_CM = 50.0 
CHECK_RIGHT_WALL_DISTANCE_MM = CHECK_RIGHT_WALL_DISTANCE_CM * 10

# --- [ใหม่!] ค่าคงที่สำหรับแก้ปัญหา ---
ACQUISITION_NUDGE_CM = 20.0 # <<<<<<<<<<<< ระยะที่จะให้หุ่นเดินหน้าทื่อๆ หลังเลี้ยว (ปรับค่านี้ได้)
ACQUISITION_NUDGE_SPEED = 0.5 # <<<<<<<<<<<< ความเร็วในการเดินหน้าเพื่อเข้าเลนใหม่

# --- PD Controller สำหรับการหมุน (แกน z) ---
KP_DISTANCE = 2
KP_ANGLE = 2.5
KD = 8.0
MAX_ROTATE_SPEED = 30.0
GIMBAL_ROTATE_SPEED = 120 # ความเร็วในการหัน Gimbal

# --- ตัวแปร Global ---
last_error = 0.0
tof_distances = {}

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
    print("=== Wall Following (Gimbal Look-Ahead) ===")
    print(f"Target Distance: {TARGET_DISTANCE_CM} cm")
    print(f"Check Right Distance: {CHECK_RIGHT_WALL_DISTANCE_CM} cm")
    print("=============================================")
    time.sleep(3)

    try:
        while True:
            # --- 1. อ่านค่าจากเซ็นเซอร์ ---
            dist_front = convert_adc_to_cm(sensor_adaptor.get_adc(id=IR_FRONT_LEFT_ID, port=IR_FRONT_LEFT_PORT))
            dist_rear = convert_adc_to_cm(sensor_adaptor.get_adc(id=IR_REAR_LEFT_ID, port=IR_REAR_LEFT_PORT))
            front_tof_mm = tof_distances.get('front', float('inf'))

            # --- 2. ตรวจสอบ, หัน Gimbal, และตัดสินใจ ---
            if front_tof_mm <= STOP_DISTANCE_MM:
                print("\n[STEP 1] Obstacle detected. Stopping chassis...")
                chassis.drive_speed(x=0, y=0, z=0)
                time.sleep(0.5)

                print("[STEP 2] Looking right with gimbal...")
                ep_gimbal.move(yaw=-90, yaw_speed=GIMBAL_ROTATE_SPEED).wait_for_completed()
                
                time.sleep(1.0) 
                right_side_dist_mm = tof_distances.get('front', float('inf'))
                print(f"[STEP 3] Measured distance to the right: {right_side_dist_mm / 10.0:.1f} cm.")

                if right_side_dist_mm < CHECK_RIGHT_WALL_DISTANCE_MM:
                    print("[RESULT] Wall found on the right. Mission complete.")
                    ep_gimbal.recenter().wait_for_completed()
                    break
                else:
                    print("[STEP 4] Path is clear. Turning chassis...")
                    chassis.move(z=-90, z_speed=MAX_ROTATE_SPEED).wait_for_completed()
                    
                    print("Resetting gimbal to forward position...")
                    ep_gimbal.recenter().wait_for_completed()
                    
                    # <<<<<<<<<<<<<<<<<<<<<<<< [ส่วนแก้ไขที่สำคัญ!] >>>>>>>>>>>>>>>>>>>>>>>>
                    print(f"[STEP 5] Nudging forward {ACQUISITION_NUDGE_CM} cm to acquire new wall...")
                    # เดินหน้าทื่อๆ เป็นระยะทางสั้นๆ เพื่อให้พ้นมุม
                    chassis.move(x=(ACQUISITION_NUDGE_CM / 100.0), xy_speed=ACQUISITION_NUDGE_SPEED).wait_for_completed()
                    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                    
                    print("Resuming wall following in new direction.")
                    last_error = 0.0
                    time.sleep(0.5)
                    continue

            # --- 3. คำนวณ Error ---
            error_distance = TARGET_DISTANCE_CM - dist_front
            error_angle = (dist_rear + SENSOR_OFFSET_CM) - dist_front
            
            # --- 4. คำนวณ "ผลรวมของ Error ที่ผ่านการถ่วงน้ำหนักแล้ว" ---
            total_error = (KP_DISTANCE * error_distance) + (KP_ANGLE * error_angle)
            
            # --- 5. คำนวณค่า Derivative ---
            derivative = total_error - last_error
            
            # --- 6. คำนวณความเร็วในการหมุน z_speed ---
            z_speed = (total_error + (KD * derivative))
            
            last_error = total_error

            # --- 7. จำกัดความเร็วสูงสุด ---
            z_speed = max(-MAX_ROTATE_SPEED, min(MAX_ROTATE_SPEED, z_speed))

            # --- 8. สั่งให้หุ่นยนต์เคลื่อนที่ ---
            chassis.drive_speed(x=FORWARD_SPEED, y=0, z=z_speed)

            # --- 9. พิมพ์สถานะ ---
            print(f"Dist F:{dist_front:5.1f} R:{dist_rear:5.1f} | Err D:{error_distance:5.1f} A:{error_angle:5.1f} | Z_Speed:{z_speed:5.1f}", end='\r')
            
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\nProgram stopped by user.")

    finally:
        print("\nStopping robot...")
        if 'chassis' in locals() and chassis:
            chassis.drive_speed(x=0, y=0, z=0)
        if 'sensor' in locals() and sensor:
            sensor.unsub_distance()
        if ep_robot:
            ep_robot.set_robot_mode(mode=robomaster.robot.FREE) 
            ep_robot.close()
            print("Connection closed.")

if __name__ == '__main__':
    main()