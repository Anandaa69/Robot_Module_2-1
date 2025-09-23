# -*-coding:utf-8-*-

import robomaster
from robomaster import robot
import time
import statistics

# --- ฟังก์ชัน Callback และฟังก์ชันแปลงค่า (ไม่เปลี่ยนแปลง) ---
current_yaw = [0.0]

def sub_attitude_info_handler(attitude_info, yaw_storage):
    """Callback function ที่จะอัปเดตค่า Yaw ล่าสุด"""
    yaw = attitude_info[0]
    yaw_storage[0] = yaw

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
# === ฟังก์ชันหลักที่นำกลับมาใช้ใหม่สำหรับการปรับตำแหน่ง ===
# =======================================================================
def adjust_position_to_target(chassis, sensor_adaptor, sensor_id, sensor_port, 
                              target_distance_cm, initial_yaw, side_name):
    """
    ฟังก์ชันที่รวมกระบวนการทั้งหมด: วัด, คำนวณ, และเคลื่อนที่เพื่อปรับตำแหน่ง
    สำหรับเซ็นเซอร์เพียงด้านเดียวจนกว่าจะเข้าเป้าหมาย
    """
    print(f"\n--- Adjusting {side_name} Side ---")
    
    # --- พารามิเตอร์ควบคุม (สามารถปรับแยกในนี้ได้ถ้าต้องการ) ---
    TOLERANCE_CM = 0.5
    KP_SLIDE = 0.08  # ลด Kp กลับมาเล็กน้อยเพื่อความนุ่มนวล
    MAX_SLIDE_SPEED = 0.15
    TOLERANCE_YAW = 0.8 # เพิ่มค่าเผื่อ Yaw เล็กน้อย
    KP_YAW = 0.030
    MAX_ROTATION_SPEED = 20
    MAX_EXECUTION_TIME = 15 # Timeout 15 วินาที

    # --- วัดและคำนวณ ---
    initial_distance = get_stable_distance(sensor_adaptor, sensor_id, sensor_port)
    print(f"[{side_name}] Initial Distance: {initial_distance:.2f} cm, Target: {target_distance_cm:.2f} cm")
    
    distance_error = initial_distance - target_distance_cm
    
    # ถ้ามีข้อผิดพลาดใดๆ ให้เริ่มเคลื่อนที่
    if abs(distance_error) > TOLERANCE_CM:
        print(f"[{side_name}] Executing movement and stabilization...")
        start_time = time.time()

        while True:
            current_adc = sensor_adaptor.get_adc(id=sensor_id, port=sensor_port)
            current_distance = convert_adc_to_cm(current_adc)
            latest_yaw = current_yaw[0]
            
            distance_error = target_distance_cm - current_distance
            yaw_error = initial_yaw - latest_yaw
            
            if yaw_error > 180: yaw_error -= 360
            elif yaw_error < -180: yaw_error += 360

            is_distance_ok = abs(distance_error) <= TOLERANCE_CM
            is_yaw_ok = abs(yaw_error) <= TOLERANCE_YAW
            
            if is_distance_ok and is_yaw_ok:
                print(f"\n[{side_name}] Target position and orientation reached!")
                break
            
            if time.time() - start_time > MAX_EXECUTION_TIME:
                print(f"\n[{side_name}] Movement timed out!")
                break

            slide_speed_y = 0.0
            if not is_distance_ok:
                slide_speed_y = KP_SLIDE * distance_error
                slide_speed_y = max(min(slide_speed_y, MAX_SLIDE_SPEED), -MAX_SLIDE_SPEED)
            
            rotational_speed_z = 0.0
            if not is_yaw_ok:
                rotational_speed_z = KP_YAW * yaw_error
                rotational_speed_z = max(min(rotational_speed_z, MAX_ROTATION_SPEED), -MAX_ROTATION_SPEED)

            chassis.drive_speed(x=0, y=slide_speed_y, z=rotational_speed_z)
            print(f"Adjusting {side_name}... DistErr: {distance_error:5.2f}cm | YawErr: {yaw_error:5.2f}deg", end='\r')
            time.sleep(0.05)
    else:
        print(f"[{side_name}] Already at the target position.")
    
    # หยุดสนิทเมื่อจบภารกิจของด้านนี้
    chassis.drive_speed(x=0, y=0, z=0)
    time.sleep(0.5)


def main():
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    sensor_adaptor = ep_robot.sensor_adaptor
    chassis = ep_robot.chassis
    
    # ==========================================================
    # === กำหนดค่าสำหรับเซ็นเซอร์และระยะเป้าหมายแต่ละด้าน ===
    # ==========================================================
    SENSOR_ADAPTOR_ID = 1
    
    # --- การตั้งค่าสำหรับด้านซ้าย ---
    LEFT_SENSOR_PORT = 2 # Port ของเซ็นเซอร์ Sharp ด้านซ้าย
    LEFT_TARGET_CM = 22.5  # ระยะเป้าหมายของด้านซ้าย (ตัวแปรแยก)

    # --- การตั้งค่าสำหรับด้านขวา ---
    RIGHT_SENSOR_PORT = 3 # !!สำคัญ!! แก้ไข Port ให้ตรงกับที่ต่อเซ็นเซอร์ด้านขวา
    RIGHT_TARGET_CM = 23.0 # ระยะเป้าหมายของด้านขวา (ตัวแปรแยก)

    # --- ตั้งค่าเริ่มต้น ---
    chassis.sub_attitude(freq=20, callback=sub_attitude_info_handler, yaw_storage=current_yaw)
    print("Capturing initial robot orientation...")
    time.sleep(1.0)
    initial_yaw = current_yaw[0]
    print(f"Initial orientation captured: {initial_yaw:.2f} degrees. This will be maintained.")
    
    try:
        # --- ลำดับที่ 1: ปรับตำแหน่งด้านซ้าย ---
        adjust_position_to_target(chassis, sensor_adaptor, SENSOR_ADAPTOR_ID,
                                  LEFT_SENSOR_PORT, LEFT_TARGET_CM, initial_yaw, "Left")

        print("\nLeft side adjustment complete. Pausing before right side check...")
        time.sleep(2.0) # หยุดพัก 2 วินาทีเพื่อให้เห็นชัดเจน

        # --- ลำดับที่ 2: ปรับตำแหน่งด้านขวา ---
        adjust_position_to_target(chassis, sensor_adaptor, SENSOR_ADAPTOR_ID,
                                  RIGHT_SENSOR_PORT, RIGHT_TARGET_CM, initial_yaw, "Right")
        
        print("\nSequence complete.")

    except KeyboardInterrupt:
        print("\n\nProgram stopped by user.")

    finally:
        # --- คืนค่าและปิดการเชื่อมต่อ ---
        print("Unsubscribing from attitude data...")
        chassis.unsub_attitude()
        print("Stopping robot movement...")
        chassis.drive_speed(x=0, y=0, z=0)
        time.sleep(0.5)
        print("Closing robot connection.")
        ep_robot.close()
        print("Connection closed.")

if __name__ == '__main__':
    main()