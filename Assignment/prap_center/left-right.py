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
# === ฟังก์ชันปรับตำแหน่ง (เวอร์ชันตัด Yaw ออก) ===
# =======================================================================
def adjust_position_to_target_simple(chassis, sensor_adaptor, sensor_id, sensor_port, 
                                     target_distance_cm, side_name):
    """
    ฟังก์ชันที่เคลื่อนที่ด้านข้างอย่างเดียว (แกน Y) จนกว่าจะถึงระยะเป้าหมาย
    ไม่มีการควบคุมการหมุน (Yaw)
    """
    print(f"\n--- Adjusting {side_name} Side (Simple Slide) ---")
    
    # --- พารามิเตอร์ควบคุม ---
    TOLERANCE_CM = 0.5
    KP_SLIDE = 0.03
    MAX_SLIDE_SPEED = 0.15
    MAX_EXECUTION_TIME = 5 # Timeout 15 วินาที

    # --- วัดและคำนวณ ---
    initial_distance = get_stable_distance(sensor_adaptor, sensor_id, sensor_port)
    print(f"[{side_name}] Initial Distance: {initial_distance:.2f} cm, Target: {target_distance_cm:.2f} cm")
    
    distance_error = initial_distance - target_distance_cm
    
    if abs(distance_error) > TOLERANCE_CM:
        print(f"[{side_name}] Executing simple slide movement...")
        start_time = time.time()

        while True:
            current_adc = sensor_adaptor.get_adc(id=sensor_id, port=sensor_port)
            current_distance = convert_adc_to_cm(current_adc)
            
            distance_error = target_distance_cm - current_distance
            
            # --- ตรวจสอบเงื่อนไขการจบการทำงาน (เช็คแค่ระยะทาง) ---
            if abs(distance_error) <= TOLERANCE_CM:
                print(f"\n[{side_name}] Target distance reached!")
                break
            
            if time.time() - start_time > MAX_EXECUTION_TIME:
                print(f"\n[{side_name}] Movement timed out!")
                break

            # --- คำนวณความเร็วสไลด์อย่างเดียว ---
            slide_speed_y = KP_SLIDE * distance_error
            slide_speed_y = max(min(slide_speed_y, MAX_SLIDE_SPEED), -MAX_SLIDE_SPEED)
            
            # --- ส่งคำสั่งเคลื่อนที่โดยไม่หมุน (z=0 เสมอ) ---
            chassis.drive_speed(x=0, y=slide_speed_y, z=0)
            
            print(f"Adjusting {side_name}... DistErr: {distance_error:5.2f}cm | Speed: {slide_speed_y:4.2f}m/s", end='\r')
            time.sleep(0.05)
    else:
        print(f"[{side_name}] Already at the target position.")
    
    # หยุดสนิทเมื่อจบภารกิจ
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
    
    LEFT_SENSOR_PORT = 2
    LEFT_TARGET_CM = 22.5

    RIGHT_SENSOR_PORT = 2
    RIGHT_TARGET_CM = 26.0

    try:
        # --- ลำดับที่ 1: ปรับตำแหน่งด้านซ้าย ---
        adjust_position_to_target_simple(chassis, sensor_adaptor, SENSOR_ADAPTOR_ID,
                                         LEFT_SENSOR_PORT, LEFT_TARGET_CM, "Left")

        print("\nLeft side adjustment complete. Pausing before right side check...")
        time.sleep(2.0)

        # --- ลำดับที่ 2: ปรับตำแหน่งด้านขวา ---
        adjust_position_to_target_simple(chassis, sensor_adaptor, SENSOR_ADAPTOR_ID,
                                         RIGHT_SENSOR_PORT, RIGHT_TARGET_CM, "Right")
        
        print("\nSequence complete.")

    except KeyboardInterrupt:
        print("\n\nProgram stopped by user.")

    finally:
        # --- คืนค่าและปิดการเชื่อมต่อ ---
        print("Stopping robot movement...")
        chassis.drive_speed(x=0, y=0, z=0) # คำสั่งหยุดยังคงสำคัญมาก
        time.sleep(0.5)
        print("Closing robot connection.")
        ep_robot.close()
        print("Connection closed.")

if __name__ == '__main__':
    main()