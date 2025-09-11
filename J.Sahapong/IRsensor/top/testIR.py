# -*-coding:utf-8-*-

import robomaster
from robomaster import robot
import time

# =======================================================================
# --- ค่าคงที่ ---
# =======================================================================

# --- สำหรับเซ็นเซอร์ IR ด้านซ้าย ---
TARGET_MIN_CM = 10.0
TARGET_MAX_CM = 13.0
IR_SENSOR_ID = 1
IR_SENSOR_PORT = 2

# --- สำหรับเซ็นเซอร์ ToF ด้านหน้า ---
STOP_DISTANCE_CM = 9.0 
STOP_DISTANCE_MM = STOP_DISTANCE_CM * 10

# --- สำหรับการควบคุมการเคลื่อนที่ ---
FORWARD_SPEED = 0.15
STRAFE_SPEED = 0.1   # ความเร็วขยับด้านข้าง (ต่ำ)

# ตัวแปรสำหรับเก็บค่า ToF
tof_distances = {}

# =======================================================================
# --- ฟังก์ชันสำหรับแปลงค่า ADC เป็นระยะทาง (cm) ---
# =======================================================================
def convert_adc_to_cm(adc_value):
    if adc_value <= 0:
        return float('inf')
    A = 30263
    B = -1.352
    distance_cm = A * (adc_value ** B)
    return distance_cm

# =======================================================================
# --- ฟังก์ชัน Callback สำหรับรับค่าจาก ToF Sensor ---
# =======================================================================
def sub_distance_handler(sub_info):
    global tof_distances
    tof_distances['front'] = sub_info[0]

# =======================================================================
# --- Main Program ---
# =======================================================================
def main():
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    sensor_adaptor = ep_robot.sensor_adaptor
    chassis = ep_robot.chassis
    sensor = ep_robot.sensor

    sensor.sub_distance(freq=30, callback=sub_distance_handler)
    print("Subscribed to ToF sensor data.")
    
    print("\n=============================================")
    print("=== Wall Following (Range 10–13 cm) ===")
    print(f"Target Left Distance: {TARGET_MIN_CM}-{TARGET_MAX_CM} cm")
    print("=============================================")
    time.sleep(2)

    try:
        while True:
            # --- อ่านค่าเซ็นเซอร์ ---
            adc_value = sensor_adaptor.get_adc(id=IR_SENSOR_ID, port=IR_SENSOR_PORT) 
            left_distance_cm = convert_adc_to_cm(adc_value)
            front_tof_mm = tof_distances.get('front', float('inf'))

            # --- ตรวจสอบสิ่งกีดขวางด้านหน้า ---
            if front_tof_mm <= STOP_DISTANCE_MM:
                print(f"\nObstacle detected. Stopping.")
                chassis.drive_speed(x=0, y=0, z=0)
                break

            # --- ตรรกะการควบคุมด้านข้าง ---
            if TARGET_MIN_CM <= left_distance_cm <= TARGET_MAX_CM:
                # อยู่ในระยะที่รับได้ → ไม่ขยับด้านข้าง
                y_speed = 0
            elif left_distance_cm > TARGET_MAX_CM:
                # ไกลเกินไป → ขยับเข้าซ้าย
                y_speed = -STRAFE_SPEED
            elif left_distance_cm < TARGET_MIN_CM:
                # ใกล้เกินไป → ขยับออกขวา
                y_speed = STRAFE_SPEED
            else:
                y_speed = 0

            # --- เคลื่อนที่ไปข้างหน้า พร้อมปรับด้านข้าง ---
            chassis.drive_speed(x=FORWARD_SPEED, y=y_speed, z=0)

            # พิมพ์สถานะ
            print(f"Left: {left_distance_cm:6.2f} cm | Y Speed: {y_speed:5.2f} m/s ", end='\r')
            
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\nProgram stopped by user.")

    finally:
        print("\nStopping robot...")
        chassis.drive_speed(x=0, y=0, z=0)
        sensor.unsub_distance()
        ep_robot.close()
        print("Connection closed.")

if __name__ == '__main__':
    main()
333