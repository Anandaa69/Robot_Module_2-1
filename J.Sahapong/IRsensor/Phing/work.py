# -*-coding:utf-8-*-

import robomaster
from robomaster import robot
import time

# =======================================================================
# --- ค่าคงที่และตัวแปรสำหรับการปรับจูน (เวอร์ชัน 5 - Corrected Strafing) ---
# =======================================================================

# --- สำหรับเซ็นเซอร์ IR ด้านซ้าย ---
TARGET_DISTANCE_CM = 10.0
IR_SENSOR_ID = 1
IR_SENSOR_PORT = 2

# --- สำหรับเซ็นเซอร์ ToF ด้านหน้า ---
STOP_DISTANCE_CM = 9.0 
STOP_DISTANCE_MM = STOP_DISTANCE_CM * 10

# --- สำหรับการควบคุมการเคลื่อนที่ (PD-Controller for Strafing) ---
FORWARD_SPEED = 0.2
KP = 0.08
KD = 0.1
MAX_STRAFE_SPEED = 0.3

# ตัวแปรสำหรับ PD Controller
last_error = 0.0
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
    global last_error
    
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    sensor_adaptor = ep_robot.sensor_adaptor
    chassis = ep_robot.chassis
    sensor = ep_robot.sensor

    sensor.sub_distance(freq=20, callback=sub_distance_handler)
    print("Subscribed to ToF sensor data.")
    
    print("\n=============================================")
    print("=== Wall Following (Corrected Strafing Control) ===")
    print(f"Target Left Distance: {TARGET_DISTANCE_CM} cm")
    # ... (ส่วน print อื่นๆ เหมือนเดิม) ...
    print("=============================================")
    time.sleep(3)

    try:
        while True:
            # --- อ่านค่าเซ็นเซอร์ ---
            adc_value = sensor_adaptor.get_adc(id=IR_SENSOR_ID, port=IR_SENSOR_PORT) 
            left_distance_cm = convert_adc_to_cm(adc_value)
            front_tof_mm = tof_distances.get('front', float('inf'))

            # --- ส่วนของการตัดสินใจและควบคุม ---

            # 1. ตรวจสอบเงื่อนไขการหยุด
            if front_tof_mm <= STOP_DISTANCE_MM:
                print(f"\nObstacle detected. Stopping.")
                chassis.drive_speed(x=0, y=0, z=0)
                break

            # 2. คำนวณค่าสำหรับ PD Controller (แก้ไขตรรกะตรงนี้!)
            # error = ค่าเป้าหมาย - ค่าจริง
            # - ถ้าหุ่นไกลไป (left_distance > target) -> error จะเป็นลบ -> y_speed เป็นลบ -> สไลด์ไปในทิศทางที่ถูกต้อง (เข้าหากำแพง)
            # - ถ้าหุ่นใกล้ไป (left_distance < target) -> error จะเป็นบวก -> y_speed เป็นบวก -> สไลด์ไปในทิศทางที่ถูกต้อง (ออกจากกำแพง)
            error = TARGET_DISTANCE_CM - left_distance_cm # <--- [แก้ไข!] สลับตรรกะเพื่อให้สไลด์ถูกทิศทาง
            
            derivative = error - last_error
            y_speed = (KP * error) + (KD * derivative)
            last_error = error

            # 3. จำกัดความเร็วการสไลด์
            y_speed = max(-MAX_STRAFE_SPEED, min(MAX_STRAFE_SPEED, y_speed))

            # 4. สั่งให้หุ่นยนต์เคลื่อนที่
            chassis.drive_speed(x=FORWARD_SPEED, y=y_speed, z=0)

            # พิมพ์สถานะ
            print(f"Left: {left_distance_cm:6.2f} cm | Error: {error:6.2f} | Strafe Speed: {y_speed:5.2f} m/s ", end='\r')
            
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