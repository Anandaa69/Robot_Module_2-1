# -*-coding:utf-8-*-

import robomaster
from robomaster import robot
import time

# =======================================================================
# --- ค่าคงที่และตัวแปรสำหรับการปรับจูน (เวอร์ชัน 8 - PID Control) ---
# =======================================================================

# --- สำหรับเซ็นเซอร์ IR ด้านซ้าย ---
TARGET_DISTANCE_CM = 19.0
IR_SENSOR_ID = 3
IR_SENSOR_PORT = 1

# --- สำหรับเซ็นเซอร์ ToF ด้านหน้า ---
STOP_DISTANCE_CM = 6.5
STOP_DISTANCE_MM = STOP_DISTANCE_CM * 10

# --- สำหรับการควบคุมการเคลื่อนที่ (PID Controller for Strafing) ---
FORWARD_SPEED = 0.1
KP = 0.1                  # Proportional Gain - ตอบสนองต่อ error ปัจจุบัน
KI = 0.005                 # <--- [ตัวแปรใหม่] Integral Gain - แก้ไข error ที่สะสม (เริ่มจากค่าน้อยๆ)
KD = 0.1                # Derivative Gain - ป้องกันการส่าย/เลยเถิด
MAX_STRAFE_SPEED = 0.15

# ตัวแปรสำหรับ PID Controller
last_error = 0.0
integral = 0.0             # <--- [ตัวแปรใหม่] สำหรับเก็บค่า error สะสม
MAX_INTEGRAL = 20          # <--- [ตัวแปรใหม่] ป้องกัน Integral Windup (ค่าสะสมไม่ให้เยอะเกินไป)

# ตัวแปรสำหรับเก็บค่า ToF
tof_distances = {}

# =======================================================================
# --- ฟังก์ชัน (ไม่มีการเปลี่ยนแปลง) ---
# =======================================================================
def convert_adc_to_cm(adc_value):
    if adc_value <= 0: return float('inf')
    return 30263 * (adc_value ** -1.352)

def sub_distance_handler(sub_info):
    global tof_distances
    tof_distances['front'] = sub_info[0]

# =======================================================================
# --- Main Program ---
# =======================================================================
def main():
    global last_error, integral
    
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    sensor_adaptor = ep_robot.sensor_adaptor
    chassis = ep_robot.chassis
    sensor = ep_robot.sensor

    sensor.sub_distance(freq=20, callback=sub_distance_handler)
    print("Subscribed to ToF sensor data.")
    
    print("\n=============================================")
    print("=== Wall Following (PID Strafing Control) ===")
    print(f"Target Left Distance: {TARGET_DISTANCE_CM} cm")
    print(f"Kp: {KP}, Ki: {KI}, Kd: {KD}")
    print("=============================================")
    time.sleep(3)

    try:
        while True:
            # --- อ่านค่าเซ็นเซอร์ ---
            adc_value = sensor_adaptor.get_adc(id=IR_SENSOR_ID, port=IR_SENSOR_PORT) 
            left_distance_cm = convert_adc_to_cm(adc_value)
            front_tof_mm = tof_distances.get('front', float('inf'))

            # --- ส่วนของการตัดสินใจและควบคุม ---

            if front_tof_mm <= STOP_DISTANCE_MM:
                print(f"\nObstacle detected. Stopping.")
                chassis.drive_speed(x=0, y=0, z=0)
                break

            # 2. คำนวณค่าสำหรับ PID Controller
            error = TARGET_DISTANCE_CM - left_distance_cm
            
            # [ใหม่!] ส่วนของ Integral: สะสมค่า error ไปเรื่อยๆ
            integral += error
            
            # [ใหม่!] ป้องกัน Integral Windup: จำกัดค่า integral ไม่ให้สูง/ต่ำเกินไป
            integral = max(-MAX_INTEGRAL, min(MAX_INTEGRAL, integral))

            # ส่วนของ Derivative
            derivative = error - last_error
            
            # [แก้ไข!] รวม P, I, และ D เข้าด้วยกัน
            y_speed = (KP * error) + (KI * integral) + (KD * derivative)
            
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