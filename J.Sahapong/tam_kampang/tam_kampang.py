# -*-coding:utf-8-*-

import robomaster
from robomaster import robot
import time

# =======================================================================
# --- ค่าคงที่และตัวแปรสำหรับการปรับจูน ---
# =======================================================================

# --- สำหรับเซ็นเซอร์ IR ด้านซ้าย (Wall Following) ---
TARGET_DISTANCE_CM = 7.0
IR_SENSOR_ID = 3
IR_SENSOR_PORT = 1

# --- สำหรับเซ็นเซอร์ ToF ด้านหน้า (Stopping) ---
STOP_DISTANCE_CM = 15.0
STOP_DISTANCE_MM = STOP_DISTANCE_CM * 10

# --- สำหรับการควบคุมการเคลื่อนที่ (Wall Following PD-Controller) ---
FORWARD_SPEED = 0.1
KP = 0.06
KD = 0.3
MAX_STRAFE_SPEED = 0.2

# --- สำหรับการปรับให้ขนานกับกำแพง (Alignment using ToF on Gimbal) ---
# มุม Gimbal Yaw เทียบกับหน้าหุ่น (ซ้าย-หน้า และ ซ้าย-หลัง)
GIMBAL_ANGLE_FRONT_LEFT = -45  # องศา
GIMBAL_ANGLE_REAR_LEFT = -135 # องศา
ALIGNMENT_TOLERANCE_MM = 15    # ค่าความต่างของระยะที่ยอมรับได้ (1.5 cm)
# ค่า P-Controller สำหรับการหมุนปรับองศา (ยิ่งมากยิ่งหมุนเร็ว)
ALIGNMENT_KP = 0.1

# ตัวแปรสำหรับ PD Controller ของ Wall Following
last_error_strafe = 0.0
# ตัวแปรสำหรับเก็บค่า ToF แบบ Global
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
    # sub_info[0] คือค่า ToF จากเซ็นเซอร์บนหัว Gimbal
    tof_distances['front'] = sub_info[0]

# =======================================================================
# --- ฟังก์ชันใหม่: ปรับหุ่นยนต์ให้ขนานกับกำแพงด้านซ้าย ---
# =======================================================================
def align_with_left_wall(ep_robot):
    """
    ใช้ ToF sensor บน Gimbal เพื่อวัดระยะ 2 จุด และหมุนตัวหุ่นยนต์
    จนกระทั่งขนานกับกำแพงด้านซ้าย
    """
    print("\n=============================================")
    print("===      STARTING WALL ALIGNMENT      ===")
    print("=============================================")

    gimbal = ep_robot.gimbal
    chassis = ep_robot.chassis

    # ตั้งค่า Gimbal ให้อยู่ในโหมดอิสระ
    gimbal.set_follow_chassis_offset(0)

    while True:
        # 1. วัดระยะจุดหน้า-ซ้าย
        print(f"Measuring Front-Left point at {GIMBAL_ANGLE_FRONT_LEFT} deg...")
        gimbal.moveto(yaw=GIMBAL_ANGLE_FRONT_LEFT, pitch=0).wait_for_completed()
        time.sleep(0.5) # รอให้ค่า ToF นิ่ง
        distance_front_left = tof_distances.get('front', float('inf'))

        # 2. วัดระยะจุดหลัง-ซ้าย
        print(f"Measuring Rear-Left point at {GIMBAL_ANGLE_REAR_LEFT} deg...")
        gimbal.moveto(yaw=GIMBAL_ANGLE_REAR_LEFT, pitch=0).wait_for_completed()
        time.sleep(0.5) # รอให้ค่า ToF นิ่ง
        distance_rear_left = tof_distances.get('front', float('inf'))

        # 3. คำนวณ Error และตัดสินใจ
        # ถ้าค่าที่ได้ไม่ใช่ค่าที่ถูกต้อง (ไกลเกิน) ให้หยุด
        if distance_front_left > 4000 or distance_rear_left > 4000:
             print("Wall not detected clearly. Cannot align.")
             chassis.drive_speed(x=0, y=0, z=0)
             break

        error = distance_front_left - distance_rear_left
        
        print(f"Distances [Front: {distance_front_left} mm, Rear: {distance_rear_left} mm] | Error: {error:.2f} mm")

        # 4. ตรวจสอบว่าขนานหรือยัง
        if abs(error) <= ALIGNMENT_TOLERANCE_MM:
            print("\n--- Robot is now parallel to the wall. ---")
            chassis.drive_speed(x=0, y=0, z=0)
            break # ออกจากลูปการจัดตำแหน่ง

        # 5. สั่งหมุนด้วย P-Controller (เพื่อให้การหมุนนุ่มนวล)
        # error > 0 -> ด้านหน้าไกลกว่า -> ต้องหันซ้าย (z เป็นบวก)
        # error < 0 -> ด้านหน้าใกล้กว่า -> ต้องหันขวา (z เป็นลบ)
        rotation_speed = ALIGNMENT_KP * error
        # จำกัดความเร็วการหมุนไม่ให้เร็วเกินไป
        rotation_speed = max(-25, min(25, rotation_speed)) # จำกัดความเร็วระหว่าง -25 ถึง 25 deg/s
        
        print(f"Adjusting angle with speed: {rotation_speed:.2f} deg/s")
        chassis.drive_speed(x=0, y=0, z=rotation_speed)
        time.sleep(0.1)

    # คืนตำแหน่ง Gimbal และหยุดการเคลื่อนที่ทั้งหมด
    print("Alignment complete. Centering Gimbal.")
    gimbal.recenter().wait_for_completed()
    chassis.drive_speed(x=0, y=0, z=0)
    time.sleep(1)


# =======================================================================
# --- Main Program ---
# =======================================================================
def main():
    global last_error_strafe
    
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    sensor_adaptor = ep_robot.sensor_adaptor
    chassis = ep_robot.chassis
    sensor = ep_robot.sensor

    sensor.sub_distance(freq=20, callback=sub_distance_handler)
    print("Subscribed to ToF sensor data.")
    time.sleep(2) # รอให้มีค่า ToF เข้ามาก่อน

    # --- ขั้นตอนที่ 1: จัดตำแหน่งให้ขนานกับกำแพงก่อน ---
    align_with_left_wall(ep_robot)

    # --- ขั้นตอนที่ 2: เริ่มการเดินตามกำแพง ---
    print("\n=============================================")
    print("===     STARTING WALL FOLLOWING         ===")
    print(f"Target Left Distance: {TARGET_DISTANCE_CM} cm")
    print(f"Forward Speed: {FORWARD_SPEED} m/s")
    print("=============================================")
    time.sleep(3)

    try:
        while True:
            # --- อ่านค่าเซ็นเซอร์ ---
            adc_value = sensor_adaptor.get_adc(id=IR_SENSOR_ID, port=IR_SENSOR_PORT)
            left_distance_cm = convert_adc_to_cm(adc_value)
            # อ่านค่า ToF จาก Global variable (Gimbal ควรจะมองไปข้างหน้าตรงๆ แล้ว)
            front_tof_mm = tof_distances.get('front', float('inf'))

            # --- ส่วนของการตัดสินใจและควบคุม ---

            # 1. ตรวจสอบเงื่อนไขการหยุด
            if front_tof_mm <= STOP_DISTANCE_MM:
                print(f"\nObstacle detected at {front_tof_mm} mm. Stopping.")
                chassis.drive_speed(x=0, y=0, z=0)
                break

            # 2. คำนวณค่าสำหรับ PD Controller (Strafing)
            error = TARGET_DISTANCE_CM - left_distance_cm
            derivative = error - last_error_strafe
            y_speed = (KP * error) + (KD * derivative)
            last_error_strafe = error

            # 3. จำกัดความเร็วการสไลด์
            y_speed = max(-MAX_STRAFE_SPEED, min(MAX_STRAFE_SPEED, y_speed))

            # 4. สั่งให้หุ่นยนต์เคลื่อนที่
            chassis.drive_speed(x=FORWARD_SPEED, y=y_speed, z=0)

            # พิมพ์สถานะ
            print(f"Left IR: {left_distance_cm:6.2f} cm | Error: {error:6.2f} | Strafe Speed: {y_speed:5.2f} m/s ", end='\r')
            
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