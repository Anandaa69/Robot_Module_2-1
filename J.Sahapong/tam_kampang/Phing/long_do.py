# -*-coding:utf-8-*-

import robomaster
from robomaster import robot
import time
import math

# =======================================================================
# --- ค่าคงที่และตัวแปรสำหรับการปรับจูน ---
# =======================================================================

# --- สถานะของหุ่นยนต์ ---
STATE_FOLLOWING_WALL = "FOLLOWING_WALL"
STATE_TURNING_RIGHT = "TURNING_RIGHT"
STATE_TURNING_LEFT = "TURNING_LEFT"

# --- สำหรับเซ็นเซอร์ IR ด้านซ้าย (2 ตัว) ---
IR_FRONT_LEFT_ID = 1
IR_FRONT_LEFT_PORT = 1
IR_BACK_LEFT_ID = 2
IR_BACK_LEFT_PORT = 1

TARGET_DISTANCE_CM = 5.0
WALL_LOST_THRESHOLD_CM = 15.0 # ระยะที่ถือว่ากำแพงด้านซ้ายหายไป

# --- สำหรับเซ็นเซอร์ ToF ด้านหน้า ---
STOP_DISTANCE_CM = 10.0
STOP_DISTANCE_MM = STOP_DISTANCE_CM * 10

# --- สำหรับการควบคุมการเคลื่อนที่ ---
FORWARD_SPEED = 0.15

# --- PD-Controller สำหรับรักษาระยะห่าง (Strafing) ---
KP_DISTANCE = 0.08
KD_DISTANCE = 0.4
MAX_STRAFE_SPEED = 0.25

# --- PD-Controller สำหรับรักษามุมด้วย IMU (Rotation) ---
KP_ANGLE_IMU = 2.5  # P gain for IMU angle (z_speed)
KD_ANGLE_IMU = 6.0  # D gain for IMU angle (z_speed)
MAX_ROTATE_SPEED = 45 # จำกัดความเร็วหมุนสูงสุด

# --- ค่าคงที่สำหรับการเลี้ยว ---
TURN_ANGLE_DEGREES = 90.0
TURN_COMPLETION_THRESHOLD_DEGREES = 3.0 # ค่าความคลาดเคลื่อนที่ยอมรับได้เมื่อเลี้ยวเสร็จ

# --- ตัวแปร Global ---
tof_distances = {}
imu_data = {}
current_state = STATE_FOLLOWING_WALL
target_yaw = 0.0
last_distance_error = 0.0
last_angle_error = 0.0

# =======================================================================
# --- ฟังก์ชัน Callback และฟังก์ชันเสริม ---
# =======================================================================
def convert_adc_to_cm(adc_value):
    if adc_value <= 0:
        return float('inf')
    A = 30263
    B = -1.352
    distance_cm = A * (adc_value ** B)
    return distance_cm

def sub_distance_handler(sub_info):
    global tof_distances
    tof_distances['front'] = sub_info[0]

def sub_imu_handler(sub_info):
    global imu_data
    # sub_info[0] = pitch, sub_info[1] = roll, sub_info[2] = yaw
    imu_data['yaw'] = sub_info[2]

def normalize_angle(angle):
    """ ทำให้มุมอยู่ในช่วง -180 ถึง 180 องศา """
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle

# =======================================================================
# --- Main Program ---
# =======================================================================
def main():
    global last_distance_error, last_angle_error, current_state, target_yaw
    
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    sensor_adaptor = ep_robot.sensor_adaptor
    chassis = ep_robot.chassis
    sensor = ep_robot.sensor
    gimbal = ep_robot.gimbal

    # รีเซ็ต Gimbal และสมัครรับข้อมูลเซ็นเซอร์
    gimbal.recenter().wait_for_completed()
    sensor.sub_distance(freq=20, callback=sub_distance_handler)
    chassis.sub_imu(freq=50, callback=sub_imu_handler)
    print("Subscribed to ToF and IMU sensor data.")
    
    time.sleep(2) # รอให้มีข้อมูลจาก IMU เข้ามาครั้งแรก

    # ตั้งค่ามุม Yaw เริ่มต้นเป็นเป้าหมาย
    if 'yaw' in imu_data:
        target_yaw = imu_data['yaw']
        print(f"Initial Yaw set to: {target_yaw:.2f} degrees")
    else:
        print("Warning: Could not get initial IMU yaw. Assuming 0.")
        target_yaw = 0.0
    
    print(f"\n=== Wall Following (State Machine + IMU Control) ===")
    print(f"Target Left Distance: {TARGET_DISTANCE_CM} cm")
    print(f"Stop Distance (Front): {STOP_DISTANCE_CM} cm")
    print("=============================================")
    time.sleep(3)

    try:
        while True:
            # --- อ่านค่าเซ็นเซอร์ ---
            adc_front_left = sensor_adaptor.get_adc(id=IR_FRONT_LEFT_ID, port=IR_FRONT_LEFT_PORT) 
            adc_back_left = sensor_adaptor.get_adc(id=IR_BACK_LEFT_ID, port=IR_BACK_LEFT_PORT)
            front_left_cm = convert_adc_to_cm(adc_front_left)
            back_left_cm = convert_adc_to_cm(adc_back_left)
            average_distance = (front_left_cm + back_left_cm) / 2.0
            
            front_tof_mm = tof_distances.get('front', float('inf'))
            current_yaw = imu_data.get('yaw', target_yaw)

            # --- State Machine ---
            if current_state == STATE_FOLLOWING_WALL:
                # 1. ตรวจสอบเงื่อนไขเพื่อเปลี่ยนสถานะ
                if front_tof_mm <= STOP_DISTANCE_MM:
                    print(f"\n[STATE CHANGE] Wall ahead. -> TURNING_RIGHT")
                    current_state = STATE_TURNING_RIGHT
                    # ตั้งเป้าหมายการหมุนขวา 90 องศา
                    target_yaw = normalize_angle(target_yaw - TURN_ANGLE_DEGREES)
                    continue
                
                if average_distance > WALL_LOST_THRESHOLD_CM:
                    print(f"\n[STATE CHANGE] Wall lost. -> TURNING_LEFT")
                    current_state = STATE_TURNING_LEFT
                    # ตั้งเป้าหมายการหมุนซ้าย 90 องศา
                    target_yaw = normalize_angle(target_yaw + TURN_ANGLE_DEGREES)
                    continue

                # 2. คำนวณการควบคุม
                # ควบคุมระยะห่าง (y_speed) - เหมือนเดิม
                distance_error = TARGET_DISTANCE_CM - average_distance
                distance_derivative = distance_error - last_distance_error
                y_speed = (KP_DISTANCE * distance_error) + (KD_DISTANCE * distance_derivative)
                last_distance_error = distance_error

                # ควบคุมมุม (z_speed) - ใช้ IMU
                angle_error = normalize_angle(target_yaw - current_yaw)
                angle_derivative = angle_error - last_angle_error
                z_speed = (KP_ANGLE_IMU * angle_error) + (KD_ANGLE_IMU * angle_derivative)
                last_angle_error = angle_error
                
                # 3. สั่งการเคลื่อนที่
                x_speed = FORWARD_SPEED

            elif current_state == STATE_TURNING_RIGHT or current_state == STATE_TURNING_LEFT:
                # 1. คำนวณการควบคุมเพื่อหมุน
                x_speed = 0 # หยุดเดินหน้า
                y_speed = 0 # หยุดสไลด์
                
                angle_error = normalize_angle(target_yaw - current_yaw)
                
                # 2. ตรวจสอบว่าเลี้ยวเสร็จหรือยัง
                if abs(angle_error) < TURN_COMPLETION_THRESHOLD_DEGREES:
                    print(f"\n[STATE CHANGE] Turn complete. -> FOLLOWING_WALL")
                    current_state = STATE_FOLLOWING_WALL
                    last_angle_error = 0 # รีเซ็ตค่า error
                    last_distance_error = 0
                    chassis.drive_speed(x=0, y=0, z=0) # หยุดนิ่งแป๊บนึง
                    time.sleep(0.5)
                    continue

                # 3. สั่งการหมุน
                angle_derivative = angle_error - last_angle_error
                z_speed = (KP_ANGLE_IMU * angle_error) + (KD_ANGLE_IMU * angle_derivative)
                last_angle_error = angle_error

            # --- จำกัดความเร็วและสั่งการ ---
            y_speed = max(-MAX_STRAFE_SPEED, min(MAX_STRAFE_SPEED, y_speed))
            z_speed = max(-MAX_ROTATE_SPEED, min(MAX_ROTATE_SPEED, z_speed))
            chassis.drive_speed(x=x_speed, y=y_speed, z=z_speed)
            
            # พิมพ์สถานะ
            print(f"State: {current_state:16s} | Yaw: {current_yaw:6.1f} (Tgt: {target_yaw:6.1f}) | Dist: {average_distance:5.1f} | X:{x_speed:4.2f} Y:{y_speed:5.2f} Z:{z_speed:5.1f}", end='\r')
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n\nProgram stopped by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

    finally:
        print("\nStopping robot...")
        chassis.drive_speed(x=0, y=0, z=0)
        sensor.unsub_distance()
        chassis.unsub_imu()
        ep_robot.close()
        print("Connection closed.")

if __name__ == '__main__':
    main()