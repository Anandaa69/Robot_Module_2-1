# -*-coding:utf--*-

import robomaster
from robomaster import robot
import time
from collections import deque

# =======================================================================
# --- ค่าคงที่และตัวแปรสำหรับการปรับจูน (เวอร์ชันแก้ไขทิศทางที่ถูกต้อง) ---
# =======================================================================

# --- การตั้งค่าเซ็นเซอร์ IR ด้านซ้าย ---
IR_FRONT_LEFT_ID = 3
IR_FRONT_LEFT_PORT = 1
IR_REAR_LEFT_ID = 1
IR_REAR_LEFT_PORT = 1
SENSOR_OFFSET_CM = 1.0 

# --- การตั้งค่าเป้าหมายการเคลื่อนที่ ---
TARGET_DISTANCE_CM = 6.0
FORWARD_SPEED = 0.18
STOP_DISTANCE_CM = 32.0
STOP_DISTANCE_MM = STOP_DISTANCE_CM * 10

# --- PD Controller สำหรับการหมุน (แกน z) ---
KP_DISTANCE = 2
KP_ANGLE = 2.5
KD = 8.0
MAX_ROTATE_SPEED = 30.0

# --- การตั้งค่าสำหรับการวิเคราะห์แนวโน้ม ---
HISTORY_SIZE = 8  # เก็บค่าย้อนหลัง 8 ค่า
SHARP_CORNER_SLOPE = 0.7  # เกณฑ์มุมฉากแหลม (ต้องเลี้ยว 90°)
GENTLE_CORNER_SLOPE = 0.3  # เกณฑ์มุมเฉียง (ใช้ Aggressive PD)
CORNER_DETECTION_THRESHOLD = 2.5  # เกณฑ์ StdDev สำหรับมุมฉาก
MIN_READINGS_FOR_ANALYSIS = 5  # จำนวนข้อมูลขั้นต่ำก่อนวิเคราะห์

# --- Aggressive PD Mode Parameters ---
AGGRESSIVE_KP_ANGLE = 4.0
AGGRESSIVE_KD = 12.0
AGGRESSIVE_FORWARD_SPEED = 0.12
AGGRESSIVE_MAX_ROTATE_SPEED = 45.0
aggressive_mode = False
aggressive_timer = 0

# --- ตัวแปร Global ---
last_error = 0.0
tof_distances = {}
left_wall_history = deque(maxlen=HISTORY_SIZE)  # เก็บประวัติระยะกำแพงซ้าย
reading_count = 0

# =======================================================================
# --- ฟังก์ชันต่างๆ ---
# =======================================================================
def convert_adc_to_cm(adc_value):
    if adc_value <= 0: return float('inf')
    A = 30263
    B = -1.352
    return A * (adc_value ** B)

def sub_distance_handler(sub_info):
    global tof_distances
    tof_distances['front'] = sub_info[0]

def calculate_trend_slope(distance_history):
    """
    คำนวณความชันของแนวโน้มการเปลี่ยนแปลงระยะกำแพง
    ใช้ linear regression แบบง่าย
    """
    if len(distance_history) < 3:
        return 0.0
    
    n = len(distance_history)
    x_values = list(range(n))
    y_values = list(distance_history)
    
    # คำนวณ slope ด้วย least squares method
    sum_x = sum(x_values)
    sum_y = sum(y_values)
    sum_xy = sum(x * y for x, y in zip(x_values, y_values))
    sum_x2 = sum(x * x for x in x_values)
    
    if n * sum_x2 - sum_x * sum_x == 0:
        return 0.0
        
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
    return slope

def analyze_wall_pattern(distance_history, current_front_distance):
    """
    วิเคราะห์รูปแบบของกำแพงเพื่อแยกแยะระหว่างกำแพงเอียงกับมุมฉาก
    
    Returns:
    - 'normal': กำแพงเอียงปกติ
    - 'sharp_corner': มุมฉากแหลม (ต้องเลี้ยว 90°)
    - 'gentle_corner': มุมเฉียง (ใช้ Aggressive PD)  
    - 'unknown': ข้อมูลไม่เพียงพอ
    """
    if len(distance_history) < MIN_READINGS_FOR_ANALYSIS:
        return 'unknown'
    
    # คำนวณความชันของแนวโน้ม
    slope = calculate_trend_slope(distance_history)
    
    # คำนวณความแปรปรวนของข้อมูล
    recent_distances = list(distance_history)[-5:]  # ใช้ 5 ค่าล่าสุด
    avg_distance = sum(recent_distances) / len(recent_distances)
    variance = sum((d - avg_distance) ** 2 for d in recent_distances) / len(recent_distances)
    std_dev = variance ** 0.5
    
    abs_slope = abs(slope)
    is_front_blocked = current_front_distance < STOP_DISTANCE_CM
    
    print(f"\n[ANALYSIS] Slope: {slope:.3f}, StdDev: {std_dev:.2f}, Front: {current_front_distance:.1f}cm")
    
    if not is_front_blocked:
        return 'unknown'
    
    # มุมฉากแหลม: เปลี่ยนแปลงเร็วและผันผวนสูง
    if abs_slope > SHARP_CORNER_SLOPE and std_dev > CORNER_DETECTION_THRESHOLD:
        print("[DECISION] Sharp corner detected - Need to turn 90°")
        return 'sharp_corner'
    
    # มุมเฉียง: เปลี่ยนแปลงปานกลาง
    elif GENTLE_CORNER_SLOPE <= abs_slope <= SHARP_CORNER_SLOPE:
        print("[DECISION] Gentle corner detected - Use Aggressive PD")
        return 'gentle_corner'
    
    # กำแพงเอียงธรรมดา
    else:
        print("[DECISION] Normal slanted wall - Continue normally")
        return 'normal'

# =======================================================================
# --- Main Program ---
# =======================================================================
def main():
    global last_error, reading_count, aggressive_mode, aggressive_timer
    
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
    print("=== Enhanced Wall Following with Trend Analysis ===")
    print(f"Target Distance: {TARGET_DISTANCE_CM} cm")
    print("Control Scheme: +z = Clockwise (Right), -z = Counter-Clockwise (Left)")
    print("=============================================")
    time.sleep(3)

    try:
        while True:
            # --- 1. อ่านค่าจากเซ็นเซอร์ ---
            dist_front = convert_adc_to_cm(sensor_adaptor.get_adc(id=IR_FRONT_LEFT_ID, port=IR_FRONT_LEFT_PORT))
            dist_rear = convert_adc_to_cm(sensor_adaptor.get_adc(id=IR_REAR_LEFT_ID, port=IR_REAR_LEFT_PORT))
            front_tof_mm = tof_distances.get('front', float('inf'))
            front_tof_cm = front_tof_mm / 10.0
            
            reading_count += 1
            
            # --- 2. เก็บประวัติระยะกำแพงด้านซ้าย ---
            if dist_front != float('inf'):
                left_wall_history.append(dist_front)

            # --- 3. ตรวจสอบเงื่อนไขการหยุด หรือ เลี้ยวขวา (ใช้การวิเคราะห์แนวโน้ม) ---
            if front_tof_cm <= STOP_DISTANCE_CM:
                print(f"\nObstacle detected in front at {front_tof_cm:.1f}cm. Analyzing wall pattern...")
                chassis.drive_speed(x=0, y=0, z=0)
                time.sleep(0.5)

                # วิเคราะห์รูปแบบของกำแพง
                wall_pattern = analyze_wall_pattern(left_wall_history, front_tof_cm)
                
                if wall_pattern == 'sharp_corner':
                    # เป็นมุมฉากแหลม ต้องเลี้ยวขวา 90°
                    print("==> TURNING 90°: Sharp corner detected")
                    chassis.move(x=0, y=0, z=-90, z_speed=90).wait_for_completed()
                    time.sleep(1)

                    # ตรวจสอบเส้นทางด้านขวา
                    distance_on_right = tof_distances.get('front', float('inf'))
                    print(f"Distance to the right: {distance_on_right / 10.0:.1f} cm")

                    if distance_on_right > STOP_DISTANCE_MM:
                        print("Path to the right is clear. Resuming wall following.")
                        last_error = 0.0
                        left_wall_history.clear()
                        aggressive_mode = False
                        continue
                    else:
                        print("Path to the right is also blocked. Stopping program.")
                        break
                        
                elif wall_pattern == 'gentle_corner':
                    # เป็นมุมเฉียง เข้าโหมด Aggressive PD
                    print("==> AGGRESSIVE MODE: Gentle corner detected")
                    aggressive_mode = True
                    aggressive_timer = 20  # ใช้โหมด aggressive เป็นเวลา 20 cycles
                    continue
                    
                elif wall_pattern == 'normal':
                    # เป็นกำแพงเอียงธรรมดา ปรับมุมเล็กน้อย
                    print("==> SMALL ADJUSTMENT: Normal slanted wall")
                    chassis.move(x=0, y=0, z=-15, z_speed=30).wait_for_completed()
                    time.sleep(0.3)
                    continue
                    
                else:
                    # ข้อมูลไม่เพียงพอ ใช้วิธีเดิม
                    print("==> INSUFFICIENT DATA: Using traditional method")
                    chassis.move(x=0, y=0, z=-90, z_speed=90).wait_for_completed()
                    time.sleep(1)

                    distance_on_right = tof_distances.get('front', float('inf'))
                    if distance_on_right > STOP_DISTANCE_MM:
                        print("Path clear. Resuming.")
                        last_error = 0.0
                        left_wall_history.clear()
                        continue
                    else:
                        print("No path available. Stopping.")
                        break

            # --- 4. ตรวจสอบโหมด Aggressive ---
            if aggressive_mode:
                aggressive_timer -= 1
                if aggressive_timer <= 0:
                    aggressive_mode = False
                    print("\n[MODE] Exiting Aggressive Mode")

            # --- 5. เลือกพารามิเตอร์ตามโหมด ---
            if aggressive_mode:
                current_kp_angle = AGGRESSIVE_KP_ANGLE
                current_kd = AGGRESSIVE_KD
                current_forward_speed = AGGRESSIVE_FORWARD_SPEED
                current_max_rotate_speed = AGGRESSIVE_MAX_ROTATE_SPEED
                mode_indicator = "AGG"
            else:
                current_kp_angle = KP_ANGLE
                current_kd = KD
                current_forward_speed = FORWARD_SPEED
                current_max_rotate_speed = MAX_ROTATE_SPEED
                mode_indicator = "NOR"

            # --- 6. คำนวณ Error (ใช้พารามิเตอร์ตามโหมด) ---
            error_distance = TARGET_DISTANCE_CM - dist_front
            error_angle = (dist_rear + SENSOR_OFFSET_CM) - dist_front
            
            # --- 7. คำนวณ "ผลรวมของ Error ที่ผ่านการถ่วงน้ำหนักแล้ว" ---
            total_error = (KP_DISTANCE * error_distance) + (current_kp_angle * error_angle)
            
            # --- 8. คำนวณค่า Derivative ---
            derivative = total_error - last_error
            
            # --- 9. คำนวณความเร็วในการหมุน z_speed ---
            z_speed = (total_error + (current_kd * derivative))
            
            last_error = total_error

            # --- 10. จำกัดความเร็วสูงสุด ---
            z_speed = max(-current_max_rotate_speed, min(current_max_rotate_speed, z_speed))

            # --- 11. สั่งให้หุ่นยนต์เคลื่อนที่ ---
            chassis.drive_speed(x=current_forward_speed, y=0, z=z_speed)

            # --- 12. พิมพ์สถานะ (เพิ่มข้อมูลโหมด) ---
            slope = calculate_trend_slope(left_wall_history) if len(left_wall_history) >= 3 else 0
            print(f"[{mode_indicator}] F:{dist_front:4.1f} R:{dist_rear:4.1f} ToF:{front_tof_cm:4.1f} | Slope:{slope:5.2f} | Z:{z_speed:5.1f} | T:{aggressive_timer}", end='\r')
            
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