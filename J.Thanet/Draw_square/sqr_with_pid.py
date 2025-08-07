import time
import math
from robomaster import robot

# === PID Parameters ===
Kp = 0.85
Ki = 0.055
Kd = 0.08

# === Movement Parameters ===
SIDE_LENGTH = 0.5       # ด้านของสี่เหลี่ยม (เมตร)
BASE_WALK_SPEED = 0.5   # ความเร็วพื้นฐานในการเดิน (m/s)
TURN_SPEED = 45         # ความเร็วในการหมุน (degrees/s)
MOVEMENT_TIMEOUT = 8    # Timeout สำหรับการเคลื่อนไหวแต่ละครั้ง (วินาที)

# === Global variables for position tracking ===
current_x = 0.0
current_y = 0.0
current_z = 0.0
position_updated = False

def position_callback(position_info):
    """Callback function to receive position data"""
    global current_x, current_y, current_z, position_updated
    current_x = position_info[0]  # X position in meters
    current_y = position_info[1]  # Y position in meters
    current_z = position_info[2]  # Z rotation (yaw) in degrees
    position_updated = True

def calculate_pid_speed(target_distance, current_distance, integral, previous_error):
    """คำนวณความเร็วจาก PID สำหรับปรับแต่ง move() command"""
    error = target_distance - current_distance
    integral += error * 0.1
    derivative = (error - previous_error) / 0.1
    
    # จำกัด integral windup
    integral = max(min(integral, 5), -5)
    
    # PID output
    pid_output = Kp * error + Ki * integral + Kd * derivative
    
    # แปลงเป็นค่าความเร็วที่เหมาะสม
    speed_adjustment = min(max(pid_output * 0.1, -0.3), 0.3)  # ปรับแต่งความเร็ว ±0.3 m/s
    adjusted_speed = BASE_WALK_SPEED + speed_adjustment
    adjusted_speed = max(min(adjusted_speed, 1.0), 0.2)  # จำกัดความเร็วระหว่าง 0.2-1.0 m/s
    
    return adjusted_speed, integral, error

def move_forward_with_pid(ep_chassis, target_distance, side_number):
    """เดินไปข้างหน้าพร้อมใช้ PID ปรับความเร็ว"""
    print(f"Moving forward {target_distance}m on side {side_number}")
    
    # บันทึกตำแหน่งเริ่มต้น
    start_x = current_x
    start_y = current_y
    
    # PID Variables
    integral = 0
    previous_error = target_distance  # error เริ่มต้น = ระยะทางทั้งหมด
    
    # เดินด้วยความเร็วที่ปรับจาก PID
    distance_traveled = 0
    attempt = 0
    max_attempts = 3
    
    while distance_traveled < target_distance * 0.95 and attempt < max_attempts:  # ยอมรับ 95% ความแม่นยำ
        attempt += 1
        
        # คำนวณระยะที่เดินแล้ว
        distance_traveled = math.sqrt((current_x - start_x)**2 + (current_y - start_y)**2)
        remaining_distance = target_distance - distance_traveled
        
        if remaining_distance <= 0.05:  # ถ้าเหลือน้อยกว่า 5cm ถือว่าเสร็จ
            break
        
        # ใช้ PID คำนวณความเร็วที่เหมาะสม
        adjusted_speed, integral, previous_error = calculate_pid_speed(
            target_distance, distance_traveled, integral, previous_error
        )
        
        print(f"  Attempt {attempt}: Distance={distance_traveled:.3f}m, "
              f"Remaining={remaining_distance:.3f}m, Speed={adjusted_speed:.3f}m/s")
        
        # เดินระยะที่เหลือด้วยความเร็วที่ปรับแล้ว
        actual_move_distance = min(remaining_distance, 0.3)  # เดินทีละไม่เกิน 30cm
        
        result = ep_chassis.move(
            x=actual_move_distance, 
            y=0, 
            z=0, 
            xy_speed=adjusted_speed
        ).wait_for_completed(timeout=MOVEMENT_TIMEOUT)
        
        if not result:
            print(f"  ❌ Movement timeout on attempt {attempt}")
            break
        
        # รอให้ตำแหน่งอัพเดท
        time.sleep(0.2)
    
    # คำนวณระยะสุดท้าย
    final_distance = math.sqrt((current_x - start_x)**2 + (current_y - start_y)**2)
    accuracy = (final_distance / target_distance) * 100
    
    if final_distance >= target_distance * 0.95:
        print(f"✅ Side {side_number} completed! Distance: {final_distance:.3f}m ({accuracy:.1f}%)")
    else:
        print(f"⚠️ Side {side_number} incomplete. Distance: {final_distance:.3f}m ({accuracy:.1f}%)")
    
    # หยุดล้อ
    ep_chassis.stop()
    time.sleep(1)

def turn_left_90_with_feedback(ep_chassis):
    """หมุนซ้าย 90 องศาพร้อมตรวจสอบผล"""
    print("Turning left 90 degrees")
    start_angle = current_z
    
    # หมุนซ้าย 90 องศา
    result = ep_chassis.move(x=0, y=0, z=90, z_speed=TURN_SPEED).wait_for_completed(timeout=MOVEMENT_TIMEOUT)
    
    # รอให้มุมอัพเดท
    time.sleep(0.5)
    
    angle_turned = current_z - start_angle
    # ปรับให้อยู่ในช่วง -180 ถึง 180
    if angle_turned > 180:
        angle_turned -= 360
    elif angle_turned < -180:
        angle_turned += 360
    
    if result and abs(angle_turned - 90) < 10:  # ยอมรับความคลาดเคลื่อน 10 องศา
        print(f"✅ Turn completed! Actual turn: {angle_turned:.1f}°")
    else:
        print(f"⚠️ Turn may be inaccurate. Actual turn: {angle_turned:.1f}°")
    
    # หยุดการหมุน
    ep_chassis.stop()
    time.sleep(1)

def print_current_position():
    """แสดงตำแหน่งปัจจุบัน"""
    print(f"📍 Current position: X={current_x:.3f}m, Y={current_y:.3f}m, Angle={current_z:.1f}°")

# === Initialize Robot ===
ep_robot = robot.Robot()

try:
    # เชื่อมต่อ robot
    print("Connecting to robot...")
    ep_robot.initialize(conn_type="ap")
    print("✅ Robot connected successfully")
    
    ep_chassis = ep_robot.chassis
    
    # เริ่ม subscription สำหรับตำแหน่งพร้อม callback
    ep_chassis.sub_position(freq=10, callback=position_callback)
    
    # รอให้ได้รับข้อมูลตำแหน่งครั้งแรก
    print("Waiting for initial position data...")
    timeout_count = 0
    while not position_updated and timeout_count < 50:
        time.sleep(0.1)
        timeout_count += 1
    
    if not position_updated:
        print("❌ Failed to get initial position data!")
        exit(1)
    
    print("✅ Position data received")
    print_current_position()
    
    # บันทึกตำแหน่งเริ่มต้น
    start_x = current_x
    start_y = current_y
    start_angle = current_z
    
    print("\n" + "="*60)
    print("STARTING SQUARE PATH WITH PID-ENHANCED MOVEMENT")
    print(f"Square side length: {SIDE_LENGTH} meters")
    print(f"Base speed: {BASE_WALK_SPEED} m/s (PID will adjust)")
    print(f"PID Parameters: Kp={Kp}, Ki={Ki}, Kd={Kd}")
    print("="*60)
    
    # รอการเริ่มต้น
    time.sleep(2)
    
    # === เดินสี่เหลี่ยมจัตุรัสด้วย PID-Enhanced Movement ===
    
    for side in range(1, 5):  # 4 ด้าน
        print(f"\n--- SIDE {side} ---")
        
        # เดินไปข้างหน้าด้วย PID control
        move_forward_with_pid(ep_chassis, SIDE_LENGTH, side)
        print_current_position()
        
        if side < 4:  # ไม่หมุนหลังด้านสุดท้าย
            # หมุนซ้าย 90 องศาเพื่อเตรียมเดินด้านต่อไป
            turn_left_90_with_feedback(ep_chassis)
            print_current_position()
    
    # คำนวณความแม่นยำสุดท้าย
    final_x = current_x
    final_y = current_y
    final_angle = current_z
    
    distance_from_start = math.sqrt((final_x - start_x)**2 + (final_y - start_y)**2)
    angle_difference = abs(final_angle - start_angle)
    if angle_difference > 180:
        angle_difference = 360 - angle_difference
    
    print("\n" + "="*60)
    print("🎉 PID-ENHANCED SQUARE PATH COMPLETED!")
    print("="*60)
    print(f"📍 Start position: X={start_x:.3f}m, Y={start_y:.3f}m, Angle={start_angle:.1f}°")
    print(f"📍 Final position: X={final_x:.3f}m, Y={final_y:.3f}m, Angle={final_angle:.1f}°")
    print(f"📏 Distance from start: {distance_from_start:.3f} meters")
    print(f"🧭 Angle difference: {angle_difference:.1f} degrees")
    
    # คำนวณเปอร์เซ็นต์ความแม่นยำ
    position_accuracy = max(0, (1 - distance_from_start/SIDE_LENGTH) * 100)
    angle_accuracy = max(0, (1 - angle_difference/180) * 100)
    
    print(f"🎯 Position accuracy: {position_accuracy:.1f}%")
    print(f"🎯 Angle accuracy: {angle_accuracy:.1f}%")
    print(f"🏆 Overall accuracy: {(position_accuracy + angle_accuracy)/2:.1f}%")
    print("🤖 PID helped optimize movement speed for better accuracy!")

except Exception as e:
    print(f"❌ Error occurred: {e}")
    import traceback
    traceback.print_exc()

finally:
    # ทำความสะอาด
    try:
        print("\nCleaning up...")
        ep_chassis.stop()
        ep_chassis.unsub_position()
        ep_robot.close()
        print("✅ Robot connection closed safely")
    except:
        print("❌ Error during cleanup")
        pass