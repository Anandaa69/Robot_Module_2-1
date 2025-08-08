from robomaster import robot
import time
import csv
from datetime import datetime

# -------------------------
# เตรียมไฟล์ CSV สำหรับบันทึกข้อมูล
# -------------------------
KP = 3
KI = 0.3
KD = 11
# BEST PID = 1.6 0.3 3

KP_str = str(KP).replace('.', '-')
KI_str = str(KI).replace('.', '-')
KD_str = str(KD).replace('.', '-')

log_filename = f"F:\Coder\Year2-1\Robot_Module\J.Thanet\Draw_square\data/robot_log_{datetime.now().strftime('%H_%M_%S')}_P{KP_str}_I{KI_str}_D{KD_str}.csv"
with open(log_filename, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["time", "x", "y", "z", "pid_output", "target_distance", "relative_position"])

# -------------------------
# PID Controller Function
# -------------------------
class PID:
    def __init__(self, Kp, Ki, Kd, setpoint=0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.prev_error = 0
        self.integral = 0
        self.integral_max = 1.0 # Limit integral term

    def compute(self, current, dt):
        error = self.setpoint - current
        self.integral += error * dt
        
        # Anti-windup: limit the integral term
        if self.integral > self.integral_max:
            self.integral = self.integral_max
        elif self.integral < -self.integral_max:
            self.integral = -self.integral_max
            
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output
    
    def reset(self):
        """Reset PID controller state"""
        self.prev_error = 0
        self.integral = 0

# -------------------------
# Callback Variables
# -------------------------
current_x = 0.0
current_y = 0.0
current_z = 0.0
last_pid_output = None
current_target = None
current_relative = None

# -------------------------
# Position Callback
# -------------------------
def sub_position_handler(position_info):
    global current_x, current_y, current_z, last_pid_output, current_target, current_relative
    current_x = position_info[0]
    current_y = position_info[1]
    current_z = position_info[2]

    # บันทึกลง CSV ทุกครั้งที่มีข้อมูลตำแหน่งใหม่
    with open(log_filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            time.time(),
            current_x,
            current_y,
            current_z,
            last_pid_output if last_pid_output is not None else "",
            current_target if current_target is not None else "",
            current_relative if current_relative is not None else ""
        ])

    print("chassis position: x:{0}, y:{1}, z:{2}".format(current_x, current_y, current_z))

# -------------------------
# Movement Function with PID and Ramp-up
# -------------------------
def move_forward_with_pid(ep_chassis, target_distance, axis, direction=1):
    global last_pid_output, current_target, current_relative
    
    pid = PID(Kp=KP, Ki=KI, Kd=KP)
    
    start_time = time.time()
    last_time = start_time
    target_reached = False
    
    # -------------------------
    # NEW: Ramp-up parameters
    # -------------------------
    ramp_up_time = 0.3  # เวลาในการเร่งความเร็ว (หน่วยเป็นวินาที)
    max_speed = 0.5     # ความเร็วสูงสุดที่ต้องการ (หน่วย m/s)
    
    if axis == 'x':
        start_position = current_x
    else:
        start_position = current_y

    current_target = target_distance

    print(f"Moving forward {target_distance}m (tracking {axis}-axis, direction: {direction}) from position {start_position:.2f}")
    
    try:
        while not target_reached:
            now = time.time()
            dt = now - last_time
            last_time = now
            
            elapsed_time = now - start_time
            
            if axis == 'x':
                current_position = current_x
            else:
                current_position = current_y
            
            relative_position = abs(current_position - start_position)
            current_relative = relative_position

            # -------------------------
            # NEW: Ramp-up Logic
            # -------------------------
            # คำนวณความเร็วเป้าหมาย (target_speed) โดยค่อยๆ เพิ่มขึ้น
            if elapsed_time < ramp_up_time:
                # ถ้ายังอยู่ในช่วง ramp-up, คำนวณความเร็วตามสัดส่วน
                target_speed = (elapsed_time / ramp_up_time) * max_speed
            else:
                # ถ้าพ้นช่วง ramp-up แล้ว ใช้ความเร็วสูงสุด
                target_speed = max_speed
            
            # -------------------------
            # PID Logic (ควบคุมความเร็ว)
            # -------------------------
            # PID จะพยายามควบคุมให้ความเร็วปัจจุบันตรงกับ target_speed
            current_speed = (current_position - pid.prev_error) / dt if dt > 0 else 0 # Crude speed estimate
            
            # ผมขอเปลี่ยนหลักการ PID เล็กน้อยเพื่อให้ควบคุมความเร็ว แทนที่จะควบคุมตำแหน่ง
            # แต่ถ้ายังอยากควบคุมตำแหน่งอยู่ ให้ใช้ PID ตัวเดิมแต่เปลี่ยนการคำนวณ output
            
            # เราสามารถใช้ PID สองชั้น (cascaded PID) ได้ คือ
            # PID ชั้นนอกควบคุมตำแหน่ง -> ได้ความเร็วที่ต้องการ
            # PID ชั้นในควบคุมความเร็ว -> ได้แรงบิด/power ที่ต้องการ
            
            # เนื่องจากโค้ดเดิมใช้ PID ควบคุมตำแหน่ง ผมจะยังคงใช้หลักการเดิม แต่ปรับการส่งค่า
            # PID output (แรงบิด/power) ให้มี ramp-up ในช่วงแรก
            
            # (ทางเลือกที่ 1) ใช้ PID เดิม แต่จำกัด output ในช่วงแรก
            # ซึ่งก็คือการทำให้ PID output ในช่วงแรกมีค่าไม่เกินค่า ramp up
            
            pid.setpoint = target_distance
            output = pid.compute(relative_position, dt)
            
            # NEW: ค่อยๆ เพิ่ม speed ตาม ramp_up_time
            if elapsed_time < ramp_up_time:
                # Interpolate speed to ramp up smoothly
                speed = (elapsed_time / ramp_up_time) * max_speed
            else:
                # After ramp up, use PID output but cap it
                speed = max(min(output, max_speed), -max_speed)
                
            last_pid_output = speed
            
            ep_chassis.drive_speed(x=speed * direction, y=0, z=0, timeout=1)

            if abs(relative_position - target_distance) < 0.02:
                print(f"Target reached. Final position: {current_position:.2f}")
                ep_chassis.drive_speed(x=0, y=0, z=0, timeout=1)
                target_reached = True
                break
                
    except KeyboardInterrupt:
        print("Program interrupted by user.")
        ep_chassis.drive_speed(x=0, y=0, z=0, timeout=1)

# -------------------------
# Rotation Function
# -------------------------
def rotate_90_degrees(ep_chassis):
    """Rotate 90 degrees clockwise"""
    print("=== Rotation: 90 degrees ===")
    time.sleep(0.5)
    
    print("Now turning 90 degrees...")
    rotation_time = 2.15
    ep_chassis.drive_speed(x=0, y=0, z=45, timeout=rotation_time)
    time.sleep(rotation_time + 0.5)
    
    ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.5)
    time.sleep(0.5)
    print("Rotation completed!")

# -------------------------
# Main
# -------------------------
if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_chassis = ep_robot.chassis

    ep_chassis.move(x=0, y=0, z=0, xy_speed=0.5).wait_for_completed()
    ep_chassis.sub_position(freq=20, callback=sub_position_handler)
    time.sleep(0.5)

    try:
        print("=== Side 1: Moving along +X axis ===")
        move_forward_with_pid(ep_chassis, 0.6, 'x', direction=1)
        
        rotate_90_degrees(ep_chassis)
        
        print("=== Side 2: Moving along +Y axis ===")
        time.sleep(0.5)
        move_forward_with_pid(ep_chassis, 0.6, 'y', direction=1)
        
        rotate_90_degrees(ep_chassis)
        
        print("=== Side 3: Moving forward (X will decrease) ===")
        time.sleep(0.5)
        move_forward_with_pid(ep_chassis, 0.6, 'x', direction=-1) # Direction corrected to -1
        
        rotate_90_degrees(ep_chassis)
        
        print("=== Side 4: Moving forward (Y will decrease) ===")
        time.sleep(0.5)
        move_forward_with_pid(ep_chassis, 0.6, 'y', direction=-1) # Direction corrected to -1
        
        rotate_90_degrees(ep_chassis)

        print("=== Square completed! ===")
        
    except KeyboardInterrupt:
        print("Program interrupted by user.")
    
    finally:
        print("All movements completed!")
        try:
            ep_chassis.unsub_position()
        except:
            pass
        time.sleep(0.5)
        ep_robot.close()
        print(f"Data saved to {log_filename}")