from robomaster import robot
import time
import csv
from datetime import datetime

KP = 2.1
KI = 0.3
KD = 10
RAMP_UP_TIME = 0.5  # เวลาที่ใช้ในการ ramp-up
ROTATE = 2.11 #MOVE RIGT
ROTATE_LEFT = 1.9
# BEST PID = 2.1 0.3 10 ramp 0.7 - 0.5

KP_str = str(KP).replace('.', '-')
KI_str = str(KI).replace('.', '-')
KD_str = str(KD).replace('.', '-')
RAMP_UP_TIME_str = str(RAMP_UP_TIME).replace('.', '-')

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

# -------------------------
# Callback Variables
# -------------------------
# current_x = 0.0
# current_y = 0.0
# current_z = 0.0
# last_pid_output = None
# current_target = None
# current_relative = None

# -------------------------
# Position Callback
# -------------------------
def sub_position_handler(position_info):
    global current_x, current_y, current_z, last_pid_output, current_target, current_relative
    current_x = position_info[0]
    current_y = position_info[1]
    current_z = position_info[2]

    print("chassis position: x:{0}, y:{1}, z:{2}".format(current_x, current_y, current_z))

# -------------------------
# Movement Function with PID and Improved Ramp-up
# -------------------------
def move_forward_with_pid(ep_chassis, target_distance, axis, direction=1):
    global last_pid_output, current_target, current_relative
    
    pid = PID(Kp=KP, Ki=KI, Kd=KD, setpoint=target_distance)  # ใช้ KD แทน KP
    
    start_time = time.time()
    last_time = start_time
    target_reached = False
    
    # -------------------------
    # Ramp-up parameters
    # -------------------------
    ramp_up_time = RAMP_UP_TIME   # เพิ่มเวลา ramp-up เล็กน้อย
    min_speed = 0.1      # ความเร็วเริ่มต้น
    max_speed = 1.5      # ความเร็วสูงสุด (เหมือนโค้ดเดิม)
    
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

            # คำนวณ PID output
            output = pid.compute(relative_position, dt)
            
            # -------------------------
            # Improved Ramp-up Logic
            # -------------------------
            # คำนวณ ramp-up multiplier
            if elapsed_time < ramp_up_time:
                # เริ่มจาก min_speed และค่อยๆ เพิ่มไป max_speed
                ramp_multiplier = min_speed + (elapsed_time / ramp_up_time) * (1.0 - min_speed)
            else:
                ramp_multiplier = 1.0
            
            # นำ PID output มาคูณกับ ramp_multiplier
            ramped_output = output * ramp_multiplier
            
            # จำกัดความเร็วสูงสุด
            speed = max(min(ramped_output, max_speed), -max_speed)
            
            last_pid_output = speed
            
            ep_chassis.drive_speed(x=speed * direction, y=0, z=0, timeout=1)

            # เงื่อนไขหยุด - เหมือนโค้ดเดิม
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
def rotate_90_degrees_right(ep_chassis):
    """Rotate 90 degrees clockwise"""
    print("=== Rotation: 90 degrees ===")
    time.sleep(0.25)
    
    print("Now turning 90 degrees...")
    rotation_time = ROTATE
    ep_chassis.drive_speed(x=0, y=0, z=45, timeout=rotation_time)
    time.sleep(rotation_time + 0.2)
    
    ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.5)
    time.sleep(0.25)
    print("Rotation completed!")

def rotate_90_degrees_left(ep_chassis):
    """Rotate 90 degrees clockwise"""
    print("=== Rotation: 90 degrees ===")
    time.sleep(0.25)
    rotate_left = ROTATE_LEFT
    print("Now turning 90 degrees...")
    rotation_time = rotate_left
    ep_chassis.drive_speed(x=0, y=0, z=-45, timeout=rotation_time)
    time.sleep(rotation_time + 0.2)
    
    ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.5)
    time.sleep(0.25)
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
    time.sleep(0.25)

    try:
        move_forward_with_pid(ep_chassis, 0.6, 'x', direction=1)
        time.sleep(0.5)
        move_forward_with_pid(ep_chassis, 0.6, 'x', direction=-1)
        
        rotate_90_degrees_right(ep_chassis)
        rotate_90_degrees_left(ep_chassis)

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