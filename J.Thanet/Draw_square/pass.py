from robomaster import robot
import time

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
# Movement Function with PID
# -------------------------
def move_forward_with_pid(ep_chassis, target_distance, axis, direction=1):
    """
    Move forward using PID control
    target_distance: distance to move (meters)
    axis: 'x' or 'y' - which axis to track for distance measurement
    direction: 1 for positive direction, -1 for negative direction
    """
    # Setup PID for this movement
    pid = PID(Kp=1.6, Ki=0.3, Kd=3, setpoint=target_distance)
    
    #best 1.6 0.3 3
    
    # Control loop for forward movement
    last_time = time.time()
    target_reached = False
    
    # Get starting position
    if axis == 'x':
        start_position = current_x
    else:
        start_position = current_y
    
    print(f"Moving forward {target_distance}m (tracking {axis}-axis, direction: {direction}) from position {start_position:.2f}")
    
    try:
        while not target_reached:
            now = time.time()
            dt = now - last_time
            last_time = now
            
            # Get current position and calculate relative position from start
            if axis == 'x':
                current_position = current_x
            else:
                current_position = current_y
            
            # Calculate distance traveled (always positive)
            relative_position = abs(current_position - start_position)
                
            output = pid.compute(relative_position, dt)

            max_speed = 1.5
            speed = max(min(output, max_speed), -max_speed)
            
            # Always use x for forward movement (robot's local coordinate)
            # direction determines if we go forward (+) or backward (-)
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
    # หยุดการ subscribe position ก่อนหมุน
    ep_chassis.unsub_position()
    time.sleep(0.5)
    
    print("Now turning 90 degrees...")
    rotation_time = 2.15
    ep_chassis.drive_speed(x=0, y=0, z=45, timeout=rotation_time)
    time.sleep(rotation_time + 0.5)
    
    # หยุดการหมุน
    ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.5)
    time.sleep(0.5)
    print("Rotation completed!")

# -------------------------
# Callback and Main Logic
# -------------------------
current_x = 0.0
current_y = 0.0

def sub_position_handler(position_info):
    global current_x, current_y
    current_x = position_info[0]
    current_y = position_info[1]
    x, y, z = position_info
    print("chassis position: x:{0}, y:{1}, z:{2}".format(x, y, z))

if __name__ == '__main__':
    # Connect to robot
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_chassis = ep_robot.chassis

    # Reset position to 0 before starting
    ep_chassis.move(x=0, y=0, z=0, xy_speed=0.5).wait_for_completed()
    ep_chassis.sub_position(freq=20, callback=sub_position_handler)
    time.sleep(0.5)

    try:
        # ================================
        # ด้าน 1: ไป +X (ไปข้างหน้า)
        # ================================
        print("=== Side 1: Moving along +X axis ===")
        move_forward_with_pid(ep_chassis, 0.6, 'x', direction=1)
        
        # หมุน 90 องศา → หันไปทาง +Y
        rotate_90_degrees(ep_chassis)
        
        # ================================
        # ด้าน 2: ไป +Y (ไปข้างหน้า)
        # ================================
        print("=== Side 2: Moving along +Y axis ===")
        # เริ่ม position subscription ใหม่
        ep_chassis.sub_position(freq=20, callback=sub_position_handler)
        time.sleep(0.5)
        
        move_forward_with_pid(ep_chassis, 0.6, 'y', direction=1)
        
        # หมุน 90 องศา → หันไปทาง -X
        rotate_90_degrees(ep_chassis)
        
        # ================================
        # ด้าน 3: ไปข้างหน้า (จะทำให้ current_x ลดลง)
        # ================================
        print("=== Side 3: Moving forward (X will decrease) ===")
        # เริ่ม position subscription ใหม่
        ep_chassis.sub_position(freq=20, callback=sub_position_handler)
        time.sleep(0.5)
        
        move_forward_with_pid(ep_chassis, 0.6, 'x', direction=1)
        
        # หมุน 90 องศา → หันไปทาง -Y
        rotate_90_degrees(ep_chassis)
        
        # ================================
        # ด้าน 4: ไปข้างหน้า (จะทำให้ current_y ลดลง)
        # ================================
        print("=== Side 4: Moving forward (Y will decrease) ===")
        # เริ่ม position subscription ใหม่
        ep_chassis.sub_position(freq=20, callback=sub_position_handler)
        time.sleep(0.5)
        
        move_forward_with_pid(ep_chassis, 0.6, 'y', direction=1)
        
        # หมุน 90 องศา → หันไปทาง -Y
        rotate_90_degrees(ep_chassis)

        print("=== Square completed! ===")
        
    except KeyboardInterrupt:
        print("Program interrupted by user.")
    
    finally:
        # ================================
        # จบการทำงาน
        # ================================
        print("All movements completed!")
        
        # หยุด position subscription และปิดการเชื่อมต่อ
        try:
            ep_chassis.unsub_position()
        except:
            pass
        time.sleep(0.5)
        ep_robot.close()