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
def move_forward_with_pid(ep_chassis, target_distance, current_pos_var):
    """
    Move forward using PID control
    target_distance: distance to move (meters)
    current_pos_var: variable name to track current position
    """
    # Setup PID for this movement
    pid = PID(Kp=1.8, Ki=0.1, Kd=4, setpoint=target_distance)
    
    # Control loop for forward movement
    last_time = time.time()
    target_reached = False
    start_position = globals()[current_pos_var]  # Get starting position
    
    print(f"Moving forward {target_distance}m from position {start_position:.2f}")
    
    try:
        while not target_reached:
            now = time.time()
            dt = now - last_time
            last_time = now
            
            # Calculate relative position from start
            relative_position = globals()[current_pos_var] - start_position
            output = pid.compute(relative_position, dt)

            max_speed = 1.5
            speed = max(min(output, max_speed), -max_speed)

            ep_chassis.drive_speed(x=speed, y=0, z=0, timeout=1)

            if abs(relative_position - target_distance) < 0.02:
                print("Target reached. Final position: {:.2f}".format(globals()[current_pos_var]))
                ep_chassis.drive_speed(x=0, y=0, z=0, timeout=1)
                target_reached = True
                break
                
    except KeyboardInterrupt:
        print("Program interrupted by user.")
        ep_chassis.drive_speed(x=0, y=0, z=0, timeout=1)

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

    # ================================
    # การเคลื่อนที่ครั้งที่ 1 - ไปข้างหน้า
    # ================================
    print("=== First Movement: Forward ===")
    move_forward_with_pid(ep_chassis, 0.6, 'current_x')
    # ================================
    # การหมุน 90 องศา
    # ================================
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
    
    # ================================
    # การเคลื่อนที่ครั้งที่ 2 - ไปข้างหน้าอีกครั้ง
    # ================================
    print("=== Second Movement: Forward ===")
    # เริ่ม position subscription ใหม่
    ep_chassis.sub_position(freq=20, callback=sub_position_handler)
    time.sleep(0.5)  # รอให้ position update
    
    # เคลื่อนที่ไปข้างหน้าอีกครั้ง (หลังจากหมุนแล้ว current_y จะเป็นทิศทางใหม่)
    move_forward_with_pid(ep_chassis, 0.6, 'current_y')

    # ================================
    # การหมุน 90 องศา
    # ================================
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

    # ================================
    # การเคลื่อนที่ครั้งที่ 3 - ไปข้างหน้าอีกครั้ง
    # ================================
    print("=== Second Movement: Forward ===")
    # เริ่ม position subscription ใหม่
    ep_chassis.sub_position(freq=20, callback=sub_position_handler)
    time.sleep(0.5)  # รอให้ position update
    
    # เคลื่อนที่ไปข้างหน้าอีกครั้ง (หลังจากหมุนแล้ว current_y จะเป็นทิศทางใหม่)
    move_forward_with_pid(ep_chassis, 0.6, 'current_x')

    # ================================
    # การหมุน 90 องศา
    # ================================
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

    # ================================
    # จบการทำงาน
    # ================================
    print("All movements completed!")
    
    # หยุด position subscription และปิดการเชื่อมต่อ
    ep_chassis.unsub_position()
    time.sleep(0.5)
    ep_robot.close()