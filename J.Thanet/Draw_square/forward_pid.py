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

# -------------------------
# Callback and Main Logic
# -------------------------
current_x = 0.0

def sub_position_handler(position_info):
    global current_x
    current_x = position_info[0]
    # print("Current X: {:.2f}".format(current_x)) # Comment out to reduce print spam

if __name__ == '__main__':
    # Connect to robot
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_chassis = ep_robot.chassis

    # Reset position to 0 before starting
    ep_chassis.move(x=0, y=0, z=0, xy_speed=0.5).wait_for_completed()
    ep_chassis.sub_position(freq=20, callback=sub_position_handler)
    time.sleep(1) # Give it time to get the first position reading

    # Target distance (meters)
    target_x = 1.2

    # Setup PID
    pid = PID(Kp=2.5, Ki=0.1, Kd=4, setpoint=target_x)

    # Control loop
    last_time = time.time()
    
    # Add a variable to track if we've reached the target
    target_reached = False
    
    try:
        while not target_reached:
            now = time.time()
            dt = now - last_time
            last_time = now

            # Compute PID output based on current position
            output = pid.compute(current_x, dt)

            # Cap the speed to avoid overshooting and instability
            max_speed = 1.5
            speed = max(min(output, max_speed), -max_speed)

            # Send speed command directly
            ep_chassis.drive_speed(x=speed, y=0, z=0, timeout=1)

            # Condition to stop: close enough to the target
            if abs(current_x - target_x) < 0.02:
                print("Target reached. Final position: {:.2f}".format(current_x))
                ep_chassis.drive_speed(x=0, y=0, z=0, timeout=1) # Stop the robot
                target_reached = True # Exit the loop
                
    except KeyboardInterrupt:
        print("Program interrupted by user.")
    finally:
        # Stop the robot and unsubscribe
        ep_chassis.drive_speed(x=0, y=0, z=0, timeout=1)
        ep_chassis.unsub_position()
        ep_robot.close()