from robomaster import robot
import time

# -------------------------
# PID Controller Class
# -------------------------
class PID:
    def __init__(self, Kp, Ki, Kd, setpoint=0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.prev_error = 0.0
        self.integral = 0.0
        self.integral_max = 1.0  # Limit integral term to prevent windup

    def compute(self, current_value, dt):
        """
        Calculates the PID output based on the current value and time delta.
        """
        error = self.setpoint - current_value
        
        # Proportional term
        p_term = self.Kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        if self.integral > self.integral_max:
            self.integral = self.integral_max
        elif self.integral < -self.integral_max:
            self.integral = -self.integral_max
        i_term = self.Ki * self.integral
        
        # Derivative term
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        d_term = self.Kd * derivative
        
        output = p_term + i_term + d_term
        
        self.prev_error = error
        return output

# -------------------------
# Main Control Logic
# -------------------------
if __name__ == '__main__':
    # Initialize robot and chassis module
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_chassis = ep_robot.chassis
    
    # Target distance in meters
    target_distance = 1.2
    
    # PID tuning parameters
    kp = 0.5
    ki = 0.05
    kd = 0.1
    
    # Initialize PID controller
    pid = PID(Kp=kp, Ki=ki, Kd=kd, setpoint=target_distance)
    
    # Global variable to store current position
    current_x = 0.0
    
    # Callback function to update position data
    def position_handler(pos_info):
        """This function is called automatically by the robot."""
        global current_x
        current_x = pos_info[0]

    try:
        # Reset position to zero and subscribe to position updates
        print("Resetting position to (0, 0, 0)...")
        ep_chassis.move(x=0, y=0, z=0, xy_speed=0.1).wait_for_completed()
        ep_chassis.sub_position(freq=20, callback=position_handler)
        time.sleep(1) # Give it a moment to receive the first position reading
        
        print(f"Starting to move to target distance: {target_distance:.2f} meters.")
        
        # Control loop
        last_time = time.time()
        start_time = last_time
        
        # Variables for soft start and stopping condition
        target_reached = False
        max_speed = 0.8  # Maximum speed allowed
        
        while not target_reached:
            now = time.time()
            dt = now - last_time
            last_time = now
            
            # Use soft start to gradually increase speed at the beginning
            run_time = now - start_time
            current_max_speed = max_speed
            if run_time < 1.0:
                current_max_speed = max_speed * (run_time / 1.0)
            
            # Compute PID output to get the required speed
            pid_output = pid.compute(current_x, dt)
            
            # Clamp the output speed to the current maximum speed
            speed = max(min(pid_output, current_max_speed), -current_max_speed)
            
            # Send speed command to the robot chassis
            ep_chassis.drive_speed(x=speed, y=0, z=0, timeout=1)
            
            # Check if the robot has reached the target
            if abs(current_x - target_distance) < 0.02:
                print(f"Target reached! Final position: {current_x:.2f} meters.")
                ep_chassis.drive_speed(x=0, y=0, z=0)  # Stop the robot
                target_reached = True
                
            time.sleep(0.01) # Small delay to avoid excessive CPU usage

    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    finally:
        # Clean up: stop the robot and unsubscribe from position updates
        print("Cleaning up...")
        ep_chassis.drive_speed(x=0, y=0, z=0)
        ep_chassis.unsub_position()
        ep_robot.close()
        print("Done.")