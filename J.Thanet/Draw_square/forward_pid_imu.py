# -*-coding:utf-8-*-
import time
from robomaster import robot

# -------------------------
# PID Controller Class
# -------------------------
class PID:
    def __init__(self, Kp, Ki, Kd, setpoint=0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.prev_error = 0
        self.integral = 0
        self.integral_max = 1.0

    def compute(self, current, dt):
        error = self.setpoint - current
        self.integral += error * dt
        
        # Anti-windup
        if self.integral > self.integral_max:
            self.integral = self.integral_max
        elif self.integral < -self.integral_max:
            self.integral = -self.integral_max
            
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

# -------------------------
# Callback Handlers
# -------------------------
current_x = 0.0
yaw_angle = 0.0

def sub_position_handler(position_info):
    global current_x
    current_x = position_info[0]

def sub_imu_info_handler(imu_info):
    global yaw_angle
    # Gyroscope Z-axis is the rotation around the vertical axis
    # The IMU data is in radians/second, we need to integrate it to get the angle
    # Or, for simplicity, we can use the yaw angle from `ep_chassis.sub_attitude`
    # For this example, let's assume we can get a yaw angle directly.
    # The `sub_imu` callback provides gyro rates, not angles.
    # A better approach is to use `sub_attitude` for yaw angle.
    # Let's switch the code to use `sub_attitude` for a cleaner example.
    pass # This function will not be used in the final example below.

def sub_attitude_info_handler(attitude_info):
    global yaw_angle
    yaw_angle = attitude_info[0] # Get the yaw angle (z-axis rotation)

# -------------------------
# Main Logic
# -------------------------
if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_chassis = ep_robot.chassis

    # Reset position and attitude before starting
    ep_chassis.move(x=0, y=0, z=0, xy_speed=0.1).wait_for_completed()
    ep_chassis.sub_position(freq=20, callback=sub_position_handler)
    ep_chassis.sub_attitude(freq=20, callback=sub_attitude_info_handler) # Subscribe to yaw angle
    time.sleep(1)

    target_x = 1.2

    # PID for distance control (X-axis)
    pid_x = PID(Kp=0.5, Ki=0.05, Kd=0.1, setpoint=target_x)

    # PID for yaw angle control (Z-axis)
    # Tune these values carefully. Kp should be high enough to correct,
    # but not so high that it oscillates.
    pid_yaw = PID(Kp=0.2, Ki=0.01, Kd=0.05, setpoint=0) 

    last_time = time.time()
    target_reached = False
    
    try:
        while not target_reached:
            now = time.time()
            dt = now - last_time
            last_time = now

            # PID output for forward speed
            speed_x = pid_x.compute(current_x, dt)
            max_speed = 0.8
            speed_x = max(min(speed_x, max_speed), -max_speed)

            # PID output for rotational speed
            speed_z = pid_yaw.compute(yaw_angle, dt)
            max_rot_speed = 30 # degrees/second
            speed_z = max(min(speed_z, max_rot_speed), -max_rot_speed)

            # Send combined speed commands
            ep_chassis.drive_speed(x=speed_x, y=0, z=speed_z, timeout=1)

            if abs(current_x - target_x) < 0.02 and abs(yaw_angle) < 0.02: # Check both position and orientation
                print(f"Target reached. Final position: {current_x:.2f}m, Final angle: {yaw_angle:.2f} degrees")
                ep_chassis.drive_speed(x=0, y=0, z=0, timeout=1)
                target_reached = True
                
    except KeyboardInterrupt:
        print("Program interrupted by user.")
    finally:
        ep_chassis.drive_speed(x=0, y=0, z=0, timeout=1)
        ep_chassis.unsub_position()
        ep_chassis.unsub_attitude() # Unsubscribe from attitude
        ep_robot.close()