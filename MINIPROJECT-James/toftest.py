import time
import robomaster
from robomaster import robot
import numpy as np
from scipy.ndimage import median_filter
from datetime import datetime
import json
from collections import deque

# ===== PID Controller =====
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
        
        if self.integral > self.integral_max:
            self.integral = self.integral_max
        elif self.integral < -self.integral_max:
            self.integral = -self.integral_max
            
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

from collections import deque
from scipy.ndimage import median_filter
import numpy as np
from datetime import datetime

class ToFSensorHandler:
    def __init__(self):
        self.CALIBRATION_SLOPE = 0.0894 
        self.CALIBRATION_Y_INTERCEPT = 3.8409
        self.WINDOW_SIZE = 5
        self.WALL_THRESHOLD = 50.00
        
        # ใช้ deque จำกัดความยาว buffer
        self.tof_buffer = deque(maxlen=self.WINDOW_SIZE)

        # readings จะเก็บข้อมูลการสแกนแต่ละทิศ
        self.readings = {
            'front': [],
            'left': [],
            'right': []
        }
        
        self.current_scan_direction = None
        self.collecting_data = False
        
    def calibrate_tof_value(self, raw_tof_mm):
        calibrated_cm = (self.CALIBRATION_SLOPE * raw_tof_mm) + self.CALIBRATION_Y_INTERCEPT
        return calibrated_cm
    
    def apply_median_filter(self, data):
        if len(data) == 0:
            return 0.0
        elif len(data) == 1:
            return data[0]
        else:
            filtered = median_filter(list(data), size=len(data))
            return filtered[-1]
    
    def tof_data_handler(self, tof_data):
        raw_tof_mm = tof_data[0]
        calibrated_tof_cm = self.calibrate_tof_value(raw_tof_mm)

        # เก็บค่าล่าสุดแบบ real-time
        self.latest_calibrated = calibrated_tof_cm

        # ใส่ลง buffer เพื่อ filter ต่อไป
        self.tof_buffer.append(calibrated_tof_cm)

        # median filter
        filtered_distance = median_filter(
            np.array(self.tof_buffer), size=self.WINDOW_SIZE)[-1]

        # เก็บ readings เพื่อ debug
        self.readings[self.current_direction].append({
            'raw_mm': raw_tof_mm,
            'calibrated_cm': calibrated_tof_cm,
            'filtered_cm': filtered_distance
        })

        print(f"[{self.current_direction.upper()}] "
            f"Raw: {raw_tof_mm}mm | Calibrated: {calibrated_tof_cm:.2f}cm "
            f"(Filtered: {filtered_distance:.2f}cm)")

        
    def start_scanning(self, direction):
        self.current_scan_direction = direction
        self.tof_buffer.clear()
        self.readings[direction] = []
        self.collecting_data = True
        
    def stop_scanning(self, unsub_distance_func):
        self.collecting_data = False
        try:
            unsub_distance_func()
        except:
            pass
    
    def get_latest_distance(self):
        """คืนค่าระยะล่าสุดที่ผ่าน median filter"""
        return self.apply_median_filter(self.tof_buffer)
    
    def get_average_distance(self, direction):
        if direction not in self.readings or len(self.readings[direction]) == 0:
            return 0.0
        
        filtered_values = [reading['filtered_cm'] for reading in self.readings[direction]]
        
        if len(filtered_values) > 4:
            q1 = np.percentile(filtered_values, 25)
            q3 = np.percentile(filtered_values, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            filtered_values = [x for x in filtered_values if lower_bound <= x <= upper_bound]
        
        return np.mean(filtered_values) if filtered_values else 0.0
    
    def is_wall_detected(self, direction, use_realtime=False):
        if use_realtime and self.latest_calibrated is not None:
            return self.latest_calibrated <= self.WALL_THRESHOLD

        # เดิม: ใช้ filtered
        if self.readings[direction]:
            filtered = self.readings[direction][-1]['filtered_cm']
            return filtered <= self.WALL_THRESHOLD
        return False

# ===== Movement Controller =====
class MovementController:
    def __init__(self, chassis):
        self.chassis = chassis
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        
        # PID Parameters
        self.KP = 1.5
        self.KI = 0.3
        self.KD = 4
        self.RAMP_UP_TIME = 0.7
        self.ROTATE_TIME = 2.11  # Right turn
        self.ROTATE_LEFT_TIME = 1.9  # Left turn
        
        # Subscribe to position updates
        self.chassis.sub_position(freq=20, callback=self.position_handler)
        time.sleep(0.25)
    
    def position_handler(self, position_info):
        self.current_x = position_info[0]
        self.current_y = position_info[1]
        self.current_z = position_info[2]
    
    def move_forward_with_pid(self, target_distance, axis, direction=1, gimbal=None, tof_handler=None, sensor=None, allow_backup=True):
        """Move forward/backward using PID control with gimbal obstacle check and automatic backup"""
        pid = PID(Kp=self.KP, Ki=self.KI, Kd=self.KD, setpoint=target_distance)

        start_time = time.time()
        last_time = start_time
        target_reached = False

        min_speed = 0.1
        max_speed = 1.5

        if axis == 'x':
            start_position = self.current_x
        else:
            start_position = self.current_y

        direction_text = "FORWARD" if direction == 1 else "BACKWARD"
        print(f"🚀 Moving {direction_text} {target_distance}m on {axis}-axis, direction: {direction}")

        # === Set gimbal direction ===
        if gimbal:
            yaw_angle = 0 if direction == 1 else 180
            gimbal.moveto(pitch=0, yaw=yaw_angle, pitch_speed=300, yaw_speed=300).wait_for_completed()
            time.sleep(0.1)

        # === Subscribe ToF for continuous obstacle check ===
        if tof_handler and sensor:
            tof_handler.start_scanning('front')
            sensor.sub_distance(freq=25, callback=tof_handler.tof_data_handler)

        try:
            while not target_reached:
                now = time.time()
                dt = now - last_time
                last_time = now
                elapsed_time = now - start_time

                if axis == 'x':
                    current_position = self.current_x
                else:
                    current_position = self.current_y

                relative_position = abs(current_position - start_position)

                # === Enhanced Obstacle Check with Real-time Distance ===
                if tof_handler and direction == 1 and allow_backup:  # Only check when moving forward
                    # Get the latest filtered ToF reading directly from buffer
                    current_distance = 0.0
                    current_distance = tof_handler.latest_calibrated if hasattr(tof_handler, 'latest_calibrated') else 0.0
                    
                    # Log current distance every 1 second (reduce spam)
                    if int(elapsed_time * 2) % 2 == 0 and elapsed_time > 0.5:  # Log every 1 second after 0.5s
                        print(f"📏 Real-time distance: {current_distance:.2f}cm (Buffer size: {len(tof_handler.tof_buffer)})")
                    
                    # Check if obstacle is too close (< 20cm)
                    if current_distance > 0 and current_distance < 30.0:
                        print(f"🚨 OBSTACLE TOO CLOSE! Distance: {current_distance:.2f}cm")
                        print("🛑 Stopping forward movement...")
                        
                        # Stop current movement
                        self.chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
                        
                        # Stop ToF scanning for current movement
                        tof_handler.stop_scanning(sensor.unsub_distance)
                        
                        # Calculate backup distance (move back to 25cm from obstacle)
                        backup_distance = max(0.1, (30.0 - current_distance) / 100.0)  # Convert to meters
                        
                        print(f"🔄 Backing up {backup_distance:.2f}m to maintain 25cm clearance...")
                        
                        # Recursive call to move backward (with allow_backup=False to prevent infinite recursion)
                        self.move_forward_with_pid(
                            target_distance=backup_distance, 
                            axis=axis, 
                            direction=-1,  # Backward
                            gimbal=gimbal,
                            tof_handler=tof_handler,
                            sensor=sensor,
                            allow_backup=False  # Prevent recursive backup
                        )
                        
                        print("✅ Backup completed. Original movement terminated.")
                        return  # Exit the original forward movement
                    
                    # Regular wall check only if no close obstacle
                    elif current_distance > 20.0 and tof_handler.is_wall_detected('front'):
                        print("🛑 Wall detected ahead! Stopping movement.")
                        self.chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
                        break

                # PID calculation
                output = pid.compute(relative_position, dt)

                # Ramp-up logic
                if elapsed_time < self.RAMP_UP_TIME:
                    ramp_multiplier = min_speed + (elapsed_time / self.RAMP_UP_TIME) * (1.0 - min_speed)
                else:
                    ramp_multiplier = 1.0

                ramped_output = output * ramp_multiplier
                speed = max(min(ramped_output, max_speed), -max_speed)

                # Apply movement
                if axis == 'x':
                    self.chassis.drive_speed(x=speed * direction, y=0, z=0, timeout=1)
                else:
                    self.chassis.drive_speed(x=0, y=speed * direction, z=0, timeout=1)

                # Log movement progress
                if int(elapsed_time * 10) % 10 == 0:  # Log every 1 second
                    print(f"📍 Position: {current_position:.3f}, Target: {target_distance:.3f}, Progress: {relative_position:.3f}m")

                # Check if target reached
                if abs(relative_position - target_distance) < 0.02:
                    print(f"✅ Target reached! Final position: {current_position:.3f}")
                    self.chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
                    target_reached = True
                    break

        except KeyboardInterrupt:
            print("Movement interrupted by user.")
        finally:
            # Stop ToF scanning
            if tof_handler and sensor:
                tof_handler.stop_scanning(sensor.unsub_distance)

# ===== Main Execution =====
if __name__ == '__main__':
    # Initialize RoboMaster
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    
    # Get components
    ep_chassis = ep_robot.chassis
    ep_gimbal = ep_robot.gimbal
    ep_sensor = ep_robot.sensor
    
    # Initialize handlers
    tof_handler = ToFSensorHandler()
    movement_controller = MovementController(ep_chassis)
    
    # Test axis (usually 'x' for forward/backward)
    axis_test = 'x'
    
    try:
        print("🤖 Starting RoboMaster movement with obstacle avoidance...")
        print("📏 System will check ToF sensor and backup if obstacle < 20cm")
        
        # Move forward 0.6m with obstacle detection and auto-backup
        movement_controller.move_forward_with_pid(
            target_distance=0.6,
            axis=axis_test, 
            direction=1,  # Forward
            gimbal=ep_gimbal,
            tof_handler=tof_handler,
            sensor=ep_sensor
        )
        
        print("✅ Movement sequence completed!")
        
    except KeyboardInterrupt:
        print("🛑 Program interrupted by user")
    except Exception as e:
        print(f"❌ Error occurred: {e}")
    finally:
        # Cleanup
        try:
            ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
            ep_robot.close()
            print("🔧 Robot connection closed")
        except:
            pass