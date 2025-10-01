import time
import robomaster
from robomaster import robot
import numpy as np
import threading

# Global variables representing the robot's state
ROBOT_FACE = 1
CURRENT_TARGET_YAW = 0.0

# --- ค่าจากการสอบเทียบ ToF (จากไฟล์ tof_linear_test.py) ---
CALIBRATION_SLOPE = 0.0894
CALIBRATION_Y_INTERCEPT = 3.8409 # หน่วยเป็น cm

### NEW ###
class TofController:
    """Manages ToF sensor data subscription and calibration."""
    def __init__(self, sensor_module):
        self.ep_sensor = sensor_module
        # เราจะใช้ ToF ตัวหน้าเป็นหลัก (tof1)
        self.calibrated_distance_cm = 0.0
        self.is_monitoring = False

    def _sub_data_handler(self, sub_info):
        """Callback function to handle raw ToF data."""
        raw_tof1 = sub_info[0] # ToF sensor ที่ด้านหน้า
        # แปลงค่าดิบ (mm) เป็น cm และสอบเทียบ
        calibrated_value_cm = (CALIBRATION_SLOPE * raw_tof1) + CALIBRATION_Y_INTERCEPT
        self.calibrated_distance_cm = calibrated_value_cm

    def get_distance_m(self):
        """Returns the calibrated distance in meters."""
        return self.calibrated_distance_cm / 100.0

    def start_monitoring(self):
        if not self.is_monitoring:
            self.ep_sensor.sub_distance(freq=20, callback=self._sub_data_handler)
            self.is_monitoring = True
            print("[ToF Controller] Started monitoring.")
            time.sleep(0.2) # รอให้ค่าแรกเข้ามา

    def stop_monitoring(self):
        if self.is_monitoring:
            try:
                self.ep_sensor.unsub_distance()
                self.is_monitoring = False
                print("[ToF Controller] Stopped monitoring.")
            except Exception as e:
                print(f"[ToF Controller] Error stopping: {e}")


class MovementTracker:
    """Tracks consecutive movements."""
    def __init__(self):
        self.consecutive_forward_moves = 0
        self.consecutive_backward_moves = 0
        self.last_movement_type = None

    def record_movement(self, movement_type):
        if movement_type == 'forward':
            self.consecutive_forward_moves += 1
            self.consecutive_backward_moves = 0
        elif movement_type == 'backward':
            self.consecutive_backward_moves += 1
            self.consecutive_forward_moves = 0
        elif movement_type == 'rotation':
            self.consecutive_forward_moves = 0
            self.consecutive_backward_moves = 0
        self.last_movement_type = movement_type

class AttitudeHandler:
    """Manages subscribing to and handling the robot's attitude (yaw)."""
    def __init__(self):
        self.current_yaw = 0.0
        self.is_monitoring = False

    def attitude_handler(self, attitude_info):
        if self.is_monitoring:
            self.current_yaw = attitude_info[0]

    def start_monitoring(self, chassis):
        self.is_monitoring = True
        chassis.sub_attitude(freq=20, callback=self.attitude_handler)

    def stop_monitoring(self, chassis):
        self.is_monitoring = False
        try:
            chassis.unsub_attitude()
        except Exception:
            pass

    def normalize_angle(self, angle):
        while angle > 180: angle -= 360
        while angle <= -180: angle += 360
        return angle

class PID:
    """A simple PID controller."""
    def __init__(self, Kp, Ki, Kd, setpoint=0):
        self.Kp, self.Ki, self.Kd, self.setpoint = Kp, Ki, Kd, setpoint
        self.prev_error, self.integral, self.integral_max = 0, 0, 1.0

    def compute(self, current, dt):
        error = self.setpoint - current
        self.integral += error * dt
        if self.integral > self.integral_max: self.integral = self.integral_max
        elif self.integral < -self.integral_max: self.integral = -self.integral_max
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

class MovementController:
    """Handles all robot movement logic, combining PID and Yaw Lock."""
    def __init__(self, chassis, attitude_handler, tof_controller): ### NEW ###
        self.chassis = chassis
        self.attitude_handler = attitude_handler
        self.tof_controller = tof_controller ### NEW ###
        self.current_x, self.current_y = 0.0, 0.0
        self.KP, self.KI, self.KD = 1.9, 0.25, 10
        self.RAMP_UP_TIME = 0.8
        self.MOVE_TIMEOUT = 5.0
        self.movement_tracker = MovementTracker()
        self.chassis.sub_position(freq=20, callback=self.position_handler)
        time.sleep(0.1)

    def position_handler(self, position_info):
        self.current_x, self.current_y = position_info[0], position_info[1]

    def _calculate_yaw_correction(self, target_yaw):
        """Calculates the corrective rotational speed to maintain a target yaw."""
        KP_YAW = 0.8
        MAX_YAW_SPEED = 25
        yaw_error = target_yaw - self.attitude_handler.current_yaw
        if yaw_error > 180: yaw_error -= 360
        elif yaw_error < -180: yaw_error += 360
        rotation_speed_z = KP_YAW * yaw_error
        return max(min(rotation_speed_z, MAX_YAW_SPEED), -MAX_YAW_SPEED)

    def hold_position(self):
        """
        Sends a command to counteract any rotation, effectively holding the robot's heading.
        This is called repeatedly by the stabilizer thread.
        """
        global CURRENT_TARGET_YAW
        rotation_speed_z = self._calculate_yaw_correction(CURRENT_TARGET_YAW)
        self.chassis.drive_speed(x=0, y=0, z=rotation_speed_z)

    ### MODIFIED ###
    def move_forward_with_pid(self, target_distance, axis, direction=1):
        """Moves the robot forward using PID and ToF for final distance confirmation."""
        self.movement_tracker.record_movement('forward' if direction == 1 else 'backward')
        
        target_yaw_for_move = self.attitude_handler.normalize_angle(CURRENT_TARGET_YAW)
        
        # --- ToF Setup ---
        initial_tof_dist_m = self.tof_controller.get_distance_m()
        if initial_tof_dist_m <= 0:
            print("\n🔥🔥 ERROR: Invalid initial ToF distance. Cannot move accurately.")
            return

        target_tof_dist_m = initial_tof_dist_m - (target_distance * direction)
        print(f"\n--- Moving with PID, Confirmed by ToF ---")
        print(f"Target Move Distance: {target_distance}m on {axis}-axis")
        print(f"Locking Target Yaw at: {target_yaw_for_move:.2f}°")
        print(f"Initial ToF Distance: {initial_tof_dist_m:.3f}m")
        print(f"Target ToF Distance : {target_tof_dist_m:.3f}m")
        
        # --- Odometry & PID Setup ---
        pid = PID(Kp=self.KP, Ki=self.KI, Kd=self.KD, setpoint=target_distance)
        start_time, last_time = time.time(), time.time()
        start_position = self.current_x if axis == 'x' else self.current_y
        max_speed = 1.5
        
        target_reached = False
        while not target_reached:
            if time.time() - start_time > self.MOVE_TIMEOUT:
                print(f"\n🔥🔥 MOVEMENT TIMEOUT!")
                break
                
            now = time.time(); dt = now - last_time; last_time = now
            
            # --- Check Stop Condition using ToF ---
            current_tof_dist_m = self.tof_controller.get_distance_m()
            if (direction == 1 and current_tof_dist_m <= target_tof_dist_m) or \
               (direction == -1 and current_tof_dist_m >= target_tof_dist_m):
                target_reached = True
                print("\n✅ Target confirmed by ToF sensor!")
                break

            # --- Calculate speed using Odometry PID (for smooth control) ---
            current_position = self.current_x if axis == 'x' else self.current_y
            relative_position = abs(current_position - start_position)
            
            output = pid.compute(relative_position, dt)
            ramp_multiplier = min(1.0, (time.time() - start_time) / self.RAMP_UP_TIME)
            forward_speed = max(-max_speed, min(max_speed, output)) * ramp_multiplier
            rotation_speed_z = self._calculate_yaw_correction(target_yaw_for_move)
            
            self.chassis.drive_speed(
                x=forward_speed * direction, 
                y=0, 
                z=rotation_speed_z, 
                timeout=0.1
            )
            print(f"Moving... Odometry: {relative_position:.3f}m | ToF: {current_tof_dist_m:.3f}m -> Target: {target_tof_dist_m:.3f}m", end='\r')

        self.chassis.drive_speed(x=0, y=0, z=self._calculate_yaw_correction(target_yaw_for_move), timeout=0.2)
        self.chassis.drive_speed(x=0, y=0, z=0)
        time.sleep(0.05)

        final_pos = self.current_x if axis == 'x' else self.current_y
        moved_dist = abs(final_pos - start_position)
        print(f"\nDistance moved (by Odometry): {moved_dist:.3f}m")
        final_tof_dist_m = self.tof_controller.get_distance_m()
        actual_moved_dist = abs(initial_tof_dist_m - final_tof_dist_m)
        print(f"Distance moved (by ToF)    : {actual_moved_dist:.3f}m")

        if not target_reached: print(f"⚠️ Target possibly not reached (Timeout).")
            
    def cleanup(self):
        try:
            self.chassis.unsub_position()
        except:
            pass

def stabilizer_func(controller, stop_event):
    """
    This function runs in a separate thread. Its only job is to call
    hold_position repeatedly to keep the robot stable.
    """
    print("[Stabilizer Thread] Started.")
    while not stop_event.is_set():
        controller.hold_position()
        time.sleep(0.05)
    print("[Stabilizer Thread] Stopped.")

# =============================================================================
# ===== MAIN EXECUTION BLOCK ==================================================
# =============================================================================
if __name__ == '__main__':
    ep_robot = None
    stabilizer_thread = None
    stop_stabilizer_event = threading.Event()

    try:
        print("🤖 Connecting to robot...")
        ep_robot = robot.Robot()
        ep_robot.initialize(conn_type="ap")
        
        attitude_handler = AttitudeHandler()
        attitude_handler.start_monitoring(ep_robot.chassis)
        
        ### NEW ###
        tof_controller = TofController(ep_robot.sensor)
        tof_controller.start_monitoring()

        movement_controller = MovementController(ep_robot.chassis, attitude_handler, tof_controller)
        
        ep_robot.gimbal.recenter(pitch_speed=100, yaw_speed=100).wait_for_completed()
        
        print("Syncing initial direction...")
        time.sleep(0.5)
        CURRENT_TARGET_YAW = attitude_handler.current_yaw
        print(f"Initial direction synced. Target Yaw is now: {CURRENT_TARGET_YAW:.2f}°")

        stabilizer_thread = threading.Thread(
            target=stabilizer_func, 
            args=(movement_controller, stop_stabilizer_event)
        )
        stabilizer_thread.start()

        print("\n" + "="*50)
        print("✅ ROBOT IS STABLE AND READY")
        print("   >>> Place robot facing a flat wall <<<")
        print("   - Type 'yes' and press Enter to move forward 0.6m.")
        print("   - Type 'exit' to quit.")
        print("="*50 + "\n")

        while True:
            command = input(">> Enter command: ").lower().strip()

            if command == 'yes':
                print("[Main Thread] Command received. Pausing stabilizer for movement...")
                stop_stabilizer_event.set()
                stabilizer_thread.join()

                axis_to_move = 'y' if ROBOT_FACE % 2 == 0 else 'x'
                movement_controller.move_forward_with_pid(
                    target_distance=0.6, axis=axis_to_move, direction=1
                )
                
                print("\n[Main Thread] Move complete. Restarting stabilizer...")
                stop_stabilizer_event.clear()
                stabilizer_thread = threading.Thread(
                    target=stabilizer_func, 
                    args=(movement_controller, stop_stabilizer_event)
                )
                stabilizer_thread.start()
                
                print("\n✅ Ready for the next command.")
            
            else:
                print("🛑 Exiting control loop.")
                break

    except KeyboardInterrupt:
        print("\n⚠️ User interrupted.")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🔌 Cleaning up...")
        if stabilizer_thread and stabilizer_thread.is_alive():
            print("Stopping stabilizer thread...")
            stop_stabilizer_event.set()
            stabilizer_thread.join()

        if ep_robot:
            ep_robot.chassis.drive_speed(x=0, y=0, z=0, timeout=0.5)
            if 'tof_controller' in locals() and tof_controller.is_monitoring: ### NEW ###
                tof_controller.stop_monitoring()
            if 'attitude_handler' in locals() and attitude_handler.is_monitoring:
                attitude_handler.stop_monitoring(ep_robot.chassis)
            if 'movement_controller' in locals():
                movement_controller.cleanup()
            ep_robot.close()
            print("🔌 Connection closed.")