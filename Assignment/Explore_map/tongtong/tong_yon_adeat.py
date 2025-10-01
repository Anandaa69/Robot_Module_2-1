import time
import robomaster
from robomaster import robot
import numpy as np
import threading

# Global variables representing the robot's state
ROBOT_FACE = 1 
CURRENT_TARGET_YAW = 0.0

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
    def __init__(self, chassis, attitude_handler):
        self.chassis = chassis
        self.attitude_handler = attitude_handler
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

    def move_forward_with_pid(self, target_distance, axis, direction=1):
        """Moves the robot forward for a specific distance using PID and Yaw Lock."""
        self.movement_tracker.record_movement('forward' if direction == 1 else 'backward')
        
        target_yaw_for_move = self.attitude_handler.normalize_angle(CURRENT_TARGET_YAW)
        print(f"\n--- Moving with PID and Yaw Lock ---")
        print(f"Target Distance: {target_distance}m on {axis}-axis")
        print(f"Locking Target Yaw at: {target_yaw_for_move:.2f}°")
        
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
            current_position = self.current_x if axis == 'x' else self.current_y
            relative_position = abs(current_position - start_position)
            
            if relative_position >= target_distance - 0.02:
                target_reached = True
                break

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
            print(f"Moving... Dist: {relative_position:.3f}/{target_distance:.2f} m | YawErr: {target_yaw_for_move - self.attitude_handler.current_yaw:5.2f}°", end='\r')

        self.chassis.drive_speed(x=0, y=0, z=self._calculate_yaw_correction(target_yaw_for_move), timeout=0.2)
        self.chassis.drive_speed(x=0, y=0, z=0)
        time.sleep(0.05)

        final_pos = self.current_x if axis == 'x' else self.current_y
        moved_dist = abs(final_pos - start_position)
        print(f"\nMoved a total distance of {moved_dist:.3f}m")
        if target_reached: print(f"✅ Target reached!")
        else: print(f"⚠️ Target possibly not reached.")
            
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
        time.sleep(0.05) # Run this loop 20 times per second
    print("[Stabilizer Thread] Stopped.")

# =============================================================================
# ===== MAIN EXECUTION BLOCK (Multithreaded Version) ========================
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
        movement_controller = MovementController(ep_robot.chassis, attitude_handler)
        
        ep_robot.gimbal.recenter(pitch_speed=100, yaw_speed=100).wait_for_completed()
        
        print("Syncing initial direction...")
        time.sleep(0.5) 
        CURRENT_TARGET_YAW = attitude_handler.current_yaw
        print(f"Initial direction synced. Target Yaw is now: {CURRENT_TARGET_YAW:.2f}°")

        # Start the stabilizer thread to keep the robot still from the very beginning
        stabilizer_thread = threading.Thread(
            target=stabilizer_func, 
            args=(movement_controller, stop_stabilizer_event)
        )
        stabilizer_thread.start()

        print("\n" + "="*50)
        print("✅ ROBOT IS STABLE AND READY")
        print("   - Type 'yes' and press Enter to move forward 0.6m.")
        print("   - Type 'exit' to quit.")
        print("="*50 + "\n")

        while True:
            # The Main Thread will block here, waiting for input.
            # Meanwhile, the Stabilizer Thread is running in the background.
            command = input(">> Enter command: ").lower().strip()

            if command == 'yes':
                # --- Manage threads to perform movement ---
                print("[Main Thread] Command received. Pausing stabilizer for movement...")
                # 1. Signal the stabilizer thread to stop.
                stop_stabilizer_event.set()
                stabilizer_thread.join() # Wait for the thread to fully stop.

                # 2. Now that the main thread has full control, execute the movement.
                axis_to_move = 'y' if ROBOT_FACE % 2 == 0 else 'x'
                movement_controller.move_forward_with_pid(
                    target_distance=0.6, axis=axis_to_move, direction=1
                )
                
                print("\n[Main Thread] Move complete. Restarting stabilizer...")
                # 3. Restart the stabilizer thread to hold the new position.
                stop_stabilizer_event.clear() # Clear the stop signal for the new thread.
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
        # Always make sure to stop the background thread before exiting.
        if stabilizer_thread and stabilizer_thread.is_alive():
            print("Stopping stabilizer thread...")
            stop_stabilizer_event.set()
            stabilizer_thread.join()

        if ep_robot:
            ep_robot.chassis.drive_speed(x=0, y=0, z=0, timeout=0.5)
            if 'attitude_handler' in locals() and attitude_handler.is_monitoring:
                attitude_handler.stop_monitoring(ep_robot.chassis)
            if 'movement_controller' in locals():
                movement_controller.cleanup()
            ep_robot.close()
            print("🔌 Connection closed.")