import time
import robomaster
from robomaster import robot
import numpy as np
from datetime import datetime
import json
import os

# =============================================================================
# ===== CONFIGURATION =========================================================
# =============================================================================
# ### ATTENTION: คุณต้องใช้ IR Sensor 2 ตัวที่ ID ไม่ซ้ำกัน ###
# แก้ไข ID ให้ตรงกับเซ็นเซอร์ที่ต่อไว้จริง
LEFT_IR_SENSOR_ID = 3
RIGHT_IR_SENSOR_ID = 1 # <--- ตัวอย่าง: สมมติว่าตัวขวาคือ ID 4

CURRENT_POSITION = (3, 3)
CURRENT_DIRECTION = 0
CURRENT_TARGET_YAW = 0.0

# --- Helper function for JSON serialization ---
def convert_to_json_serializable(obj):
    if isinstance(obj, np.bool_): return bool(obj)
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_to_json_serializable(i) for i in obj]
    return obj

# =============================================================================
# ===== CORE ROBOT CONTROL CLASSES (No changes) ===============================
# =============================================================================
class AttitudeHandler:
    def __init__(self):
        self.current_yaw, self.yaw_tolerance, self.is_monitoring = 0.0, 3.0, False
    def attitude_handler(self, attitude_info):
        if self.is_monitoring: self.current_yaw = attitude_info[0]
    def start_monitoring(self, chassis):
        self.is_monitoring = True; chassis.sub_attitude(freq=20, callback=self.attitude_handler)
    def stop_monitoring(self, chassis):
        self.is_monitoring = False
        try: chassis.unsub_attitude()
        except Exception: pass
    def normalize_angle(self, angle):
        while angle > 180: angle -= 360
        while angle <= -180: angle += 360
        return angle
    def correct_yaw_to_target(self, chassis, target_yaw=0.0):
        normalized_target = self.normalize_angle(target_yaw); time.sleep(0.2)
        robot_rotation = -self.normalize_angle(normalized_target - self.current_yaw)
        print(f"\n🔧 Correcting Yaw: {self.current_yaw:.1f}° -> {target_yaw}°. Rotating: {robot_rotation:.1f}°")
        if abs(robot_rotation) > self.yaw_tolerance:
            chassis.move(x=0, y=0, z=robot_rotation, z_speed=60).wait_for_completed(timeout=3)
        time.sleep(0.3)
        final_error = abs(self.normalize_angle(normalized_target - self.current_yaw))
        if final_error <= self.yaw_tolerance:
            print(f"✅ Yaw Correction Success: {self.current_yaw:.1f}°"); return True
        else:
            print(f"🔥🔥 Yaw Correction FAILED. Final Yaw: {self.current_yaw:.1f}°"); return False

class PID:
    def __init__(self, Kp, Ki, Kd, setpoint=0):
        self.Kp, self.Ki, self.Kd, self.setpoint = Kp, Ki, Kd, setpoint
        self.prev_error, self.integral, self.integral_max = 0, 0, 1.0
    def compute(self, current, dt):
        error = self.setpoint - current
        self.integral += error * dt
        self.integral = max(min(self.integral, self.integral_max), -self.integral_max)
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error; return output

class MovementController:
    def __init__(self, chassis):
        self.chassis = chassis
        self.current_x_pos, self.current_y_pos = 0.0, 0.0
        self.KP, self.KI, self.KD = 1.9, 0.25, 10
        self.MOVE_TIMEOUT = 5.0
        self.chassis.sub_position(freq=20, callback=self.position_handler)
    def position_handler(self, position_info):
        self.current_x_pos, self.current_y_pos = position_info[0], position_info[1]
    def move_one_grid_cell(self):
        target_distance = 0.60
        pid = PID(Kp=self.KP, Ki=self.KI, Kd=self.KD, setpoint=target_distance)
        start_time, last_time = time.time(), time.time()
        self.chassis.move(x=0, y=0, z=0, xy_speed=0).wait_for_completed()
        start_position = self.current_x_pos
        print(f"🚀 Moving FORWARD {target_distance}m")
        while time.time() - start_time < self.MOVE_TIMEOUT:
            now = time.time(); dt = now - last_time; last_time = now
            relative_position = abs(self.current_x_pos - start_position)
            if abs(relative_position - target_distance) < 0.03:
                print("✅ Move complete!"); break
            output = pid.compute(relative_position, dt)
            speed = max(-1.0, min(1.0, output))
            self.chassis.drive_speed(x=speed, y=0, z=0, timeout=1)
        self.chassis.drive_speed(x=0, y=0, z=0, timeout=0.1); time.sleep(0.5)
    def rotate_90_degrees_right(self, attitude_handler):
        global CURRENT_TARGET_YAW, CURRENT_DIRECTION
        print("🔄 Rotating 90° RIGHT..."); CURRENT_TARGET_YAW = attitude_handler.normalize_angle(CURRENT_TARGET_YAW + 90)
        attitude_handler.correct_yaw_to_target(self.chassis, CURRENT_TARGET_YAW); CURRENT_DIRECTION = (CURRENT_DIRECTION + 1) % 4
    def rotate_90_degrees_left(self, attitude_handler):
        global CURRENT_TARGET_YAW, CURRENT_DIRECTION
        print("🔄 Rotating 90° LEFT..."); CURRENT_TARGET_YAW = attitude_handler.normalize_angle(CURRENT_TARGET_YAW - 90)
        attitude_handler.correct_yaw_to_target(self.chassis, CURRENT_TARGET_YAW); CURRENT_DIRECTION = (CURRENT_DIRECTION - 1 + 4) % 4
    def cleanup(self):
        try: self.chassis.unsub_position()
        except Exception: pass

# =============================================================================
# ===== SENSOR HANDLER (STABLE VERSION FOR S1) ================================
# =============================================================================
class EnvironmentScanner:
    def __init__(self, sensor_adaptor, tof_sensor):
        self.sensor_adaptor = sensor_adaptor
        self.tof_sensor = tof_sensor
        self.tof_wall_threshold_cm = 50.0
        self.last_tof_distance_mm = 0
        # <<< MODIFIED: เริ่ม subscribe ToF แค่ครั้งเดียว
        self.tof_sensor.sub_distance(freq=5, callback=self._tof_data_handler)
        print(" SENSOR: ToF distance stream started.")

    def _tof_data_handler(self, sub_info):
        self.last_tof_distance_mm = sub_info[0]

    def get_wall_status(self):
        results, raw_values = {}, {}

        # 1. Read Front (using latest ToF value from continuous stream)
        # <<< MODIFIED: ไม่ต้อง sub/unsub แล้ว
        tof_distance_cm = self.last_tof_distance_mm / 10.0
        results['front'] = tof_distance_cm < self.tof_wall_threshold_cm and self.last_tof_distance_mm > 0
        raw_values['front_cm'] = f"{tof_distance_cm:.1f}"
        print(f"[SCAN] Front (ToF): {tof_distance_cm:.1f}cm -> {'WALL' if results['front'] else 'FREE'}")

        # 2. Read Left and Right IR Sensors
        # <<< MODIFIED: แก้ไขการรับค่าจาก get_io และลบ 'port' ออก
        left_val = self.sensor_adaptor.get_io(id=LEFT_IR_SENSOR_ID)
        right_val = self.sensor_adaptor.get_io(id=RIGHT_IR_SENSOR_ID)

        results['left'] = (left_val == 0)
        results['right'] = (right_val == 0)
        raw_values['left_ir'] = left_val
        raw_values['right_ir'] = right_val
        print(f"[SCAN] Left (IR ID {LEFT_IR_SENSOR_ID}): val={left_val} -> {'WALL' if results['left'] else 'FREE'}")
        print(f"[SCAN] Right (IR ID {RIGHT_IR_SENSOR_ID}): val={right_val} -> {'WALL' if results['right'] else 'FREE'}")

        return results, raw_values
    
    def cleanup(self):
        # <<< MODIFIED: unsub ToF ตอนจบ
        try:
            self.tof_sensor.unsub_distance()
            print(" SENSOR: ToF distance stream stopped.")
        except Exception: pass

# =============================================================================
# ===== OGM & EXPLORATION LOGIC (No changes needed) ===========================
# =============================================================================
class OccupancyGridMap:
    def __init__(self, width, height):
        self.width, self.height = width, height; self.grid = np.zeros((height, width))
        self.l_occ = np.log(0.8 / 0.2); self.l_free = np.log(0.2 / 0.8)
    def update_cell(self, y, x, is_occupied):
        if 0 <= y < self.height and 0 <= x < self.width:
            self.grid[y, x] += self.l_occ if is_occupied else self.l_free
    def get_probability_grid(self): return 1 - (1 / (1 + np.exp(self.grid)))
    def display_map(self, robot_pos, robot_dir):
        print("\n" + "="*7 + " REAL-TIME MAP " + "="*7)
        prob_grid = self.get_probability_grid()
        dir_symbols = {0: '^', 1: '>', 2: 'v', 3: '<'}
        for r in range(self.height):
            row_str = "|"
            for c in range(self.width):
                if (r, c) == robot_pos: row_str += f" {dir_symbols.get(robot_dir, '?')} |"
                else:
                    p = prob_grid[r, c]
                    if p > 0.65: row_str += "███|"
                    elif p < 0.35: row_str += "   |"
                    else: row_str += " . |"
            print(row_str)
        print("="*27)

def explore_with_ogm(scanner, movement_controller, attitude_handler, og_map, max_steps=40):
    global CURRENT_POSITION, CURRENT_DIRECTION
    print("\n🚀 === STARTING AUTONOMOUS EXPLORATION WITH OGM ===")
    with open("experiment_log.txt", "w") as log_file:
        log_file.write("Step\tRobot Pos(y,x)\tIR Left\tToF Front (cm)\tIR Right\n")
        for step in range(max_steps):
            print(f"\n--- Step {step + 1} at {CURRENT_POSITION}, Facing: {['North', 'East', 'South', 'West'][CURRENT_DIRECTION]} ---")
            wall_status, raw_values = scanner.get_wall_status()
            log_file.write(f"{step+1}\t({CURRENT_POSITION[0]},{CURRENT_POSITION[1]})\t\t{raw_values['left_ir']}\t{raw_values['front_cm']}\t\t{raw_values['right_ir']}\n")
            
            dir_vectors = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            front_dir, right_dir, left_dir = CURRENT_DIRECTION, (CURRENT_DIRECTION + 1) % 4, (CURRENT_DIRECTION - 1 + 4) % 4
            front_cell, right_cell, left_cell = (CURRENT_POSITION[0] + dir_vectors[front_dir][0], CURRENT_POSITION[1] + dir_vectors[front_dir][1]), (CURRENT_POSITION[0] + dir_vectors[right_dir][0], CURRENT_POSITION[1] + dir_vectors[right_dir][1]), (CURRENT_POSITION[0] + dir_vectors[left_dir][0], CURRENT_POSITION[1] + dir_vectors[left_dir][1])
            og_map.update_cell(front_cell[0], front_cell[1], wall_status['front']); og_map.update_cell(right_cell[0], right_cell[1], wall_status['right']); og_map.update_cell(left_cell[0], left_cell[1], wall_status['left'])
            og_map.display_map(CURRENT_POSITION, CURRENT_DIRECTION)
            
            if not wall_status['right']: print("Decision: Turning RIGHT."); movement_controller.rotate_90_degrees_right(attitude_handler)
            elif not wall_status['front']: print("Decision: Moving FORWARD.")
            elif not wall_status['left']: print("Decision: Turning LEFT."); movement_controller.rotate_90_degrees_left(attitude_handler)
            else: print("Decision: DEAD END. Turning around."); movement_controller.rotate_90_degrees_right(attitude_handler); movement_controller.rotate_90_degrees_right(attitude_handler)
            
            movement_controller.move_one_grid_cell()
            
            new_pos_y, new_pos_x = CURRENT_POSITION[0] + dir_vectors[CURRENT_DIRECTION][0], CURRENT_POSITION[1] + dir_vectors[CURRENT_DIRECTION][1]
            if not (0 <= new_pos_y < og_map.height and 0 <= new_pos_x < og_map.width):
                print(f"🔥🔥 ERROR: Robot moved out of bounds to ({new_pos_y}, {new_pos_x}). Stopping."); break
            CURRENT_POSITION = (new_pos_y, new_pos_x)
    
    print("\n🎉 === EXPLORATION FINISHED ==="); og_map.display_map(CURRENT_POSITION, CURRENT_DIRECTION)
    with open("final_occupancy_map.json", "w") as f: json.dump({"occupancy_probability_grid": convert_to_json_serializable(og_map.get_probability_grid())}, f, indent=2)
    print("✅ Final map and log file saved.")

# =============================================================================
# ===== MAIN EXECUTION BLOCK ==================================================
# =============================================================================
if __name__ == '__main__':
    ep_robot = None; og_map = OccupancyGridMap(width=4, height=4); attitude_handler = AttitudeHandler()
    movement_controller = None; scanner = None; ep_chassis = None
    try:
        print("🤖 Connecting to robot..."); ep_robot = robot.Robot(); ep_robot.initialize(conn_type="ap")
        
        ep_chassis = ep_robot.chassis
        ep_gimbal = ep_robot.gimbal
        ep_tof_sensor = ep_robot.sensor
        ep_sensor_adaptor = ep_robot.sensor_adaptor
        
        print(" GIMBAL: Centering gimbal..."); ep_gimbal.recenter().wait_for_completed(); print(" GIMBAL: Centered.")
        
        scanner = EnvironmentScanner(ep_sensor_adaptor, ep_tof_sensor)
        movement_controller = MovementController(ep_chassis)
        attitude_handler.start_monitoring(ep_chassis)
        
        explore_with_ogm(scanner, movement_controller, attitude_handler, og_map)

    except KeyboardInterrupt: print("\n⚠️ User interrupted exploration.")
    except Exception as e: print(f"\n❌ An error occurred: {e}"); import traceback; traceback.print_exc()
    finally:
        if ep_robot:
            print("🔌 Cleaning up and closing connection...")
            if scanner: scanner.cleanup()
            if attitude_handler and attitude_handler.is_monitoring: attitude_handler.stop_monitoring(ep_chassis)
            if movement_controller: movement_controller.cleanup()
            ep_robot.close()
            print("🔌 Connection closed.")