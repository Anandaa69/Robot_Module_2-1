import time
import robomaster
from robomaster import robot
import numpy as np
from datetime import datetime
import json
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import deque

# =============================================================================
# ===== CONFIGURATION & GLOBAL STATE ==========================================
# =============================================================================
LEFT_IR_SENSOR_ID = 3
RIGHT_IR_SENSOR_ID = 1
CURRENT_POSITION = (3, 3)
CURRENT_DIRECTION = 0
CURRENT_TARGET_YAW = 0.0

def convert_to_json_serializable(obj):
    if isinstance(obj, np.bool_): return bool(obj)
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_to_json_serializable(i) for i in obj]
    return obj

# =============================================================================
# ===== OGM & VISUALIZATION (BAYESIAN) ========================================
# =============================================================================
class OccupancyGridMap:
    """แผนที่แบบความน่าจะเป็นที่ใช้ Bayes' Theorem ผ่าน Log-Odds"""
    def __init__(self, width, height):
        self.width, self.height = width, height
        # Grid เก็บค่า Log-Odds, เริ่มต้นที่ 0 (P=0.5, Unknown)
        self.grid = np.zeros((height, width))
        # ค่า P(occupied|sensed) = 0.8, P(free|sensed) = 0.2
        self.l_occ = np.log(0.8 / 0.2)
        self.l_free = np.log(0.2 / 0.8)

    def update_cell(self, r, c, is_occupied):
        """อัปเดตความเชื่อของช่องด้วย Bayes' update rule (ในรูป Log-Odds)"""
        if 0 <= r < self.height and 0 <= c < self.width:
            self.grid[r, c] += self.l_occ if is_occupied else self.l_free

    def get_probability_grid(self):
        """แปลง Log-Odds กลับเป็นความน่าจะเป็น (0.0 ถึง 1.0)"""
        return 1 - (1 / (1 + np.exp(self.grid)))

class RealTimeVisualizer:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.UNKNOWN, self.FREE, self.OCCUPIED, self.ROBOT, self.PATH = 0, 1, 2, 3, 4
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        # สีตามโจทย์: Unknown(เขียวอ่อน), Free(เทาอ่อน), Occupied(แดงเข้ม)
        self.cmap = ListedColormap(['#90EE90', '#D3D3D3', '#8B0000', '#0000FF', '#FFFF00'])
        self.image = self.ax.imshow(np.zeros((grid_size, grid_size)), cmap=self.cmap, vmin=0, vmax=4)
        self.ax.set_xticks(np.arange(-0.5, self.grid_size, 1)); self.ax.set_yticks(np.arange(-0.5, self.grid_size, 1))
        self.ax.set_xticklabels([]); self.ax.set_yticklabels([])
        self.ax.grid(which='major', color='k', linestyle='-', linewidth=2)
        self.ax.set_title("Real-time Occupancy Grid Map (Bayesian)"); self.fig.tight_layout(); plt.show()

    def update_plot(self, og_map, robot_pos, path=None):
        prob_grid = og_map.get_probability_grid()
        display_grid = np.full((self.grid_size, self.grid_size), self.UNKNOWN, dtype=int)
        
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                p = prob_grid[r, c]
                if p > 0.7: display_grid[r, c] = self.OCCUPIED
                elif p < 0.3: display_grid[r, c] = self.FREE
        
        if path:
            for r, c in path:
                if display_grid[r, c] != self.ROBOT: display_grid[r, c] = self.PATH
        if robot_pos:
            r, c = robot_pos
            display_grid[r, c] = self.ROBOT
            
        self.image.set_data(display_grid)
        self.fig.canvas.draw(); self.fig.canvas.flush_events(); plt.pause(0.01)

# =============================================================================
# ===== CORE ROBOT CONTROL CLASSES ============================================
# =============================================================================
class AttitudeHandler:
    def __init__(self):
        self.current_yaw, self.yaw_tolerance, self.is_monitoring = 0.0, 3.0, False
    def attitude_handler(self, attitude_info):
        if self.is_monitoring: self.current_yaw = attitude_info[0]
    def start_monitoring(self, chassis):
        self.is_monitoring = True; chassis.sub_attitude(freq=20, callback=self.attitude_handler)
    def stop_monitoring(self, chassis):
        self.is_monitoring = False;
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
            time.sleep(0.2)
        final_error = abs(self.normalize_angle(normalized_target - self.current_yaw))
        if final_error <= self.yaw_tolerance: print(f"✅ Yaw Correction Success: {self.current_yaw:.1f}°"); return True
        print(f"⚠️ First attempt incomplete. Current: {self.current_yaw:.1f}°. Fine-tuning...")
        remaining_rotation = -self.normalize_angle(normalized_target - self.current_yaw)
        if abs(remaining_rotation) > 0.5 and abs(remaining_rotation) < 20:
            chassis.move(x=0, y=0, z=remaining_rotation, z_speed=40).wait_for_completed(timeout=2)
            time.sleep(0.2)
        final_error = abs(self.normalize_angle(normalized_target - self.current_yaw))
        if final_error <= self.yaw_tolerance: print(f"✅ Yaw Fine-tuning Success: {self.current_yaw:.1f}°"); return True
        else: print(f"🔥🔥 Yaw Correction FAILED. Final Yaw: {self.current_yaw:.1f}°"); return False

class PID:
    def __init__(self, Kp, Ki, Kd, setpoint=0):
        self.Kp, self.Ki, self.Kd, self.setpoint = Kp, Ki, Kd, setpoint
        self.prev_error, self.integral, self.integral_max = 0, 0, 1.0
    def compute(self, current, dt):
        error = self.setpoint - current
        self.integral += error * dt; self.integral = max(min(self.integral, self.integral_max), -self.integral_max)
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error; return output

class MovementController:
    def __init__(self, chassis):
        self.chassis = chassis
        self.current_x_pos, self.current_y_pos = 0.0, 0.0
        self.KP, self.KI, self.KD = 1.8, 0.25, 12
        self.MOVE_TIMEOUT = 2.5
        self.RAMP_UP_TIME = 1.0
        self.chassis.sub_position(freq=20, callback=self.position_handler)
    
    def position_handler(self, position_info):
        self.current_x_pos, self.current_y_pos = position_info[0], position_info[1]
    
    def move_forward_one_grid(self, axis, attitude_handler):
        print("--- Pre-move Yaw Correction ---")
        attitude_handler.correct_yaw_to_target(self.chassis, CURRENT_TARGET_YAW)
        target_distance = 0.6
        pid = PID(Kp=self.KP, Ki=self.KI, Kd=self.KD, setpoint=target_distance)
        start_time, last_time = time.time(), time.time()
        start_position = self.current_x_pos if axis == 'x' else self.current_y_pos
        print(f"🚀 Moving FORWARD 0.55m, monitoring GLOBAL AXIS '{axis}'")
        while time.time() - start_time < self.MOVE_TIMEOUT:
            now = time.time(); dt = now - last_time; last_time = now
            current_position = self.current_x_pos if axis == 'x' else self.current_y_pos
            relative_position = abs(current_position - start_position)
            if abs(relative_position - target_distance) < 0.03: print("✅ Move complete!"); break
            output = pid.compute(relative_position, dt)
            ramp_multiplier = min(1.0, 0.1 + ((now - start_time) / self.RAMP_UP_TIME) * 0.9)
            speed = max(-1.0, min(1.0, output * ramp_multiplier))
            self.chassis.drive_speed(x=speed, y=0, z=0, timeout=1)
        self.chassis.drive_speed(x=0, y=0, z=0, timeout=0.1); time.sleep(0.5)

    def nudge_robot(self, y_speed, duration):
        print(f"🔩 Nudging robot: y_speed={y_speed}, duration={duration}s")
        self.chassis.drive_speed(x=0, y=y_speed, z=0, timeout=duration + 0.5)
        time.sleep(duration)
        self.chassis.drive_speed(x=0, y=0, z=0, timeout=0.5)
        time.sleep(0.2)

    def rotate_to_direction(self, target_direction, attitude_handler):
        global CURRENT_DIRECTION
        if CURRENT_DIRECTION == target_direction: return
        diff = (target_direction - CURRENT_DIRECTION + 4) % 4
        if diff == 1: self.rotate_90_degrees_right(attitude_handler)
        elif diff == 3: self.rotate_90_degrees_left(attitude_handler)
        elif diff == 2: self.rotate_90_degrees_right(attitude_handler); self.rotate_90_degrees_right(attitude_handler)

    def rotate_90_degrees_right(self, attitude_handler):
        global CURRENT_TARGET_YAW, CURRENT_DIRECTION
        print("🔄 Rotating 90° RIGHT...")
        CURRENT_TARGET_YAW = attitude_handler.normalize_angle(CURRENT_TARGET_YAW + 90)
        attitude_handler.correct_yaw_to_target(self.chassis, CURRENT_TARGET_YAW)
        CURRENT_DIRECTION = (CURRENT_DIRECTION + 1) % 4
    
    def rotate_90_degrees_left(self, attitude_handler):
        global CURRENT_TARGET_YAW, CURRENT_DIRECTION
        print("🔄 Rotating 90° LEFT...")
        CURRENT_TARGET_YAW = attitude_handler.normalize_angle(CURRENT_TARGET_YAW - 90)
        attitude_handler.correct_yaw_to_target(self.chassis, CURRENT_TARGET_YAW)
        CURRENT_DIRECTION = (CURRENT_DIRECTION - 1 + 4) % 4
    
    def cleanup(self):
        try: self.chassis.unsub_position()
        except Exception: pass

# =============================================================================
# ===== SENSOR HANDLER ========================================================
# =============================================================================
class EnvironmentScanner:
    def __init__(self, sensor_adaptor, tof_sensor, gimbal):
        self.sensor_adaptor, self.tof_sensor, self.gimbal = sensor_adaptor, tof_sensor, gimbal
        self.tof_wall_threshold_cm = 50.0; self.last_tof_distance_mm = 0
        self.tof_sensor.sub_distance(freq=5, callback=self._tof_data_handler)
        print(" SENSOR: ToF distance stream started.")
    def _tof_data_handler(self, sub_info): self.last_tof_distance_mm = sub_info[0]
    def get_wall_status(self):
        self.gimbal.moveto(pitch=0, yaw=0).wait_for_completed()
        results, raw_values = {}, {}
        time.sleep(0.1)
        tof_distance_cm = self.last_tof_distance_mm / 10.0
        results['front'] = tof_distance_cm < self.tof_wall_threshold_cm and self.last_tof_distance_mm > 0
        raw_values['front_cm'] = f"{tof_distance_cm:.1f}"
        print(f"[SCAN] Front (ToF): {tof_distance_cm:.1f}cm -> {'WALL' if results['front'] else 'FREE'}")
        left_val = self.sensor_adaptor.get_io(id=LEFT_IR_SENSOR_ID)
        right_val = self.sensor_adaptor.get_io(id=RIGHT_IR_SENSOR_ID)
        results['left'] = (left_val == 0); results['right'] = (right_val == 0)
        raw_values['left_ir'] = left_val; raw_values['right_ir'] = right_val
        print(f"[SCAN] Left (IR ID {LEFT_IR_SENSOR_ID}): val={left_val} -> {'WALL' if results['left'] else 'FREE'}")
        print(f"[SCAN] Right (IR ID {RIGHT_IR_SENSOR_ID}): val={right_val} -> {'WALL' if results['right'] else 'FREE'}")
        return results, raw_values
    def cleanup(self):
        try: self.tof_sensor.unsub_distance(); print(" SENSOR: ToF distance stream stopped.")
        except Exception: pass

# =============================================================================
# ===== PATHFINDING & BACKTRACKING (Probabilistic) ============================
# =============================================================================
def find_path_bfs(og_map, start, end):
    prob_grid = og_map.get_probability_grid()
    queue = deque([[start]])
    visited = {start}
    while queue:
        path = queue.popleft()
        r, c = path[-1]
        if (r, c) == end:
            return path
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < og_map.height and 0 <= nc < og_map.width and (nr, nc) not in visited and prob_grid[nr, nc] < 0.4:
                visited.add((nr, nc))
                new_path = list(path)
                new_path.append((nr, nc))
                queue.append(new_path)
    return None

def find_nearest_frontier(og_map, current_pos, visited_cells):
    prob_grid = og_map.get_probability_grid()
    frontiers = []
    for r in range(og_map.height):
        for c in range(og_map.width):
            if prob_grid[r, c] < 0.4 and (r, c) not in visited_cells: # ถ้าเป็นช่องว่างและยังไม่เคยไป
                is_frontier = False
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < og_map.height and 0 <= nc < og_map.width and abs(prob_grid[nr, nc] - 0.5) < 0.1: # ติดกับช่อง Unknown
                        is_frontier = True; break
                if is_frontier:
                    frontiers.append((r, c))
    if not frontiers: return None
    shortest_path = None
    for frontier in frontiers:
        path = find_path_bfs(og_map, current_pos, frontier)
        if path:
            if not shortest_path or len(path) < len(shortest_path):
                shortest_path = path
    return shortest_path

def execute_backtrack_path(path, movement_controller, attitude_handler, visualizer, og_map):
    global CURRENT_POSITION
    print(f"Executing backtrack path: {path}")
    dir_vectors = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}
    for i in range(len(path) - 1):
        visualizer.update_plot(og_map, path[i], path[i:])
        dr, dc = path[i+1][0] - path[i][0], path[i+1][1] - path[i][1]
        target_direction = dir_vectors[(dr, dc)]
        movement_controller.rotate_to_direction(target_direction, attitude_handler)
        axis_to_monitor = 'x' if CURRENT_DIRECTION == 0 or CURRENT_DIRECTION == 2 else 'y'
        movement_controller.move_forward_one_grid(axis=axis_to_monitor, attitude_handler=attitude_handler)
        CURRENT_POSITION = path[i+1]
    visualizer.update_plot(og_map, CURRENT_POSITION)

def perform_centering_nudge(movement_controller, scanner):
    print("--- Performing Centering Nudge ---")
    wall_status, _ = scanner.get_wall_status()
    has_left_wall, has_right_wall = wall_status['left'], wall_status['right']
    if has_left_wall and not has_right_wall:
        print("   Left wall detected. Nudging RIGHT."); movement_controller.nudge_robot(y_speed=0.15, duration=0.3)
    elif not has_left_wall and has_right_wall:
        print("   Right wall detected. Nudging LEFT."); movement_controller.nudge_robot(y_speed=-0.15, duration=0.3)
    else:
        print("   No single wall detected for nudging. Skipping.")

# =============================================================================
# ===== EXPLORATION LOGIC (BAYESIAN) ==========================================
# =============================================================================
def explore_with_ogm(scanner, movement_controller, attitude_handler, og_map, visualizer, max_steps=40):
    global CURRENT_POSITION, CURRENT_DIRECTION
    
    node_visit_count = 0
    CENTERING_INTERVAL = 2
    visited_cells = {CURRENT_POSITION}

    print("\n🚀 === STARTING AUTONOMOUS EXPLORATION WITH OGM (BAYESIAN) ===")
    with open("experiment_log.txt", "w") as log_file:
        log_file.write("Step\tRobot Pos(y,x)\tIR Left\tToF Front (cm)\tIR Right\n")
        for step in range(max_steps):
            r, c = CURRENT_POSITION
            print(f"\n--- Step {step + 1} at {CURRENT_POSITION}, Facing: {['North', 'East', 'South', 'West'][CURRENT_DIRECTION]} ---")
            wall_status, raw_values = scanner.get_wall_status()
            log_file.write(f"{step+1}\t({r},{c})\t\t{raw_values['left_ir']}\t{raw_values['front_cm']}\t\t{raw_values['right_ir']}\n")
            
            # --- Bayes' Update ---
            dir_map_rel_abs = {'front': CURRENT_DIRECTION, 'right': (CURRENT_DIRECTION + 1) % 4, 'left': (CURRENT_DIRECTION - 1 + 4) % 4}
            dir_vectors = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            for rel_dir, has_wall in wall_status.items():
                abs_dir_idx = dir_map_rel_abs[rel_dir]
                r_update, c_update = r + dir_vectors[abs_dir_idx][0], c + dir_vectors[abs_dir_idx][1]
                og_map.update_cell(r_update, c_update, has_wall)

            visualizer.update_plot(og_map, CURRENT_POSITION)
            
            # --- Decision Logic ---
            next_move_found = False
            priority_dirs = [(CURRENT_DIRECTION + 1) % 4, CURRENT_DIRECTION, (CURRENT_DIRECTION - 1 + 4) % 4]
            prob_grid = og_map.get_probability_grid()
            for target_dir in priority_dirs:
                r_next, c_next = r + dir_vectors[target_dir][0], c + dir_vectors[target_dir][1]
                if 0 <= r_next < og_map.height and 0 <= c_next < og_map.width and prob_grid[r_next, c_next] < 0.4 and (r_next, c_next) not in visited_cells:
                    print(f"Found local unvisited path to {['North', 'East', 'South', 'West'][target_dir]}.")
                    movement_controller.rotate_to_direction(target_dir, attitude_handler)
                    next_move_found = True; break
            
            if not next_move_found:
                print("No local unvisited path. Searching for nearest frontier...")
                path = find_nearest_frontier(og_map, CURRENT_POSITION, visited_cells)
                if path:
                    execute_backtrack_path(path, movement_controller, attitude_handler, visualizer, og_map)
                    visited_cells.update(path)
                    node_visit_count = 0
                    continue
                else:
                    print("🎉 EXPLORATION COMPLETE! No more frontiers reachable.")
                    break
            
            axis_to_monitor = 'x' if CURRENT_DIRECTION == 0 or CURRENT_DIRECTION == 2 else 'y'
            movement_controller.move_forward_one_grid(axis=axis_to_monitor, attitude_handler=attitude_handler)
            
            node_visit_count += 1
            CURRENT_POSITION = (r + dir_vectors[CURRENT_DIRECTION][0], c + dir_vectors[CURRENT_DIRECTION][1])
            visited_cells.add(CURRENT_POSITION)

            if node_visit_count % CENTERING_INTERVAL == 0:
                perform_centering_nudge(movement_controller, scanner)
    
    print("\n🎉 === EXPLORATION FINISHED ==="); visualizer.update_plot(og_map, CURRENT_POSITION)
    with open("final_occupancy_map.json", "w") as f:
        json.dump({"probability_grid": convert_to_json_serializable(og_map.get_probability_grid())}, f, indent=2)
    print("✅ Final map and log file saved.")
    print("... You can close the heatmap window now ...")
    plt.ioff(); plt.show()

# =============================================================================
# ===== MAIN EXECUTION BLOCK ==================================================
# =============================================================================
if __name__ == '__main__':
    ep_robot = None; og_map = OccupancyGridMap(width=4, height=4); attitude_handler = AttitudeHandler()
    movement_controller = None; scanner = None; ep_chassis = None
    try:
        visualizer = RealTimeVisualizer(grid_size=4)
        print("🤖 Connecting to robot..."); ep_robot = robot.Robot(); ep_robot.initialize(conn_type="ap")
        
        ep_chassis, ep_gimbal = ep_robot.chassis, ep_robot.gimbal
        ep_tof_sensor, ep_sensor_adaptor = ep_robot.sensor, ep_robot.sensor_adaptor
        
        print(" GIMBAL: Centering gimbal..."); ep_gimbal.recenter().wait_for_completed(); print(" GIMBAL: Centered.")
        
        scanner = EnvironmentScanner(ep_sensor_adaptor, ep_tof_sensor, ep_gimbal)
        movement_controller = MovementController(ep_chassis)
        attitude_handler.start_monitoring(ep_chassis)
        
        explore_with_ogm(scanner, movement_controller, attitude_handler, og_map, visualizer)

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