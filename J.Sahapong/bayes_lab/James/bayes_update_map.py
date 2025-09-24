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
# ===== STRUCTURAL MAP & ADVANCED VISUALIZATION ===============================
# =============================================================================
class MapCell:
    def __init__(self):
        self.walls = {'north': False, 'south': False, 'east': False, 'west': False}
        self.visited = False

class MazeMap:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[MapCell() for _ in range(width)] for _ in range(height)]

    def set_wall(self, r, c, direction, has_wall):
        if 0 <= r < self.height and 0 <= c < self.width:
            self.grid[r][c].walls[direction] = has_wall
            if direction == 'north' and r > 0: self.grid[r-1][c].walls['south'] = has_wall
            if direction == 'south' and r < self.height - 1: self.grid[r+1][c].walls['north'] = has_wall
            if direction == 'west' and c > 0: self.grid[r][c-1].walls['east'] = has_wall
            if direction == 'east' and c < self.width - 1: self.grid[r][c+1].walls['west'] = has_wall

    def has_wall(self, r, c, direction):
        if 0 <= r < self.height and 0 <= c < self.width:
            return self.grid[r][c].walls[direction]
        return True

    def set_visited(self, r, c):
        if 0 <= r < self.height and 0 <= c < self.width:
            self.grid[r][c].visited = True

class RealTimeVisualizer:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        
        self.colors = {
            "visited": "#D0D0FF", # สีฟ้าอ่อน
            "unvisited": "#A0A0A0", # สีเทา
            "robot": "#0000FF", # สีน้ำเงิน
            "path": "#FFFF00" # สีเหลือง
        }

    def update_plot(self, maze_map, robot_pos, path=None):
        self.ax.clear()
        self.ax.set_title("Real-time Structural Map")
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.ax.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax.set_ylim(self.grid_size - 0.5, -0.5) # Invert Y-axis

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                color = self.colors['visited'] if maze_map.grid[r][c].visited else self.colors['unvisited']
                self.ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=color, edgecolor='k', lw=0.5))

                # วาดกำแพงเป็นเส้นหนา
                if maze_map.has_wall(r, c, 'north'): self.ax.plot([c-0.5, c+0.5], [r-0.5, r-0.5], color='black', linewidth=4)
                if maze_map.has_wall(r, c, 'south'): self.ax.plot([c-0.5, c+0.5], [r+0.5, r+0.5], color='black', linewidth=4)
                if maze_map.has_wall(r, c, 'west'): self.ax.plot([c-0.5, c-0.5], [r-0.5, r+0.5], color='black', linewidth=4)
                if maze_map.has_wall(r, c, 'east'): self.ax.plot([c+0.5, c+0.5], [r-0.5, r+0.5], color='black', linewidth=4)
        
        if path:
            for r, c in path:
                 self.ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=self.colors['path'], edgecolor='k', lw=0.5, alpha=0.7))

        if robot_pos:
            r, c = robot_pos
            self.ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=self.colors['robot'], edgecolor='k', lw=0.5))

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
        self.chassis.drive_speed(x=0, y=0, z=0, timeout=0.5); time.sleep(0.2)

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
# ===== PATHFINDING & BACKTRACKING LOGIC ======================================
# =============================================================================
def find_path_bfs(maze_map, start, end):
    queue = deque([[start]])
    visited = {start}
    dir_map = {'north': (-1, 0), 'south': (1, 0), 'west': (0, -1), 'east': (0, 1)}
    while queue:
        path = queue.popleft()
        r, c = path[-1]
        if (r, c) == end: return path
        for direction, (dr, dc) in dir_map.items():
            if not maze_map.has_wall(r, c, direction):
                nr, nc = r + dr, c + dc
                if 0 <= nr < maze_map.height and 0 <= nc < maze_map.width and (nr, nc) not in visited:
                    visited.add((nr, nc)); new_path = list(path); new_path.append((nr, nc)); queue.append(new_path)
    return None

def find_nearest_unvisited(maze_map, current_pos):
    unvisited_cells = [ (r,c) for r in range(maze_map.height) for c in range(maze_map.width) if not maze_map.grid[r][c].visited ]
    if not unvisited_cells: return None
    shortest_path = None
    for cell in unvisited_cells:
        path = find_path_bfs(maze_map, current_pos, cell)
        if path and (not shortest_path or len(path) < len(shortest_path)):
            shortest_path = path
    return shortest_path

def execute_backtrack_path(path, movement_controller, attitude_handler, visualizer, maze_map):
    global CURRENT_POSITION
    print(f"Executing backtrack path: {path}")
    dir_vectors = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}
    for i in range(len(path) - 1):
        visualizer.update_plot(maze_map, path[i], path[i:])
        dr, dc = path[i+1][0] - path[i][0], path[i+1][1] - path[i][1]
        target_direction = dir_vectors[(dr, dc)]
        movement_controller.rotate_to_direction(target_direction, attitude_handler)
        axis_to_monitor = 'x' if CURRENT_DIRECTION == 0 or CURRENT_DIRECTION == 2 else 'y'
        movement_controller.move_forward_one_grid(axis=axis_to_monitor, attitude_handler=attitude_handler)
        CURRENT_POSITION = path[i+1]
    visualizer.update_plot(maze_map, CURRENT_POSITION)

def perform_centering_nudge(movement_controller, scanner, initial_wall_status):
    print("--- Performing Centering Nudge ---")
    has_left_wall = initial_wall_status['left']
    has_right_wall = initial_wall_status['right']
    
    # ทำงานก็ต่อเมื่อมีกำแพงแค่ด้านเดียว
    if has_left_wall and not has_right_wall:
        print("   Left wall detected. Nudging RIGHT to check for opposite wall.")
        movement_controller.nudge_robot(y_speed=0.2, duration=0.4) # ขยับขวา
        new_wall_status, _ = scanner.get_wall_status()
        if new_wall_status['right']:
            print("   Opposite wall found. Nudging back to center.")
            movement_controller.nudge_robot(y_speed=-0.1, duration=0.4) # กลับมาซ้ายครึ่งนึง
        else:
            print("   No opposite wall. Returning to original position.")
            movement_controller.nudge_robot(y_speed=-0.2, duration=0.4) # กลับมาซ้ายเท่าเดิม
    elif not has_left_wall and has_right_wall:
        print("   Right wall detected. Nudging LEFT to check for opposite wall.")
        movement_controller.nudge_robot(y_speed=-0.2, duration=0.4) # ขยับซ้าย
        new_wall_status, _ = scanner.get_wall_status()
        if new_wall_status['left']:
            print("   Opposite wall found. Nudging back to center.")
            movement_controller.nudge_robot(y_speed=0.1, duration=0.4) # กลับมาขวาครึ่งนึง
        else:
            print("   No opposite wall. Returning to original position.")
            movement_controller.nudge_robot(y_speed=0.2, duration=0.4) # กลับมาขวาเท่าเดิม
    else:
        print("   Corridor or open space. Skipping nudge.")
        return # คืนค่าทันที ไม่ต้องทำอะไรต่อ

# =============================================================================
# ===== EXPLORATION LOGIC =====================================================
# =============================================================================
def explore_with_ogm(scanner, movement_controller, attitude_handler, maze_map, visualizer, max_steps=40):
    global CURRENT_POSITION, CURRENT_DIRECTION
    
    print("\n🚀 === STARTING AUTONOMOUS EXPLORATION WITH OGM ===")
    with open("experiment_log.txt", "w") as log_file:
        log_file.write("Step\tRobot Pos(y,x)\tIR Left\tToF Front (cm)\tIR Right\n")
        for step in range(max_steps):
            r, c = CURRENT_POSITION
            maze_map.set_visited(r, c)
            print(f"\n--- Step {step + 1} at {CURRENT_POSITION}, Facing: {['North', 'East', 'South', 'West'][CURRENT_DIRECTION]} ---")

            # 1. Initial Scan
            initial_wall_status, _ = scanner.get_wall_status()

            # 2. Perform Centering Nudge (if applicable)
            perform_centering_nudge(movement_controller, scanner, initial_wall_status)

            # 3. Final, definitive scan from the centered position
            print("--- Performing Final Scan for Mapping ---")
            final_wall_status, raw_values = scanner.get_wall_status()
            log_file.write(f"{step+1}\t({r},{c})\t\t{raw_values['left_ir']}\t{raw_values['front_cm']}\t\t{raw_values['right_ir']}\n")
            
            # 4. Update map with the most accurate data
            dir_map_rel_abs = {'front': CURRENT_DIRECTION, 'right': (CURRENT_DIRECTION + 1) % 4, 'left': (CURRENT_DIRECTION - 1 + 4) % 4}
            dir_map_idx_name = {0: 'north', 1: 'east', 2: 'south', 3: 'west'}
            for rel_dir, has_wall in final_wall_status.items():
                abs_dir_idx = dir_map_rel_abs[rel_dir]
                abs_dir_name = dir_map_idx_name[abs_dir_idx]
                maze_map.set_wall(r, c, abs_dir_name, has_wall)

            visualizer.update_plot(maze_map, CURRENT_POSITION)
            
            # 5. Decision Logic using the accurate data
            next_move_found = False
            priority_dirs = [(CURRENT_DIRECTION + 1) % 4, CURRENT_DIRECTION, (CURRENT_DIRECTION - 1 + 4) % 4]
            dir_vectors = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            for target_dir in priority_dirs:
                r_next, c_next = r + dir_vectors[target_dir][0], c + dir_vectors[target_dir][1]
                if 0 <= r_next < maze_map.height and 0 <= c_next < maze_map.width:
                    if not maze_map.has_wall(r, c, dir_map_idx_name[target_dir]) and not maze_map.grid[r_next][c_next].visited:
                        print(f"Found local unvisited path to {dir_map_idx_name[target_dir]}.")
                        movement_controller.rotate_to_direction(target_dir, attitude_handler)
                        next_move_found = True; break
            
            if not next_move_found:
                print("No local unvisited path. Searching for nearest unvisited cell...")
                path = find_nearest_unvisited(maze_map, CURRENT_POSITION)
                if path:
                    execute_backtrack_path(path, movement_controller, attitude_handler, visualizer, maze_map)
                    continue
                else:
                    print("🎉 EXPLORATION COMPLETE! No unvisited cells reachable.")
                    break
            
            axis_to_monitor = 'x' if CURRENT_DIRECTION == 0 or CURRENT_DIRECTION == 2 else 'y'
            movement_controller.move_forward_one_grid(axis=axis_to_monitor, attitude_handler=attitude_handler)
            
            CURRENT_POSITION = (r + dir_vectors[CURRENT_DIRECTION][0], c + dir_vectors[CURRENT_DIRECTION][1])
    
    print("\n🎉 === EXPLORATION FINISHED ==="); visualizer.update_plot(maze_map, CURRENT_POSITION)
    with open("final_structural_map.json", "w") as f:
        json_map = [[{'visited': cell.visited, 'walls': cell.walls} for cell in row] for row in maze_map.grid]
        json.dump(json_map, f, indent=2)
    print("✅ Final map and log file saved.")
    print("... You can close the heatmap window now ...")
    plt.ioff(); plt.show()

# =============================================================================
# ===== MAIN EXECUTION BLOCK ==================================================
# =============================================================================
if __name__ == '__main__':
    ep_robot = None; maze_map = MazeMap(width=4, height=4); attitude_handler = AttitudeHandler()
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
        
        explore_with_ogm(scanner, movement_controller, attitude_handler, maze_map, visualizer)

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