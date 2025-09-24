import time
import robomaster
from robomaster import robot
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import deque

# =============================================================================
# ===== CONFIGURATION & GLOBAL STATE ==========================================
# =============================================================================
LEFT_IR_SENSOR_ID = 3
RIGHT_IR_SENSOR_ID = 1

CURRENT_POSITION = (3, 3) # (แถว, คอลัมน์)
CURRENT_DIRECTION = 0     # 0:North, 1:East, 2:South, 3:West
CURRENT_TARGET_YAW = 0.0

# --- OGM Configuration ---
PROB_OCCUPIED_THRESHOLD = 0.80  # Prob > 0.80 is considered an obstacle
PROB_FREE_THRESHOLD = 0.20      # Prob < 0.20 is considered free space
# Values for Bayesian Update in Log-Odds space
# Corresponds to a strong belief update
L_OCC = 2.2  # log(0.9 / 0.1)
L_FREE = -1.9 # log(0.15 / 0.85)

# =============================================================================
# ===== OGM MAP & VISUALIZATION CLASSES (REFACTORED) ==========================
# =============================================================================
class MazeMap:
    """Represents the maze using a probabilistic Occupancy Grid Map."""
    def __init__(self, width, height):
        self.width = width
        self.height = height
        # Log-odds grid. 0 means 0.5 probability (unknown).
        self.log_odds_grid = np.zeros((height, width))
        self.robot_visit_count = np.zeros((height, width))

    def _log_odds_to_prob(self, l):
        return 1.0 - 1.0 / (1.0 + np.exp(l))

    def get_probability(self, r, c):
        if 0 <= r < self.height and 0 <= c < self.width:
            return self._log_odds_to_prob(self.log_odds_grid[r, c])
        return 1.0 # Boundaries are always occupied

    def update_cell(self, r, c, is_hit):
        """Updates a cell's probability using Bayesian update in log-odds."""
        if 0 <= r < self.height and 0 <= c < self.width:
            if is_hit:
                self.log_odds_grid[r, c] += L_OCC
            else:
                self.log_odds_grid[r, c] += L_FREE
            # Clamp values to prevent extreme overconfidence
            self.log_odds_grid[r, c] = np.clip(self.log_odds_grid[r, c], -10, 10)

    def is_occupied(self, r, c):
        return self.get_probability(r, c) > PROB_OCCUPIED_THRESHOLD

    def is_free(self, r, c):
        return self.get_probability(r, c) < PROB_FREE_THRESHOLD

    def is_unexplored(self, r, c):
        prob = self.get_probability(r, c)
        return PROB_FREE_THRESHOLD <= prob <= PROB_OCCUPIED_THRESHOLD

class RealTimeVisualizer:
    """Handles the real-time Matplotlib heatmap for the OGM."""
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.heatmap_data = np.zeros((self.grid_size, self.grid_size))
        self.UNEXPLORED, self.FREE, self.OCCUPIED, self.ROBOT, self.PATH = 0, 1, 2, 3, 4
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.cmap = ListedColormap(['#A0A0A0', '#FFFFFF', '#000000', '#0000FF', '#FFFF00'])
        self.image = self.ax.imshow(self.heatmap_data, cmap=self.cmap, vmin=0, vmax=4)
        self.ax.set_xticks(np.arange(-.5, self.grid_size, 1), minor=True)
        self.ax.set_yticks(np.arange(-.5, self.grid_size, 1), minor=True)
        self.ax.grid(which="minor", color="k", linestyle='-', linewidth=1)
        self.ax.tick_params(which="minor", size=0)
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.ax.set_title("Real-time Occupancy Grid Map"); self.fig.tight_layout(); plt.show()

    def update_plot(self, maze_map, robot_pos, path=None):
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if maze_map.is_occupied(r, c): self.heatmap_data[r, c] = self.OCCUPIED
                elif maze_map.is_free(r, c): self.heatmap_data[r, c] = self.FREE
                else: self.heatmap_data[r, c] = self.UNEXPLORED
        if path:
            for r, c in path:
                self.heatmap_data[r, c] = self.PATH
        if robot_pos:
            r, c = robot_pos
            self.heatmap_data[r, c] = self.ROBOT
        self.image.set_data(self.heatmap_data)
        self.fig.canvas.draw(); self.fig.canvas.flush_events(); plt.pause(0.01)

# =============================================================================
# ===== CORE ROBOT CONTROL CLASSES (UNCHANGED) ================================
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
        distance_pid = PID(Kp=self.KP, Ki=self.KI, Kd=self.KD, setpoint=target_distance)
        yaw_pid = PID(Kp=1.2, Ki=0.1, Kd=0.8, setpoint=CURRENT_TARGET_YAW)
        start_time, last_time = time.time(), time.time()
        start_position = self.current_x_pos if axis == 'x' else self.current_y_pos
        print(f"🚀 Moving FORWARD 0.6m with YAW STABILIZATION, monitoring GLOBAL AXIS '{axis}'")
        while time.time() - start_time < self.MOVE_TIMEOUT:
            now = time.time(); dt = now - last_time
            if dt == 0: continue
            last_time = now
            current_position = self.current_x_pos if axis == 'x' else self.current_y_pos
            relative_position = abs(current_position - start_position)
            if abs(relative_position - target_distance) < 0.03: print("✅ Move complete!"); break
            forward_output = distance_pid.compute(relative_position, dt)
            ramp_multiplier = min(1.0, 0.1 + ((now - start_time) / self.RAMP_UP_TIME) * 0.9)
            forward_speed = max(-1.0, min(1.0, forward_output * ramp_multiplier))
            current_yaw = attitude_handler.current_yaw
            yaw_error = attitude_handler.normalize_angle(CURRENT_TARGET_YAW - current_yaw)
            z_speed_correction = yaw_pid.compute(-yaw_error, dt)
            z_speed_correction = max(-60, min(60, z_speed_correction))
            self.chassis.drive_speed(x=forward_speed, y=0, z=z_speed_correction, timeout=1)
        self.chassis.drive_speed(x=0, y=0, z=0, timeout=0.1); time.sleep(0.5)
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
# ===== SENSOR HANDLER (ADAPTED FOR OGM) ======================================
# =============================================================================
class EnvironmentScanner:
    def __init__(self, sensor_adaptor, tof_sensor, gimbal):
        self.sensor_adaptor, self.tof_sensor, self.gimbal = sensor_adaptor, tof_sensor, gimbal
        self.tof_wall_threshold_cm = 50.0; self.last_tof_distance_mm = 0
        self.tof_sensor.sub_distance(freq=5, callback=self._tof_data_handler)
        print(" SENSOR: ToF distance stream started.")
    def _tof_data_handler(self, sub_info): self.last_tof_distance_mm = sub_info[0]
    
    def scan_surroundings(self, robot_pos, robot_dir):
        """
        Scans surroundings and returns a list of (coordinates, is_hit) tuples.
        This decouples sensor logic from map updating.
        """
        self.gimbal.moveto(pitch=0, yaw=0).wait_for_completed()
        time.sleep(0.1)
        
        r, c = robot_pos
        dir_vectors = [(-1, 0), (0, 1), (1, 0), (0, -1)] # N, E, S, W
        
        # Relative directions: 0=front, 1=right, 2=back, 3=left
        front_dir = robot_dir
        right_dir = (robot_dir + 1) % 4
        left_dir = (robot_dir - 1 + 4) % 4
        
        # Map relative directions to absolute cell coordinates
        cells_to_scan = {
            'front': (r + dir_vectors[front_dir][0], c + dir_vectors[front_dir][1]),
            'right': (r + dir_vectors[right_dir][0], c + dir_vectors[right_dir][1]),
            'left': (r + dir_vectors[left_dir][0], c + dir_vectors[left_dir][1])
        }
        
        # Get raw sensor values
        tof_dist_cm = self.last_tof_distance_mm / 10.0
        front_hit = tof_dist_cm < self.tof_wall_threshold_cm and self.last_tof_distance_mm > 0
        
        left_val = self.sensor_adaptor.get_io(id=LEFT_IR_SENSOR_ID)
        right_val = self.sensor_adaptor.get_io(id=RIGHT_IR_SENSOR_ID)
        left_hit = (left_val == 0)
        right_hit = (right_val == 0)
        
        print(f"[SCAN] Front (ToF): {tof_dist_cm:.1f}cm -> {'HIT' if front_hit else 'MISS'}")
        print(f"[SCAN] Left (IR): val={left_val} -> {'HIT' if left_hit else 'MISS'}")
        print(f"[SCAN] Right (IR): val={right_val} -> {'HIT' if right_hit else 'MISS'}")
        
        # Return list of observations: [( (r,c), is_hit ), ...]
        return [
            (cells_to_scan['front'], front_hit),
            (cells_to_scan['left'], left_hit),
            (cells_to_scan['right'], right_hit)
        ]

    def cleanup(self):
        try: self.tof_sensor.unsub_distance(); print(" SENSOR: ToF distance stream stopped.")
        except Exception: pass

# =============================================================================
# ===== PATHFINDING & BACKTRACKING LOGIC (ADAPTED FOR OGM) ====================
# =============================================================================
def find_path_bfs(maze_map, start, end):
    """Finds the shortest path in the OGM from start to end."""
    queue = deque([[start]])
    visited = {start}
    dir_vectors = [(-1, 0), (0, 1), (1, 0), (0, -1)] # N, E, S, W

    while queue:
        path = queue.popleft()
        r, c = path[-1]
        if (r, c) == end:
            return path
            
        for dr, dc in dir_vectors:
            nr, nc = r + dr, c + dc
            # Pathfind through cells that are considered FREE
            if not maze_map.is_occupied(nr, nc) and (nr, nc) not in visited:
                visited.add((nr, nc))
                new_path = list(path)
                new_path.append((nr, nc))
                queue.append(new_path)
    return None

def find_nearest_unexplored(maze_map, current_pos):
    """Finds the nearest cell that is still considered unexplored."""
    unexplored_cells = []
    for r in range(maze_map.height):
        for c in range(maze_map.width):
            if maze_map.is_unexplored(r, c):
                # Prioritize cells with fewer robot visits
                unexplored_cells.append(((r, c), maze_map.robot_visit_count[r,c]))
    
    if not unexplored_cells:
        return None
        
    # Sort by visit count first, then find path
    unexplored_cells.sort(key=lambda x: x[1])

    shortest_path = None
    # Only check the top N most promising cells to save computation
    for (cell, visits) in unexplored_cells[:10]:
        path = find_path_bfs(maze_map, current_pos, cell)
        if path:
            if not shortest_path or len(path) < len(shortest_path):
                shortest_path = path
    return shortest_path

def execute_path(path, movement_controller, attitude_handler, visualizer, maze_map):
    global CURRENT_POSITION
    print(f"Executing path: {path}")
    dir_vectors = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}
    for i in range(len(path) - 1):
        visualizer.update_plot(maze_map, path[i], path[i:])
        dr, dc = path[i+1][0] - path[i][0], path[i+1][1] - path[i][1]
        target_direction = dir_vectors[(dr, dc)]
        movement_controller.rotate_to_direction(target_direction, attitude_handler)
        axis_to_monitor = 'x' if CURRENT_DIRECTION % 2 == 0 else 'y'
        movement_controller.move_forward_one_grid(axis=axis_to_monitor, attitude_handler=attitude_handler)
        CURRENT_POSITION = path[i+1]
        maze_map.robot_visit_count[CURRENT_POSITION] += 1
    visualizer.update_plot(maze_map, CURRENT_POSITION)

# =============================================================================
# ===== EXPLORATION LOGIC (REFACTORED FOR OGM) ================================
# =============================================================================
def explore_with_ogm(scanner, movement_controller, attitude_handler, maze_map, visualizer, max_steps=40):
    global CURRENT_POSITION
    
    print("\n🚀 === STARTING AUTONOMOUS EXPLORATION WITH BAYESIAN OGM ===")
    
    for step in range(max_steps):
        r, c = CURRENT_POSITION
        maze_map.robot_visit_count[r,c] += 1 # Mark current cell as visited
        
        print(f"\n--- Step {step + 1} at {CURRENT_POSITION}, Facing: {['North', 'East', 'South', 'West'][CURRENT_DIRECTION]} ---")
        
        # 1. Scan surroundings and update the map with new information
        observations = scanner.scan_surroundings(CURRENT_POSITION, CURRENT_DIRECTION)
        for (cell_coords, is_hit) in observations:
            maze_map.update_cell(cell_coords[0], cell_coords[1], is_hit)
        
        # Update free space in front of robot (raycasting)
        if not observations[0][1]: # If front is not a hit
             maze_map.update_cell(r, c, is_hit=False)

        visualizer.update_plot(maze_map, CURRENT_POSITION)
        
        # 2. Decide next move: find the nearest unexplored cell
        print("Searching for nearest unexplored cell...")
        path_to_explore = find_nearest_unexplored(maze_map, CURRENT_POSITION)
        
        # 3. Execute the plan
        if path_to_explore:
            print(f"Found path to unexplored cell {path_to_explore[-1]}.")
            execute_path(path_to_explore, movement_controller, attitude_handler, visualizer, maze_map)
        else:
            print("🎉 EXPLORATION COMPLETE! No unexplored cells reachable.")
            break
            
    print("\n🎉 === EXPLORATION FINISHED ===")
    visualizer.update_plot(maze_map, CURRENT_POSITION)
    
    # Save the final probability map
    with open("final_probability_map.json", "w") as f:
        prob_map = [[maze_map.get_probability(r, c) for c in range(maze_map.width)] for r in range(maze_map.height)]
        json.dump(prob_map, f, indent=2)
    print("✅ Final probability map saved.")
    print("... You can close the heatmap window now ...")
    plt.ioff(); plt.show()

# =============================================================================
# ===== MAIN EXECUTION BLOCK (UNCHANGED) ======================================
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