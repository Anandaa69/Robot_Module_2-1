# -*-coding:utf-8-*-

import time
import robomaster
from robomaster import robot
import numpy as np
import math
import json
import matplotlib.pyplot as plt
from collections import deque

# =============================================================================
# ===== CONFIGURATION & BAYESIAN PARAMETERS ===================================
# =============================================================================
LEFT_IR_SENSOR_ID = 3
RIGHT_IR_SENSOR_ID = 1

# --- Logical state for the grid map ---
CURRENT_POSITION = (3, 3) # (แถว, คอลัมน์)
CURRENT_DIRECTION = 0     # 0:North, 1:East, 2:South, 3:West
TARGET_DESTINATION = (3, 0)

# --- Physical state for the robot ---
CURRENT_TARGET_YAW = 0.0
ROBOT_FACE = 1 # 1,3,5.. = X axis, 2,4,6.. = Y axis

# --- Occupancy Grid Parameters ---
PROB_OCC_GIVEN_OCC = {'tof': 0.95, 'ir': 0.85}
PROB_OCC_GIVEN_FREE = {'tof': 0.05, 'ir': 0.15}

LOG_ODDS_OCC = {
    'tof': math.log(PROB_OCC_GIVEN_OCC['tof'] / (1 - PROB_OCC_GIVEN_OCC['tof'])),
    'ir': math.log(PROB_OCC_GIVEN_OCC['ir'] / (1 - PROB_OCC_GIVEN_OCC['ir']))
}
LOG_ODDS_FREE = {
    'tof': math.log(PROB_OCC_GIVEN_FREE['tof'] / (1 - PROB_OCC_GIVEN_FREE['tof'])),
    'ir': math.log(PROB_OCC_GIVEN_FREE['ir'] / (1 - PROB_OCC_GIVEN_FREE['ir']))
}

# --- Decision Thresholds ---
OCCUPANCY_THRESHOLD = 0.7
FREE_THRESHOLD = 0.3

# =============================================================================
# ===== OCCUPANCY GRID MAP & VISUALIZATION ====================================
# =============================================================================
class OGMCell:
    def __init__(self):
        self.log_odds = 0.0
    def get_probability(self):
        return 1.0 - 1.0 / (1.0 + math.exp(self.log_odds))

class OccupancyGridMap:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[OGMCell() for _ in range(width)] for _ in range(height)]
    
    def update_cell(self, r, c, is_occupied_reading, sensor_type='ir'):
        if 0 <= r < self.height and 0 <= c < self.width:
            if is_occupied_reading:
                self.grid[r][c].log_odds += LOG_ODDS_OCC[sensor_type]
            else:
                self.grid[r][c].log_odds += LOG_ODDS_FREE[sensor_type]
            self.grid[r][c].log_odds = max(min(self.grid[r][c].log_odds, 10), -10)

    def get_probability(self, r, c):
        if 0 <= r < self.height and 0 <= c < self.width:
            return self.grid[r][c].get_probability()
        return 1.0

    def is_occupied(self, r, c):
        return self.get_probability(r, c) > OCCUPANCY_THRESHOLD

    def is_free(self, r, c):
        return self.get_probability(r, c) < FREE_THRESHOLD

class RealTimeVisualizer:
    def __init__(self, grid_size, target_dest=None):
        self.grid_size = grid_size
        self.target_dest = target_dest
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.colors = {"robot": "#0000FF", "target": "#FFD700", "path": "#FFFF00"}
    
    def update_plot(self, occupancy_map, robot_pos, path=None):
        self.ax.clear()
        self.ax.set_title("Real-time Occupancy Grid Map")
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.ax.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax.set_ylim(self.grid_size - 0.5, -0.5)

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                prob = occupancy_map.get_probability(r, c)
                if prob > OCCUPANCY_THRESHOLD: color = '#8B0000'
                elif prob < FREE_THRESHOLD: color = '#D3D3D3'
                else: color = '#90EE90'
                
                self.ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=color, edgecolor='k', lw=0.5))
                self.ax.text(c, r, f"{prob:.2f}", ha="center", va="center", color="black", fontsize=8)

        if self.target_dest:
             r, c = self.target_dest
             self.ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=self.colors['target'], edgecolor='k', lw=2, alpha=0.8))

        if path:
            for r_p, c_p in path:
                 self.ax.add_patch(plt.Rectangle((c_p - 0.5, r_p - 0.5), 1, 1, facecolor=self.colors['path'], edgecolor='k', lw=0.5, alpha=0.7))
        
        if robot_pos:
            r_r, c_r = robot_pos
            self.ax.add_patch(plt.Rectangle((c_r - 0.5, r_r - 0.5), 1, 1, facecolor=self.colors['robot'], edgecolor='k', lw=2))

        legend_elements = [ plt.Rectangle((0,0),1,1, facecolor='#8B0000', label=f'Occupied (P>{OCCUPANCY_THRESHOLD})'), plt.Rectangle((0,0),1,1, facecolor='#90EE90', label=f'Unknown'), plt.Rectangle((0,0),1,1, facecolor='#D3D3D3', label=f'Free (P<{FREE_THRESHOLD})'), plt.Rectangle((0,0),1,1, facecolor=self.colors['robot'], label='Robot'), plt.Rectangle((0,0),1,1, facecolor=self.colors['target'], label='Target') ]
        self.ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.45, 1.0))
        self.fig.tight_layout(rect=[0, 0, 0.8, 1])
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
        # ... (โค้ดส่วนนี้เหมือนเดิม) ...
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
        self.chassis.sub_position(freq=20, callback=self.position_handler)
    def position_handler(self, position_info):
        self.current_x_pos, self.current_y_pos = position_info[0], position_info[1]
    
    def move_forward_one_grid(self, axis, attitude_handler):
        attitude_handler.correct_yaw_to_target(self.chassis, CURRENT_TARGET_YAW)
        target_distance = 0.6
        pid = PID(Kp=1.8, Ki=0.25, Kd=12, setpoint=target_distance)
        start_time, last_time = time.time(), time.time()
        start_position = self.current_x_pos if axis == 'x' else self.current_y_pos
        print(f"🚀 Moving FORWARD 0.6m, monitoring GLOBAL AXIS '{axis}'")
        while time.time() - start_time < 2.5:
            now = time.time(); dt = now - last_time; last_time = now
            current_position = self.current_x_pos if axis == 'x' else self.current_y_pos
            relative_position = abs(current_position - start_position)
            if abs(relative_position - target_distance) < 0.03: print("✅ Move complete!"); break
            output = pid.compute(relative_position, dt)
            ramp_multiplier = min(1.0, 0.1 + ((now - start_time) / 1.0) * 0.9)
            speed = max(-1.0, min(1.0, output * ramp_multiplier))
            self.chassis.drive_speed(x=speed, y=0, z=0, timeout=1)
        self.chassis.drive_speed(x=0, y=0, z=0, timeout=0.1); time.sleep(0.5)

    # ========== START: โค้ดสไลด์ที่นำกลับมา ==========
    def _calculate_yaw_correction(self, attitude_handler, target_yaw):
        KP_YAW = 0.8; MAX_YAW_SPEED = 30
        yaw_error = attitude_handler.normalize_angle(target_yaw - attitude_handler.current_yaw)
        rotation_speed_z = KP_YAW * yaw_error
        return max(min(rotation_speed_z, MAX_YAW_SPEED), -MAX_YAW_SPEED)

    def slide_with_yaw_lock(self, y_speed, duration, attitude_handler):
        global CURRENT_TARGET_YAW
        print(f"🔩 Sliding with Yaw Lock: y_speed={y_speed:.2f}, duration={duration}s")
        target_yaw = CURRENT_TARGET_YAW
        start_time = time.time()
        while time.time() - start_time < duration:
            rotation_speed_z = self._calculate_yaw_correction(attitude_handler, target_yaw)
            self.chassis.drive_speed(x=0, y=y_speed, z=rotation_speed_z)
            time.sleep(0.02)
        print("\n   Slide complete.")
        self.chassis.drive_speed(x=0, y=0, z=0)
        time.sleep(0.2)
        attitude_handler.correct_yaw_to_target(self.chassis, target_yaw)
    # ========== END: โค้ดสไลด์ที่นำกลับมา ==========
    
    def rotate_to_direction(self, target_direction, attitude_handler):
        # ... (โค้ดส่วนนี้เหมือนเดิม) ...
        global CURRENT_DIRECTION
        if CURRENT_DIRECTION == target_direction: return
        diff = (target_direction - CURRENT_DIRECTION + 4) % 4
        if diff == 1: self.rotate_90_degrees_right(attitude_handler)
        elif diff == 3: self.rotate_90_degrees_left(attitude_handler)
        elif diff == 2: self.rotate_90_degrees_right(attitude_handler); self.rotate_90_degrees_right(attitude_handler)

    def rotate_90_degrees_right(self, attitude_handler):
        global CURRENT_TARGET_YAW, CURRENT_DIRECTION, ROBOT_FACE
        print("🔄 Rotating 90° RIGHT...")
        CURRENT_TARGET_YAW = attitude_handler.normalize_angle(CURRENT_TARGET_YAW + 90)
        attitude_handler.correct_yaw_to_target(self.chassis, CURRENT_TARGET_YAW)
        CURRENT_DIRECTION = (CURRENT_DIRECTION + 1) % 4; ROBOT_FACE += 1
    def rotate_90_degrees_left(self, attitude_handler):
        global CURRENT_TARGET_YAW, CURRENT_DIRECTION, ROBOT_FACE
        print("🔄 Rotating 90° LEFT...")
        CURRENT_TARGET_YAW = attitude_handler.normalize_angle(CURRENT_TARGET_YAW - 90)
        attitude_handler.correct_yaw_to_target(self.chassis, CURRENT_TARGET_YAW)
        CURRENT_DIRECTION = (CURRENT_DIRECTION - 1 + 4) % 4; ROBOT_FACE -= 1
        if ROBOT_FACE < 1: ROBOT_FACE += 4
    def cleanup(self):
        try: self.chassis.unsub_position()
        except Exception: pass

class EnvironmentScanner:
    def __init__(self, sensor_adaptor, tof_sensor, gimbal, chassis):
        self.sensor_adaptor, self.tof_sensor, self.gimbal, self.chassis = sensor_adaptor, tof_sensor, gimbal, chassis
        self.tof_wall_threshold_cm = 50.0; self.last_tof_distance_mm = 0
        self.tof_sensor.sub_distance(freq=5, callback=self._tof_data_handler)
    def _tof_data_handler(self, sub_info): self.last_tof_distance_mm = sub_info[0]
    
    def get_sensor_readings(self, quiet=False):
        self.chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0); time.sleep(0.2)
        self.gimbal.moveto(pitch=0, yaw=0).wait_for_completed(); time.sleep(0.2)
        
        readings = {}
        tof_distance_cm = self.last_tof_distance_mm / 10.0
        readings['front'] = (tof_distance_cm < self.tof_wall_threshold_cm and self.last_tof_distance_mm > 0)
        if not quiet: print(f"[SCAN] Front (ToF): {tof_distance_cm:.1f}cm -> {'OCCUPIED' if readings['front'] else 'FREE'}")
        
        left_val = self.sensor_adaptor.get_io(id=LEFT_IR_SENSOR_ID)
        right_val = self.sensor_adaptor.get_io(id=RIGHT_IR_SENSOR_ID)
        readings['left'] = (left_val == 0)
        readings['right'] = (right_val == 0)
        if not quiet: print(f"[SCAN] Left (IR): val={left_val} -> {'OCCUPIED' if readings['left'] else 'FREE'}")
        if not quiet: print(f"[SCAN] Right (IR): val={right_val} -> {'OCCUPIED' if readings['right'] else 'FREE'}")
        
        return readings

    def get_front_tof_cm(self):
        time.sleep(0.2) 
        return self.last_tof_distance_mm / 10.0 if self.last_tof_distance_mm > 0 else 999.0
    def cleanup(self):
        try: self.tof_sensor.unsub_distance()
        except Exception: pass

# =============================================================================
# ===== PATHFINDING & EXPLORATION LOGIC =======================================
# =============================================================================
def find_path_bfs(occupancy_map, start, end):
    # ... (โค้ดส่วนนี้เหมือนเดิม) ...
    queue = deque([[start]]); visited = {start}
    moves = [(-1, 0), (0, 1), (1, 0), (0, -1)] 
    while queue:
        path = queue.popleft()
        r, c = path[-1]
        if (r, c) == end: return path
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if not occupancy_map.is_occupied(nr, nc) and (nr, nc) not in visited:
                visited.add((nr, nc))
                new_path = list(path)
                new_path.append((nr, nc))
                queue.append(new_path)
    return None

def execute_path(path, movement_controller, attitude_handler, visualizer, occupancy_map):
    # ... (โค้ดส่วนนี้เหมือนเดิม) ...
    pass # Implementation is fine

# ========== START: ฟังก์ชัน Nudge ที่นำกลับมา ==========
def perform_centering_nudge(movement_controller, scanner, initial_readings, attitude_handler):
    print("--- Performing Centering Nudge ---")
    has_left_wall, has_right_wall = initial_readings['left'], initial_readings['right']
    nudge_dist, nudge_dur = 0.1, 0.25
    y_speed = nudge_dist / nudge_dur
    
    if has_left_wall and not has_right_wall:
        print("   [Nudge] Left wall detected. Nudging RIGHT to check for opposite wall.")
        movement_controller.slide_with_yaw_lock(y_speed=y_speed, duration=nudge_dur, attitude_handler=attitude_handler)
        # Check if right wall appears after nudge
        readings_after, _ = scanner.get_sensor_readings(quiet=True), scanner.get_sensor_readings(quiet=True)
        if readings_after['right']:
            print("   -> Found opposite wall. Centering.")
            movement_controller.slide_with_yaw_lock(y_speed=-(y_speed/2), duration=nudge_dur/2, attitude_handler=attitude_handler)
        else:
            print("   -> No opposite wall. Returning to safe offset.")
            movement_controller.slide_with_yaw_lock(y_speed=-(y_speed*0.9), duration=nudge_dur*0.9, attitude_handler=attitude_handler)
    elif not has_left_wall and has_right_wall:
        print("   [Nudge] Right wall detected. Nudging LEFT to check.")
        movement_controller.slide_with_yaw_lock(y_speed=-y_speed, duration=nudge_dur, attitude_handler=attitude_handler)
        readings_after, _ = scanner.get_sensor_readings(quiet=True), scanner.get_sensor_readings(quiet=True)
        if readings_after['left']:
            print("   -> Found opposite wall. Centering.")
            movement_controller.slide_with_yaw_lock(y_speed=(y_speed/2), duration=nudge_dur/2, attitude_handler=attitude_handler)
        else:
            print("   -> No opposite wall. Returning to safe offset.")
            movement_controller.slide_with_yaw_lock(y_speed=(y_speed*0.9), duration=nudge_dur*0.9, attitude_handler=attitude_handler)
    else:
        print("   [Nudge] In open space or corridor. Skipping nudge.")
# ========== END: ฟังก์ชัน Nudge ที่นำกลับมา ==========


def explore_with_ogm(scanner, movement_controller, attitude_handler, occupancy_map, visualizer, max_steps=40):
    global CURRENT_POSITION, CURRENT_DIRECTION
    visited_cells = set()
    
    for step in range(max_steps):
        r, c = CURRENT_POSITION
        
        print(f"\n--- Step {step + 1} at {CURRENT_POSITION}, Facing: {['N', 'E', 'S', 'W'][CURRENT_DIRECTION]} ---")
        
        # ========== START: LOGIC ที่เพิ่มเข้ามาใหม่ ==========
        # 1. Perform Centering Nudge before official scan
        initial_readings_for_nudge = scanner.get_sensor_readings(quiet=True)
        perform_centering_nudge(movement_controller, scanner, initial_readings_for_nudge, attitude_handler)
        # ========== END: LOGIC ที่เพิ่มเข้ามาใหม่ ==========

        # 2. Scan environment and update map
        print("--- Performing Official Scan for Mapping ---")
        sensor_readings = scanner.get_sensor_readings()
        
        dir_map_rel_abs = {'front': CURRENT_DIRECTION, 'left': (CURRENT_DIRECTION - 1 + 4) % 4, 'right': (CURRENT_DIRECTION + 1) % 4}
        dir_vectors = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        front_dr, front_dc = dir_vectors[dir_map_rel_abs['front']]
        occupancy_map.update_cell(r + front_dr, c + front_dc, sensor_readings['front'], 'tof')
        left_dr, left_dc = dir_vectors[dir_map_rel_abs['left']]
        occupancy_map.update_cell(r + left_dr, c + left_dc, sensor_readings['left'], 'ir')
        right_dr, right_dc = dir_vectors[dir_map_rel_abs['right']]
        occupancy_map.update_cell(r + right_dr, c + right_dc, sensor_readings['right'], 'ir')
        
        # Mark current cell as free (since the robot is in it)
        occupancy_map.update_cell(r, c, False, 'tof') 
        visited_cells.add((r, c))
        visualizer.update_plot(occupancy_map, CURRENT_POSITION)

        # 3. Decide next move
        priority_dirs = [(CURRENT_DIRECTION + 1) % 4, CURRENT_DIRECTION, (CURRENT_DIRECTION - 1 + 4) % 4]
        moved = False
        for target_dir in priority_dirs:
            target_r, target_c = r + dir_vectors[target_dir][0], c + dir_vectors[target_dir][1]
            
            if not occupancy_map.is_occupied(target_r, target_c) and (target_r, target_c) not in visited_cells:
                print(f"Path to {['N','E','S','W'][target_dir]} at ({target_r},{target_c}) seems clear. Attempting move.")
                
                movement_controller.rotate_to_direction(target_dir, attitude_handler)
                
                print("    Confirming path forward with ToF...")
                is_blocked = scanner.get_front_tof_cm() < scanner.tof_wall_threshold_cm
                occupancy_map.update_cell(target_r, target_c, is_blocked, 'tof')
                print(f"    ToF confirmation: Path is {'BLOCKED' if is_blocked else 'CLEAR'}.")
                visualizer.update_plot(occupancy_map, CURRENT_POSITION)

                if not occupancy_map.is_occupied(target_r, target_c):
                    axis_to_monitor = 'x' if ROBOT_FACE % 2 != 0 else 'y'
                    movement_controller.move_forward_one_grid(axis=axis_to_monitor, attitude_handler=attitude_handler)
                    CURRENT_POSITION = (target_r, target_c)
                    moved = True
                    break
                else:
                    print(f"    Confirmation failed. Path to {['N','E','S','W'][target_dir]} is blocked. Re-evaluating.")
        
        if not moved:
            print("No immediate unvisited path. Turning around.")
            movement_controller.rotate_to_direction((CURRENT_DIRECTION + 2) % 4, attitude_handler)
    
    print("\n🎉 === EXPLORATION PHASE FINISHED ===")

# =============================================================================
# ===== MAIN EXECUTION BLOCK ==================================================
# =============================================================================
if __name__ == '__main__':
    # ... (โค้ดส่วนนี้เหมือนเดิม) ...
    ep_robot = None
    occupancy_map = OccupancyGridMap(width=4, height=4)
    attitude_handler = AttitudeHandler()
    movement_controller = None
    scanner = None
    ep_chassis = None
    
    try:
        visualizer = RealTimeVisualizer(grid_size=4, target_dest=TARGET_DESTINATION)
        print("🤖 Connecting to robot..."); ep_robot = robot.Robot(); ep_robot.initialize(conn_type="ap")
        ep_chassis, ep_gimbal = ep_robot.chassis, ep_robot.gimbal
        ep_tof_sensor, ep_sensor_adaptor = ep_robot.sensor, ep_robot.sensor_adaptor
        
        print(" GIMBAL: Centering gimbal..."); ep_gimbal.recenter().wait_for_completed()
        
        scanner = EnvironmentScanner(ep_sensor_adaptor, ep_tof_sensor, ep_gimbal, ep_chassis)
        movement_controller = MovementController(ep_chassis)
        attitude_handler.start_monitoring(ep_chassis)
        
        explore_with_ogm(scanner, movement_controller, attitude_handler, occupancy_map, visualizer)
        
    except KeyboardInterrupt: print("\n⚠️ User interrupted exploration.")
    except Exception as e: print(f"\n⚌ An error occurred: {e}"); import traceback; traceback.print_exc()
    finally:
        if ep_robot:
            print("🔌 Cleaning up and closing connection...")
            if scanner: scanner.cleanup()
            if attitude_handler and attitude_handler.is_monitoring: attitude_handler.stop_monitoring(ep_chassis)
            if movement_controller: movement_controller.cleanup()
            ep_robot.close()
            print("🔌 Connection closed.")
        
        final_map_data = [[cell.get_probability() for cell in row] for row in occupancy_map.grid]
        with open("final_occupancy_map.json", "w") as f:
            json.dump(final_map_data, f, indent=2)
        print("✅ Final Occupancy Grid Map saved to final_occupancy_map.json")
        print("... You can close the plot window now ...")
        plt.ioff()
        plt.show()