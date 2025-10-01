# -*-coding:utf-8-*-

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

# --- Logical state for the grid map ---
CURRENT_POSITION = (3, 3) # (แถว, คอลัมน์)
CURRENT_DIRECTION = 0     # 0:North, 1:East, 2:South, 3:West

# --- TARGET DESTINATION (เพิ่มใหม่) ---
TARGET_DESTINATION = (3, 0)  # พิกัดที่ต้องการไปหยุดหลังจากสำรวจเสร็จ (แถว, คอลัมน์)

# --- Physical state for the robot (from slide_kak.py) ---
CURRENT_TARGET_YAW = 0.0
ROBOT_FACE = 1 # ใช้อ้างอิงแกนการเคลื่อนที่ (1,3,5.. = X axis), (2,4,6.. = Y axis)

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
            if not has_wall and self.grid[r][c].walls[direction]:
                return
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
    def __init__(self, grid_size, target_dest=None):
        self.grid_size = grid_size
        self.target_dest = target_dest
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.colors = {"visited": "#D0D0FF", "unvisited": "#A0A0A0", "robot": "#0000FF", 
                      "path": "#FFFF00", "target": "#FF0000"}
    def update_plot(self, maze_map, robot_pos, path=None):
        self.ax.clear()
        self.ax.set_title("Real-time Structural Map")
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.ax.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax.set_ylim(self.grid_size - 0.5, -0.5)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                color = self.colors['visited'] if maze_map.grid[r][c].visited else self.colors['unvisited']
                if self.target_dest and (r, c) == self.target_dest: color = self.colors['target']
                self.ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=color, edgecolor='k', lw=0.5))
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
        legend_elements = [ plt.Rectangle((0,0),1,1, facecolor=self.colors['robot'], label='Robot'), plt.Rectangle((0,0),1,1, facecolor=self.colors['target'], label='Target'), plt.Rectangle((0,0),1,1, facecolor=self.colors['visited'], label='Visited'), plt.Rectangle((0,0),1,1, facecolor=self.colors['unvisited'], label='Unvisited'), plt.Rectangle((0,0),1,1, facecolor=self.colors['path'], label='Path') ]
        self.ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
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

    def move_precise_distance(self, distance_m, axis, attitude_handler):
        """
        เคลื่อนที่ไปข้างหน้าหรือถอยหลังเป็นระยะทางสั้นๆ อย่างแม่นยำ
        distance_m: ระยะทางเป็นเมตร (ค่าลบหมายถึงถอยหลัง)
        """
        print(f"🔩 Moving precise distance: {distance_m * 100:.1f} cm on axis '{axis}'")
        Kp = 2.0; max_speed = 0.25; timeout = 3.0
        start_time = time.time()
        start_position = self.current_x_pos if axis == 'x' else self.current_y_pos
        target_relative_dist = abs(distance_m)
        while time.time() - start_time < timeout:
            current_position = self.current_x_pos if axis == 'x' else self.current_y_pos
            relative_dist = abs(current_position - start_position)
            if relative_dist >= target_relative_dist: print("✅ Precise move complete."); break
            error = target_relative_dist - relative_dist
            speed = Kp * error
            drive_speed = np.sign(distance_m) * min(max_speed, speed)
            self.chassis.drive_speed(x=drive_speed, y=0, z=0, timeout=0.2)
            time.sleep(0.05)
        self.chassis.drive_speed(x=0, y=0, z=0, timeout=0.1); time.sleep(0.3)
        attitude_handler.correct_yaw_to_target(self.chassis, CURRENT_TARGET_YAW)

    # ========== START: โค้ดที่แก้ไข/เพิ่มใหม่ ==========
    def _calculate_yaw_correction(self, attitude_handler, target_yaw):
        """คำนวณความเร็วในการหมุนเพื่อรักษาองศา Yaw ให้คงที่"""
        KP_YAW = 0.8  # ค่า P-gain สำหรับควบคุม Yaw
        MAX_YAW_SPEED = 20 # ความเร็วการหมุนสูงสุด

        yaw_error = attitude_handler.normalize_angle(target_yaw - attitude_handler.current_yaw)
        rotation_speed_z = KP_YAW * yaw_error
        
        # จำกัดความเร็วสูงสุด
        return max(min(rotation_speed_z, MAX_YAW_SPEED), -MAX_YAW_SPEED)

    def slide_with_yaw_lock(self, y_speed, duration, attitude_handler):
        """
        สไลด์ไปด้านข้าง (แกน y) พร้อมกับล็อกองศา (Yaw) ให้นิ่ง
        y_speed: ความเร็วในการสไลด์ (ค่าบวกคือขวา, ค่าลบคือซ้าย)
        duration: ระยะเวลาที่ต้องการให้สไลด์ (วินาที)
        attitude_handler: อ็อบเจกต์สำหรับจัดการเรื่ององศา
        """
        global CURRENT_TARGET_YAW
        print(f"🔩 Sliding with Yaw Lock: y_speed={y_speed:.2f}, duration={duration}s")
        
        target_yaw = CURRENT_TARGET_YAW
        print(f"   Locking target Yaw at: {target_yaw:.2f} degrees")

        start_time = time.time()
        while time.time() - start_time < duration:
            # คำนวณค่าการปรับแก้ Yaw ในทุกๆ รอบของลูป
            rotation_speed_z = self._calculate_yaw_correction(attitude_handler, target_yaw)
            
            # ส่งคำสั่งเคลื่อนที่โดยมี x=0, y=y_speed, และ z=ค่าที่คำนวณได้
            self.chassis.drive_speed(x=0, y=y_speed, z=rotation_speed_z)
            
            # พิมพ์สถานะเพื่อ debug (สามารถเอาออกได้)
            print(f"Sliding... YawErr: {target_yaw - attitude_handler.current_yaw:5.2f}° | Y_Spd: {y_speed:4.2f} | Z_Spd: {rotation_speed_z:5.1f}", end='\r')
            time.sleep(0.02) # หน่วงเวลาเล็กน้อยเพื่อให้ลูปทำงานที่ความถี่ประมาณ 50Hz

        # หยุดการเคลื่อนที่หลังจากครบกำหนดเวลา
        print("\n   Slide complete.                                 ")
        self.chassis.drive_speed(x=0, y=0, z=0)
        time.sleep(0.2)
        # ปรับแก้ Yaw ครั้งสุดท้ายเพื่อให้มั่นใจว่าหุ่นยนต์หันตรง
        attitude_handler.correct_yaw_to_target(self.chassis, target_yaw)
    # ========== END: โค้ดที่แก้ไข/เพิ่มใหม่ ==========

    def rotate_to_direction(self, target_direction, attitude_handler):
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

# =============================================================================
# ===== SENSOR HANDLER ========================================================
# =============================================================================
class EnvironmentScanner:
    def __init__(self, sensor_adaptor, tof_sensor, gimbal, chassis):
        self.sensor_adaptor, self.tof_sensor, self.gimbal, self.chassis = sensor_adaptor, tof_sensor, gimbal, chassis
        self.tof_wall_threshold_cm = 50.0; self.last_tof_distance_mm = 0
        self.tof_sensor.sub_distance(freq=5, callback=self._tof_data_handler)
        print(" SENSOR: ToF distance stream started.")
    def _tof_data_handler(self, sub_info): self.last_tof_distance_mm = sub_info[0]
    def get_wall_status(self):
        self.chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0); time.sleep(0.2)
        self.gimbal.moveto(pitch=0, yaw=0).wait_for_completed()
        results, raw_values = {}, {}; time.sleep(0.1)
        tof_distance_cm = self.last_tof_distance_mm / 10.0
        results['front'] = tof_distance_cm < self.tof_wall_threshold_cm and self.last_tof_distance_mm > 0
        raw_values['front_cm'] = f"{tof_distance_cm:.1f}"
        print(f"[SCAN] Front (ToF): {tof_distance_cm:.1f}cm -> {'WALL' if results['front'] else 'FREE'}")
        left_val, right_val = self.sensor_adaptor.get_io(id=LEFT_IR_SENSOR_ID), self.sensor_adaptor.get_io(id=RIGHT_IR_SENSOR_ID)
        results['left'], results['right'] = (left_val == 0), (right_val == 0)
        raw_values['left_ir'], raw_values['right_ir'] = left_val, right_val
        print(f"[SCAN] Left (IR ID {LEFT_IR_SENSOR_ID}): val={left_val} -> {'WALL' if results['left'] else 'FREE'}")
        print(f"[SCAN] Right (IR ID {RIGHT_IR_SENSOR_ID}): val={right_val} -> {'WALL' if results['right'] else 'FREE'}")
        return results, raw_values
    
    def get_front_tof_cm(self):
        """
        ดึงค่าระยะห่างจากเซ็นเซอร์ ToF ด้านหน้าเป็นเซนติเมตรโดยตรง
        """
        time.sleep(0.1) 
        if self.last_tof_distance_mm > 0:
            return self.last_tof_distance_mm / 10.0
        return 999.0

    def cleanup(self):
        try: self.tof_sensor.unsub_distance(); print(" SENSOR: ToF distance stream stopped.")
        except Exception: pass

# =============================================================================
# ===== PATHFINDING & BACKTRACKING LOGIC ======================================
# =============================================================================
def find_path_bfs(maze_map, start, end):
    queue = deque([[start]]); visited = {start}
    dir_map = {'north': (-1, 0), 'south': (1, 0), 'west': (0, -1), 'east': (0, 1)}
    while queue:
        path = queue.popleft(); r, c = path[-1]
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
        if path and (not shortest_path or len(path) < len(shortest_path)): shortest_path = path
    return shortest_path

def execute_path(path, movement_controller, attitude_handler, visualizer, maze_map):
    global CURRENT_POSITION
    print(f"🎯 Executing path: {path}")
    dir_vectors = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}
    for i in range(len(path) - 1):
        visualizer.update_plot(maze_map, path[i], path[i:])
        dr, dc = path[i+1][0] - path[i][0], path[i+1][1] - path[i][1]
        target_direction = dir_vectors[(dr, dc)]
        movement_controller.rotate_to_direction(target_direction, attitude_handler)
        axis_to_monitor = 'x' if ROBOT_FACE % 2 != 0 else 'y'
        movement_controller.move_forward_one_grid(axis=axis_to_monitor, attitude_handler=attitude_handler)
        CURRENT_POSITION = path[i+1]
    visualizer.update_plot(maze_map, CURRENT_POSITION)

def go_to_target_destination(maze_map, current_pos, target_dest, movement_controller, attitude_handler, visualizer):
    global CURRENT_POSITION
    if current_pos == target_dest: print(f"🎯 Already at target destination {target_dest}!"); return True
    print(f"\n🎯 === NAVIGATING TO TARGET DESTINATION {target_dest} ===")
    path_to_target = find_path_bfs(maze_map, current_pos, target_dest)
    if path_to_target:
        print(f"📍 Found path to target: {path_to_target}"); print(f"📏 Path length: {len(path_to_target) - 1} steps")
        execute_path(path_to_target, movement_controller, attitude_handler, visualizer, maze_map)
        print(f"🎉 Successfully reached target destination {target_dest}!"); return True
    else:
        print(f"❌ No path found to target destination {target_dest}"); print("   Target may be unreachable due to walls or out of bounds"); return False

def perform_consistency_nudge(scanner, movement_controller, maze_map, current_pos, current_dir, attitude_handler):
    print("--- Performing Consistency Nudge Check ---")
    r, c = current_pos
    dir_map_rel_abs = {'left': (current_dir - 1 + 4) % 4, 'right': (current_dir + 1) % 4}
    dir_map_idx_name = {0: 'north', 1: 'east', 2: 'south', 3: 'west'}
    dir_vectors = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    abs_dir_left = dir_map_rel_abs['left']; abs_dir_left_name = dir_map_idx_name[abs_dir_left]; nr_left, nc_left = r + dir_vectors[abs_dir_left][0], c + dir_vectors[abs_dir_left][1]
    if 0 <= nr_left < maze_map.height and 0 <= nc_left < maze_map.width and maze_map.grid[nr_left][nc_left].visited:
        if maze_map.has_wall(r, c, abs_dir_left_name):
            current_scan, _ = scanner.get_wall_status()
            if not current_scan['left']:
                print(f"   [Consistency] Map says wall on the LEFT, but sensor doesn't see it. Nudging LEFT to check.")
                movement_controller.slide_with_yaw_lock(y_speed=-0.25, duration=0.2, attitude_handler=attitude_handler)
                time.sleep(0.2)
    abs_dir_right = dir_map_rel_abs['right']; abs_dir_right_name = dir_map_idx_name[abs_dir_right]; nr_right, nc_right = r + dir_vectors[abs_dir_right][0], c + dir_vectors[abs_dir_right][1]
    if 0 <= nr_right < maze_map.height and 0 <= nc_right < maze_map.width and maze_map.grid[nr_right][nc_right].visited:
        if maze_map.has_wall(r, c, abs_dir_right_name):
            current_scan, _ = scanner.get_wall_status()
            if not current_scan['right']:
                print(f"   [Consistency] Map says wall on the RIGHT, but sensor doesn't see it. Nudging RIGHT to check.")
                movement_controller.slide_with_yaw_lock(y_speed=0.25, duration=0.2, attitude_handler=attitude_handler)
                time.sleep(0.2)

def perform_centering_nudge(movement_controller, scanner, initial_wall_status, attitude_handler):
    print("--- Performing Advanced Centering Nudge ---")
    has_left_wall_initial, has_right_wall_initial = initial_wall_status['left'], initial_wall_status['right']
    nudge_dist, nudge_dur = 0.1, 0.25
    y_speed = nudge_dist / nudge_dur
    
    if has_left_wall_initial and not has_right_wall_initial:
        print("   [Single Wall] Left wall detected initially. Nudging RIGHT to check for opposite wall.")
        movement_controller.slide_with_yaw_lock(y_speed=y_speed, duration=nudge_dur, attitude_handler=attitude_handler)
        status_after_nudge, _ = scanner.get_wall_status()
        if status_after_nudge['right']:
            print("   -> Found opposite (right) wall. Now centering between the two walls.")
            movement_controller.slide_with_yaw_lock(y_speed=-(y_speed/2), duration=nudge_dur/2, attitude_handler=attitude_handler)
        else:
            print("   -> No opposite wall found. Returning to a safe offset from the initial left wall.")
            movement_controller.slide_with_yaw_lock(y_speed=-(y_speed*0.9), duration=nudge_dur*0.9, attitude_handler=attitude_handler)
        return
        
    if not has_left_wall_initial and has_right_wall_initial:
        print("   [Single Wall] Right wall detected initially. Nudging LEFT to check for opposite wall.")
        movement_controller.slide_with_yaw_lock(y_speed=-y_speed, duration=nudge_dur, attitude_handler=attitude_handler)
        status_after_nudge, _ = scanner.get_wall_status()
        if status_after_nudge['left']:
            print("   -> Found opposite (left) wall. Now centering between the two walls.")
            movement_controller.slide_with_yaw_lock(y_speed=(y_speed/2), duration=nudge_dur/2, attitude_handler=attitude_handler)
        else:
            print("   -> No opposite wall found. Returning to a safe offset from the initial right wall.")
            movement_controller.slide_with_yaw_lock(y_speed=(y_speed*0.9), duration=nudge_dur*0.9, attitude_handler=attitude_handler)
        return
        
    if not has_left_wall_initial and not has_right_wall_initial:
        print("   [Open Space] Probing for nearby walls.")
        print("   -> Probing RIGHT...")
        movement_controller.slide_with_yaw_lock(y_speed=y_speed, duration=nudge_dur, attitude_handler=attitude_handler)
        right_probe, _ = scanner.get_wall_status()
        if right_probe['right']:
            print("   -> Found a wall on the right. Stopping here.")
            return
        print("   -> Right is clear. Probing LEFT...")
        movement_controller.slide_with_yaw_lock(y_speed=-(y_speed*2), duration=nudge_dur*2, attitude_handler=attitude_handler)
        left_probe, _ = scanner.get_wall_status()
        if left_probe['left']:
            print("   -> Found a wall on the left. Stopping here.")
            return
        print("   -> No walls found on either side. Returning to original starting point.")
        movement_controller.slide_with_yaw_lock(y_speed=y_speed, duration=nudge_dur, attitude_handler=attitude_handler)
        return
    print("   [Corridor] Walls detected on both sides initially. Skipping nudge.")

# =============================================================================
# ===== EXPLORATION LOGIC =====================================================
# =============================================================================
def explore_with_ogm(scanner, movement_controller, attitude_handler, maze_map, visualizer, max_steps=40):
    global CURRENT_POSITION, CURRENT_DIRECTION, TARGET_DESTINATION
    print(f"\n🚀 === STARTING AUTONOMOUS EXPLORATION WITH OGM ==="); print(f"🎯 Target destination set to: {TARGET_DESTINATION}")
    with open("experiment_log.txt", "w") as log_file:
        log_file.write("Step\tRobot Pos(y,x)\tIR Left\tToF Front (cm)\tIR Right\n")
        for step in range(max_steps):
            r, c = CURRENT_POSITION
            maze_map.set_visited(r, c)
            print(f"\n--- Step {step + 1} at {CURRENT_POSITION}, Facing: {['North', 'East', 'South', 'West'][CURRENT_DIRECTION]} (ROBOT_FACE: {ROBOT_FACE}) ---")
            
            perform_consistency_nudge(scanner, movement_controller, maze_map, CURRENT_POSITION, CURRENT_DIRECTION, attitude_handler)
            initial_wall_status, _ = scanner.get_wall_status()
            perform_centering_nudge(movement_controller, scanner, initial_wall_status, attitude_handler)
            
            print("--- Performing Final Scan for Mapping ---")
            final_wall_status, raw_values = scanner.get_wall_status()
            log_file.write(f"{step+1}\t({r},{c})\t\t{raw_values['left_ir']}\t{raw_values['front_cm']}\t\t{raw_values['right_ir']}\n")
            dir_map_rel_abs = {'front': CURRENT_DIRECTION, 'right': (CURRENT_DIRECTION + 1) % 4, 'left': (CURRENT_DIRECTION - 1 + 4) % 4}
            dir_map_idx_name = {0: 'north', 1: 'east', 2: 'south', 3: 'west'}
            for rel_dir, has_wall in final_wall_status.items():
                abs_dir_idx = dir_map_rel_abs[rel_dir]; abs_dir_name = dir_map_idx_name[abs_dir_idx]
                maze_map.set_wall(r, c, abs_dir_name, has_wall)

            visualizer.update_plot(maze_map, CURRENT_POSITION)
            next_move_found = False
            priority_dirs = [(CURRENT_DIRECTION + 1) % 4, CURRENT_DIRECTION, (CURRENT_DIRECTION - 1 + 4) % 4]
            dir_vectors = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            for target_dir in priority_dirs:
                r_next, c_next = r + dir_vectors[target_dir][0], c + dir_vectors[target_dir][1]
                if 0 <= r_next < maze_map.height and 0 <= c_next < maze_map.width:
                    if not maze_map.has_wall(r, c, dir_map_idx_name[target_dir]) and not maze_map.grid[r_next][c_next].visited:
                        print(f"Found local unvisited path to {dir_map_idx_name[target_dir]}."); movement_controller.rotate_to_direction(target_dir, attitude_handler); next_move_found = True; break
            if not next_move_found:
                print("No local unvisited path. Searching for nearest unvisited cell...")
                path = find_nearest_unvisited(maze_map, CURRENT_POSITION)
                if path: execute_path(path, movement_controller, attitude_handler, visualizer, maze_map); continue
                else: print("🎉 EXPLORATION COMPLETE! No unvisited cells reachable."); break
            
            axis_to_monitor = 'x' if ROBOT_FACE % 2 != 0 else 'y'
            movement_controller.move_forward_one_grid(axis=axis_to_monitor, attitude_handler=attitude_handler)

            # ===== LOGIC ที่เพิ่มใหม่: ตรวจสอบและปรับระยะห่างด้านหน้า =====
            print("--- Performing Post-Move Front Distance Check ---")
            front_distance_cm = scanner.get_front_tof_cm()
            print(f"   Current front distance: {front_distance_cm:.1f} cm")
            if front_distance_cm < 17.6 and front_distance_cm > 0:
                correction_cm = 17.6 - front_distance_cm
                correction_m = correction_cm / 100.0
                print(f"   Distance is too close! Moving BACKWARD by {correction_cm:.1f} cm.")
                movement_controller.move_precise_distance(-correction_m, axis=axis_to_monitor, attitude_handler=attitude_handler)
                final_dist_cm = scanner.get_front_tof_cm()
                print(f"   Correction complete. New front distance: {final_dist_cm:.1f} cm.")
            
            CURRENT_POSITION = (r + dir_vectors[CURRENT_DIRECTION][0], c + dir_vectors[CURRENT_DIRECTION][1])
    
    print("\n🎉 === EXPLORATION PHASE FINISHED ===")
    print(f"\n🎯 === STARTING NAVIGATION TO TARGET DESTINATION ===")
    target_r, target_c = TARGET_DESTINATION
    if not (0 <= target_r < maze_map.height and 0 <= target_c < maze_map.width):
        print(f"❌ ERROR: Target destination {TARGET_DESTINATION} is out of map bounds!"); print(f"   Map size: {maze_map.height} x {maze_map.width}")
    else:
        success = go_to_target_destination(maze_map, CURRENT_POSITION, TARGET_DESTINATION, movement_controller, attitude_handler, visualizer)
        if success: print(f"🎉 === MISSION ACCOMPLISHED! ==="); print(f"   Robot successfully reached target destination {TARGET_DESTINATION}")
        else: print(f"⚠️ === MISSION INCOMPLETE ==="); print(f"   Could not reach target destination {TARGET_DESTINATION}")
    
    visualizer.update_plot(maze_map, CURRENT_POSITION)
    with open("final_structural_map.json", "w") as f:
        json_map = [[{'visited': cell.visited, 'walls': cell.walls} for cell in row] for row in maze_map.grid]
        final_data = {'map': json_map, 'final_position': CURRENT_POSITION, 'target_destination': TARGET_DESTINATION, 'mission_completed': CURRENT_POSITION == TARGET_DESTINATION}
        json.dump(final_data, f, indent=2)
    print("✅ Final map and log file saved."); print("... You can close the heatmap window now ..."); plt.ioff(); plt.show()

# =============================================================================
# ===== MAIN EXECUTION BLOCK ==================================================
# =============================================================================
if __name__ == '__main__':
    ep_robot = None; maze_map = MazeMap(width=4, height=4); attitude_handler = AttitudeHandler()
    movement_controller = None; scanner = None; ep_chassis = None
    print(f"🎯 Target destination: {TARGET_DESTINATION}"); print(f"🏁 Starting position: {CURRENT_POSITION}")
    try:
        visualizer = RealTimeVisualizer(grid_size=4, target_dest=TARGET_DESTINATION)
        print("🤖 Connecting to robot..."); ep_robot = robot.Robot(); ep_robot.initialize(conn_type="ap")
        ep_chassis, ep_gimbal = ep_robot.chassis, ep_robot.gimbal
        ep_tof_sensor, ep_sensor_adaptor = ep_robot.sensor, ep_robot.sensor_adaptor
        print(" ROBOT: Locking wheels for stable gimbal recentering..."); ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0); time.sleep(0.5)
        print(" GIMBAL: Centering gimbal..."); ep_gimbal.recenter().wait_for_completed()
        print(" ROBOT: Re-locking wheels post-recenter to ensure stability."); ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0); time.sleep(0.5)
        print(" GIMBAL: Centered. Robot is stable.")
        scanner = EnvironmentScanner(ep_sensor_adaptor, ep_tof_sensor, ep_gimbal, ep_chassis)
        movement_controller = MovementController(ep_chassis)
        attitude_handler.start_monitoring(ep_chassis)
        explore_with_ogm(scanner, movement_controller, attitude_handler, maze_map, visualizer)
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