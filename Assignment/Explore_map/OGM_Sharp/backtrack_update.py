# -*-coding:utf-8-*-

import time
import robomaster
from robomaster import robot
import numpy as np
import math
import json
import matplotlib.pyplot as plt
from collections import deque
import traceback
import statistics

# =============================================================================
# ===== CONFIGURATION & PARAMETERS ============================================
# =============================================================================

# --- Sharp Distance Sensor Configuration ---
LEFT_SHARP_SENSOR_ID = 1
LEFT_SHARP_SENSOR_PORT = 1
LEFT_TARGET_CM = 15.0

RIGHT_SHARP_SENSOR_ID = 2
RIGHT_SHARP_SENSOR_PORT = 1
RIGHT_TARGET_CM = 15.0

# --- IR Sensor Configuration ---
LEFT_IR_SENSOR_ID = 1
LEFT_IR_SENSOR_PORT = 2
RIGHT_IR_SENSOR_ID = 2
RIGHT_IR_SENSOR_PORT = 2

# --- Sharp Sensor Detection Thresholds ---
SHARP_WALL_THRESHOLD_CM = 45.0  # ระยะสูงสุดที่จะถือว่าเจอผนัง
SHARP_STDEV_THRESHOLD = 0.25      # ค่าเบี่ยงเบนมาตรฐานสูงสุดที่ยอมรับได้ เพื่อกรองค่าที่แกว่ง

# --- Logical state for the grid map (from map_suay.py) ---
CURRENT_POSITION = (4, 0)  # (แถว, คอลัมน์)
CURRENT_DIRECTION = 0      # 0:North, 1:East, 2:South, 3:West
TARGET_DESTINATION = (4, 0)

# --- Physical state for the robot ---
CURRENT_TARGET_YAW = 0.0
ROBOT_FACE = 1  # 1,3,5.. = X axis, 2,4,6.. = Y axis

# --- Occupancy Grid Parameters ---
PROB_OCC_GIVEN_OCC = {'tof': 0.95, 'sharp': 0.90} # เพิ่ม 'sharp'
PROB_OCC_GIVEN_FREE = {'tof': 0.05, 'sharp': 0.10} # เพิ่ม 'sharp'

LOG_ODDS_OCC = {
    'tof': math.log(PROB_OCC_GIVEN_OCC['tof'] / (1 - PROB_OCC_GIVEN_OCC['tof'])),
    'sharp': math.log(PROB_OCC_GIVEN_OCC['sharp'] / (1 - PROB_OCC_GIVEN_OCC['sharp']))
}
LOG_ODDS_FREE = {
    'tof': math.log(PROB_OCC_GIVEN_FREE['tof'] / (1 - PROB_OCC_GIVEN_FREE['tof'])),
    'sharp': math.log(PROB_OCC_GIVEN_FREE['sharp'] / (1 - PROB_OCC_GIVEN_FREE['sharp']))
}

# --- Decision Thresholds ---
OCCUPANCY_THRESHOLD = 0.7
FREE_THRESHOLD = 0.3

# =============================================================================
# ===== HELPER FUNCTIONS ======================================================
# =============================================================================
def convert_adc_to_cm(adc_value):
    """Converts ADC value from Sharp sensor to centimeters."""
    if adc_value <= 0: return float('inf')
    # This formula is specific to the GP2Y0A21YK0F sensor.
    # You may need to re-calibrate for your specific sensor.
    return 30263 * (adc_value ** -1.352)

# =============================================================================
# ===== OCCUPANCY GRID MAP & VISUALIZATION (from map_suay.py) =================
# =============================================================================

class WallBelief:
    """Class to manage the belief of a 'wall'."""
    def __init__(self):
        self.log_odds = 0.0

    def update(self, is_occupied_reading, sensor_type):
        if is_occupied_reading:
            self.log_odds += LOG_ODDS_OCC[sensor_type]
        else:
            self.log_odds += LOG_ODDS_FREE[sensor_type]
        self.log_odds = max(min(self.log_odds, 10), -10)

    def get_probability(self):
        return 1.0 - 1.0 / (1.0 + math.exp(self.log_odds))

    def is_occupied(self):
        return self.get_probability() > OCCUPANCY_THRESHOLD

class OGMCell:
    """Class for a cell that stores beliefs about 'space' and 'walls'."""
    def __init__(self):
        self.log_odds_occupied = 0.0
        self.walls = {'N': None, 'E': None, 'S': None, 'W': None}

    def get_node_probability(self):
        return 1.0 - 1.0 / (1.0 + math.exp(self.log_odds_occupied))

    def is_node_occupied(self):
        return self.get_node_probability() > OCCUPANCY_THRESHOLD

class OccupancyGridMap:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[OGMCell() for _ in range(width)] for _ in range(height)]
        self._link_walls()

    def _link_walls(self):
        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r][c].walls['N'] is None:
                    wall = WallBelief()
                    self.grid[r][c].walls['N'] = wall
                    if r > 0: self.grid[r-1][c].walls['S'] = wall
                if self.grid[r][c].walls['W'] is None:
                    wall = WallBelief()
                    self.grid[r][c].walls['W'] = wall
                    if c > 0: self.grid[r][c-1].walls['E'] = wall
                if self.grid[r][c].walls['S'] is None:
                    self.grid[r][c].walls['S'] = WallBelief()
                if self.grid[r][c].walls['E'] is None:
                    self.grid[r][c].walls['E'] = WallBelief()

    def update_wall(self, r, c, direction_char, is_occupied_reading, sensor_type):
        if 0 <= r < self.height and 0 <= c < self.width:
            wall = self.grid[r][c].walls.get(direction_char)
            if wall:
                wall.update(is_occupied_reading, sensor_type)

    def update_node(self, r, c, is_occupied_reading, sensor_type='tof'):
        if 0 <= r < self.height and 0 <= c < self.width:
            if is_occupied_reading:
                self.grid[r][c].log_odds_occupied += LOG_ODDS_OCC[sensor_type]
            else:
                self.grid[r][c].log_odds_occupied += LOG_ODDS_FREE[sensor_type]

    def is_path_clear(self, r1, c1, r2, c2):
        dr, dc = r2 - r1, c2 - c1
        if abs(dr) + abs(dc) != 1: return False
        if dr == -1: wall_char = 'N'
        elif dr == 1: wall_char = 'S'
        elif dc == 1: wall_char = 'E'
        elif dc == -1: wall_char = 'W'
        else: return False
        wall = self.grid[r1][c1].walls.get(wall_char)
        if wall and wall.is_occupied(): return False
        if 0 <= r2 < self.height and 0 <= c2 < self.width:
            if self.grid[r2][c2].is_node_occupied(): return False
        else: return False
        return True

class RealTimeVisualizer:
    def __init__(self, grid_size, target_dest=None):
        self.grid_size = grid_size
        self.target_dest = target_dest
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 7))
        self.colors = {"robot": "#0000FF", "target": "#FFD700", "path": "#FFFF00", "wall": "#000000", "wall_prob": "#000080"}

    def update_plot(self, occupancy_map, robot_pos, path=None):
        self.ax.clear()
        self.ax.set_title("Real-time Hybrid Belief Map (Nodes & Walls)")
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.ax.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax.set_ylim(self.grid_size - 0.5, -0.5)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                prob = occupancy_map.grid[r][c].get_node_probability()
                if prob > OCCUPANCY_THRESHOLD: color = '#8B0000'
                elif prob < FREE_THRESHOLD: color = '#D3D3D3'
                else: color = '#90EE90'
                self.ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=color, edgecolor='k', lw=0.5))
                self.ax.text(c, r, f"{prob:.2f}", ha="center", va="center", color="black", fontsize=8)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell = occupancy_map.grid[r][c]
                prob_n = cell.walls['N'].get_probability()
                if abs(prob_n - 0.5) > 0.01: self.ax.text(c, r - 0.5, f"{prob_n:.2f}", ha="center", va="center", color=self.colors['wall_prob'], fontsize=6, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1', edgecolor='none'))
                prob_w = cell.walls['W'].get_probability()
                if abs(prob_w - 0.5) > 0.01: self.ax.text(c - 0.5, r, f"{prob_w:.2f}", ha="center", va="center", color=self.colors['wall_prob'], fontsize=6, rotation=90, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1', edgecolor='none'))
        for c in range(self.grid_size):
            r_edge = self.grid_size - 1
            prob_s = occupancy_map.grid[r_edge][c].walls['S'].get_probability()
            if abs(prob_s - 0.5) > 0.01: self.ax.text(c, r_edge + 0.5, f"{prob_s:.2f}", ha="center", va="center", color=self.colors['wall_prob'], fontsize=6, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1', edgecolor='none'))
        for r in range(self.grid_size):
            c_edge = self.grid_size - 1
            prob_e = occupancy_map.grid[r][c_edge].walls['E'].get_probability()
            if abs(prob_e - 0.5) > 0.01: self.ax.text(c_edge + 0.5, r, f"{prob_e:.2f}", ha="center", va="center", color=self.colors['wall_prob'], fontsize=6, rotation=90, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1', edgecolor='none'))
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell = occupancy_map.grid[r][c]
                if cell.walls['N'].is_occupied(): self.ax.plot([c - 0.5, c + 0.5], [r - 0.5, r - 0.5], color=self.colors['wall'], linewidth=4)
                if cell.walls['W'].is_occupied(): self.ax.plot([c - 0.5, c - 0.5], [r - 0.5, r + 0.5], color=self.colors['wall'], linewidth=4)
                if r == self.grid_size - 1 and cell.walls['S'].is_occupied(): self.ax.plot([c - 0.5, c + 0.5], [r + 0.5, r + 0.5], color=self.colors['wall'], linewidth=4)
                if c == self.grid_size - 1 and cell.walls['E'].is_occupied(): self.ax.plot([c + 0.5, c + 0.5], [r - 0.5, r + 0.5], color=self.colors['wall'], linewidth=4)
        if self.target_dest:
            r_t, c_t = self.target_dest
            self.ax.add_patch(plt.Rectangle((c_t - 0.5, r_t - 0.5), 1, 1, facecolor=self.colors['target'], edgecolor='k', lw=2, alpha=0.8))
        if path:
            for r_p, c_p in path: self.ax.add_patch(plt.Rectangle((c_p - 0.5, r_p - 0.5), 1, 1, facecolor=self.colors['path'], edgecolor='k', lw=0.5, alpha=0.7))
        if robot_pos:
            r_r, c_r = robot_pos
            self.ax.add_patch(plt.Rectangle((c_r - 0.5, r_r - 0.5), 1, 1, facecolor=self.colors['robot'], edgecolor='k', lw=2))
        legend_elements = [ plt.Rectangle((0,0),1,1, facecolor='#8B0000', label=f'Node Occupied (P>{OCCUPANCY_THRESHOLD})'), plt.Rectangle((0,0),1,1, facecolor='#90EE90', label=f'Node Unknown'), plt.Rectangle((0,0),1,1, facecolor='#D3D3D3', label=f'Node Free (P<{FREE_THRESHOLD})'), plt.Line2D([0], [0], color=self.colors['wall'], lw=4, label='Wall Occupied'), plt.Rectangle((0,0),1,1, facecolor=self.colors['robot'], label='Robot'), plt.Rectangle((0,0),1,1, facecolor=self.colors['target'], label='Target') ]
        self.ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.55, 1.0))
        self.fig.tight_layout(rect=[0, 0, 0.75, 1])
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
        print(f"\n🔧 Correcting Yaw: {self.current_yaw:.1f}° -> {target_yaw:.1f}°. Rotating: {robot_rotation:.1f}°")
        if abs(robot_rotation) > self.yaw_tolerance:
            chassis.move(x=0, y=0, z=robot_rotation, z_speed=60).wait_for_completed(timeout=2)
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

    def _calculate_yaw_correction(self, attitude_handler, target_yaw):
        KP_YAW = 1.8; MAX_YAW_SPEED = 25
        yaw_error = attitude_handler.normalize_angle(target_yaw - attitude_handler.current_yaw)
        speed = KP_YAW * yaw_error
        return max(min(speed, MAX_YAW_SPEED), -MAX_YAW_SPEED)

    def move_forward_one_grid(self, axis, attitude_handler):
        attitude_handler.correct_yaw_to_target(self.chassis, CURRENT_TARGET_YAW)
        target_distance = 0.6
        pid = PID(Kp=1.8, Ki=0.25, Kd=12, setpoint=target_distance)
        start_time, last_time = time.time(), time.time()
        start_position = self.current_x_pos if axis == 'x' else self.current_y_pos
        print(f"🚀 Moving FORWARD 0.6m, monitoring GLOBAL AXIS '{axis}'")
        while time.time() - start_time < 3.5: # Increased timeout
            now = time.time(); dt = now - last_time; last_time = now
            current_position = self.current_x_pos if axis == 'x' else self.current_y_pos
            relative_position = abs(current_position - start_position)
            if abs(relative_position - target_distance) < 0.03:
                print("\n✅ Move complete!"); break
            output = pid.compute(relative_position, dt)
            ramp_multiplier = min(1.0, 0.1 + ((now - start_time) / 1.0) * 0.9)
            speed = max(-1.0, min(1.0, output * ramp_multiplier))
            yaw_correction = self._calculate_yaw_correction(attitude_handler, CURRENT_TARGET_YAW)
            self.chassis.drive_speed(x=speed, y=0, z=yaw_correction, timeout=1)
            print(f"Moving... Dist: {relative_position:.3f}/{target_distance:.2f} m", end='\r')
        self.chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0); time.sleep(0.5)

    def adjust_position_to_wall(self, sensor_adaptor, attitude_handler, side, sensor_config, target_distance_cm, direction_multiplier):
        """
        เวอร์ชันแก้ไข: อ่านค่า ADC โดยตรงเพื่อการตอบสนองที่รวดเร็ว (เหมือน std.py)
        """
        print(f"\n--- Adjusting {side} Side (Yaw locked at {CURRENT_TARGET_YAW:.2f}°) ---")
        
        print(f"   -> Config: ID={sensor_config['sharp_id']}, Port={sensor_config['sharp_port']}, Target={target_distance_cm}cm")

        TOLERANCE_CM, MAX_EXEC_TIME, KP_SLIDE, MAX_SLIDE_SPEED = 0.8, 8, 0.045, 0.18 # เพิ่ม KP และ Speed เล็กน้อย
        start_time = time.time()
        
        while time.time() - start_time < MAX_EXEC_TIME:
            adc_val = sensor_adaptor.get_adc(id=sensor_config["sharp_id"], port=sensor_config["sharp_port"])
            current_dist = convert_adc_to_cm(adc_val)

            dist_error = target_distance_cm - current_dist
            
            if abs(dist_error) <= TOLERANCE_CM:
                print(f"\n[{side}] Target distance reached! Final distance: {current_dist:.2f} cm")
                break

            slide_speed = max(min(direction_multiplier * KP_SLIDE * dist_error, MAX_SLIDE_SPEED), -MAX_SLIDE_SPEED)
            yaw_correction = self._calculate_yaw_correction(attitude_handler, CURRENT_TARGET_YAW)
            
            self.chassis.drive_speed(x=0, y=slide_speed, z=yaw_correction)
            
            print(f"Adjusting {side}... Current: {current_dist:5.2f}cm, Target: {target_distance_cm:4.1f}cm, Error: {dist_error:5.2f}cm, Speed: {slide_speed:5.3f}", end='\r')
            time.sleep(0.02)
        else:
             print(f"\n[{side}] Movement timed out!")
        
        self.chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
        time.sleep(0.2)

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

class EnvironmentScanner:
    """ NEW EnvironmentScanner using ToF (front) and Sharp Sensors (sides) """
    def __init__(self, sensor_adaptor, tof_sensor, gimbal, chassis):
        self.sensor_adaptor, self.tof_sensor, self.gimbal, self.chassis = sensor_adaptor, tof_sensor, gimbal, chassis
        self.tof_wall_threshold_cm = 50.0; self.last_tof_distance_mm = 0
        self.tof_sensor.sub_distance(freq=5, callback=self._tof_data_handler)
        
        self.side_sensors = {
            "Left":  {
                "sharp_id": LEFT_SHARP_SENSOR_ID, "sharp_port": LEFT_SHARP_SENSOR_PORT,
                "ir_id": LEFT_IR_SENSOR_ID, "ir_port": LEFT_IR_SENSOR_PORT
            },
            "Right": {
                "sharp_id": RIGHT_SHARP_SENSOR_ID, "sharp_port": RIGHT_SHARP_SENSOR_PORT,
                "ir_id": RIGHT_IR_SENSOR_ID, "ir_port": RIGHT_IR_SENSOR_PORT
            }
        }

    def _tof_data_handler(self, sub_info): self.last_tof_distance_mm = sub_info[0]

    def _get_stable_reading_cm(self, side, duration=0.7):
        """Reads a Sharp sensor for a duration to get a stable value."""
        sensor_info = self.side_sensors.get(side)
        if not sensor_info: return None, None
        
        readings = []
        start_time = time.time()
        while time.time() - start_time < duration:
            adc = self.sensor_adaptor.get_adc(id=sensor_info["sharp_id"], port=sensor_info["sharp_port"])
            readings.append(convert_adc_to_cm(adc))
            time.sleep(0.05)
        
        if len(readings) < 5: return None, None

        avg_distance = statistics.mean(readings)
        std_dev = statistics.stdev(readings)
        return avg_distance, std_dev

    def get_sensor_readings(self):
        """
        Performs a full scan: Front (ToF) and Sides (Sharp + IR).
        Returns a dictionary indicating if a wall is present.
        """
        self.chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0); time.sleep(0.2)
        self.gimbal.moveto(pitch=0, yaw=0).wait_for_completed(); time.sleep(0.2)
        
        readings = {}
        # Front ToF check
        tof_distance_cm = self.last_tof_distance_mm / 10.0
        readings['front'] = (tof_distance_cm < self.tof_wall_threshold_cm and self.last_tof_distance_mm > 0)
        print(f"[SCAN] Front (ToF): {tof_distance_cm:.1f}cm -> {'OCCUPIED' if readings['front'] else 'FREE'}")
        
        # Side Sharp + IR sensor checks
        for side in ["Left", "Right"]:
            avg_dist, std_dev = self._get_stable_reading_cm(side)
            
            if avg_dist is None:
                print(f"[{side.upper()}] Wall Check Error: Not enough sensor data.")
                readings[side.lower()] = False
                continue
            
            # 1. ตรวจสอบจาก Sharp sensor
            is_sharp_detecting_wall = (avg_dist < SHARP_WALL_THRESHOLD_CM and std_dev < SHARP_STDEV_THRESHOLD)
            
            # 2. อ่านค่าจาก IR sensor
            sensor_config = self.side_sensors[side]
            ir_value = self.sensor_adaptor.get_io(id=sensor_config["ir_id"], port=sensor_config["ir_port"])
            is_ir_detecting_wall = (ir_value == 0) # 0 คือเจอ, 1 คือไม่เจอ

            # 3. ตรรกะการตัดสินใจใหม่:
            #    - ถ้า Sharp เจอกำแพง -> เชื่อ Sharp ทันที (is_wall = True)
            #    - ถ้า Sharp ไม่เจอ -> ให้ IR เป็นตัวตัดสินสุดท้าย
            #    ซึ่งสามารถสรุปเป็นเงื่อนไข OR ได้
            is_wall = is_sharp_detecting_wall or is_ir_detecting_wall
            readings[side.lower()] = is_wall
            
            # ปรับปรุง Log เพื่อให้เห็นค่าจากทั้ง 2 sensors และอธิบายผลลัพธ์
            print(f"[SCAN] {side} (Sharp) -> Avg Dist: {avg_dist:.2f} cm, Std Dev: {std_dev:.2f} -> Sharp says: {'WALL' if is_sharp_detecting_wall else 'FREE'}")
            print(f"    -> {side} (IR)   -> Value: {ir_value} -> IR says: {'WALL' if is_ir_detecting_wall else 'FREE'}")

            if is_sharp_detecting_wall:
                print("    -> Final Result: WALL DETECTED (Based on Sharp sensor)")
            elif is_ir_detecting_wall:
                print("    -> Final Result: WALL DETECTED (Sharp was clear, but IR override detected an object)")
            else:
                print("    -> Final Result: NO WALL (Both sensors agree the path is clear)")
            
        return readings

    def get_front_tof_cm(self):
        self.gimbal.moveto(pitch=0, yaw=0).wait_for_completed()
        time.sleep(0.2)
        return self.last_tof_distance_mm / 10.0 if self.last_tof_distance_mm > 0 else 999.0

    def cleanup(self):
        try: self.tof_sensor.unsub_distance()
        except Exception: pass

# =============================================================================
# ===== PATHFINDING & EXPLORATION LOGIC =======================================
# =============================================================================

def find_path_bfs(occupancy_map, start, end):
    queue = deque([[start]]); visited = {start}
    moves = [(-1, 0), (0, 1), (1, 0), (0, -1)] # N, E, S, W
    while queue:
        path = queue.popleft()
        r, c = path[-1]
        if (r, c) == end: return path
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < occupancy_map.height and 0 <= nc < occupancy_map.width:
                if occupancy_map.is_path_clear(r, c, nr, nc) and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    new_path = list(path)
                    new_path.append((nr, nc))
                    queue.append(new_path)
    return None

def find_nearest_unvisited_path(occupancy_map, start_pos, visited_cells):
    h, w = occupancy_map.height, occupancy_map.width
    unvisited_cells_coords = []
    for r in range(h):
        for c in range(w):
            if (r, c) not in visited_cells and not occupancy_map.grid[r][c].is_node_occupied():
                unvisited_cells_coords.append((r, c))
    if not unvisited_cells_coords: return None
    shortest_path = None
    for target_pos in unvisited_cells_coords:
        path = find_path_bfs(occupancy_map, start_pos, target_pos)
        if path:
            if shortest_path is None or len(path) < len(shortest_path):
                shortest_path = path
    return shortest_path

def execute_path(path, movement_controller, attitude_handler, scanner, visualizer, occupancy_map, path_name="Backtrack"):
    global CURRENT_POSITION
    print(f"🎯 Executing {path_name} Path: {path}")
    dir_vectors_map = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}
    for i in range(len(path) - 1):
        visualizer.update_plot(occupancy_map, path[i], path)
        current_r, current_c = path[i]
        
        # --- ส่วนของการเคลื่อนที่ (เหมือนเดิม) ---
        if i + 1 < len(path):
            next_r, next_c = path[i+1]
            dr, dc = next_r - current_r, next_c - current_c
            target_direction = dir_vectors_map[(dr, dc)]
            
            movement_controller.rotate_to_direction(target_direction, attitude_handler)
            
            axis_to_monitor = 'x' if ROBOT_FACE % 2 != 0 else 'y'
            movement_controller.move_forward_one_grid(axis=axis_to_monitor, attitude_handler=attitude_handler)
            
            CURRENT_POSITION = (next_r, next_c)
            visualizer.update_plot(occupancy_map, CURRENT_POSITION, path) # อัปเดตภาพหลังเดิน
            
            # --- [ส่วนที่เพิ่มเข้ามา] ---
            # หลังจากเดินไปถึงช่องใหม่แล้ว ให้ทำการตรวจสอบกำแพงด้านข้างและปรับตำแหน่ง
            print(f"   -> [{path_name}] Performing side alignment at new position {CURRENT_POSITION}")
            perform_side_alignment_and_mapping(movement_controller, scanner, attitude_handler, occupancy_map, visualizer)
            # ---------------------------

    print(f"✅ {path_name} complete.")

def perform_side_alignment_and_mapping(movement_controller, scanner, attitude_handler, occupancy_map, visualizer):
    """
    NEW function to check for walls, update the map, and physically align the robot.
    Replaces the old 'perform_centering_nudge'.
    """
    print("\n--- Stage: Wall Detection & Side Alignment ---")
    r, c = CURRENT_POSITION
    dir_map_abs_char = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}
    
    # Get a stable reading for both sides
    side_walls_present = scanner.get_sensor_readings()

    # --- LEFT WALL ---
    left_dir_abs = (CURRENT_DIRECTION - 1 + 4) % 4
    occupancy_map.update_wall(r, c, dir_map_abs_char[left_dir_abs], side_walls_present['left'], 'sharp')
    visualizer.update_plot(occupancy_map, CURRENT_POSITION)
    if side_walls_present['left']:
        # Call the updated adjustment function
        movement_controller.adjust_position_to_wall(
            scanner.sensor_adaptor, attitude_handler, "Left", 
            scanner.side_sensors["Left"], LEFT_TARGET_CM, direction_multiplier=1
        )
    
    # --- RIGHT WALL ---
    right_dir_abs = (CURRENT_DIRECTION + 1) % 4
    occupancy_map.update_wall(r, c, dir_map_abs_char[right_dir_abs], side_walls_present['right'], 'sharp')
    visualizer.update_plot(occupancy_map, CURRENT_POSITION)
    if side_walls_present['right']:
        # Call the updated adjustment function
        movement_controller.adjust_position_to_wall(
            scanner.sensor_adaptor, attitude_handler, "Right", 
            scanner.side_sensors["Right"], RIGHT_TARGET_CM, direction_multiplier=-1
        )

    if not side_walls_present['left'] and not side_walls_present['right']:
        print("\n⚠️  WARNING: No side walls detected. Skipping alignment.")
    
    # Final check to ensure robot is perfectly straight
    attitude_handler.correct_yaw_to_target(movement_controller.chassis, CURRENT_TARGET_YAW)
    time.sleep(0.3)


def explore_with_ogm(scanner, movement_controller, attitude_handler, occupancy_map, visualizer, max_steps=40):
    global CURRENT_POSITION, CURRENT_DIRECTION
    visited_cells = set()
    
    for step in range(max_steps):
        r, c = CURRENT_POSITION
        print(f"\n--- Step {step + 1} at {CURRENT_POSITION}, Facing: {['N', 'E', 'S', 'W'][CURRENT_DIRECTION]} ---")
        
        # --- [ส่วนที่เพิ่มเข้ามา] ---
        # เพื่อความแน่นอน: ทำการบังคับตั้งตรงทุกครั้งที่เริ่ม Step ใหม่
        # เพื่อล้างค่า Error ของ Yaw ที่อาจเกิดขึ้นจากการหยุดนิ่งใน Step ที่แล้ว
        print("   -> Resetting Yaw to ensure perfect alignment before new step...")
        attitude_handler.correct_yaw_to_target(movement_controller.chassis, CURRENT_TARGET_YAW)
        # ---------------------------

        # 1. Perform side wall detection, mapping, and physical alignment
        perform_side_alignment_and_mapping(movement_controller, scanner, attitude_handler, occupancy_map, visualizer)

        # 2. Perform official scan for front wall and map it
        print("--- Performing Scan for Mapping (Front ToF Only) ---")
        is_front_occupied = scanner.get_front_tof_cm() < scanner.tof_wall_threshold_cm
        dir_map_abs_char = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}
        occupancy_map.update_wall(r, c, dir_map_abs_char[CURRENT_DIRECTION], is_front_occupied, 'tof')
        
        # 3. Mark current node as visited and free
        occupancy_map.update_node(r, c, False, 'tof')
        visited_cells.add((r, c))
        visualizer.update_plot(occupancy_map, CURRENT_POSITION)
        
        # 4. Decide on the next move
        priority_dirs = [(CURRENT_DIRECTION + 1) % 4, CURRENT_DIRECTION, (CURRENT_DIRECTION - 1 + 4) % 4]
        moved = False
        dir_vectors = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        for target_dir in priority_dirs:
            target_r, target_c = r + dir_vectors[target_dir][0], c + dir_vectors[target_dir][1]
            
            if occupancy_map.is_path_clear(r, c, target_r, target_c) and (target_r, target_c) not in visited_cells:
                print(f"Path to {['N','E','S','W'][target_dir]} at ({target_r},{target_c}) seems clear. Attempting move.")
                movement_controller.rotate_to_direction(target_dir, attitude_handler)
                
                print("    Confirming path forward with ToF...")
                is_blocked = scanner.get_front_tof_cm() < scanner.tof_wall_threshold_cm
                
                # Update map with this new information before final decision
                occupancy_map.update_wall(r, c, dir_map_abs_char[CURRENT_DIRECTION], is_blocked, 'tof')
                print(f"    ToF confirmation: Wall belief updated. Path is {'BLOCKED' if is_blocked else 'CLEAR'}.")
                visualizer.update_plot(occupancy_map, CURRENT_POSITION)

                if occupancy_map.is_path_clear(r, c, target_r, target_c):
                    axis_to_monitor = 'x' if ROBOT_FACE % 2 != 0 else 'y'
                    movement_controller.move_forward_one_grid(axis=axis_to_monitor, attitude_handler=attitude_handler)
                    CURRENT_POSITION = (target_r, target_c)
                    moved = True
                    break
                else:
                    print(f"    Confirmation failed. Path to {['N','E','S','W'][target_dir]} is blocked. Re-evaluating.")
        
        if not moved:
            print("No immediate unvisited path. Initiating backtrack...")
            backtrack_path = find_nearest_unvisited_path(occupancy_map, CURRENT_POSITION, visited_cells)

            if backtrack_path and len(backtrack_path) > 1:
                execute_path(backtrack_path, movement_controller, attitude_handler, scanner, visualizer, occupancy_map)
                print("Backtrack to new area complete. Resuming exploration.")
                continue
            else:
                print("🎉 EXPLORATION COMPLETE! No reachable unvisited cells remain.")
                break
    
    print("\n🎉 === EXPLORATION PHASE FINISHED ===")

# =============================================================================
# ===== MAIN EXECUTION BLOCK ==================================================
# =============================================================================
if __name__ == '__main__':
    ep_robot = None
    occupancy_map = OccupancyGridMap(width=5, height=5)
    attitude_handler = AttitudeHandler()
    movement_controller = None
    scanner = None
    ep_chassis = None
    
    try:
        visualizer = RealTimeVisualizer(grid_size=5, target_dest=TARGET_DESTINATION)
        print("🤖 Connecting to robot..."); ep_robot = robot.Robot(); ep_robot.initialize(conn_type="ap")
        ep_chassis, ep_gimbal = ep_robot.chassis, ep_robot.gimbal
        ep_tof_sensor, ep_sensor_adaptor = ep_robot.sensor, ep_robot.sensor_adaptor
        
        print(" GIMBAL: Centering gimbal..."); ep_gimbal.recenter().wait_for_completed()
        
        scanner = EnvironmentScanner(ep_sensor_adaptor, ep_tof_sensor, ep_gimbal, ep_chassis)
        movement_controller = MovementController(ep_chassis)
        attitude_handler.start_monitoring(ep_chassis)
        
        explore_with_ogm(scanner, movement_controller, attitude_handler, occupancy_map, visualizer)
        
        print(f"\n\n--- NAVIGATION TO TARGET PHASE: From {CURRENT_POSITION} to {TARGET_DESTINATION} ---")
        
        if CURRENT_POSITION == TARGET_DESTINATION:
            print("🎉 Robot is already at the target destination!")
        else:
            path_to_target = find_path_bfs(occupancy_map, CURRENT_POSITION, TARGET_DESTINATION)
            if path_to_target and len(path_to_target) > 1:
                print(f"✅ Path found to target: {path_to_target}")
                execute_path(path_to_target, movement_controller, attitude_handler, scanner, visualizer, occupancy_map, path_name="Final Navigation")
                print(f"🎉🎉 Robot has arrived at the target destination: {TARGET_DESTINATION}!")
            else:
                print(f"⚠️ Could not find a path from {CURRENT_POSITION} to {TARGET_DESTINATION}.")
        
    except KeyboardInterrupt: print("\n⚠️ User interrupted exploration.")
    except Exception as e: print(f"\n⚌ An error occurred: {e}"); traceback.print_exc()
    finally:
        if ep_robot:
            print("🔌 Cleaning up and closing connection...")
            if scanner: scanner.cleanup()
            if attitude_handler and attitude_handler.is_monitoring: attitude_handler.stop_monitoring(ep_chassis)
            if movement_controller: movement_controller.cleanup()
            ep_robot.close()
            print("🔌 Connection closed.")
        
        final_map_data = {
            'nodes': [[cell.get_node_probability() for cell in row] for row in occupancy_map.grid]
        }
        with open("final_hybrid_map.json", "w") as f:
            json.dump(final_map_data, f, indent=2)
        print("✅ Final Hybrid Belief Map saved to final_hybrid_map.json")
        print("... You can close the plot window now ...")
        plt.ioff()
        plt.show()