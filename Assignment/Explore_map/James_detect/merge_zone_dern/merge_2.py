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
import cv2
import sys
import threading

# =============================================================================
# ===== CONFIGURATION & PARAMETERS ============================================
# =============================================================================
# --- Robot Movement Config ---
SPEED_ROTATE = 480

# --- Sharp Distance Sensor Configuration ---
LEFT_SHARP_SENSOR_ID = 1
LEFT_SHARP_SENSOR_PORT = 1
LEFT_TARGET_CM = 13.0
RIGHT_SHARP_SENSOR_ID = 2
RIGHT_SHARP_SENSOR_PORT = 1
RIGHT_TARGET_CM = 13.0

# --- IR Sensor Configuration ---
LEFT_IR_SENSOR_ID = 1
LEFT_IR_SENSOR_PORT = 2
RIGHT_IR_SENSOR_ID = 2
RIGHT_IR_SENSOR_PORT = 2

# --- Sensor Thresholds ---
SHARP_WALL_THRESHOLD_CM = 60.0
SHARP_STDEV_THRESHOLD = 0.3
TOF_WALL_THRESHOLD_CM = 60.0 # From scanner
OBJECT_DETECTION_WALL_CHECK_CM = 80.0 # NEW: ToF distance to check for wall before object detection

# --- ToF Centering Configuration ---
TOF_ADJUST_SPEED = 0.1
TOF_CALIBRATION_SLOPE = 0.0894
TOF_CALIBRATION_Y_INTERCEPT = 3.8409

# --- Logical state for the grid map ---
CURRENT_POSITION = (4, 0)
CURRENT_DIRECTION = 1  # 0:North, 1:East, 2:South, 3:West
TARGET_DESTINATION = (4, 0)

# --- Physical state for the robot ---
CURRENT_TARGET_YAW = 0.0
ROBOT_FACE = 1  # 1,3,5.. = X axis, 2,4,6.. = Y axis

# --- IMU Drift Compensation Parameters ---
IMU_COMPENSATION_START_NODE_COUNT = 7
IMU_COMPENSATION_NODE_INTERVAL = 10
IMU_COMPENSATION_DEG_PER_INTERVAL = -2.0
IMU_DRIFT_COMPENSATION_DEG = 0.0

# --- Occupancy Grid Parameters ---
PROB_OCC_GIVEN_OCC = {'tof': 0.95, 'sharp': 0.90}
PROB_OCC_GIVEN_FREE = {'tof': 0.05, 'sharp': 0.10}
LOG_ODDS_OCC = {
    'tof': math.log(PROB_OCC_GIVEN_OCC['tof'] / (1 - PROB_OCC_GIVEN_OCC['tof'])),
    'sharp': math.log(PROB_OCC_GIVEN_OCC['sharp'] / (1 - PROB_OCC_GIVEN_OCC['sharp']))
}
LOG_ODDS_FREE = {
    'tof': math.log(PROB_OCC_GIVEN_FREE['tof'] / (1 - PROB_OCC_GIVEN_FREE['tof'])),
    'sharp': math.log(PROB_OCC_GIVEN_FREE['sharp'] / (1 - PROB_OCC_GIVEN_FREE['sharp']))
}
OCCUPANCY_THRESHOLD = 0.7
FREE_THRESHOLD = 0.3

# =============================================================================
# ===== HELPER FUNCTIONS (Movement & Sensors) =================================
# =============================================================================
def convert_adc_to_cm(adc_value):
    if adc_value <= 0: return float('inf')
    return 30263 * (adc_value ** -1.352)

def calibrate_tof_value(raw_tof_value):
    try:
        if raw_tof_value is None or raw_tof_value <= 0: return float('inf')
        return (TOF_CALIBRATION_SLOPE * raw_tof_value) + TOF_CALIBRATION_Y_INTERCEPT
    except Exception: return float('inf')

def get_compensated_target_yaw():
    return CURRENT_TARGET_YAW + IMU_DRIFT_COMPENSATION_DEG

# =============================================================================
# ===== NEW: IMAGE PROCESSING & OBJECT DETECTION (from zone.py) ===============
# =============================================================================
def apply_awb(bgr):
    # This function remains the same as in zone.py
    if hasattr(cv2, "xphoto") and hasattr(cv2.xphoto, "createLearningBasedWB"):
        wb = cv2.xphoto.createLearningBasedWB()
        try: wb.setSaturationThreshold(0.99)
        except Exception: pass
        return wb.balanceWhite(bgr)
    return bgr

def retinex_msrcp(bgr, sigmas=(15, 80, 250), eps=1e-6):
    # This function remains the same as in zone.py
    img = bgr.astype(np.float32) + 1.0; intensity = img.mean(axis=2)
    log_I = np.log(intensity + eps); msr = np.zeros_like(intensity, dtype=np.float32)
    for s in sigmas:
        blur = cv2.GaussianBlur(intensity, (0, 0), s)
        msr += (log_I - np.log(blur + eps))
    msr /= float(len(sigmas)); msr -= msr.min(); msr /= (msr.max() + eps)
    msr = (msr * 255.0).astype(np.float32); scale = (msr + 1.0) / (intensity + eps)
    out = np.clip(img * scale[..., None], 0, 255).astype(np.uint8)
    return out

def night_enhance_pipeline(bgr):
    # This function remains the same as in zone.py
    den = cv2.fastNlMeansDenoisingColored(bgr, None, 7, 7, 7, 21)
    awb = apply_awb(den); ret = retinex_msrcp(awb)
    return ret

class ObjectTracker:
    # This class is copied directly from zone.py with no changes
    def __init__(self, template_paths):
        print("🖼️  Loading and processing template images...")
        self.templates = self._load_templates(template_paths)
        if not self.templates: sys.exit("❌ Could not load template files.")
        print("✅ Templates loaded successfully.")

    def _load_templates(self, template_paths):
        processed_templates = {}
        for shape_name, path in template_paths.items():
            template_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if template_img is None: continue
            blurred = cv2.GaussianBlur(template_img, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours: processed_templates[shape_name] = max(contours, key=cv2.contourArea)
        return processed_templates

    def _get_angle(self, pt1, pt2, pt0):
        dx1 = pt1[0] - pt0[0]; dy1 = pt1[1] - pt0[1]
        dx2 = pt2[0] - pt0[0]; dy2 = pt2[1] - pt0[1]
        dot_product = dx1 * dx2 + dy1 * dy2; magnitude1 = math.sqrt(dx1 * dx1 + dy1 * dy1)
        magnitude2 = math.sqrt(dx2 * dx2 + dy2 * dy2)
        if magnitude1 * magnitude2 == 0: return 0
        angle_rad = math.acos(dot_product / (magnitude1 * magnitude2)); return math.degrees(angle_rad)

    def _get_raw_detections(self, frame):
        enhanced = night_enhance_pipeline(frame)
        enhanced = cv2.GaussianBlur(enhanced, (5, 5), 0)
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV); h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)); v_eq = clahe.apply(v)
        normalized_hsv = cv2.merge([h, s, v_eq])
        color_ranges = {'Red': [np.array([0, 80, 40]), np.array([10, 255, 255]), np.array([170, 80, 40]), np.array([180, 255, 255])],'Yellow': [np.array([20, 60, 40]), np.array([35, 255, 255])],'Green': [np.array([35, 40, 30]), np.array([85, 255, 255])],'Blue': [np.array([90, 40, 30]), np.array([130, 255, 255])]}
        masks = {}; lower1, upper1, lower2, upper2 = color_ranges['Red']
        masks['Red'] = cv2.inRange(normalized_hsv, lower1, upper1) | cv2.inRange(normalized_hsv, lower2, upper2)
        for name in ['Yellow', 'Green', 'Blue']: lower, upper = color_ranges[name]; masks[name] = cv2.inRange(normalized_hsv, lower, upper)
        combined_mask = masks['Red'] | masks['Yellow'] | masks['Green'] | masks['Blue']
        kernel = np.ones((5, 5), np.uint8)
        cleaned_mask = cv2.morphologyEx(cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1), cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); raw_detections = []
        
        # --- NEW: Get image dimensions for border check ---
        img_h, img_w = frame.shape[:2]

        for cnt in contours:
            # --- FILTER 1: Area (already exists) ---
            area = cv2.contourArea(cnt)
            if area < 1500: 
                continue

            # --- NEW FILTER 2: Aspect Ratio & Bounding Box ---
            # กรองวัตถุที่ยาวหรือแบนเกินไป
            x, y, w, h = cv2.boundingRect(cnt)
            if w == 0 or h == 0: continue
            aspect_ratio = w / float(h)
            if aspect_ratio > 4.0 or aspect_ratio < 0.25:
                # print(f"Filtered out by Aspect Ratio: {aspect_ratio:.2f}")
                continue

            # --- NEW FILTER 3: Solidity ---
            # กรองวัตถุที่ไม่ "ทึบ" (มีรูปร่างไม่สมบูรณ์)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0: continue
            solidity = float(area) / hull_area
            if solidity < 0.85:
                # print(f"Filtered out by Solidity: {solidity:.2f}")
                continue

            # --- NEW FILTER 4: Border Contact ---
            # กรองวัตถุที่สัมผัสขอบภาพ (มักจะเป็นขอบ ROI)
            # ใช้ margin เล็กน้อย (เช่น 2 pixels) เพื่อความแน่นอน
            if x <= 2 or y <= 2 or (x + w) >= (img_w - 2) or (y + h) >= (img_h - 2):
                # print(f"Filtered out by Border Contact")
                continue

            # --- If all filters pass, proceed with detection ---
            contour_mask = np.zeros(frame.shape[:2], dtype="uint8"); cv2.drawContours(contour_mask, [cnt], -1, 255, -1)
            max_mean, found_color = 0, "Unknown"
            for color_name, m in masks.items():
                mean_val = cv2.mean(m, mask=contour_mask)[0]
                if mean_val > max_mean: max_mean, found_color = mean_val, color_name
            if max_mean <= 20: continue

            shape = "Uncertain"; peri = cv2.arcLength(cnt, True); approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            circularity = (4 * np.pi * area) / (peri * peri) if peri > 0 else 0
            
            # ปรับปรุงการเช็ค Shape เล็กน้อยเพื่อให้แม่นยำขึ้น
            if circularity > 0.84: 
                shape = "Circle"
            elif len(approx) == 4 and solidity > 0.9: # เพิ่มเช็ค solidity สำหรับสี่เหลี่ยม
                points = [tuple(p[0]) for p in approx]
                angles = [self._get_angle(points[(i - 1) % 4], points[(i + 1) % 4], p) for i, p in enumerate(points)]
                if all(75 <= ang <= 105 for ang in angles):
                    _, (rect_w, rect_h), _ = cv2.minAreaRect(cnt)
                    if rect_w == 0 or rect_h == 0: continue
                    aspect_ratio_rect = max(rect_w, rect_h) / min(rect_w, rect_h)
                    if 0.90 <= aspect_ratio_rect <= 1.10:
                        shape = "Square"
                    elif rect_w > rect_h:
                        shape = "Rectangle_H"
                    else:
                        shape = "Rectangle_V"

            raw_detections.append({'contour': cnt, 'shape': shape, 'color': found_color})
            
        return raw_detections

# =============================================================================
# ===== OCCUPANCY GRID MAP & VISUALIZATION (HEAVILY MODIFIED) =================
# =============================================================================
class WallBelief:
    # This class remains the same
    def __init__(self): self.log_odds = 0.0
    def update(self, is_occupied_reading, sensor_type):
        if is_occupied_reading: self.log_odds += LOG_ODDS_OCC[sensor_type]
        else: self.log_odds += LOG_ODDS_FREE[sensor_type]
        self.log_odds = max(min(self.log_odds, 10), -10)
    def get_probability(self): return 1.0 - 1.0 / (1.0 + math.exp(self.log_odds))
    def is_occupied(self): return self.get_probability() > OCCUPANCY_THRESHOLD

class OGMCell:
    # MODIFIED: Added self.objects to store detected object info
    def __init__(self):
        self.log_odds_occupied = 0.0
        self.walls = {'N': None, 'E': None, 'S': None, 'W': None}
        self.objects = [] # NEW: List to store {'shape': str, 'color': str, 'zone': str, 'is_target': bool}

    def get_node_probability(self): return 1.0 - 1.0 / (1.0 + math.exp(self.log_odds_occupied))
    def is_node_occupied(self): return self.get_node_probability() > OCCUPANCY_THRESHOLD

class OccupancyGridMap:
    # MODIFIED: Added a method to add objects to a cell
    def __init__(self, width, height):
        self.width = width; self.height = height
        self.grid = [[OGMCell() for _ in range(width)] for _ in range(height)]
        self._link_walls()

    def _link_walls(self):
        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r][c].walls['N'] is None:
                    wall = WallBelief(); self.grid[r][c].walls['N'] = wall
                    if r > 0: self.grid[r-1][c].walls['S'] = wall
                if self.grid[r][c].walls['W'] is None:
                    wall = WallBelief(); self.grid[r][c].walls['W'] = wall
                    if c > 0: self.grid[r][c-1].walls['E'] = wall
                if self.grid[r][c].walls['S'] is None: self.grid[r][c].walls['S'] = WallBelief()
                if self.grid[r][c].walls['E'] is None: self.grid[r][c].walls['E'] = WallBelief()

    def update_wall(self, r, c, direction_char, is_occupied_reading, sensor_type):
        if 0 <= r < self.height and 0 <= c < self.width:
            wall = self.grid[r][c].walls.get(direction_char)
            if wall: wall.update(is_occupied_reading, sensor_type)

    def update_node(self, r, c, is_occupied_reading, sensor_type='tof'):
        if 0 <= r < self.height and 0 <= c < self.width:
            if is_occupied_reading: self.grid[r][c].log_odds_occupied += LOG_ODDS_OCC[sensor_type]
            else: self.grid[r][c].log_odds_occupied += LOG_ODDS_FREE[sensor_type]

    # NEW: Method to add a detected object to a specific grid cell
    def add_object_to_cell(self, r, c, object_info):
        if 0 <= r < self.height and 0 <= c < self.width:
            self.grid[r][c].objects.append(object_info)
            print(f"🗺️  Object '{object_info['color']} {object_info['shape']}' added to map at ({r}, {c})")

    def is_path_clear(self, r1, c1, r2, c2):
        dr, dc = r2 - r1, c2 - c1;
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
    # HEAVILY MODIFIED: Now displays both the map and the last captured image
    def __init__(self, grid_size, target_dest=None, target_object_info=None):
        self.grid_size = grid_size
        self.target_dest = target_dest
        self.target_object_info = target_object_info
        plt.ion()
        self.fig = plt.figure(figsize=(10, 14))
        self.ax_map = self.fig.add_subplot(2, 1, 1)
        self.ax_image = self.fig.add_subplot(2, 1, 2)
        self.colors = {"robot": "#0000FF", "target": "#FFD700", "path": "#FFFF00", "wall": "#000000", "wall_prob": "#000080"}

        # --- MODIFIED: Separate shape markers from colors ---
        self.shape_to_marker = {
            'Circle': 'o',
            'Square': 's',
            'Rectangle_H': '<',
            'Rectangle_V': '^',
        }
        self.color_to_code = {
            'Red': 'red',
            'Blue': 'blue',
            'Green': 'green',
            'Yellow': 'gold', # Use 'gold' for better visibility than 'y'
        }
        # --- END MODIFICATION ---

    def update_plot(self, occupancy_map, robot_pos, last_processed_image=None, path=None):
        # --- 1. Update Map Plot (ax_map) ---
        self.ax_map.clear()
        self.ax_map.set_title("Real-time Hybrid Belief Map (Nodes, Walls & Objects)")
        self.ax_map.set_xticks([]); self.ax_map.set_yticks([])
        self.ax_map.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax_map.set_ylim(self.grid_size - 0.5, -0.5)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                prob = occupancy_map.grid[r][c].get_node_probability()
                if prob > OCCUPANCY_THRESHOLD: color = '#8B0000'
                elif prob < FREE_THRESHOLD: color = '#D3D3D3'
                else: color = '#90EE90'
                self.ax_map.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=color, edgecolor='k', lw=0.5))
                self.ax_map.text(c, r, f"{prob:.2f}", ha="center", va="center", color="black", fontsize=8)
                
                # --- MODIFIED: Draw objects using separate shape and color ---
                for obj in occupancy_map.grid[r][c].objects:
                    shape_marker = self.shape_to_marker.get(obj['shape'], '*') # Default to star
                    marker_color = self.color_to_code.get(obj['color'], 'black')
                    offset_x = -0.25 if 'Left' in obj['zone'] else 0.25 if 'Right' in obj['zone'] else 0
                    ms = 12 if obj['is_target'] else 8
                    self.ax_map.plot(c + offset_x, r, 
                                     marker=shape_marker, 
                                     color=marker_color,
                                     markersize=ms, 
                                     markeredgecolor='black',
                                     linestyle='None') # Add this to only show the marker
                # --- END MODIFICATION ---

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell = occupancy_map.grid[r][c]
                if cell.walls['N'].is_occupied(): self.ax_map.plot([c - 0.5, c + 0.5], [r - 0.5, r - 0.5], color=self.colors['wall'], linewidth=4)
                if cell.walls['W'].is_occupied(): self.ax_map.plot([c - 0.5, c - 0.5], [r - 0.5, r + 0.5], color=self.colors['wall'], linewidth=4)
                if r == self.grid_size - 1 and cell.walls['S'].is_occupied(): self.ax_map.plot([c - 0.5, c + 0.5], [r + 0.5, r + 0.5], color=self.colors['wall'], linewidth=4)
                if c == self.grid_size - 1 and cell.walls['E'].is_occupied(): self.ax_map.plot([c + 0.5, c + 0.5], [r - 0.5, r + 0.5], color=self.colors['wall'], linewidth=4)
        
        if self.target_dest: self.ax_map.add_patch(plt.Rectangle((self.target_dest[1] - 0.5, self.target_dest[0] - 0.5), 1, 1, facecolor=self.colors['target'], edgecolor='k', lw=2, alpha=0.8))
        if path:
            for r_p, c_p in path: self.ax_map.add_patch(plt.Rectangle((c_p - 0.5, r_p - 0.5), 1, 1, facecolor=self.colors['path'], edgecolor='k', lw=0.5, alpha=0.7))
        if robot_pos: self.ax_map.add_patch(plt.Rectangle((robot_pos[1] - 0.5, robot_pos[0] - 0.5), 1, 1, facecolor=self.colors['robot'], edgecolor='k', lw=2))

        # --- MODIFIED: Create legend using separate shape and color ---
        target_shape_marker = self.shape_to_marker.get(self.target_object_info['shape'], '*')
        target_marker_color = self.color_to_code.get(self.target_object_info['color'], 'black')
        
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor='#8B0000', label=f'Node Occupied'),
            plt.Rectangle((0,0),1,1, facecolor='#90EE90', label=f'Node Unknown'),
            plt.Rectangle((0,0),1,1, facecolor='#D3D3D3', label=f'Node Free'),
            plt.Line2D([0], [0], color=self.colors['wall'], lw=4, label='Wall Occupied'),
            plt.Rectangle((0,0),1,1, facecolor=self.colors['robot'], label='Robot'),
            plt.Rectangle((0,0),1,1, facecolor=self.colors['target'], label='Target Dest.'),
            plt.Line2D([0], [0], marker=target_shape_marker, color='w', markerfacecolor=target_marker_color, label=f"Target Object", markersize=12, markeredgecolor='k'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', label=f"Other Object", markersize=8, markeredgecolor='k'),
        ]
        # --- END MODIFICATION ---
        self.ax_map.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1.0))

        self.ax_image.clear()
        self.ax_image.set_title("Last Object Scan Result")
        self.ax_image.axis('off')
        if last_processed_image is not None:
            img_rgb = cv2.cvtColor(last_processed_image, cv2.COLOR_BGR2RGB)
            self.ax_image.imshow(img_rgb)
        else:
            self.ax_image.text(0.5, 0.5, "No image processed yet", ha="center", va="center", fontsize=16)

        self.fig.tight_layout(rect=[0, 0, 0.85, 1])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

# =============================================================================
# ===== CORE ROBOT CONTROL CLASSES (Unchanged from debug_img_3-10) ============
# =============================================================================
class AttitudeHandler:
    # This class remains the same
    def __init__(self): self.current_yaw, self.yaw_tolerance, self.is_monitoring = 0.0, 3.0, False
    def attitude_handler(self, attitude_info):
        if self.is_monitoring: self.current_yaw = attitude_info[0]
    def start_monitoring(self, chassis): self.is_monitoring = True; chassis.sub_attitude(freq=20, callback=self.attitude_handler)
    def stop_monitoring(self, chassis):
        self.is_monitoring = False;
        try: chassis.unsub_attitude()
        except Exception: pass
    def normalize_angle(self, angle):
        while angle > 180: angle -= 360
        while angle <= -180: angle += 360
        return angle
    def correct_yaw_to_target(self, chassis, target_yaw=0.0):
        normalized_target = self.normalize_angle(target_yaw); time.sleep(0.1)
        robot_rotation = -self.normalize_angle(normalized_target - self.current_yaw)
        print(f"\n🔧 Correcting Yaw: {self.current_yaw:.1f}° -> {target_yaw:.1f}°. Rotating: {robot_rotation:.1f}°")
        if abs(robot_rotation) > self.yaw_tolerance:
            chassis.move(x=0, y=0, z=robot_rotation, z_speed=60).wait_for_completed(timeout=2); time.sleep(0.1)
        final_error = abs(self.normalize_angle(normalized_target - self.current_yaw))
        if final_error <= self.yaw_tolerance: print(f"✅ Yaw Correction Success: {self.current_yaw:.1f}°"); return True
        print(f"⚠️ First attempt incomplete. Current: {self.current_yaw:.1f}°. Fine-tuning...")
        remaining_rotation = -self.normalize_angle(normalized_target - self.current_yaw)
        if abs(remaining_rotation) > 0.5 and abs(remaining_rotation) < 20:
            chassis.move(x=0, y=0, z=remaining_rotation, z_speed=40).wait_for_completed(timeout=2); time.sleep(0.1)
        final_error = abs(self.normalize_angle(normalized_target - self.current_yaw))
        if final_error <= self.yaw_tolerance: print(f"✅ Yaw Fine-tuning Success: {self.current_yaw:.1f}°"); return True
        else: print(f"🔥🔥 Yaw Correction FAILED. Final Yaw: {self.current_yaw:.1f}°"); return False

class PID:
    # This class remains the same
    def __init__(self, Kp, Ki, Kd, setpoint=0):
        self.Kp, self.Ki, self.Kd, self.setpoint = Kp, Ki, Kd, setpoint
        self.prev_error, self.integral, self.integral_max = 0, 0, 1.0
    def compute(self, current, dt):
        error = self.setpoint - current; self.integral += error * dt
        self.integral = max(min(self.integral, self.integral_max), -self.integral_max)
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error; return output

class MovementController:
    # This class remains the same
    def __init__(self, chassis):
        self.chassis = chassis
        self.current_x_pos, self.current_y_pos = 0.0, 0.0
        self.chassis.sub_position(freq=20, callback=self.position_handler)
    def position_handler(self, position_info): self.current_x_pos, self.current_y_pos = position_info[0], position_info[1]
    def _calculate_yaw_correction(self, attitude_handler, target_yaw):
        KP_YAW = 1.8; MAX_YAW_SPEED = 25
        yaw_error = attitude_handler.normalize_angle(target_yaw - attitude_handler.current_yaw)
        speed = KP_YAW * yaw_error; return max(min(speed, MAX_YAW_SPEED), -MAX_YAW_SPEED)
    def move_forward_one_grid(self, axis, attitude_handler):
        attitude_handler.correct_yaw_to_target(self.chassis, get_compensated_target_yaw())
        target_distance = 0.6; pid = PID(Kp=1.8, Ki=0.25, Kd=12, setpoint=target_distance)
        start_time, last_time = time.time(), time.time()
        start_position = self.current_x_pos if axis == 'x' else self.current_y_pos
        print(f"🚀 Moving FORWARD 0.6m, monitoring GLOBAL AXIS '{axis}'")
        while time.time() - start_time < 3.5:
            now = time.time(); dt = now - last_time; last_time = now
            current_position = self.current_x_pos if axis == 'x' else self.current_y_pos
            relative_position = abs(current_position - start_position)
            if abs(relative_position - target_distance) < 0.03: print("\n✅ Move complete!"); break
            output = pid.compute(relative_position, dt)
            ramp_multiplier = min(1.0, 0.1 + ((now - start_time) / 1.0) * 0.9)
            speed = max(-1.0, min(1.0, output * ramp_multiplier))
            yaw_correction = self._calculate_yaw_correction(attitude_handler, get_compensated_target_yaw())
            self.chassis.drive_speed(x=speed, y=0, z=yaw_correction, timeout=1)
            print(f"Moving... Dist: {relative_position:.3f}/{target_distance:.2f} m", end='\r')
        self.chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0); time.sleep(0.5)
    def adjust_position_to_wall(self, sensor_adaptor, attitude_handler, side, sensor_config, target_distance_cm, direction_multiplier):
        compensated_yaw = get_compensated_target_yaw()
        print(f"\n--- Adjusting {side} Side (Yaw locked at {compensated_yaw:.2f}°) ---")
        TOLERANCE_CM, MAX_EXEC_TIME, KP_SLIDE, MAX_SLIDE_SPEED = 0.8, 8, 0.045, 0.18
        start_time = time.time()
        while time.time() - start_time < MAX_EXEC_TIME:
            adc_val = sensor_adaptor.get_adc(id=sensor_config["sharp_id"], port=sensor_config["sharp_port"])
            current_dist = convert_adc_to_cm(adc_val); dist_error = target_distance_cm - current_dist
            if abs(dist_error) <= TOLERANCE_CM: print(f"\n[{side}] Target distance reached! Final distance: {current_dist:.2f} cm"); break
            slide_speed = max(min(direction_multiplier * KP_SLIDE * dist_error, MAX_SLIDE_SPEED), -MAX_SLIDE_SPEED)
            yaw_correction = self._calculate_yaw_correction(attitude_handler, compensated_yaw)
            self.chassis.drive_speed(x=0, y=slide_speed, z=yaw_correction)
            print(f"Adjusting {side}... Current: {current_dist:5.2f}cm, Target: {target_distance_cm:4.1f}cm", end='\r')
            time.sleep(0.02)
        else: print(f"\n[{side}] Movement timed out!")
        self.chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0); time.sleep(0.1)
    def center_in_node_with_tof(self, scanner, attitude_handler, target_cm=19, tol_cm=1.0, max_adjust_time=6.0):
        if scanner.is_performing_full_scan: print("[ToF Centering] SKIPPED: A critical side-scan is in progress."); return
        print("\n--- Stage: Centering in Node with ToF ---"); time.sleep(0.1)
        tof_dist = scanner.last_tof_distance_cm
        if tof_dist is None or math.isinf(tof_dist): print("[ToF] ❌ No valid ToF data available. Skipping centering."); return
        if tof_dist >= 50: print("[ToF] Distance >= 50cm, likely open space. Skipping centering."); return
        direction = 0
        if tof_dist > target_cm + tol_cm: direction = abs(TOF_ADJUST_SPEED)
        elif tof_dist < 22: direction = -abs(TOF_ADJUST_SPEED)
        else: direction = abs(TOF_ADJUST_SPEED)
        if direction == 0: print(f"[ToF] Already centered within tolerance ({tof_dist:.2f} cm). Skipping."); return
        start_time = time.time(); compensated_yaw = get_compensated_target_yaw()
        while time.time() - start_time < max_adjust_time:
            yaw_correction = self._calculate_yaw_correction(attitude_handler, compensated_yaw)
            self.chassis.drive_speed(x=direction, y=0, z=yaw_correction, timeout=0.1); time.sleep(0.12)
            self.chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0); time.sleep(0.08)
            current_tof = scanner.last_tof_distance_cm
            if current_tof is None or math.isinf(current_tof): continue
            print(f"[ToF] Adjusting... Current Distance: {current_tof:.2f} cm", end="\r")
            if abs(current_tof - target_cm) <= tol_cm: print(f"\n[ToF] ✅ Centering complete. Final distance: {current_tof:.2f} cm"); break
            if (direction > 0 and current_tof < target_cm - tol_cm) or (direction < 0 and current_tof > target_cm + tol_cm):
                direction *= -1; print("\n[ToF] 🔄 Overshot target. Reversing direction.")
        else: print(f"\n[ToF] ⚠️ Centering timed out. Final distance: {scanner.last_tof_distance_cm:.2f} cm")
        self.chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0); time.sleep(0.1)
    def rotate_to_direction(self, target_direction, attitude_handler):
        global CURRENT_DIRECTION
        if CURRENT_DIRECTION == target_direction: return
        diff = (target_direction - CURRENT_DIRECTION + 4) % 4
        if diff == 1: self.rotate_90_degrees_right(attitude_handler)
        elif diff == 3: self.rotate_90_degrees_left(attitude_handler)
        elif diff == 2: self.rotate_90_degrees_right(attitude_handler); self.rotate_90_degrees_right(attitude_handler)
    def rotate_90_degrees_right(self, attitude_handler):
        global CURRENT_TARGET_YAW, CURRENT_DIRECTION, ROBOT_FACE
        print("🔄 Rotating 90° RIGHT..."); CURRENT_TARGET_YAW = attitude_handler.normalize_angle(CURRENT_TARGET_YAW + 90)
        attitude_handler.correct_yaw_to_target(self.chassis, get_compensated_target_yaw())
        CURRENT_DIRECTION = (CURRENT_DIRECTION + 1) % 4; ROBOT_FACE += 1
    def rotate_90_degrees_left(self, attitude_handler):
        global CURRENT_TARGET_YAW, CURRENT_DIRECTION, ROBOT_FACE
        print("🔄 Rotating 90° LEFT..."); CURRENT_TARGET_YAW = attitude_handler.normalize_angle(CURRENT_TARGET_YAW - 90)
        attitude_handler.correct_yaw_to_target(self.chassis, get_compensated_target_yaw())
        CURRENT_DIRECTION = (CURRENT_DIRECTION - 1 + 4) % 4; ROBOT_FACE -= 1
        if ROBOT_FACE < 1: ROBOT_FACE += 4
    def cleanup(self):
        try: self.chassis.unsub_position()
        except Exception: pass

class EnvironmentScanner:
    # This class remains the same
    def __init__(self, sensor_adaptor, tof_sensor, gimbal, chassis):
        self.sensor_adaptor, self.tof_sensor, self.gimbal, self.chassis = sensor_adaptor, tof_sensor, gimbal, chassis
        self.tof_wall_threshold_cm = TOF_WALL_THRESHOLD_CM; self.last_tof_distance_cm = float('inf')
        self.side_tof_reading_cm = float('inf'); self.is_gimbal_centered = True
        self.is_performing_full_scan = False; self.tof_sensor.sub_distance(freq=10, callback=self._tof_data_handler)
        self.side_sensors = {"Left": {"sharp_id": 1, "sharp_port": 1, "ir_id": 1, "ir_port": 2},"Right": {"sharp_id": 2, "sharp_port": 1, "ir_id": 2, "ir_port": 2}}
    def _tof_data_handler(self, sub_info):
        calibrated_cm = calibrate_tof_value(sub_info[0])
        if self.is_gimbal_centered: self.last_tof_distance_cm = calibrated_cm
        else: self.side_tof_reading_cm = calibrated_cm
    def _get_stable_reading_cm(self, side, duration=0.35):
        sensor_info = self.side_sensors.get(side);
        if not sensor_info: return None, None
        readings = []; start_time = time.time()
        while time.time() - start_time < duration:
            adc = self.sensor_adaptor.get_adc(id=sensor_info["sharp_id"], port=sensor_info["sharp_port"])
            readings.append(convert_adc_to_cm(adc)); time.sleep(0.05)
        if len(readings) < 5: return None, None
        return statistics.mean(readings), statistics.stdev(readings)
    def get_sensor_readings(self):
        self.is_performing_full_scan = True
        try:
            self.gimbal.moveto(pitch=0, yaw=0, yaw_speed=SPEED_ROTATE).wait_for_completed(); time.sleep(0.15)
            readings = {}; readings['front'] = (self.last_tof_distance_cm < self.tof_wall_threshold_cm)
            print(f"[SCAN] Front (ToF): {self.last_tof_distance_cm:.1f}cm -> {'OCCUPIED' if readings['front'] else 'FREE'}")
            for side in ["Left", "Right"]:
                avg_dist, std_dev = self._get_stable_reading_cm(side)
                if avg_dist is None: readings[side.lower()] = False; continue
                is_sharp_detecting_wall = (avg_dist < SHARP_WALL_THRESHOLD_CM and std_dev < SHARP_STDEV_THRESHOLD)
                ir_value = self.sensor_adaptor.get_io(id=self.side_sensors[side]["ir_id"], port=self.side_sensors[side]["ir_port"])
                is_ir_detecting_wall = (ir_value == 0)
                print(f"\n[SCAN] {side} Side Analysis: Sharp suggests {'WALL' if is_sharp_detecting_wall else 'FREE'}, IR suggests {'WALL' if is_ir_detecting_wall else 'FREE'}")
                if is_sharp_detecting_wall == is_ir_detecting_wall: is_wall = is_sharp_detecting_wall
                else:
                    print("    -> Ambiguity detected! Confirming with ToF...")
                    target_gimbal_yaw = -90 if side == "Left" else 90
                    try:
                        self.is_gimbal_centered = False
                        self.gimbal.moveto(pitch=0, yaw=target_gimbal_yaw, yaw_speed=SPEED_ROTATE).wait_for_completed(); time.sleep(0.1)
                        is_wall = (self.side_tof_reading_cm < self.tof_wall_threshold_cm)
                        print(f"    -> ToF Confirmation at {target_gimbal_yaw}°: {'WALL' if is_wall else 'NO WALL'}")
                    finally:
                        self.gimbal.moveto(pitch=0, yaw=0, yaw_speed=SPEED_ROTATE).wait_for_completed()
                        self.is_gimbal_centered = True; time.sleep(0.1)
                readings[side.lower()] = is_wall
            return readings
        finally: self.is_performing_full_scan = False
    def get_front_tof_cm(self):
        self.gimbal.moveto(pitch=0, yaw=0, yaw_speed=SPEED_ROTATE).wait_for_completed(); time.sleep(0.1)
        return self.last_tof_distance_cm
    def cleanup(self):
        try: self.tof_sensor.unsub_distance()
        except Exception: pass

# =============================================================================
# ===== NEW: MULTI-THREADED OBJECT DETECTION MANAGER ==========================
# =============================================================================
class ObjectDetectionManager:
    def __init__(self, ep_camera, object_tracker, target_info, total_targets_to_find, occupancy_map, visualizer):
        self.camera = ep_camera
        self.tracker = object_tracker
        self.target_shape = target_info['shape']
        self.target_color = target_info['color']
        self.total_targets = total_targets_to_find
        self.occupancy_map = occupancy_map
        self.visualizer = visualizer
        
        self.detection_thread = None
        self.is_running = False
        self.lock = threading.Lock()
        
        # Shared resources that the thread will update
        self.last_processed_image = None
        self.targets_found_count = 0
        self.detection_in_progress = False

    def start(self):
        self.is_running = True
        print("👁️  Object Detection Manager started.")

    def stop(self):
        self.is_running = False
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join()
        print("👁️  Object Detection Manager stopped.")

    def needs_to_scan(self):
        return self.targets_found_count < self.total_targets

    def _detection_thread_worker(self, current_pos, current_dir, front_tof_dist_cm):
        with self.lock:
            self.detection_in_progress = True
        
        print("\n--- 📸 Starting Object Detection Scan (in thread) ---")
        try:
            # 1. Capture Image
            print("   -> Starting camera stream...")
            self.camera.start_video_stream(display=False, resolution="720p")
            time.sleep(2) # Stabilize
            captured_frame = self.camera.read_cv2_image(timeout=5)
            print("   -> Frame captured.")
        except Exception as e:
            print(f"   -> ❌ Error capturing frame: {e}")
            captured_frame = None
        finally:
            print("   -> Stopping camera stream.")
            self.camera.stop_video_stream()

        if captured_frame is None:
            with self.lock:
                self.detection_in_progress = False
            return
            
        # 2. Process Image (based on zone.py logic)
        output_image, detections = self._scan_and_process_frame(captured_frame)
        with self.lock:
            self.last_processed_image = output_image # Store image for visualizer

        # 3. Analyze detections and update map
        print("   -> Analyzing detections and updating map...")
        wall_is_close = front_tof_dist_cm < OBJECT_DETECTION_WALL_CHECK_CM
        dir_vectors = [(-1, 0), (0, 1), (1, 0), (0, -1)] # N, E, S, W
        dr, dc = dir_vectors[current_dir]
        target_r, target_c = current_pos[0] + dr, current_pos[1] + dc
        
        for det in detections:
            # Logic: If wall is far, ignore 'Center' zone.
            if not wall_is_close and 'Center' in det['zone']:
                print(f"   -> Ignoring Center object ({det['color']} {det['shape']}) because front wall is far ({front_tof_dist_cm:.1f}cm).")
                continue

            # Logic: Save confident objects and uncertain objects from side zones
            if det['shape'] != 'Uncertain' or ('Left' in det['zone'] or 'Right' in det['zone']):
                self.occupancy_map.add_object_to_cell(target_r, target_c, det)
                if det['is_target']:
                    with self.lock:
                        if self.targets_found_count < self.total_targets:
                            self.targets_found_count += 1
                            print(f"   -> 🎉 TARGET FOUND! ({self.targets_found_count}/{self.total_targets})")

        print("--- ✅ Object Detection Scan Complete ---")
        with self.lock:
            self.detection_in_progress = False

    def trigger_scan(self, current_pos, current_dir, front_tof_dist_cm):
        if not self.is_running or self.detection_in_progress:
            print("👁️  Skipping scan: Manager not running or a scan is already in progress.")
            return

        # Start the detection process in a new thread
        self.detection_thread = threading.Thread(target=self._detection_thread_worker, args=(current_pos, current_dir, front_tof_dist_cm))
        self.detection_thread.start()

    def _scan_and_process_frame(self, frame):
        # This is a modified version of scan_and_process_maze_wall from zone.py
        if frame is None: return None, []
        
        frame = frame[352:352+360, 14:14+1215]
        output_image = frame.copy(); height, width, _ = frame.shape
        DIVIDER_X1, DIVIDER_X2 = 400, 800
        cv2.line(output_image, (DIVIDER_X1, 0), (DIVIDER_X1, height), (255, 255, 0), 2)
        cv2.line(output_image, (DIVIDER_X2, 0), (DIVIDER_X2, height), (255, 255, 0), 2)

        raw_detections = self.tracker._get_raw_detections(frame)
        processed_results = []

        for det in raw_detections:
            shape, color, contour = det['shape'], det['color'], det['contour']
            x, y, w, h = cv2.boundingRect(contour)
            obj_start_x, obj_end_x = x, x + w

            zone_label = "Unknown"
            if obj_end_x < DIVIDER_X1: zone_label = "Left"
            elif obj_start_x >= DIVIDER_X2: zone_label = "Right"
            elif obj_start_x >= DIVIDER_X1 and obj_end_x < DIVIDER_X2: zone_label = "Center"
            
            is_target = (shape == self.target_shape and color == self.target_color)
            box_color = (0, 0, 255) if is_target else (0, 255, 0)
            
            processed_results.append({ "color": color, "shape": shape, "zone": zone_label, "is_target": is_target })

            cv2.rectangle(output_image, (x, y), (x + w, y + h), box_color, 3)
            label_text = f"{color} {shape}" + (" (TARGET)" if is_target else "")
            cv2.putText(output_image, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
            
        return output_image, processed_results

# =============================================================================
# ===== PATHFINDING & EXPLORATION LOGIC (MODIFIED for object detection) =======
# =============================================================================
def find_path_bfs(occupancy_map, start, end):
    # This function remains the same
    queue = deque([[start]]); visited = {start}
    moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    while queue:
        path = queue.popleft()
        r, c = path[-1]
        if (r, c) == end: return path
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < occupancy_map.height and 0 <= nc < occupancy_map.width:
                if occupancy_map.is_path_clear(r, c, nr, nc) and (nr, nc) not in visited:
                    visited.add((nr, nc)); new_path = list(path); new_path.append((nr, nc)); queue.append(new_path)
    return None

def find_nearest_unvisited_path(occupancy_map, start_pos, visited_cells):
    # This function remains the same
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

def execute_path(path, movement_controller, attitude_handler, scanner, visualizer, object_detector, occupancy_map, path_name="Backtrack"):
    # MODIFIED: Pass object_detector to update visualizer correctly
    global CURRENT_POSITION
    print(f"🎯 Executing {path_name} Path: {path}")
    dir_vectors_map = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}
    dir_map_abs_char = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}
    for i in range(len(path) - 1):
        visualizer.update_plot(occupancy_map, path[i], object_detector.last_processed_image, path)
        current_r, current_c = path[i]
        if i + 1 < len(path):
            next_r, next_c = path[i+1]; dr, dc = next_r - current_r, next_c - current_c
            target_direction = dir_vectors_map[(dr, dc)]
            movement_controller.rotate_to_direction(target_direction, attitude_handler)
            print(f"   -> [{path_name}] Confirming path to ({next_r},{next_c}) with ToF...")
            scanner.gimbal.moveto(pitch=0, yaw=0, yaw_speed=SPEED_ROTATE).wait_for_completed(); time.sleep(0.15)
            is_blocked = scanner.get_front_tof_cm() < scanner.tof_wall_threshold_cm
            occupancy_map.update_wall(current_r, current_c, dir_map_abs_char[CURRENT_DIRECTION], is_blocked, 'tof')
            print(f"   -> [{path_name}] Real-time ToF check: Path is {'BLOCKED' if is_blocked else 'CLEAR'}.")
            visualizer.update_plot(occupancy_map, CURRENT_POSITION, object_detector.last_processed_image)
            if is_blocked:
                print(f"   -> 🔥 [{path_name}] IMMEDIATE STOP. Real-time sensor detected an obstacle. Aborting path.")
                break
            axis_to_monitor = 'x' if ROBOT_FACE % 2 != 0 else 'y'
            movement_controller.move_forward_one_grid(axis=axis_to_monitor, attitude_handler=attitude_handler)
            movement_controller.center_in_node_with_tof(scanner, attitude_handler)
            CURRENT_POSITION = (next_r, next_c)
            visualizer.update_plot(occupancy_map, CURRENT_POSITION, object_detector.last_processed_image, path)
            print(f"   -> [{path_name}] Performing side alignment at new position {CURRENT_POSITION}")
            perform_side_alignment_and_mapping(movement_controller, scanner, attitude_handler, occupancy_map, visualizer, object_detector)
    print(f"✅ {path_name} complete.")

def perform_side_alignment_and_mapping(movement_controller, scanner, attitude_handler, occupancy_map, visualizer, object_detector):
    # MODIFIED: Pass object_detector to update visualizer correctly
    print("\n--- Stage: Wall Detection & Side Alignment ---")
    r, c = CURRENT_POSITION; dir_map_abs_char = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}
    side_walls_present = scanner.get_sensor_readings()
    left_dir_abs = (CURRENT_DIRECTION - 1 + 4) % 4
    occupancy_map.update_wall(r, c, dir_map_abs_char[left_dir_abs], side_walls_present['left'], 'sharp')
    visualizer.update_plot(occupancy_map, CURRENT_POSITION, object_detector.last_processed_image)
    if side_walls_present['left']:
        movement_controller.adjust_position_to_wall(scanner.sensor_adaptor, attitude_handler, "Left", scanner.side_sensors["Left"], LEFT_TARGET_CM, direction_multiplier=1)
    right_dir_abs = (CURRENT_DIRECTION + 1) % 4
    occupancy_map.update_wall(r, c, dir_map_abs_char[right_dir_abs], side_walls_present['right'], 'sharp')
    visualizer.update_plot(occupancy_map, CURRENT_POSITION, object_detector.last_processed_image)
    if side_walls_present['right']:
        movement_controller.adjust_position_to_wall(scanner.sensor_adaptor, attitude_handler, "Right", scanner.side_sensors["Right"], RIGHT_TARGET_CM, direction_multiplier=-1)
    if not side_walls_present['left'] and not side_walls_present['right']: print("\n⚠️  WARNING: No side walls detected. Skipping alignment.")
    attitude_handler.correct_yaw_to_target(movement_controller.chassis, get_compensated_target_yaw()); time.sleep(0.1)

def explore_with_ogm(scanner, movement_controller, attitude_handler, object_detector, occupancy_map, visualizer, max_steps=40):
    # MODIFIED: Added object_detector interaction
    global CURRENT_POSITION, CURRENT_DIRECTION, IMU_DRIFT_COMPENSATION_DEG
    visited_cells = set()
    for step in range(max_steps):
        r, c = CURRENT_POSITION
        print(f"\n--- Step {step + 1} at {CURRENT_POSITION}, Facing: {['N', 'E', 'S', 'W'][CURRENT_DIRECTION]} ---")
        attitude_handler.correct_yaw_to_target(movement_controller.chassis, get_compensated_target_yaw())
        
        # Perform side alignment first
        perform_side_alignment_and_mapping(movement_controller, scanner, attitude_handler, occupancy_map, visualizer, object_detector)

        # Update current node as visited
        occupancy_map.update_node(r, c, False, 'tof')
        if (r, c) not in visited_cells:
            # --- NEW: OBJECT DETECTION LOGIC ---
            if object_detector.needs_to_scan():
                print("\n--- Checking for objects in front ---")
                front_dist = scanner.get_front_tof_cm()
                object_detector.trigger_scan(CURRENT_POSITION, CURRENT_DIRECTION, front_dist)
                # Give the thread a moment to start and update the visualizer
                time.sleep(1) 
                visualizer.update_plot(occupancy_map, CURRENT_POSITION, object_detector.last_processed_image)
            else:
                print("\n[Object Scan] Skipped: All targets have been found.")
            # --- END OF NEW LOGIC ---
        
        visited_cells.add((r, c))

        # IMU drift compensation
        nodes_visited = len(visited_cells)
        if nodes_visited >= IMU_COMPENSATION_START_NODE_COUNT:
            compensation_intervals = nodes_visited // IMU_COMPENSATION_NODE_INTERVAL
            new_compensation = compensation_intervals * IMU_COMPENSATION_DEG_PER_INTERVAL
            if new_compensation != IMU_DRIFT_COMPENSATION_DEG:
                IMU_DRIFT_COMPENSATION_DEG = new_compensation
                print(f"🔩 IMU Drift Compensation Updated: Visited {nodes_visited} nodes. New offset is {IMU_DRIFT_COMPENSATION_DEG:.1f}°")

        # Decision making for next move
        priority_dirs = [(CURRENT_DIRECTION + 1) % 4, CURRENT_DIRECTION, (CURRENT_DIRECTION - 1 + 4) % 4]
        moved = False; dir_vectors = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        for target_dir in priority_dirs:
            target_r, target_c = r + dir_vectors[target_dir][0], c + dir_vectors[target_dir][1]
            if occupancy_map.is_path_clear(r, c, target_r, target_c) and (target_r, target_c) not in visited_cells:
                print(f"Path to {['N','E','S','W'][target_dir]} at ({target_r},{target_c}) seems clear. Attempting move.")
                movement_controller.rotate_to_direction(target_dir, attitude_handler)
                scanner.gimbal.moveto(pitch=0, yaw=0, yaw_speed=SPEED_ROTATE).wait_for_completed(); time.sleep(0.1)
                is_blocked = scanner.get_front_tof_cm() < scanner.tof_wall_threshold_cm
                dir_map_abs_char = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}
                occupancy_map.update_wall(r, c, dir_map_abs_char[CURRENT_DIRECTION], is_blocked, 'tof')
                visualizer.update_plot(occupancy_map, CURRENT_POSITION, object_detector.last_processed_image)
                if not is_blocked:
                    axis_to_monitor = 'x' if ROBOT_FACE % 2 != 0 else 'y'
                    movement_controller.move_forward_one_grid(axis=axis_to_monitor, attitude_handler=attitude_handler)
                    movement_controller.center_in_node_with_tof(scanner, attitude_handler)
                    CURRENT_POSITION = (target_r, target_c); moved = True; break
                else: print(f"    Confirmation failed. Path to {['N','E','S','W'][target_dir]} is blocked.")
        
        if not moved:
            print("No immediate unvisited path. Initiating backtrack...")
            backtrack_path = find_nearest_unvisited_path(occupancy_map, CURRENT_POSITION, visited_cells)
            if backtrack_path and len(backtrack_path) > 1:
                execute_path(backtrack_path, movement_controller, attitude_handler, scanner, visualizer, object_detector, occupancy_map)
                print("Backtrack to new area complete. Resuming exploration.")
                continue
            else: print("🎉 EXPLORATION COMPLETE! No reachable unvisited cells remain."); break
    print("\n🎉 === EXPLORATION PHASE FINISHED ===")

# =============================================================================
# ===== MAIN EXECUTION BLOCK ==================================================
# =============================================================================
if __name__ == '__main__':
    ep_robot = None; occupancy_map = OccupancyGridMap(width=5, height=5)
    attitude_handler = AttitudeHandler(); movement_controller = None
    scanner = None; object_detector = None; ep_chassis = None

    try:
        # --- NEW: Get Target Info from User ---
        VALID_SHAPES = ["Circle", "Square", "Rectangle_H", "Rectangle_V"]
        VALID_COLORS = ["Red", "Yellow", "Green", "Blue"]
        print("\n--- 🎯 Define Target Characteristics ---")
        while True:
            target_shape = input(f"Select Target Shape ({'/'.join(VALID_SHAPES)}): ").strip().title()
            if target_shape in VALID_SHAPES: break
            print("⚠️ Invalid shape.")
        while True:
            target_color = input(f"Select Target Color ({'/'.join(VALID_COLORS)}): ").strip().title()
            if target_color in VALID_COLORS: break
            print("⚠️ Invalid color.")
        while True:
            try:
                num_targets = int(input("How many target objects to find?: ").strip())
                if num_targets > 0: break
            except ValueError: print("⚠️ Please enter a valid number.")
        TARGET_OBJECT_INFO = {'shape': target_shape, 'color': target_color}
        print(f"✅ Objective: Find {num_targets} x {target_color} {target_shape}(s).")
        # --- END NEW ---
        
        visualizer = RealTimeVisualizer(grid_size=5, target_dest=TARGET_DESTINATION, target_object_info=TARGET_OBJECT_INFO)
        
        # --- NEW: Initialize Object Tracker ---
        template_files = { "Circle": "./Assignment/image_processing/template/circle1.png",
        "Square": "./Assignment/image_processing/template/square.png",
        "Rectangle_H": "./Assignment/image_processing/template/rec_h.png",
        "Rectangle_V": "./Assignment/image_processing/template/rec_v.png",}
        object_tracker = ObjectTracker(template_paths=template_files)

        print("🤖 Connecting to robot..."); ep_robot = robot.Robot(); ep_robot.initialize(conn_type="ap")
        ep_chassis, ep_gimbal, ep_camera = ep_robot.chassis, ep_robot.gimbal, ep_robot.camera
        ep_tof_sensor, ep_sensor_adaptor = ep_robot.sensor, ep_robot.sensor_adaptor
        
        print(" GIMBAL: Centering gimbal..."); ep_gimbal.moveto(pitch=0, yaw=0, yaw_speed=SPEED_ROTATE).wait_for_completed()
        
        # Initialize all managers and controllers
        scanner = EnvironmentScanner(ep_sensor_adaptor, ep_tof_sensor, ep_gimbal, ep_chassis)
        movement_controller = MovementController(ep_chassis)
        object_detector = ObjectDetectionManager(ep_camera, object_tracker, TARGET_OBJECT_INFO, num_targets, occupancy_map, visualizer)
        
        attitude_handler.start_monitoring(ep_chassis)
        object_detector.start() # Start the manager
        
        explore_with_ogm(scanner, movement_controller, attitude_handler, object_detector, occupancy_map, visualizer)
        
        print(f"\n\n--- NAVIGATION TO TARGET PHASE: From {CURRENT_POSITION} to {TARGET_DESTINATION} ---")
        if CURRENT_POSITION == TARGET_DESTINATION:
            print("🎉 Robot is already at the target destination!")
        else:
            path_to_target = find_path_bfs(occupancy_map, CURRENT_POSITION, TARGET_DESTINATION)
            if path_to_target and len(path_to_target) > 1:
                print(f"✅ Path found to target: {path_to_target}")
                execute_path(path_to_target, movement_controller, attitude_handler, scanner, visualizer, object_detector, occupancy_map, path_name="Final Navigation")
                print(f"🎉🎉 Robot has arrived at the target destination: {TARGET_DESTINATION}!")
            else:
                print(f"⚠️ Could not find a path from {CURRENT_POSITION} to {TARGET_DESTINATION}.")
        
    except KeyboardInterrupt: print("\n⚠️ User interrupted exploration.")
    except Exception as e: print(f"\n⚌ An error occurred: {e}"); traceback.print_exc()
    finally:
        if ep_robot:
            print("🔌 Cleaning up and closing connection...")
            if object_detector: object_detector.stop() # Stop the detection thread gracefully
            if scanner: scanner.cleanup()
            if attitude_handler and attitude_handler.is_monitoring: attitude_handler.stop_monitoring(ep_chassis)
            if movement_controller: movement_controller.cleanup()
            ep_robot.close()
            print("🔌 Connection closed.")
        
        print("... You can close the plot window now ...")
        plt.ioff(); plt.show()