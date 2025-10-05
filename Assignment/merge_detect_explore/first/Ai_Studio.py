# -*-coding:utf-8-*-

import cv2
import numpy as np
import time
import math
import json
import matplotlib.pyplot as plt
from collections import deque
import traceback
import statistics
import os
import threading
import queue
from robomaster import robot, camera as r_camera, blaster as r_blaster

# =============================================================================
# ===== CONFIGURATION & PARAMETERS (MERGED) ===================================
# =============================================================================

# --- Robot Movement & Mapping Config (from test_cur_time_copy.py) ---
SPEED_ROTATE = 480
LEFT_SHARP_SENSOR_ID, LEFT_SHARP_SENSOR_PORT, LEFT_TARGET_CM = 1, 1, 13.0
RIGHT_SHARP_SENSOR_ID, RIGHT_SHARP_SENSOR_PORT, RIGHT_TARGET_CM = 2, 1, 13.0
LEFT_IR_SENSOR_ID, LEFT_IR_SENSOR_PORT = 1, 2
RIGHT_IR_SENSOR_ID, RIGHT_IR_SENSOR_PORT = 2, 2
SHARP_WALL_THRESHOLD_CM = 60.0
SHARP_STDEV_THRESHOLD = 0.3
TOF_ADJUST_SPEED = 0.1
TOF_CALIBRATION_SLOPE = 0.0894
TOF_CALIBRATION_Y_INTERCEPT = 3.8409
CURRENT_POSITION = (3, 2)
CURRENT_DIRECTION = 0  # 0:North, 1:East, 2:South, 3:West
TARGET_DESTINATION = CURRENT_POSITION
CURRENT_TARGET_YAW = 0.0
ROBOT_FACE = 1
IMU_COMPENSATION_START_NODE_COUNT = 7
IMU_COMPENSATION_NODE_INTERVAL = 10
IMU_COMPENSATION_DEG_PER_INTERVAL = -2.0
IMU_DRIFT_COMPENSATION_DEG = 0.0
PROB_OCC_GIVEN_OCC = {'tof': 0.95, 'sharp': 0.90}
PROB_OCC_GIVEN_FREE = {'tof': 0.05, 'sharp': 0.10}
OCCUPANCY_THRESHOLD = 0.7
FREE_THRESHOLD = 0.3
POSITION_LOG = []
DETECTED_OBJECTS_LOG = [] # NEW: For saving detected objects
RESUME_MODE = False
DATA_FOLDER = r"F:\Coder\Year2-1\Robot_Module\Assignment\dude\James_path"

LOG_ODDS_OCC = {
    'tof': math.log(PROB_OCC_GIVEN_OCC['tof'] / (1 - PROB_OCC_GIVEN_OCC['tof'])),
    'sharp': math.log(PROB_OCC_GIVEN_OCC['sharp'] / (1 - PROB_OCC_GIVEN_OCC['sharp']))
}
LOG_ODDS_FREE = {
    'tof': math.log(PROB_OCC_GIVEN_FREE['tof'] / (1 - PROB_OCC_GIVEN_FREE['tof'])),
    'sharp': math.log(PROB_OCC_GIVEN_FREE['sharp'] / (1 - PROB_OCC_GIVEN_FREE['sharp']))
}


# --- Object Detection Config (from fire_target.py) ---
TARGET_SHAPE = "Circle"
TARGET_COLOR = "Red"
PID_KP, PID_KI, PID_KD = -0.25, -0.01, -0.03
DERIV_LPF_ALPHA = 0.25
MAX_YAW_SPEED, MAX_PITCH_SPEED = 220, 180
I_CLAMP = 2000.0
PIX_ERR_DEADZONE = 6
LOCK_TOL_X, LOCK_TOL_Y = 8, 8
LOCK_STABLE_COUNT = 6
FRAME_W, FRAME_H = 960, 540
VERTICAL_FOV_DEG = 54.0
PIXELS_PER_DEG_V = FRAME_H / VERTICAL_FOV_DEG
PITCH_BIAS_DEG = 2.0
PITCH_BIAS_PIX = +PITCH_BIAS_DEG * PIXELS_PER_DEG_V
ROI_Y0, ROI_H0, ROI_X0, ROI_W0 = 264, 270, 10, 911
ROI_SHIFT_PER_DEG = 6.0
ROI_Y_MIN, ROI_Y_MAX = 0, FRAME_H - 10

# --- GPU Check ---
USE_GPU = False
try:
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print("✅ CUDA available, enabling GPU path")
        USE_GPU = True
    else:
        print("⚠️ CUDA not available, CPU path")
except Exception:
    print("⚠️ Skip CUDA check, CPU path")

# =============================================================================
# ===== SHARED STATES & THREADING (from fire_target.py) =======================
# =============================================================================
frame_queue = queue.Queue(maxsize=1)
processed_output = {"details": []}
output_lock = threading.Lock()
stop_event = threading.Event()
gimbal_angle_lock = threading.Lock()
gimbal_angles = (0.0, 0.0, 0.0, 0.0)
is_detecting_flag = {"v": False} # Start with detection OFF

def is_detecting():
    return is_detecting_flag["v"]

def sub_angle_cb(angle_info):
    global gimbal_angles
    with gimbal_angle_lock:
        gimbal_angles = tuple(angle_info)

# =============================================================================
# ===== HELPER & RESUME FUNCTIONS (from test_cur_time_copy.py) ================
# =============================================================================
def convert_adc_to_cm(adc_value):
    if adc_value <= 0: return float('inf')
    return 30263 * (adc_value ** -1.352)

def calibrate_tof_value(raw_tof_value):
    try:
        if raw_tof_value is None or raw_tof_value <= 0:
            return float('inf')
        return (TOF_CALIBRATION_SLOPE * raw_tof_value) + TOF_CALIBRATION_Y_INTERCEPT
    except Exception:
        return float('inf')

def get_compensated_target_yaw():
    return CURRENT_TARGET_YAW + IMU_DRIFT_COMPENSATION_DEG

def log_position_timestamp(position, direction, action="arrived"):
    global POSITION_LOG
    timestamp = time.time()
    direction_names = ['North', 'East', 'South', 'West']
    dt = time.gmtime(timestamp)
    microseconds = int((timestamp % 1) * 1000000)
    iso_timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", dt) + f".{microseconds:06d}Z"
    log_entry = {
        "timestamp": timestamp, "iso_timestamp": iso_timestamp, "position": list(position),
        "direction": direction_names[direction], "action": action, "yaw_angle": CURRENT_TARGET_YAW,
        "imu_compensation": IMU_DRIFT_COMPENSATION_DEG
    }
    POSITION_LOG.append(log_entry)
    print(f"📍 [{action}] Position: {position}, Direction: {direction_names[direction]}, Time: {log_entry['iso_timestamp']}")

def check_for_resume_data():
    map_file = os.path.join(DATA_FOLDER, "Mapping_Top.json")
    timestamp_file = os.path.join(DATA_FOLDER, "Robot_Position_Timestamps.json")
    return os.path.exists(map_file) and os.path.exists(timestamp_file)

def load_resume_data():
    global CURRENT_POSITION, CURRENT_DIRECTION, CURRENT_TARGET_YAW, ROBOT_FACE, IMU_DRIFT_COMPENSATION_DEG, POSITION_LOG, RESUME_MODE
    try:
        print("🔄 Loading resume data...")
        timestamp_file = os.path.join(DATA_FOLDER, "Robot_Position_Timestamps.json")
        with open(timestamp_file, "r", encoding="utf-8") as f:
            timestamp_data = json.load(f)
        if timestamp_data["position_log"]:
            last_log = timestamp_data["position_log"][-1]
            CURRENT_POSITION = tuple(last_log["position"])
            CURRENT_TARGET_YAW = last_log["yaw_angle"]
            IMU_DRIFT_COMPENSATION_DEG = last_log["imu_compensation"]
            POSITION_LOG = timestamp_data["position_log"]
            yaw = last_log["yaw_angle"]
            if -45 <= yaw <= 45: CURRENT_DIRECTION, ROBOT_FACE = 0, 1
            elif 45 < yaw <= 135: CURRENT_DIRECTION, ROBOT_FACE = 1, 2
            elif 135 < yaw or yaw <= -135: CURRENT_DIRECTION, ROBOT_FACE = 2, 3
            else: CURRENT_DIRECTION, ROBOT_FACE = 3, 4
        print(f"✅ Resume data loaded: Pos:{CURRENT_POSITION}, Dir:{['North', 'East', 'South', 'West'][CURRENT_DIRECTION]}, Yaw:{CURRENT_TARGET_YAW:.1f}°")
        RESUME_MODE = True
        return True
    except Exception as e:
        print(f"❌ Error loading resume data: {e}")
        return False

def create_occupancy_map_from_json():
    try:
        map_file = os.path.join(DATA_FOLDER, "Mapping_Top.json")
        with open(map_file, "r", encoding="utf-8") as f:
            map_data = json.load(f)
        max_row = max(node['coordinate']['row'] for node in map_data['nodes'])
        max_col = max(node['coordinate']['col'] for node in map_data['nodes'])
        occupancy_map = OccupancyGridMap(max_col + 1, max_row + 1)
        for node_data in map_data['nodes']:
            r, c = node_data['coordinate']['row'], node_data['coordinate']['col']
            cell = occupancy_map.grid[r][c]
            p_node = node_data['probability']
            cell.log_odds_occupied = math.log(p_node / (1 - p_node)) if p_node not in [0, 0.5, 1] else 0
            walls = node_data['wall_probabilities']
            dir_map = {'north': 'N', 'south': 'S', 'east': 'E', 'west': 'W'}
            for direction, prob in walls.items():
                if direction in dir_map:
                    cell.walls[dir_map[direction]].log_odds = math.log(prob / (1 - prob)) if prob not in [0, 0.5, 1] else 0
        print(f"✅ Occupancy map loaded from JSON ({occupancy_map.width}x{occupancy_map.height})")
        return occupancy_map
    except Exception as e:
        print(f"❌ Error loading occupancy map: {e}")
        return None

# =============================================================================
# ===== OBJECT DETECTION & IMAGE PROCESSING (from fire_target.py) =============
# =============================================================================
def apply_awb(bgr):
    if hasattr(cv2, "xphoto") and hasattr(cv2.xphoto, "createLearningBasedWB"):
        wb = cv2.xphoto.createLearningBasedWB()
        try: wb.setSaturationThreshold(0.99)
        except Exception: pass
        return wb.balanceWhite(bgr)
    return bgr

def night_enhance_pipeline_cpu(bgr):
    return apply_awb(bgr)

class ObjectTracker:
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        print(f"🖼️  ObjectTracker in {'GPU' if use_gpu else 'CPU'} mode")

    def _get_angle(self, pt1, pt2, pt0):
        dx1, dy1 = pt1[0] - pt0[0], pt1[1] - pt0[1]
        dx2, dy2 = pt2[0] - pt0[0], pt2[1] - pt0[1]
        dot = dx1 * dx2 + dy1 * dy2
        mag1 = (dx1 * dx1 + dy1 * dy1) ** 0.5
        mag2 = (dx2 * dx2 + dy2 * dy2) ** 0.5
        if mag1 * mag2 == 0: return 0
        return math.degrees(math.acos(max(-1, min(1, dot / (mag1 * mag2)))))

    def get_raw_detections(self, frame):
        enhanced = cv2.GaussianBlur(night_enhance_pipeline_cpu(frame), (5, 5), 0)
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        ranges = {
            'Red': ([0, 80, 40], [10, 255, 255], [170, 80, 40], [180, 255, 255]), 'Yellow': ([20, 60, 40], [35, 255, 255]),
            'Green': ([35, 40, 30], [85, 255, 255]), 'Blue': ([90, 40, 30], [130, 255, 255])}
        masks = {'Red': cv2.inRange(hsv, np.array(ranges['Red'][0]), np.array(ranges['Red'][1])) | cv2.inRange(hsv, np.array(ranges['Red'][2]), np.array(ranges['Red'][3]))}
        for name in ['Yellow', 'Green', 'Blue']:
            masks[name] = cv2.inRange(hsv, np.array(ranges[name][0]), np.array(ranges[name][1]))
        combined = masks['Red'] | masks['Yellow'] | masks['Green'] | masks['Blue']
        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        H, W = frame.shape[:2]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1500: continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w == 0 or h == 0: continue
            ar = w / float(h)
            if ar > 4.0 or ar < 0.25: continue
            hull = cv2.convexHull(cnt)
            if cv2.contourArea(hull) == 0: continue
            if area / cv2.contourArea(hull) < 0.85: continue
            if x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2: continue
            contour_mask = np.zeros((H, W), np.uint8)
            cv2.drawContours(contour_mask, [cnt], -1, 255, -1)
            max_mean, found = 0, "Unknown"
            for cname, m in masks.items():
                mv = cv2.mean(m, mask=contour_mask)[0]
                if mv > max_mean: max_mean, found = mv, cname
            if max_mean <= 20: continue
            shape, peri = "Uncertain", cv2.arcLength(cnt, True)
            if peri > 0 and (4 * math.pi * area) / (peri * peri) > 0.82:
                shape = "Circle"
            else:
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                if len(approx) == 4 and area / cv2.contourArea(hull) > 0.88:
                    pts = [tuple(p[0]) for p in approx]
                    angs = [self._get_angle(pts[(i - 1) % 4], pts[(i + 1) % 4], p) for i, p in enumerate(pts)]
                    if all(70 <= a <= 110 for a in angs):
                        _, (rw, rh), _ = cv2.minAreaRect(cnt)
                        if min(rw, rh) > 0:
                            ar2 = max(rw, rh) / min(rw, rh)
                            if 0.88 <= ar2 <= 1.12: shape = "Square"
                            elif w > h: shape = "Rectangle_H"
                            else: shape = "Rectangle_V"
            out.append({"contour": cnt, "shape": shape, "color": found, "box": (x, y, w, h)})
        return out

# =============================================================================
# ===== OCCUPANCY GRID MAP & VISUALIZATION (ENHANCED) =========================
# =============================================================================
class WallBelief:
    def __init__(self):
        self.log_odds = 0.0
    def update(self, is_occupied_reading, sensor_type):
        self.log_odds += LOG_ODDS_OCC[sensor_type] if is_occupied_reading else LOG_ODDS_FREE[sensor_type]
        self.log_odds = max(min(self.log_odds, 10), -10)
    def get_probability(self):
        return 1.0 - 1.0 / (1.0 + math.exp(self.log_odds))
    def is_occupied(self):
        return self.get_probability() > OCCUPANCY_THRESHOLD

class OGMCell:
    def __init__(self):
        self.log_odds_occupied = 0.0
        self.walls = {'N': None, 'E': None, 'S': None, 'W': None}
        self.detected_objects = [] # NEW: To store objects found in front of this cell
    def get_node_probability(self):
        return 1.0 - 1.0 / (1.0 + math.exp(self.log_odds_occupied))
    def is_node_occupied(self):
        return self.get_node_probability() > OCCUPANCY_THRESHOLD

class OccupancyGridMap:
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.grid = [[OGMCell() for _ in range(width)] for _ in range(height)]
        self._link_walls()
    def _link_walls(self):
        for r, c in np.ndindex(self.height, self.width):
            if self.grid[r][c].walls['N'] is None:
                wall = WallBelief(); self.grid[r][c].walls['N'] = wall
                if r > 0: self.grid[r - 1][c].walls['S'] = wall
            if self.grid[r][c].walls['W'] is None:
                wall = WallBelief(); self.grid[r][c].walls['W'] = wall
                if c > 0: self.grid[r][c - 1].walls['E'] = wall
            if self.grid[r][c].walls['S'] is None: self.grid[r][c].walls['S'] = WallBelief()
            if self.grid[r][c].walls['E'] is None: self.grid[r][c].walls['E'] = WallBelief()
    def update_wall(self, r, c, direction_char, is_occupied_reading, sensor_type):
        if 0 <= r < self.height and 0 <= c < self.width:
            wall = self.grid[r][c].walls.get(direction_char)
            if wall: wall.update(is_occupied_reading, sensor_type)
    def update_node(self, r, c, is_occupied_reading, sensor_type='tof'):
        if 0 <= r < self.height and 0 <= c < self.width:
            self.grid[r][c].log_odds_occupied += LOG_ODDS_OCC[sensor_type] if is_occupied_reading else LOG_ODDS_FREE[sensor_type]
    def is_path_clear(self, r1, c1, r2, c2):
        dr, dc = r2 - r1, c2 - c1
        if abs(dr) + abs(dc) != 1: return False
        if dr == -1: wall_char = 'N'
        elif dr == 1: wall_char = 'S'
        elif dc == 1: wall_char = 'E'
        else: wall_char = 'W'
        if self.grid[r1][c1].walls.get(wall_char).is_occupied(): return False
        if 0 <= r2 < self.height and 0 <= c2 < self.width and self.grid[r2][c2].is_node_occupied(): return False
        return True

class RealTimeVisualizer:
    def __init__(self, grid_size, target_dest=None):
        self.grid_size = grid_size
        self.target_dest = target_dest
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.colors = {"robot": "#0000FF", "target": "#FFD700", "path": "#FFFF00", "wall": "#000000", "wall_prob": "#000080"}
        self.obj_colors = {'Red': '#FF0000', 'Green': '#00FF00', 'Blue': '#0080FF', 'Yellow': '#FFFF00', 'Uncertain': '#808080'}

    def update_plot(self, occupancy_map, robot_pos, path=None):
        self.ax.clear()
        self.ax.set_title("Real-time Map (Nodes, Walls, Objects)")
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.ax.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax.set_ylim(self.grid_size - 0.5, -0.5)
        for r, c in np.ndindex(self.grid_size, self.grid_size):
            cell = occupancy_map.grid[r][c]
            prob = cell.get_node_probability()
            color = '#8B0000' if prob > OCCUPANCY_THRESHOLD else ('#D3D3D3' if prob < FREE_THRESHOLD else '#90EE90')
            self.ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=color, edgecolor='k', lw=0.5))
            # Draw Objects
            if cell.detected_objects:
                for obj in cell.detected_objects:
                    zone_offset = {'Left': -0.25, 'Center': 0, 'Right': 0.25}.get(obj.get('zone', 'Center'), 0)
                    marker_shape = 's' if 'Square' in obj.get('shape', '') or 'Rect' in obj.get('shape', '') else 'o'
                    self.ax.plot(c + zone_offset, r, marker=marker_shape, markersize=8,
                                 color=self.obj_colors.get(obj.get('color'), '#FFFFFF'), markeredgecolor='black')
        for r, c in np.ndindex(self.grid_size, self.grid_size):
            cell = occupancy_map.grid[r][c]
            if cell.walls['N'].is_occupied(): self.ax.plot([c - 0.5, c + 0.5], [r - 0.5, r - 0.5], color=self.colors['wall'], linewidth=4)
            if cell.walls['W'].is_occupied(): self.ax.plot([c - 0.5, c - 0.5], [r - 0.5, r + 0.5], color=self.colors['wall'], linewidth=4)
            if r == self.grid_size - 1 and cell.walls['S'].is_occupied(): self.ax.plot([c - 0.5, c + 0.5], [r + 0.5, r + 0.5], color=self.colors['wall'], linewidth=4)
            if c == self.grid_size - 1 and cell.walls['E'].is_occupied(): self.ax.plot([c + 0.5, c + 0.5], [r - 0.5, r + 0.5], color=self.colors['wall'], linewidth=4)
        if self.target_dest: self.ax.add_patch(plt.Rectangle((self.target_dest[1] - 0.5, self.target_dest[0] - 0.5), 1, 1, facecolor=self.colors['target'], edgecolor='k', lw=2))
        if path:
            for r_p, c_p in path: self.ax.add_patch(plt.Rectangle((c_p - 0.5, r_p - 0.5), 1, 1, facecolor=self.colors['path'], alpha=0.7))
        if robot_pos: self.ax.add_patch(plt.Rectangle((robot_pos[1] - 0.5, robot_pos[0] - 0.5), 1, 1, facecolor=self.colors['robot'], edgecolor='k'))
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
        normalized_target = self.normalize_angle(target_yaw); time.sleep(0.1)
        robot_rotation = -self.normalize_angle(normalized_target - self.current_yaw)
        print(f"\n🔧 Correcting Yaw: {self.current_yaw:.1f}° -> {target_yaw:.1f}°. Rotating: {robot_rotation:.1f}°")
        if abs(robot_rotation) > self.yaw_tolerance:
            chassis.move(x=0, y=0, z=robot_rotation, z_speed=60).wait_for_completed(timeout=2)
        final_error = abs(self.normalize_angle(normalized_target - self.current_yaw))
        if final_error <= self.yaw_tolerance: print(f"✅ Yaw Correction Success: {self.current_yaw:.1f}°"); return True
        remaining_rotation = -self.normalize_angle(normalized_target - self.current_yaw)
        if 0.5 < abs(remaining_rotation) < 20:
            chassis.move(x=0, y=0, z=remaining_rotation, z_speed=40).wait_for_completed(timeout=2)
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
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error

class MovementController:
    def __init__(self, chassis):
        self.chassis = chassis
        self.current_x_pos, self.current_y_pos = 0.0, 0.0
        self.chassis.sub_position(freq=20, callback=self.position_handler)
    def position_handler(self, pos_info): self.current_x_pos, self.current_y_pos = pos_info[0], pos_info[1]
    def _calculate_yaw_correction(self, attitude_handler, target_yaw):
        yaw_error = attitude_handler.normalize_angle(target_yaw - attitude_handler.current_yaw)
        return max(min(1.8 * yaw_error, 25), -25)
    def move_forward_one_grid(self, axis, attitude_handler):
        attitude_handler.correct_yaw_to_target(self.chassis, get_compensated_target_yaw())
        target_distance, pid = 0.6, PID(Kp=1.0, Ki=0.25, Kd=8, setpoint=0.6)
        start_time, last_time = time.time(), time.time()
        start_pos = self.current_x_pos if axis == 'x' else self.current_y_pos
        print(f"🚀 Moving FORWARD 0.6m on AXIS '{axis}'")
        while time.time() - start_time < 3.5:
            now = time.time(); dt = now - last_time; last_time = now
            rel_pos = abs((self.current_x_pos if axis == 'x' else self.current_y_pos) - start_pos)
            if abs(rel_pos - target_distance) < 0.03: print("\n✅ Move complete!"); break
            speed = max(-1.0, min(1.0, pid.compute(rel_pos, dt) * min(1.0, 0.1 + ((now - start_time) / 1.0) * 0.9)))
            yaw_corr = self._calculate_yaw_correction(attitude_handler, get_compensated_target_yaw())
            self.chassis.drive_speed(x=speed, y=0, z=yaw_corr, timeout=1)
            print(f"Moving... Dist: {rel_pos:.3f}/{target_distance:.2f} m", end='\r')
        self.chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0); time.sleep(0.5)
    def adjust_position_to_wall(self, sensor_adaptor, attitude_handler, side, s_cfg, target_cm, dir_mult):
        compensated_yaw = get_compensated_target_yaw()
        print(f"\n--- Adjusting {side} Side (Yaw {compensated_yaw:.2f}°) ---")
        start_time = time.time()
        while time.time() - start_time < 8:
            adc = sensor_adaptor.get_adc(id=s_cfg["sharp_id"], port=s_cfg["sharp_port"])
            dist = convert_adc_to_cm(adc)
            err = target_cm - dist
            if abs(err) <= 0.8: print(f"\n[{side}] Target reached! Final dist: {dist:.2f} cm"); break
            slide_speed = max(min(dir_mult * 0.045 * err, 0.18), -0.18)
            yaw_corr = self._calculate_yaw_correction(attitude_handler, compensated_yaw)
            self.chassis.drive_speed(x=0, y=slide_speed, z=yaw_corr)
            print(f"Adjusting {side}... Current: {dist:5.2f}cm, Target: {target_cm:4.1f}cm", end='\r')
            time.sleep(0.02)
        self.chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0); time.sleep(0.1)
    def center_in_node_with_tof(self, scanner, attitude_handler, target_cm=19, tol_cm=1.0):
        if scanner.is_performing_full_scan: print("[ToF Centering] SKIPPED: Side-scan in progress."); return
        print("\n--- Stage: Centering in Node with ToF ---")
        tof_dist = scanner.last_tof_distance_cm
        if tof_dist is None or math.isinf(tof_dist) or tof_dist >= 50: print(f"[ToF] No valid data or too far ({tof_dist:.1f}cm). Skipping."); return
        direction = abs(TOF_ADJUST_SPEED) if tof_dist > target_cm + tol_cm or 22 <= tof_dist <= target_cm else -abs(TOF_ADJUST_SPEED)
        start_time, compensated_yaw = time.time(), get_compensated_target_yaw()
        while time.time() - start_time < 6.0:
            self.chassis.drive_speed(x=direction, y=0, z=self._calculate_yaw_correction(attitude_handler, compensated_yaw), timeout=0.1)
            time.sleep(0.12); self.chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0); time.sleep(0.08)
            current_tof = scanner.last_tof_distance_cm
            if current_tof is None or math.isinf(current_tof): continue
            print(f"[ToF] Adjusting... Current: {current_tof:.2f} cm", end="\r")
            if abs(current_tof - target_cm) <= tol_cm: print(f"\n[ToF] ✅ Centering complete. Final: {current_tof:.2f} cm"); break
            if (direction > 0 and current_tof < target_cm - tol_cm) or (direction < 0 and current_tof > target_cm + tol_cm):
                direction *= -1; print("\n[ToF] 🔄 Overshot. Reversing.")
        self.chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0); time.sleep(0.1)
    def rotate_to_direction(self, target_dir, attitude_handler):
        global CURRENT_DIRECTION
        if CURRENT_DIRECTION == target_dir: return
        diff = (target_dir - CURRENT_DIRECTION + 4) % 4
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
    def __init__(self, sensor_adaptor, tof_sensor, gimbal):
        self.sensor_adaptor, self.tof_sensor, self.gimbal = sensor_adaptor, tof_sensor, gimbal
        self.tof_wall_threshold_cm = 60.0
        self.last_tof_distance_cm, self.side_tof_reading_cm = float('inf'), float('inf')
        self.is_gimbal_centered, self.is_performing_full_scan = True, False
        self.tof_sensor.sub_distance(freq=10, callback=self._tof_data_handler)
        self.side_sensors = {"Left": {"sharp_id": 1, "sharp_port": 1, "ir_id": 1, "ir_port": 2}, "Right": {"sharp_id": 2, "sharp_port": 1, "ir_id": 2, "ir_port": 2}}
    def _tof_data_handler(self, sub_info):
        calibrated_cm = calibrate_tof_value(sub_info[0])
        if self.is_gimbal_centered: self.last_tof_distance_cm = calibrated_cm
        else: self.side_tof_reading_cm = calibrated_cm
    def _get_stable_reading_cm(self, side, duration=0.35):
        s_info = self.side_sensors.get(side)
        if not s_info: return None, None
        readings = []
        start_time = time.time()
        while time.time() - start_time < duration:
            readings.append(convert_adc_to_cm(self.sensor_adaptor.get_adc(id=s_info["sharp_id"], port=s_info["sharp_port"])))
            time.sleep(0.05)
        return statistics.mean(readings), statistics.stdev(readings) if len(readings) > 1 else 0
    def get_sensor_readings(self):
        self.is_performing_full_scan = True
        try:
            self.gimbal.moveto(pitch=0, yaw=0, yaw_speed=SPEED_ROTATE).wait_for_completed(); time.sleep(0.15)
            readings = {'front': (self.last_tof_distance_cm < self.tof_wall_threshold_cm)}
            print(f"[SCAN] Front (ToF): {self.last_tof_distance_cm:.1f}cm -> {'OCCUPIED' if readings['front'] else 'FREE'}")
            for side in ["Left", "Right"]:
                avg_dist, std_dev = self._get_stable_reading_cm(side)
                if avg_dist is None: readings[side.lower()] = False; continue
                is_sharp = (avg_dist < SHARP_WALL_THRESHOLD_CM and std_dev < SHARP_STDEV_THRESHOLD)
                is_ir = (self.sensor_adaptor.get_io(id=self.side_sensors[side]["ir_id"], port=self.side_sensors[side]["ir_port"]) == 0)
                if is_sharp == is_ir: is_wall = is_sharp
                else:
                    print(f"    -> Ambiguity on {side} side! Confirming with ToF...")
                    target_gimbal_yaw = -90 if side == "Left" else 90
                    try:
                        self.is_gimbal_centered = False
                        self.gimbal.moveto(pitch=0, yaw=target_gimbal_yaw, yaw_speed=SPEED_ROTATE).wait_for_completed(); time.sleep(0.1)
                        is_wall = (self.side_tof_reading_cm < self.tof_wall_threshold_cm)
                    finally:
                        self.gimbal.moveto(pitch=0, yaw=0, yaw_speed=SPEED_ROTATE).wait_for_completed(); self.is_gimbal_centered = True; time.sleep(0.1)
                readings[side.lower()] = is_wall
                print(f"    -> Final Result for {side} side: {'WALL' if is_wall else 'FREE'}")
            return readings
        finally:
            self.is_performing_full_scan = False
    def get_front_tof_cm(self):
        self.gimbal.moveto(pitch=0, yaw=0, yaw_speed=SPEED_ROTATE).wait_for_completed(); time.sleep(0.1)
        return self.last_tof_distance_cm
    def cleanup(self):
        try: self.tof_sensor.unsub_distance()
        except Exception: pass

# =============================================================================
# ===== ROBOT CONNECTION MANAGER (from fire_target.py) ========================
# =============================================================================
class RMConnection:
    def __init__(self):
        self._lock, self._robot, self.connected = threading.Lock(), None, threading.Event()
    def connect(self):
        with self._lock:
            self._safe_close()
            print("🤖 Connecting to RoboMaster...")
            rb = robot.Robot(); rb.initialize(conn_type="ap")
            self.ep_chassis, self.ep_gimbal = rb.chassis, rb.gimbal
            self.ep_tof_sensor, self.ep_sensor_adaptor = rb.sensor, rb.sensor_adaptor
            self.ep_camera, self.ep_blaster = rb.camera, rb.blaster
            self.ep_camera.start_video_stream(display=False, resolution=r_camera.STREAM_540P)
            self.ep_gimbal.sub_angle(freq=50, callback=sub_angle_cb)
            self._robot = rb
            self.connected.set()
            print("✅ RoboMaster connected & streaming")
            self.ep_gimbal.recenter(pitch_speed=200, yaw_speed=200).wait_for_completed()
    def _safe_close(self):
        if self._robot is not None:
            try:
                self._robot.camera.stop_video_stream()
                self._robot.gimbal.unsub_angle()
                self._robot.close()
            except Exception: pass
            finally: self._robot, self.connected = None, self.connected.clear()
            print("🔌 Connection closed")
    def drop_and_reconnect(self):
        with self._lock: self._safe_close()
    def close(self):
        with self._lock: self._safe_close()

def reconnector_thread(manager: RMConnection):
    backoff = 1.0
    while not stop_event.is_set():
        if not manager.connected.is_set():
            try:
                manager.connect(); backoff = 1.0
            except Exception as e:
                print(f"♻️ Reconnect failed: {e} (retry in {backoff:.1f}s)")
                time.sleep(backoff); backoff = min(backoff * 1.6, 8.0)
        time.sleep(0.2)

# =============================================================================
# ===== CAPTURE, PROCESSING, CONTROL THREADS (from fire_target.py) ============
# =============================================================================
def capture_thread_func(manager: RMConnection, q: queue.Queue):
    print("🚀 Capture thread started")
    fail = 0
    while not stop_event.is_set():
        if not manager.connected.is_set(): time.sleep(0.1); continue
        try:
            frame = manager.ep_camera.read_cv2_image(timeout=1.0)
            if frame is not None:
                if q.full(): q.get_nowait()
                q.put(frame); fail = 0
            else: fail += 1
        except Exception: fail += 1
        if fail >= 10:
            print("⚠️ Camera errors -> reconnecting"); manager.drop_and_reconnect()
            try:
                while True: q.get_nowait()
            except queue.Empty: pass
            fail = 0
        time.sleep(0.005)
    print("🛑 Capture thread stopped")

def processing_thread_func(tracker: ObjectTracker, q: queue.Queue, roi_state):
    print("🧠 Processing thread started.")
    while not stop_event.is_set():
        if not is_detecting(): time.sleep(0.05); continue
        try:
            frame = q.get(timeout=1.0)
            with gimbal_angle_lock: pitch_deg = gimbal_angles[0]
            roi_y = int(ROI_Y0 - (max(0.0, -pitch_deg) * ROI_SHIFT_PER_DEG))
            roi_y = max(ROI_Y_MIN, min(ROI_Y_MAX, roi_y))
            roi_state["y"] = roi_y
            roi_frame = frame[roi_y:roi_y + ROI_H0, ROI_X0:ROI_X0 + ROI_W0]
            detections = tracker.get_raw_detections(roi_frame)
            results = []
            div1, div2 = int(ROI_W0 * 0.33), int(ROI_W0 * 0.66)
            for i, d in enumerate(detections):
                shape, color, (x, y, w, h) = d['shape'], d['color'], d['box']
                zone = "Center"
                if x + w < div1: zone = "Left"
                elif x >= div2: zone = "Right"
                results.append({"id": i + 1, "color": color, "shape": shape, "zone": zone, "is_target": (shape == TARGET_SHAPE and color == TARGET_COLOR), "box": (x, y, w, h)})
            with output_lock: processed_output = {"details": results}
        except queue.Empty: continue
        except Exception as e: print(f"CRITICAL: Processing error: {e}")
    print("🛑 Processing thread stopped.")

def control_thread_func(manager: RMConnection, roi_state):
    print("🎯 Control thread started.")
    prev_time, err_x_prev_f, err_y_prev_f, integ_x, integ_y = None, 0.0, 0.0, 0.0, 0.0
    lock_queue = deque(maxlen=LOCK_STABLE_COUNT)
    while not stop_event.is_set():
        if not (is_detecting() and manager.connected.is_set()): time.sleep(0.02); continue
        with output_lock: dets = list(processed_output["details"])
        target_box = None
        max_area = -1
        for det in dets:
            if det.get("is_target", False):
                x, y, w, h = det["box"]
                if w * h > max_area: max_area, target_box = w * h, (x, y, w, h)
        if target_box:
            x, y, w, h = target_box
            cx, cy = ROI_X0 + x + w / 2.0, roi_state["y"] + y + h / 2.0
            err_x, err_y = (FRAME_W / 2.0 - cx), (FRAME_H / 2.0 - cy) + PITCH_BIAS_PIX
            if abs(err_x) < PIX_ERR_DEADZONE: err_x = 0.0
            if abs(err_y) < PIX_ERR_DEADZONE: err_y = 0.0
            now = time.time()
            if prev_time is None: prev_time, err_x_prev_f, err_y_prev_f = now, err_x, err_y; continue
            dt = max(1e-3, now - prev_time); prev_time = now
            err_x_f = err_x_prev_f + DERIV_LPF_ALPHA * (err_x - err_x_prev_f); dx = (err_x_f - err_x_prev_f) / dt; err_x_prev_f = err_x_f
            err_y_f = err_y_prev_f + DERIV_LPF_ALPHA * (err_y - err_y_prev_f); dy = (err_y_f - err_y_prev_f) / dt; err_y_prev_f = err_y_f
            integ_x, integ_y = np.clip(integ_x + err_x * dt, -I_CLAMP, I_CLAMP), np.clip(integ_y + err_y * dt, -I_CLAMP, I_CLAMP)
            u_x, u_y = PID_KP * err_x + PID_KI * integ_x + PID_KD * dx, PID_KP * err_y + PID_KI * integ_y + PID_KD * dy
            u_x, u_y = float(np.clip(u_x, -MAX_YAW_SPEED, MAX_YAW_SPEED)), float(np.clip(u_y, -MAX_PITCH_SPEED, MAX_PITCH_SPEED))
            try: manager.ep_gimbal.drive_speed(pitch_speed=-u_y, yaw_speed=u_x)
            except Exception: pass
            locked = (abs(err_x) <= LOCK_TOL_X) and (abs(err_y) <= LOCK_TOL_Y)
            lock_queue.append(1 if locked else 0)
            if len(lock_queue) == LOCK_STABLE_COUNT and sum(lock_queue) == LOCK_STABLE_COUNT:
                try: manager.ep_blaster.fire(fire_type=r_blaster.WATER_FIRE); time.sleep(0.1); lock_queue.clear()
                except Exception: pass
        else:
            try: manager.ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
            except Exception: pass
            lock_queue.clear(); integ_x *= 0.98; integ_y *= 0.98
        time.sleep(0.005)
    print("🛑 Control thread stopped.")

# =============================================================================
# ===== PATHFINDING & EXPLORATION LOGIC (ENHANCED) ============================
# =============================================================================
def find_path_bfs(occupancy_map, start, end):
    queue, visited = deque([[start]]), {start}
    moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    while queue:
        path = queue.popleft()
        r, c = path[-1]
        if (r, c) == end: return path
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < occupancy_map.height and 0 <= nc < occupancy_map.width and (nr, nc) not in visited:
                if occupancy_map.is_path_clear(r, c, nr, nc):
                    visited.add((nr, nc)); new_path = list(path); new_path.append((nr, nc)); queue.append(new_path)
    return None

def find_nearest_unvisited_path(occupancy_map, start_pos, visited_cells):
    unvisited = [(r, c) for r, c in np.ndindex(occupancy_map.height, occupancy_map.width) if (r, c) not in visited_cells and not occupancy_map.grid[r][c].is_node_occupied()]
    shortest_path = None
    for target in unvisited:
        path = find_path_bfs(occupancy_map, start_pos, target)
        if path and (shortest_path is None or len(path) < len(shortest_path)): shortest_path = path
    return shortest_path

def execute_path(path, movement_controller, attitude_handler, scanner, visualizer, occupancy_map, path_name="Backtrack"):
    global CURRENT_POSITION
    print(f"🎯 Executing {path_name} Path: {path}")
    dir_vectors = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}
    dir_chars = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}
    log_position_timestamp(CURRENT_POSITION, CURRENT_DIRECTION, f"{path_name}_start")
    for i in range(len(path) - 1):
        visualizer.update_plot(occupancy_map, path[i], path)
        curr_r, curr_c = path[i]
        next_r, next_c = path[i+1]
        target_dir = dir_vectors[(next_r - curr_r, next_c - curr_c)]
        movement_controller.rotate_to_direction(target_dir, attitude_handler)
        scanner.gimbal.moveto(pitch=0, yaw=0, yaw_speed=SPEED_ROTATE).wait_for_completed(); time.sleep(0.15)
        is_blocked = scanner.get_front_tof_cm() < scanner.tof_wall_threshold_cm
        occupancy_map.update_wall(curr_r, curr_c, dir_chars[CURRENT_DIRECTION], is_blocked, 'tof')
        visualizer.update_plot(occupancy_map, CURRENT_POSITION)
        if is_blocked: print(f"   -> 🔥 [{path_name}] IMMEDIATE STOP. Obstacle detected. Aborting path."); break
        axis = 'x' if ROBOT_FACE % 2 != 0 else 'y'
        movement_controller.move_forward_one_grid(axis=axis, attitude_handler=attitude_handler)
        movement_controller.center_in_node_with_tof(scanner, attitude_handler)
        CURRENT_POSITION = (next_r, next_c)
        log_position_timestamp(CURRENT_POSITION, CURRENT_DIRECTION, f"{path_name}_moved")
        visualizer.update_plot(occupancy_map, CURRENT_POSITION, path)
        perform_side_alignment_and_mapping(movement_controller, scanner, attitude_handler, occupancy_map, visualizer)
    print(f"✅ {path_name} complete.")

def perform_side_alignment_and_mapping(movement_controller, scanner, attitude_handler, occupancy_map, visualizer):
    print("\n--- Stage: Wall Detection & Side Alignment ---")
    r, c = CURRENT_POSITION; dir_map = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}
    side_walls = scanner.get_sensor_readings()
    left_dir = (CURRENT_DIRECTION - 1 + 4) % 4
    occupancy_map.update_wall(r, c, dir_map[left_dir], side_walls['left'], 'sharp')
    if side_walls['left']: movement_controller.adjust_position_to_wall(scanner.sensor_adaptor, attitude_handler, "Left", scanner.side_sensors["Left"], LEFT_TARGET_CM, 1)
    right_dir = (CURRENT_DIRECTION + 1) % 4
    occupancy_map.update_wall(r, c, dir_map[right_dir], side_walls['right'], 'sharp')
    if side_walls['right']: movement_controller.adjust_position_to_wall(scanner.sensor_adaptor, attitude_handler, "Right", scanner.side_sensors["Right"], RIGHT_TARGET_CM, -1)
    attitude_handler.correct_yaw_to_target(movement_controller.chassis, get_compensated_target_yaw())

# NEW: Function for the object detection sequence
def perform_object_scan_and_map(scanner, occupancy_map, visualizer):
    global DETECTED_OBJECTS_LOG
    print("\n--- Stage: Object Scanning ---")
    
    # 1. Check ToF for wall in front
    front_dist_cm = scanner.get_front_tof_cm()
    is_wall_ahead = front_dist_cm < 60.0
    print(f"   -> Front ToF: {front_dist_cm:.1f}cm. Wall ahead: {is_wall_ahead}")

    # 2. Activate detection for 1 second
    print("   -> 🔴 Activating object detection for 1 second...")
    is_detecting_flag["v"] = True
    time.sleep(1)
    is_detecting_flag["v"] = False
    print("   -> ⚫️ Object detection finished.")

    # 3. Get results and filter them
    with output_lock:
        detected_details = list(processed_output["details"])
    
    valid_objects = []
    if is_wall_ahead:
        # If wall, take all zones
        valid_objects = detected_details
        print(f"   -> Wall detected. Processing all zones. Found {len(valid_objects)} object(s).")
    else:
        # No wall, only take Left and Right
        for obj in detected_details:
            if obj.get("zone") in ["Left", "Right"]:
                valid_objects.append(obj)
        print(f"   -> No wall. Processing side zones only. Found {len(valid_objects)} object(s).")

    if not valid_objects:
        print("   -> No valid objects found to map.")
        return

    # 4. Save objects to the map and log
    r, c = CURRENT_POSITION
    dir_vectors = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    dr, dc = dir_vectors[CURRENT_DIRECTION]
    target_r, target_c = r + dr, c + dc
    
    if 0 <= target_r < occupancy_map.height and 0 <= target_c < occupancy_map.width:
        target_cell = occupancy_map.grid[target_r][target_c]
        
        # Clear previous scan results for this cell and add new ones
        target_cell.detected_objects.clear()
        
        for obj in valid_objects:
            # Add to map cell
            target_cell.detected_objects.append(obj)
            
            # Add to global log
            log_entry = {
                "timestamp": time.time(),
                "robot_pos_when_scanned": [r, c],
                "robot_dir_when_scanned": ['N', 'E', 'S', 'W'][CURRENT_DIRECTION],
                "mapped_to_cell": [target_r, target_c],
                "object_details": obj
            }
            DETECTED_OBJECTS_LOG.append(log_entry)
            print(f"   -> Mapped {obj['color']} {obj['shape']} in zone {obj['zone']} to cell ({target_r}, {target_c})")
        
        visualizer.update_plot(occupancy_map, CURRENT_POSITION)
    else:
        print(f"   -> Cannot map objects, target cell ({target_r}, {target_c}) is outside map bounds.")

def explore_with_ogm(scanner, movement_controller, attitude_handler, occupancy_map, visualizer, max_steps=40):
    global CURRENT_POSITION, CURRENT_DIRECTION, IMU_DRIFT_COMPENSATION_DEG
    visited_cells = set()
    log_position_timestamp(CURRENT_POSITION, CURRENT_DIRECTION, "exploration_start")
    
    for step in range(max_steps):
        r, c = CURRENT_POSITION
        print(f"\n--- Step {step + 1} at {CURRENT_POSITION}, Facing: {['N', 'E', 'S', 'W'][CURRENT_DIRECTION]} ---")
        log_position_timestamp(CURRENT_POSITION, CURRENT_DIRECTION, f"step_{step + 1}")
        
        attitude_handler.correct_yaw_to_target(movement_controller.chassis, get_compensated_target_yaw())
        perform_side_alignment_and_mapping(movement_controller, scanner, attitude_handler, occupancy_map, visualizer)
        
        dir_map_abs_char = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}
        is_front_occupied = scanner.get_front_tof_cm() < scanner.tof_wall_threshold_cm
        occupancy_map.update_wall(r, c, dir_map_abs_char[CURRENT_DIRECTION], is_front_occupied, 'tof')
        
        occupancy_map.update_node(r, c, False, 'tof'); visited_cells.add((r, c))
        visualizer.update_plot(occupancy_map, CURRENT_POSITION)
        
        if len(visited_cells) >= IMU_COMPENSATION_START_NODE_COUNT:
            new_comp = (len(visited_cells) // IMU_COMPENSATION_NODE_INTERVAL) * IMU_COMPENSATION_DEG_PER_INTERVAL
            if new_comp != IMU_DRIFT_COMPENSATION_DEG:
                IMU_DRIFT_COMPENSATION_DEG = new_comp
                print(f"🔩 IMU Drift Compensation Updated: {IMU_DRIFT_COMPENSATION_DEG:.1f}°")

        priority_dirs = [(CURRENT_DIRECTION + 1) % 4, CURRENT_DIRECTION, (CURRENT_DIRECTION - 1 + 4) % 4]
        moved = False
        dir_vectors = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        for target_dir in priority_dirs:
            target_r, target_c = r + dir_vectors[target_dir][0], c + dir_vectors[target_dir][1]
            if occupancy_map.is_path_clear(r, c, target_r, target_c) and (target_r, target_c) not in visited_cells:
                movement_controller.rotate_to_direction(target_dir, attitude_handler)
                
                # <<< NEW INTEGRATED LOGIC >>>
                perform_object_scan_and_map(scanner, occupancy_map, visualizer)
                # <<< END OF NEW LOGIC >>>
                
                is_blocked = scanner.get_front_tof_cm() < scanner.tof_wall_threshold_cm
                occupancy_map.update_wall(r, c, dir_map_abs_char[CURRENT_DIRECTION], is_blocked, 'tof')
                visualizer.update_plot(occupancy_map, CURRENT_POSITION)

                if occupancy_map.is_path_clear(r, c, target_r, target_c):
                    axis_to_monitor = 'x' if ROBOT_FACE % 2 != 0 else 'y'
                    movement_controller.move_forward_one_grid(axis=axis_to_monitor, attitude_handler=attitude_handler)
                    movement_controller.center_in_node_with_tof(scanner, attitude_handler)
                    CURRENT_POSITION = (target_r, target_c)
                    log_position_timestamp(CURRENT_POSITION, CURRENT_DIRECTION, "moved_to_new_node")
                    moved = True
                    break
        
        if not moved:
            print("No immediate path. Backtracking...")
            backtrack_path = find_nearest_unvisited_path(occupancy_map, CURRENT_POSITION, visited_cells)
            if backtrack_path and len(backtrack_path) > 1:
                execute_path(backtrack_path, movement_controller, attitude_handler, scanner, visualizer, occupancy_map)
            else:
                print("🎉 EXPLORATION COMPLETE!"); break
    print("\n🎉 === EXPLORATION PHASE FINISHED ===")


# =============================================================================
# ===== MAIN EXECUTION BLOCK ==================================================
# =============================================================================
if __name__ == '__main__':
    occupancy_map = None
    attitude_handler = AttitudeHandler()
    movement_controller = None
    scanner = None
    
    if check_for_resume_data():
        if input("Resume from previous session? (y/n): ").lower().strip() in ['y', 'yes']:
            if load_resume_data(): occupancy_map = create_occupancy_map_from_json()
    if occupancy_map is None:
        print("🆕 Starting fresh session...")
        occupancy_map = OccupancyGridMap(width=4, height=4); RESUME_MODE = False

    manager = RMConnection()
    try:
        visualizer = RealTimeVisualizer(grid_size=4, target_dest=TARGET_DESTINATION)
        
        # --- Start Connection and Detection Threads ---
        reconn = threading.Thread(target=reconnector_thread, args=(manager,), daemon=True)
        reconn.start()
        print("Waiting for initial robot connection...")
        manager.connected.wait(timeout=15)
        if not manager.connected.is_set():
            raise ConnectionError("Failed to connect to robot on startup.")

        tracker = ObjectTracker(use_gpu=USE_GPU)
        roi_state = {"x": ROI_X0, "y": ROI_Y0, "w": ROI_W0, "h": ROI_H0}
        
        cap_t = threading.Thread(target=capture_thread_func, args=(manager, frame_queue), daemon=True)
        proc_t = threading.Thread(target=processing_thread_func, args=(tracker, frame_queue, roi_state), daemon=True)
        ctrl_t = threading.Thread(target=control_thread_func, args=(manager, roi_state), daemon=True)
        cap_t.start(); proc_t.start(); ctrl_t.start()
        
        # --- Initialize Robot Control Objects ---
        scanner = EnvironmentScanner(manager.ep_sensor_adaptor, manager.ep_tof_sensor, manager.ep_gimbal)
        movement_controller = MovementController(manager.ep_chassis)
        attitude_handler.start_monitoring(manager.ep_chassis)
        
        if RESUME_MODE: log_position_timestamp(CURRENT_POSITION, CURRENT_DIRECTION, "resume_session")
        else: log_position_timestamp(CURRENT_POSITION, CURRENT_DIRECTION, "new_session_start")
        
        explore_with_ogm(scanner, movement_controller, attitude_handler, occupancy_map, visualizer)
        
        print(f"\n--- NAVIGATION TO TARGET: {CURRENT_POSITION} to {TARGET_DESTINATION} ---")
        if CURRENT_POSITION != TARGET_DESTINATION:
            path_to_target = find_path_bfs(occupancy_map, CURRENT_POSITION, TARGET_DESTINATION)
            if path_to_target:
                execute_path(path_to_target, movement_controller, attitude_handler, scanner, visualizer, occupancy_map, "Final Navigation")
                print(f"🎉🎉 Robot arrived at target: {TARGET_DESTINATION}!")
            else:
                print(f"⚠️ Could not find a path to {TARGET_DESTINATION}.")

    except (KeyboardInterrupt, Exception) as e:
        if isinstance(e, KeyboardInterrupt): print("\n⚠️ User interrupted.")
        else: print(f"\n⚌ An error occurred: {e}"); traceback.print_exc()
    
    finally:
        stop_event.set() # Signal all threads to stop
        print("💾 Saving data before exit...")
        try:
            # Save Map
            nodes_data = []
            for r, c in np.ndindex(occupancy_map.height, occupancy_map.width):
                cell = occupancy_map.grid[r][c]
                nodes_data.append({
                    "coordinate": {"row": r, "col": c}, "probability": round(cell.get_node_probability(), 3),
                    "is_occupied": cell.is_node_occupied(),
                    "walls": {d: cell.walls[c].is_occupied() for c,d in zip(['N','S','E','W'],['north','south','east','west'])},
                    "wall_probabilities": {d: round(cell.walls[c].get_probability(), 3) for c,d in zip(['N','S','E','W'],['north','south','east','west'])}
                })
            with open(os.path.join(DATA_FOLDER, "Mapping_Top.json"), "w") as f:
                json.dump({"nodes": nodes_data}, f, indent=2)
            
            # Save Timestamps
            with open(os.path.join(DATA_FOLDER, "Robot_Position_Timestamps.json"), "w") as f:
                json.dump({"session_info": {"total_positions_logged": len(POSITION_LOG)}, "position_log": POSITION_LOG}, f, indent=2)

            # Save Detected Objects
            with open(os.path.join(DATA_FOLDER, "Detected_Objects.json"), "w") as f:
                json.dump(DETECTED_OBJECTS_LOG, f, indent=2)

            print("✅ All data saved.")
        except Exception as save_error:
            print(f"❌ Error saving data: {save_error}")
            
        if manager and manager.connected.is_set():
            print("🔌 Cleaning up and closing connection...")
            try:
                scanner.cleanup()
                attitude_handler.stop_monitoring(manager.ep_chassis)
                movement_controller.cleanup()
                manager.close()
                print("🔌 Connection closed.")
            except Exception as cleanup_error:
                print(f"⚠️ Error during cleanup: {cleanup_error}")
        
        print("... You can close the plot window now ...")
        plt.ioff(); plt.show()