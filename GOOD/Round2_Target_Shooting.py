# -*-coding:utf-8-*-
"""
ROUND 2: TARGET SHOOTING MISSION
โหลดแผนที่และข้อมูลวัตถุจากรอบที่ 1 (integrated_complete_ready.py)
แล้วเดินไปยิงเป้าหมายตามที่กำหนด โดยใช้ TSP algorithm หาเส้นทางที่สั้นที่สุด
"""

import time
import robomaster
from robomaster import robot, camera as r_camera, blaster as r_blaster
import numpy as np
import math
import json
import os
import cv2
import threading
import queue
from itertools import permutations
from collections import deque

# =============================================================================
# ===== CONFIGURATION & PARAMETERS ============================================
# =============================================================================

# Data folder (จากรอบที่ 1)
DATA_FOLDER = r"F:\Coder\Year2-1\Robot_Module\Assignment\dude\James_path"

# Grid size (ต้องตรงกับรอบที่ 1)
GRID = 4

# Robot configuration
CURRENT_POSITION = (3, 0)  # ตำแหน่งเริ่มต้น
CURRENT_DIRECTION = 0  # 0:North, 1:East, 2:South, 3:West
CURRENT_TARGET_YAW = 0.0
ROBOT_FACE = 1

# IMU Drift Compensation (เหมือนโค้ดรอบที่ 1)
IMU_COMPENSATION_START_NODE_COUNT = 7      # จำนวนโหนดขั้นต่ำก่อนเริ่มการชดเชย
IMU_COMPENSATION_NODE_INTERVAL = 10        # เพิ่มค่าชดเชยทุกๆ N โหนด
IMU_COMPENSATION_DEG_PER_INTERVAL = -2.0   # ค่าองศาที่ชดเชย (ลบเพื่อแก้การเบี้ยวขวา)
IMU_DRIFT_COMPENSATION_DEG = 0.0           # ตัวแปรเก็บค่าชดเชยปัจจุบัน

# PID Parameters for targeting
PID_KP = -0.25
PID_KI = -0.01
PID_KD = -0.03
DERIV_LPF_ALPHA = 0.25

MAX_YAW_SPEED = 220
MAX_PITCH_SPEED = 180
I_CLAMP = 2000.0

PIX_ERR_DEADZONE = 6
LOCK_TOL_X = 8
LOCK_TOL_Y = 8
LOCK_STABLE_COUNT = 6

# Camera Configuration
FRAME_W, FRAME_H = 960, 540
VERTICAL_FOV_DEG = 54.0
PIXELS_PER_DEG_V = FRAME_H / VERTICAL_FOV_DEG

PITCH_BIAS_DEG = 2.0
PITCH_BIAS_PIX = +PITCH_BIAS_DEG * PIXELS_PER_DEG_V

# ROI Configuration
ROI_Y0, ROI_H0, ROI_X0, ROI_W0 = 264, 270, 10, 911
ROI_SHIFT_PER_DEG = 6.0
ROI_Y_MIN, ROI_Y_MAX = 0, FRAME_H - 10

SPEED_ROTATE = 480

# GPU Configuration
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
# ===== SHARED VARIABLES & THREADING =========================================
# =============================================================================

frame_queue = queue.Queue(maxsize=1)
processed_output = {"details": []}
output_lock = threading.Lock()
stop_event = threading.Event()

# Gimbal angles
gimbal_angle_lock = threading.Lock()
gimbal_angles = (0.0, 0.0, 0.0, 0.0)

def sub_angle_cb(angle_info):
    global gimbal_angles
    with gimbal_angle_lock:
        gimbal_angles = tuple(angle_info)

# =============================================================================
# ===== IMU DRIFT COMPENSATION ================================================
# =============================================================================

def get_compensated_target_yaw():
    """
    Returns the current target yaw with the calculated IMU drift compensation.
    This function is now the single source of truth for the robot's target heading.
    """
    return CURRENT_TARGET_YAW + IMU_DRIFT_COMPENSATION_DEG

# =============================================================================
# ===== DATA LOADING FUNCTIONS ================================================
# =============================================================================

def load_map_data():
    """โหลดข้อมูลแผนที่จากรอบที่ 1"""
    try:
        map_file = os.path.join(DATA_FOLDER, "Mapping_Top.json")
        with open(map_file, "r", encoding="utf-8") as f:
            map_data = json.load(f)
        print(f"✅ Loaded map data from {map_file}")
        return map_data
    except Exception as e:
        print(f"❌ Error loading map data: {e}")
        return None

def load_detected_objects():
    """โหลดข้อมูลวัตถุที่ตรวจจับได้จากรอบที่ 1"""
    try:
        objects_file = os.path.join(DATA_FOLDER, "Detected_Objects.json")
        with open(objects_file, "r", encoding="utf-8") as f:
            objects_data = json.load(f)
        
        detected_objects = objects_data.get('detected_objects', [])
        print(f"✅ Loaded {len(detected_objects)} detected objects")
        
        # แสดงข้อมูลวัตถุที่โหลดมา
        for obj in detected_objects:
            pos = obj.get('cell_position', {})
            print(f"   - {obj.get('color')} {obj.get('shape')} at ({pos.get('row')}, {pos.get('col')}) [{obj.get('zone')}]")
        
        return detected_objects
    except Exception as e:
        print(f"❌ Error loading detected objects: {e}")
        return []

def reconstruct_occupancy_map(map_data):
    """สร้าง grid จากข้อมูล JSON (เก็บเฉพาะข้อมูลกำแพง)"""
    max_row = max(node['coordinate']['row'] for node in map_data['nodes'])
    max_col = max(node['coordinate']['col'] for node in map_data['nodes'])
    width = max_col + 1
    height = max_row + 1
    
    # สร้าง grid แบบง่าย
    grid = {}
    for node in map_data['nodes']:
        r = node['coordinate']['row']
        c = node['coordinate']['col']
        grid[(r, c)] = {
            'walls': node.get('walls', {}),
            'is_occupied': node.get('is_occupied', False),
            'objects': node.get('objects', [])
        }
    
    return grid, width, height

# =============================================================================
# ===== PATHFINDING FUNCTIONS =================================================
# =============================================================================

def find_path_bfs(grid, start, end, width, height):
    """หาเส้นทางด้วย BFS โดยคำนึงถึงกำแพง"""
    if start == end:
        return [start]
    
    queue = deque([[start]])
    visited = {start}
    moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # N, E, S, W
    wall_dirs = ['north', 'east', 'south', 'west']
    
    while queue:
        path = queue.popleft()
        r, c = path[-1]
        
        if (r, c) == end:
            return path
        
        for i, (dr, dc) in enumerate(moves):
            nr, nc = r + dr, c + dc
            
            # ตรวจสอบว่าอยู่ในขอบเขต
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            
            # ตรวจสอบว่าเคยเยี่ยมชมแล้ว
            if (nr, nc) in visited:
                continue
            
            # ตรวจสอบกำแพง
            if (r, c) in grid:
                if grid[(r, c)]['walls'].get(wall_dirs[i], False):
                    continue
            
            # ตรวจสอบว่าโหนดปลายทางถูกบล็อก
            if (nr, nc) in grid and grid[(nr, nc)]['is_occupied']:
                continue
            
            visited.add((nr, nc))
            new_path = list(path)
            new_path.append((nr, nc))
            queue.append(new_path)
    
    return None

def calculate_path_length(grid, positions, width, height):
    """คำนวณความยาวเส้นทางรวม"""
    if not positions:
        return float('inf')
    
    total_length = 0
    current = positions[0]
    
    for next_pos in positions[1:]:
        path = find_path_bfs(grid, current, next_pos, width, height)
        if path is None:
            return float('inf')
        total_length += len(path) - 1
        current = next_pos
    
    return total_length

def solve_tsp_optimal(grid, start_pos, target_positions, width, height):
    """แก้ปัญหา TSP เพื่อหาเส้นทางที่สั้นที่สุด"""
    if not target_positions:
        return []
    
    if len(target_positions) == 1:
        return [start_pos] + target_positions
    
    # ถ้ามีเป้าหมายมาก (>8) ใช้ greedy algorithm
    if len(target_positions) > 8:
        return solve_tsp_greedy(grid, start_pos, target_positions, width, height)
    
    # ลองทุกลำดับที่เป็นไปได้
    best_order = None
    best_length = float('inf')
    
    print(f"🔍 Calculating optimal path for {len(target_positions)} targets...")
    
    for perm in permutations(target_positions):
        path_sequence = [start_pos] + list(perm)
        length = calculate_path_length(grid, path_sequence, width, height)
        
        if length < best_length:
            best_length = length
            best_order = perm
    
    if best_order:
        print(f"✅ Optimal path length: {best_length} steps")
        return [start_pos] + list(best_order)
    else:
        return [start_pos] + target_positions

def solve_tsp_greedy(grid, start_pos, target_positions, width, height):
    """แก้ปัญหา TSP ด้วย Greedy algorithm (ใช้เมื่อมีเป้าหมายมาก)"""
    remaining = set(target_positions)
    path_sequence = [start_pos]
    current = start_pos
    
    print(f"🔍 Using greedy algorithm for {len(target_positions)} targets...")
    
    while remaining:
        nearest = None
        nearest_dist = float('inf')
        
        for target in remaining:
            path = find_path_bfs(grid, current, target, width, height)
            if path:
                dist = len(path) - 1
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest = target
        
        if nearest is None:
            print(f"⚠️ Cannot reach some targets from {current}")
            break
        
        path_sequence.append(nearest)
        remaining.remove(nearest)
        current = nearest
    
    return path_sequence

# =============================================================================
# ===== OBJECT DETECTION FUNCTIONS ==========================================
# =============================================================================

def apply_awb(bgr):
    """Apply automatic white balance"""
    if hasattr(cv2, "xphoto") and hasattr(cv2.xphoto, "createLearningBasedWB"):
        wb = cv2.xphoto.createLearningBasedWB()
        try:
            wb.setSaturationThreshold(0.99)
        except Exception:
            pass
        return wb.balanceWhite(bgr)
    return bgr

def night_enhance_pipeline_cpu(bgr):
    """Night enhancement pipeline"""
    return apply_awb(bgr)

class ObjectTracker:
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        print(f"🖼️  ObjectTracker in {'GPU' if use_gpu else 'CPU'} mode")

    def _get_angle(self, pt1, pt2, pt0):
        dx1 = pt1[0] - pt0[0]; dy1 = pt1[1] - pt0[1]
        dx2 = pt2[0] - pt0[0]; dy2 = pt2[1] - pt0[1]
        dot = dx1*dx2 + dy1*dy2
        mag1 = (dx1*dx1 + dy1*dy1)**0.5
        mag2 = (dx2*dx2 + dy2*dy2)**0.5
        if mag1*mag2 == 0:
            return 0
        return math.degrees(math.acos(max(-1, min(1, dot/(mag1*mag2)))) )

    def get_raw_detections(self, frame):
        enhanced = cv2.GaussianBlur(night_enhance_pipeline_cpu(frame), (5,5), 0)
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)

        ranges = {
            'Red': ([0,80,40],[10,255,255],[170,80,40],[180,255,255]),
            'Yellow': ([20,60,40],[35,255,255]),
            'Green': ([35,40,30],[85,255,255]),
            'Blue': ([90,40,30],[130,255,255])
        }
        masks = {}
        masks['Red'] = cv2.inRange(hsv, np.array(ranges['Red'][0]), np.array(ranges['Red'][1])) | \
                       cv2.inRange(hsv, np.array(ranges['Red'][2]), np.array(ranges['Red'][3]))
        for name in ['Yellow','Green','Blue']:
            masks[name] = cv2.inRange(hsv, np.array(ranges[name][0]), np.array(ranges[name][1]))

        combined = masks['Red'] | masks['Yellow'] | masks['Green'] | masks['Blue']
        kernel = np.ones((5,5), np.uint8)
        cleaned = cv2.morphologyEx(cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)

        contours,_ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        H,W = frame.shape[:2]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1500: continue
            x,y,w,h = cv2.boundingRect(cnt)
            if w==0 or h==0: continue
            ar = w/float(h)
            if ar>4.0 or ar<0.25: continue
            hull = cv2.convexHull(cnt); ha = cv2.contourArea(hull)
            if ha==0: continue
            solidity = area/ha
            if solidity < 0.85: continue
            if x<=2 or y<=2 or x+w>=W-2 or y+h>=H-2: continue

            contour_mask = np.zeros((H,W), np.uint8)
            cv2.drawContours(contour_mask, [cnt], -1, 255, -1)
            max_mean, found = 0, "Unknown"
            for cname, m in masks.items():
                mv = cv2.mean(m, mask=contour_mask)[0]
                if mv > max_mean:
                    max_mean, found = mv, cname
            if max_mean <= 20: continue

            shape = "Uncertain"
            peri = cv2.arcLength(cnt, True)
            circ = (4*math.pi*area)/(peri*peri) if peri>0 else 0
            if circ > 0.82:
                shape = "Circle"
            else:
                approx = cv2.approxPolyDP(cnt, 0.04*peri, True)
                if len(approx)==4 and solidity>0.88:
                    pts=[tuple(p[0]) for p in approx]
                    angs=[self._get_angle(pts[(i-1)%4], pts[(i+1)%4], p) for i,p in enumerate(pts)]
                    if all(70<=a<=110 for a in angs):
                        _,(rw,rh),_ = cv2.minAreaRect(cnt)
                        if min(rw,rh)>0:
                            ar2 = max(rw,rh)/min(rw,rh)
                            if 0.88<=ar2<=1.12: shape="Square"
                            elif w>h: shape="Rectangle_H"
                            else: shape="Rectangle_V"
            out.append({"contour":cnt,"shape":shape,"color":found,"box":(x,y,w,h)})
        return out

# =============================================================================
# ===== ROBOT CONNECTION & THREADING =========================================
# =============================================================================

class RMConnection:
    def __init__(self):
        self._lock = threading.Lock()
        self._robot = None
        self.connected = threading.Event()

    def connect(self):
        with self._lock:
            self._safe_close()
            print("🤖 Connecting to RoboMaster...")
            try:
                rb = robot.Robot()
                rb.initialize(conn_type="ap")
                time.sleep(1.0)
                rb.camera.start_video_stream(display=False, resolution=r_camera.STREAM_540P)
                time.sleep(1.0)
                try:
                    rb.gimbal.sub_angle(freq=20, callback=sub_angle_cb)
                except Exception as e:
                    print("Gimbal sub_angle error:", e)
                self._robot = rb
                self.connected.set()
                print("✅ RoboMaster connected & camera streaming")

                try:
                    rb.gimbal.recenter(pitch_speed=200, yaw_speed=200).wait_for_completed()
                except Exception as e:
                    print("Recenter error:", e)
            except Exception as e:
                print(f"❌ Connection failed: {e}")
                self._safe_close()

    def _safe_close(self):
        if self._robot is not None:
            try:
                try: self._robot.camera.stop_video_stream()
                except Exception: pass
                try:
                    try: self._robot.gimbal.unsub_angle()
                    except Exception: pass
                except Exception: pass
                try: self._robot.close()
                except Exception: pass
            finally:
                self._robot = None
                self.connected.clear()
                print("🔌 Connection closed")

    def drop_and_reconnect(self):
        with self._lock:
            self._safe_close()

    def get_camera(self):
        with self._lock:
            return None if self._robot is None else self._robot.camera

    def get_gimbal(self):
        with self._lock:
            return None if self._robot is None else self._robot.gimbal

    def get_blaster(self):
        with self._lock:
            return None if self._robot is None else self._robot.blaster

    def close(self):
        with self._lock:
            self._safe_close()

def reconnector_thread(manager: RMConnection):
    backoff = 1.0
    while not stop_event.is_set():
        if not manager.connected.is_set():
            try:
                manager.connect()
                backoff = 1.0
            except Exception as e:
                print(f"♻️ Reconnect failed: {e} (retry in {backoff:.1f}s)")
                time.sleep(backoff)
                backoff = min(backoff*1.6, 8.0)
                continue
        time.sleep(0.2)

def capture_thread_func(manager: RMConnection, q: queue.Queue):
    print("🚀 Capture thread started")
    fail = 0
    
    while not stop_event.is_set():
        if not manager.connected.is_set():
            time.sleep(0.1)
            continue
            
        cam = manager.get_camera()
        if cam is None:
            time.sleep(0.1)
            continue
            
        try:
            frame = cam.read_cv2_image(timeout=1.0)
            if frame is not None and frame.size > 0:
                if q.full():
                    try: 
                        q.get_nowait()
                    except queue.Empty: 
                        pass
                
                q.put(frame)
                fail = 0
            else:
                fail += 1
                
        except Exception as e:
            print(f"⚠️ Camera read error: {e}")
            fail += 1

        if fail >= 10:
            print("⚠️ Too many camera errors → drop & reconnect")
            manager.drop_and_reconnect()
            try:
                while True: 
                    q.get_nowait()
            except queue.Empty:
                pass
            fail = 0
            time.sleep(0.2)
            
        time.sleep(0.005)
    print("🛑 Capture thread stopped")

def processing_thread_func(tracker: ObjectTracker, q: queue.Queue,
                           target_shape, target_color,
                           roi_state):
    global processed_output
    print("🧠 Processing thread started.")

    while not stop_event.is_set():
        try:
            frame_to_process = q.get(timeout=0.3)

            with gimbal_angle_lock:
                pitch_deg = gimbal_angles[0]
            roi_y_dynamic = int(ROI_Y0 - (max(0.0, -pitch_deg) * ROI_SHIFT_PER_DEG))
            roi_y_dynamic = max(ROI_Y_MIN, min(ROI_Y_MAX, roi_y_dynamic))

            ROI_X, ROI_W = roi_state["x"], roi_state["w"]
            ROI_H = roi_state["h"]
            roi_state["y"] = roi_y_dynamic

            roi_frame = frame_to_process[roi_y_dynamic:roi_y_dynamic+ROI_H, ROI_X:ROI_X+ROI_W]
            detections = tracker.get_raw_detections(roi_frame)

            detailed_results = []

            for d in detections:
                shape, color, (x,y,w,h) = d['shape'], d['color'], d['box']
                is_target = (shape == target_shape and color == target_color)

                detailed_results.append({
                    "color": color,
                    "shape": shape,
                    "is_target": is_target,
                    "box": (x,y,w,h),
                    "center": (x + w//2, y + h//2)
                })

            with output_lock:
                processed_output = {"details": detailed_results}

        except queue.Empty:
            time.sleep(0.1)
            continue
        except Exception as e:
            print(f"⚠️ Processing error: {e}")
            time.sleep(0.1)
            try:
                while True: 
                    q.get_nowait()
            except queue.Empty:
                pass

    print("🛑 Processing thread stopped.")

# =============================================================================
# ===== ROBOT CONTROL CLASSES =================================================
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
            time.sleep(0.1)
        final_error = abs(self.normalize_angle(normalized_target - self.current_yaw))
        if final_error <= self.yaw_tolerance: print(f"✅ Yaw Correction Success: {self.current_yaw:.1f}°"); return True
        print(f"⚠️ First attempt incomplete. Current: {self.current_yaw:.1f}°. Fine-tuning...")
        remaining_rotation = -self.normalize_angle(normalized_target - self.current_yaw)
        if abs(remaining_rotation) > 0.5 and abs(remaining_rotation) < 20:
            chassis.move(x=0, y=0, z=remaining_rotation, z_speed=40).wait_for_completed(timeout=2)
            time.sleep(0.1)
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

    def move_forward_one_grid(self, axis, attitude_handler, target_yaw):
        attitude_handler.correct_yaw_to_target(self.chassis, get_compensated_target_yaw())  # ใช้ค่าชดเชย
        target_distance = 0.6
        pid = PID(Kp=1.0, Ki=0.25, Kd=8, setpoint=target_distance)
        start_time, last_time = time.time(), time.time()
        start_position = self.current_x_pos if axis == 'x' else self.current_y_pos
        print(f"🚀 Moving FORWARD 0.6m, monitoring GLOBAL AXIS '{axis}'")
        while time.time() - start_time < 3.5:
            now = time.time(); dt = now - last_time; last_time = now
            current_position = self.current_x_pos if axis == 'x' else self.current_y_pos
            relative_position = abs(current_position - start_position)
            if abs(relative_position - target_distance) < 0.03:
                print("\n✅ Move complete!"); break
            output = pid.compute(relative_position, dt)
            ramp_multiplier = min(1.0, 0.1 + ((now - start_time) / 1.0) * 0.9)
            speed = max(-1.0, min(1.0, output * ramp_multiplier))
            yaw_correction = self._calculate_yaw_correction(attitude_handler, get_compensated_target_yaw())  # ใช้ค่าชดเชย
            self.chassis.drive_speed(x=speed, y=0, z=yaw_correction, timeout=1)
            print(f"Moving... Dist: {relative_position:.3f}/{target_distance:.2f} m", end='\r')
        self.chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0); time.sleep(0.5)

    def rotate_to_direction(self, target_direction, current_direction, attitude_handler, current_yaw):
        if current_direction == target_direction: 
            return current_direction, current_yaw
        
        diff = (target_direction - current_direction + 4) % 4
        if diff == 1: 
            current_direction, current_yaw = self.rotate_90_degrees_right(current_direction, attitude_handler, current_yaw)
        elif diff == 3: 
            current_direction, current_yaw = self.rotate_90_degrees_left(current_direction, attitude_handler, current_yaw)
        elif diff == 2: 
            current_direction, current_yaw = self.rotate_90_degrees_right(current_direction, attitude_handler, current_yaw)
            current_direction, current_yaw = self.rotate_90_degrees_right(current_direction, attitude_handler, current_yaw)
        
        return current_direction, current_yaw

    def rotate_90_degrees_right(self, current_direction, attitude_handler, current_yaw):
        print("🔄 Rotating 90° RIGHT...")
        new_yaw = attitude_handler.normalize_angle(current_yaw + 90)
        attitude_handler.correct_yaw_to_target(self.chassis, new_yaw)
        new_direction = (current_direction + 1) % 4
        return new_direction, new_yaw
    
    def rotate_90_degrees_left(self, current_direction, attitude_handler, current_yaw):
        print("🔄 Rotating 90° LEFT...")
        new_yaw = attitude_handler.normalize_angle(current_yaw - 90)
        attitude_handler.correct_yaw_to_target(self.chassis, new_yaw)
        new_direction = (current_direction - 1 + 4) % 4
        return new_direction, new_yaw

    def cleanup(self):
        try: self.chassis.unsub_position()
        except Exception: pass

# =============================================================================
# ===== TARGET SHOOTING FUNCTIONS ============================================
# =============================================================================

class PIDController:
    """PID Controller สำหรับการเล็งเป้า"""
    def __init__(self, kp, ki, kd, i_clamp=1000.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.i_clamp = i_clamp
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_deriv = 0.0
        
    def compute(self, error, dt):
        if dt <= 0:
            dt = 0.02
        
        # P term
        p_term = self.kp * error
        
        # I term with clamping
        self.integral += error * dt
        self.integral = max(min(self.integral, self.i_clamp), -self.i_clamp)
        i_term = self.ki * self.integral
        
        # D term with low-pass filter
        raw_deriv = (error - self.prev_error) / dt
        filtered_deriv = DERIV_LPF_ALPHA * raw_deriv + (1 - DERIV_LPF_ALPHA) * self.prev_deriv
        d_term = self.kd * filtered_deriv
        
        self.prev_error = error
        self.prev_deriv = filtered_deriv
        
        return p_term + i_term + d_term
    
    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_deriv = 0.0

def aim_and_shoot_target(manager, target_shape, target_color, roi_state, max_lock_time=15.0):
    """เล็งและยิงเป้าหมาย"""
    print(f"\n🎯 Starting aim sequence for {target_color} {target_shape}")
    
    gimbal = manager.get_gimbal()
    blaster = manager.get_blaster()
    
    if gimbal is None or blaster is None:
        print("❌ Gimbal or Blaster not available")
        return False
    
    # Reset gimbal to center
    print("📐 Centering gimbal...")
    gimbal.recenter(pitch_speed=200, yaw_speed=200).wait_for_completed()
    time.sleep(0.5)
    
    # PID controllers
    pid_yaw = PIDController(PID_KP, PID_KI, PID_KD, I_CLAMP)
    pid_pitch = PIDController(PID_KP, PID_KI, PID_KD, I_CLAMP)
    
    lock_start_time = time.time()
    stable_count = 0
    last_time = time.time()
    
    print("🔍 Searching for target...")
    
    while time.time() - lock_start_time < max_lock_time:
        now = time.time()
        dt = now - last_time
        last_time = now
        
        # Get current gimbal angles
        with gimbal_angle_lock:
            current_pitch, current_yaw = gimbal_angles[0], gimbal_angles[1]
        
        # Get detected objects
        with output_lock:
            details = processed_output.get("details", [])
        
        # Find matching target
        target_obj = None
        for obj in details:
            if obj["shape"] == target_shape and obj["color"] == target_color:
                target_obj = obj
                break
        
        if target_obj is None:
            # Target not found - search slowly
            stable_count = 0
            pid_yaw.reset()
            pid_pitch.reset()
            time.sleep(0.05)
            continue
        
        # Calculate pixel errors
        cx, cy = target_obj["center"]
        roi_cx = roi_state["w"] // 2
        roi_cy = roi_state["h"] // 2
        
        # Add bias for pitch
        roi_cy_biased = int(roi_cy - PITCH_BIAS_PIX)
        
        err_x = cx - roi_cx
        err_y = cy - roi_cy_biased
        
        print(f"🎯 Target found! Error: X={err_x:.0f}px, Y={err_y:.0f}px, Pitch={current_pitch:.1f}°, Yaw={current_yaw:.1f}°", end='\r')
        
        # Check if locked
        if abs(err_x) < LOCK_TOL_X and abs(err_y) < LOCK_TOL_Y:
            stable_count += 1
            if stable_count >= LOCK_STABLE_COUNT:
                print(f"\n✅ Target LOCKED after {time.time() - lock_start_time:.1f}s!")
                print(f"🔫 Firing at target...")
                
                # Fire multiple shots
                try:
                    blaster.fire(fire_type=r_blaster.WATER_FIRE, times=3)
                    time.sleep(0.5)
                except Exception as e:
                    print(f"⚠️ Firing error: {e}")
                
                print("✅ Shots fired!")
                return True
        else:
            stable_count = 0
        
        # Apply deadzone
        if abs(err_x) < PIX_ERR_DEADZONE:
            err_x = 0
        if abs(err_y) < PIX_ERR_DEADZONE:
            err_y = 0
        
        # Compute PID outputs
        yaw_output = pid_yaw.compute(err_x, dt)
        pitch_output = pid_pitch.compute(err_y, dt)
        
        # Clamp speeds
        yaw_speed = max(min(yaw_output, MAX_YAW_SPEED), -MAX_YAW_SPEED)
        pitch_speed = max(min(pitch_output, MAX_PITCH_SPEED), -MAX_PITCH_SPEED)
        
        # Move gimbal
        try:
            gimbal.drive_speed(pitch_speed=pitch_speed, yaw_speed=yaw_speed)
        except Exception as e:
            print(f"\n⚠️ Gimbal control error: {e}")
            time.sleep(0.05)
        
        time.sleep(0.02)
    
    print(f"\n⚠️ Target lock timeout after {max_lock_time}s")
    return False

def execute_shooting_mission(grid, width, height, target_sequence, movement_controller, 
                            attitude_handler, manager, roi_state):
    """ดำเนินการภารกิจยิงเป้าหมายตามลำดับ"""
    global CURRENT_POSITION, CURRENT_DIRECTION, CURRENT_TARGET_YAW, ROBOT_FACE, IMU_DRIFT_COMPENSATION_DEG
    
    current_pos = target_sequence[0]
    CURRENT_POSITION = current_pos
    nodes_visited = 0  # นับจำนวนโหนดที่เดินผ่าน
    
    print(f"\n🎯 Starting shooting mission from {current_pos}")
    print(f"📍 Targets to shoot: {len(target_sequence) - 1}")
    
    for i in range(1, len(target_sequence)):
        target_info = target_sequence[i]
        target_pos = target_info['shoot_from']
        target_obj = target_info['object']
        
        print(f"\n{'='*60}")
        print(f"🎯 Target {i}/{len(target_sequence)-1}: {target_obj['color']} {target_obj['shape']}")
        print(f"📍 Moving to position: {target_pos}")
        print(f"{'='*60}")
        
        # Find path to shooting position
        path = find_path_bfs(grid, CURRENT_POSITION, target_pos, width, height)
        
        if path is None or len(path) == 0:
            print(f"❌ Cannot find path from {CURRENT_POSITION} to {target_pos}")
            continue
        
        print(f"🗺️ Path: {path}")
        
        # Execute path
        dir_vectors_map = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}
        
        for j in range(len(path) - 1):
            current_r, current_c = path[j]
            next_r, next_c = path[j+1]
            dr, dc = next_r - current_r, next_c - current_c
            
            target_direction = dir_vectors_map[(dr, dc)]
            
            # Rotate to correct direction
            CURRENT_DIRECTION, CURRENT_TARGET_YAW = movement_controller.rotate_to_direction(
                target_direction, CURRENT_DIRECTION, attitude_handler, CURRENT_TARGET_YAW
            )
            
            # Update ROBOT_FACE
            if CURRENT_DIRECTION == 0:  # North
                ROBOT_FACE = 1
            elif CURRENT_DIRECTION == 1:  # East
                ROBOT_FACE = 2
            elif CURRENT_DIRECTION == 2:  # South
                ROBOT_FACE = 3
            else:  # West
                ROBOT_FACE = 4
            
            # Move forward
            axis_to_monitor = 'x' if ROBOT_FACE % 2 != 0 else 'y'
            movement_controller.move_forward_one_grid(axis_to_monitor, attitude_handler, CURRENT_TARGET_YAW)
            
            CURRENT_POSITION = (next_r, next_c)
            nodes_visited += 1
            print(f"✅ Moved to {CURRENT_POSITION} (Node #{nodes_visited})")
            
            # Update IMU drift compensation (เหมือนโค้ดรอบที่ 1)
            if nodes_visited >= IMU_COMPENSATION_START_NODE_COUNT:
                compensation_intervals = nodes_visited // IMU_COMPENSATION_NODE_INTERVAL
                new_compensation = compensation_intervals * IMU_COMPENSATION_DEG_PER_INTERVAL
                if new_compensation != IMU_DRIFT_COMPENSATION_DEG:
                    IMU_DRIFT_COMPENSATION_DEG = new_compensation
                    print(f"🔩 IMU Drift Compensation Updated: {IMU_DRIFT_COMPENSATION_DEG:.1f}° (after {nodes_visited} nodes)")
        
        # Now at target position - aim and shoot
        print(f"🎯 At position {target_pos} - Searching for target...")
        
        # Aim and shoot
        success = aim_and_shoot_target(
            manager, 
            target_obj['shape'], 
            target_obj['color'], 
            roi_state,
            max_lock_time=15.0
        )
        
        if success:
            print(f"✅ Successfully shot target {i}!")
        else:
            print(f"❌ Failed to shoot target {i}")
        
        # Wait before next target
        time.sleep(1.0)
    
    print(f"\n{'='*60}")
    print("🎉 MISSION COMPLETE!")
    print(f"{'='*60}")

# =============================================================================
# ===== MAIN EXECUTION ========================================================
# =============================================================================

if __name__ == '__main__':
    print("="*60)
    print("🎯 ROUND 2: TARGET SHOOTING MISSION")
    print("="*60)
    
    # Load data from round 1
    print("\n📂 Loading data from Round 1...")
    map_data = load_map_data()
    detected_objects = load_detected_objects()
    
    if map_data is None or len(detected_objects) == 0:
        print("❌ Failed to load data. Please run Round 1 first.")
        exit(1)
    
    # Reconstruct map
    grid, width, height = reconstruct_occupancy_map(map_data)
    print(f"✅ Map reconstructed: {width}x{height}")
    
    # Show available objects
    print(f"\n📦 Available objects ({len(detected_objects)}):")
    unique_objects = {}
    for obj in detected_objects:
        key = f"{obj.get('color', 'Unknown')} {obj.get('shape', 'Unknown')}"
        pos = obj.get('cell_position', {})
        position = (pos.get('row', '?'), pos.get('col', '?'))
        if key not in unique_objects:
            unique_objects[key] = []
        unique_objects[key].append(position)
    
    for key, positions in unique_objects.items():
        print(f"  - {key}: {len(positions)} found at {positions}")
    
    # Get target from user
    print("\n" + "="*60)
    print("🎯 TARGET SELECTION")
    print("="*60)
    print("\nAvailable colors: Red, Yellow, Green, Blue")
    print("Available shapes: Circle, Square, Rectangle_H, Rectangle_V")
    
    target_color = input("\n🎨 Enter target color: ").strip()
    target_shape = input("📐 Enter target shape: ").strip()
    
    # Filter matching objects
    matching_targets = []
    for obj in detected_objects:
        if obj.get('shape') == target_shape and obj.get('color') == target_color:
            pos = obj.get('cell_position', {})
            if 'row' in pos and 'col' in pos:
                obj['row'] = pos['row']
                obj['col'] = pos['col']
                matching_targets.append(obj)
    
    if len(matching_targets) == 0:
        print(f"\n❌ No {target_color} {target_shape} found in map!")
        print("Please check your input and try again.")
        exit(1)
    
    print(f"\n✅ Found {len(matching_targets)} matching target(s):")
    for i, obj in enumerate(matching_targets):
        zone = obj.get('zone', 'unknown')
        print(f"  {i+1}. Position: ({obj['row']}, {obj['col']}), Zone: {zone}")
    
    # Calculate shooting positions (go directly to recorded position)
    target_positions = []
    
    for obj in matching_targets:
        obj_pos = (obj['row'], obj['col'])
        
        # Check if the position is accessible
        if obj_pos in grid and not grid[obj_pos]['is_occupied']:
            target_positions.append({
                'object': obj,
                'shoot_from': obj_pos
            })
            print(f"  → Will shoot from {obj_pos}")
        else:
            print(f"  ⚠️ Cannot access position {obj_pos} (occupied)")
    
    if len(target_positions) == 0:
        print("\n❌ Cannot find valid shooting positions!")
        exit(1)
    
    # Solve TSP to find optimal order
    print(f"\n🗺️ Planning optimal route for {len(target_positions)} target(s)...")
    
    start_pos = CURRENT_POSITION
    shoot_positions = [t['shoot_from'] for t in target_positions]
    
    optimal_sequence = solve_tsp_optimal(grid, start_pos, shoot_positions, width, height)
    
    # Create ordered target sequence
    ordered_targets = [start_pos]
    for pos in optimal_sequence[1:]:
        for t in target_positions:
            if t['shoot_from'] == pos:
                ordered_targets.append(t)
                break
    
    print(f"✅ Optimal route calculated!")
    print(f"📍 Route: {[t if isinstance(t, tuple) else t['shoot_from'] for t in ordered_targets]}")
    
    # Initialize robot and detection system
    print("\n" + "="*60)
    print("🤖 INITIALIZING ROBOT SYSTEM")
    print("="*60)
    
    tracker = ObjectTracker(use_gpu=USE_GPU)
    manager = RMConnection()
    
    roi_state = {"x": ROI_X0, "y": ROI_Y0, "w": ROI_W0, "h": ROI_H0}
    
    # Start threads
    reconn = threading.Thread(target=reconnector_thread, args=(manager,), daemon=True)
    reconn.start()
    
    cap_t = threading.Thread(target=capture_thread_func, args=(manager, frame_queue), daemon=True)
    proc_t = threading.Thread(target=processing_thread_func,
                              args=(tracker, frame_queue, target_shape, target_color, roi_state),
                              daemon=True)
    
    cap_t.start()
    proc_t.start()
    
    print("⏳ Waiting for camera connection...")
    time.sleep(3.0)
    
    if not manager.connected.is_set():
        print("❌ Camera connection failed!")
        exit(1)
    
    print("✅ Camera ready!")
    
    # Connect to robot
    try:
        ep_robot = robot.Robot()
        ep_robot.initialize(conn_type="ap")
        time.sleep(2.0)
        
        ep_chassis = ep_robot.chassis
        ep_gimbal = ep_robot.gimbal
        
        print("✅ Robot connected!")
        
        # Initialize controllers
        movement_controller = MovementController(ep_chassis)
        attitude_handler = AttitudeHandler()
        attitude_handler.start_monitoring(ep_chassis)
        
        # Center gimbal
        print("📐 Centering gimbal...")
        ep_gimbal.recenter(pitch_speed=200, yaw_speed=200).wait_for_completed()
        time.sleep(1.0)
        
        # Start mission
        print("\n" + "="*60)
        print("🚀 STARTING SHOOTING MISSION")
        print("="*60)
        
        input("\nPress ENTER to start mission...")
        
        execute_shooting_mission(
            grid, width, height,
            ordered_targets,
            movement_controller,
            attitude_handler,
            manager,
            roi_state
        )
        
    except KeyboardInterrupt:
        print("\n⚠️ Mission interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🔌 Cleaning up...")
        stop_event.set()
        
        try:
            cap_t.join(timeout=2.0)
            proc_t.join(timeout=2.0)
        except:
            pass
        
        try:
            if 'movement_controller' in locals():
                movement_controller.cleanup()
            if 'attitude_handler' in locals() and attitude_handler.is_monitoring:
                attitude_handler.stop_monitoring(ep_chassis)
            if 'manager' in locals():
                manager.close()
            if 'ep_robot' in locals():
                ep_robot.close()
        except:
            pass
        
        print("✅ Cleanup complete!")