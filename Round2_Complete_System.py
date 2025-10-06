# -*-coding:utf-8-*-

"""
Round 2: Complete Target Shooting System
ระบบยิงเป้าหมายรอบที่ 2 แบบครบครันในไฟล์เดียว
"""

import time
import robomaster
from robomaster import robot, camera as r_camera, blaster as r_blaster
import numpy as np
import math
import json
import matplotlib.pyplot as plt
import threading
import queue
from collections import deque
import traceback
import statistics
import os
import cv2
import threading
import queue
from itertools import permutations
import random

# =============================================================================
# ===== CONFIGURATION & PARAMETERS ============================================
# =============================================================================

# Data folder (จากรอบที่ 1)
DATA_FOLDER = r"./Assignment/dude/James_path"

# Grid size
GRID = 5

# Robot configuration
CURRENT_POSITION = (4, 0)
CURRENT_DIRECTION = 1  # 0:North, 1:East, 2:South, 3:West
CURRENT_TARGET_YAW = 0.0
ROBOT_FACE = 1

# IMU Drift Compensation
IMU_DRIFT_COMPENSATION_DEG = 0.0

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
PITCH_BIAS_DEG = 2.5
PITCH_BIAS_PIX = +PITCH_BIAS_DEG * PIXELS_PER_DEG_V

# ROI Configuration
ROI_Y0, ROI_H0, ROI_X0, ROI_W0 = 264, 270, 10, 911
ROI_SHIFT_PER_DEG = 6.0
ROI_Y_MIN, ROI_Y_MAX = 0, FRAME_H - 10

# Movement parameters
SPEED_ROTATE = 480
FIRE_SHOTS_COUNT = 2

# Global variables
is_tracking_mode = False
fired_targets = set()
shots_fired = 0
gimbal_angle_lock = threading.Lock()
gimbal_angles = (0.0, 0.0, 0.0, 0.0)

# Camera Threading (from Round 1)
frame_queue = queue.Queue(maxsize=1)
processed_output = {"details": []}
output_lock = threading.Lock()
stop_event = threading.Event()

# =============================================================================
# ===== DATA LOADING FUNCTIONS ===============================================
# =============================================================================

def load_data():
    """โหลดข้อมูลจากไฟล์ JSON ทั้ง 3 ไฟล์"""
    print("📂 Loading data from Round 1...")
    
    try:
        # โหลดข้อมูลแผนที่
        map_file = os.path.join(DATA_FOLDER, "Mapping_Top.json")
        with open(map_file, "r", encoding="utf-8") as f:
            map_data = json.load(f)
        print(f"✅ Map data loaded: {len(map_data['nodes'])} nodes")
        
        # โหลดข้อมูลวัตถุ
        objects_file = os.path.join(DATA_FOLDER, "Detected_Objects.json")
        with open(objects_file, "r", encoding="utf-8") as f:
            objects_data = json.load(f)
        print(f"✅ Objects data loaded: {len(objects_data['detected_objects'])} objects")
        
        # โหลดข้อมูลการเคลื่อนที่
        timestamp_file = os.path.join(DATA_FOLDER, "Robot_Position_Timestamps.json")
        with open(timestamp_file, "r", encoding="utf-8") as f:
            timestamp_data = json.load(f)
        print(f"✅ Timestamp data loaded: {len(timestamp_data['position_log'])} positions")
        
        return map_data, objects_data, timestamp_data
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None, None

def create_grid_from_map(map_data):
    """สร้างกริดจากข้อมูลแผนที่"""
    print("🗺️ Creating grid from map data...")
    
    # หาขนาดของกริด
    max_row = max(node['coordinate']['row'] for node in map_data['nodes'])
    max_col = max(node['coordinate']['col'] for node in map_data['nodes'])
    width = max_col + 1
    height = max_row + 1
    
    print(f"📊 Grid size: {width}x{height}")
    
    # สร้างกริด - เริ่มต้นด้วย 0 (free) ทั้งหมด
    grid = [[0 for _ in range(width)] for _ in range(height)]
    
    # กำหนดค่าให้แต่ละเซลล์
    for node_data in map_data['nodes']:
        r = node_data['coordinate']['row']
        c = node_data['coordinate']['col']
        
        # 0 = free, 1 = occupied
        if node_data['is_occupied']:
            grid[r][c] = 1
        else:
            grid[r][c] = 0  # free
    
    # Debug: แสดงกริด
    print("🔍 Grid layout (X=occupied, .=free):")
    for i in range(height):
        row_str = ""
        for j in range(width):
            if grid[i][j] == 0:
                row_str += "."
            elif grid[i][j] == 1:
                row_str += "X"
            else:
                row_str += "#"
        print(f"  Row {i}: {row_str}")
    
    return grid, width, height

# =============================================================================
# ===== PATHFINDING ALGORITHMS ===============================================
# =============================================================================

def find_path_bfs(grid, start, end, width, height, map_data=None):
    """ใช้ BFS หาเส้นทางสั้นที่สุด โดยคำนึงถึงกำแพง"""
    if start == end:
        return [start]
    
    queue = deque([[start]])
    visited = {start}
    moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # N, E, S, W
    move_names = ['N', 'E', 'S', 'W']
    
    while queue:
        path = queue.popleft()
        r, c = path[-1]
        
        if (r, c) == end:
            return path
        
        for i, (dr, dc) in enumerate(moves):
            nr, nc = r + dr, c + dc
            
            # ตรวจสอบขอบเขต
            if not (0 <= nr < height and 0 <= nc < width):
                continue
                
            # ตรวจสอบว่าถูกเยี่ยมแล้วหรือไม่
            if (nr, nc) in visited:
                continue
                
            # ตรวจสอบว่าเซลล์ว่างหรือไม่
            if grid[nr][nc] != 0:
                continue
            
            # ตรวจสอบกำแพงระหว่างเซลล์ปัจจุบันกับเซลล์ถัดไป
            if map_data and not can_move_between_cells(r, c, nr, nc, move_names[i], map_data):
                continue
            
            visited.add((nr, nc))
            new_path = list(path)
            new_path.append((nr, nc))
            queue.append(new_path)
    
    return None

def can_move_between_cells(from_row, from_col, to_row, to_col, direction, map_data):
    """ตรวจสอบว่าสามารถเดินจากเซลล์หนึ่งไปอีกเซลล์หนึ่งได้หรือไม่"""
    # หา node ของเซลล์ต้นทาง
    from_node = None
    for node in map_data['nodes']:
        if node['coordinate']['row'] == from_row and node['coordinate']['col'] == from_col:
            from_node = node
            break
    
    if not from_node:
        return False
    
    # ตรวจสอบกำแพงตามทิศทาง
    walls = from_node['walls']
    
    if direction == 'N' and walls['north']:
        return False
    elif direction == 'S' and walls['south']:
        return False
    elif direction == 'E' and walls['east']:
        return False
    elif direction == 'W' and walls['west']:
        return False
    
    return True

def solve_tsp_greedy(grid, start_pos, target_positions, width, height, map_data=None):
    """แก้ปัญหา TSP แบบ greedy"""
    if not target_positions:
        return []
    
    print(f"🔍 TSP Debug: Start position: {start_pos}")
    print(f"🔍 TSP Debug: Target positions: {target_positions}")
    print(f"🔍 TSP Debug: Grid size: {width}x{height}")
    
    remaining_targets = target_positions.copy()
    current_pos = start_pos
    ordered_targets = []
    
    while remaining_targets:
        min_distance = float('inf')
        nearest_target = None
        
        for target in remaining_targets:
            print(f"🔍 TSP Debug: Trying to find path from {current_pos} to {target}")
            path = find_path_bfs(grid, current_pos, target, width, height, map_data)
            if path:
                distance = len(path) - 1
                print(f"🔍 TSP Debug: Path found! Distance: {distance}, Path: {path}")
                if distance < min_distance:
                    min_distance = distance
                    nearest_target = target
            else:
                print(f"🔍 TSP Debug: No path found from {current_pos} to {target}")
        
        if nearest_target:
            print(f"🔍 TSP Debug: Nearest target: {nearest_target} (distance: {min_distance})")
            ordered_targets.append(nearest_target)
            remaining_targets.remove(nearest_target)
            current_pos = nearest_target
        else:
            print("⚠️ Cannot reach some targets, stopping TSP")
            break
    
    print(f"🎯 TSP solution: {len(ordered_targets)} targets")
    return ordered_targets

# =============================================================================
# ===== OBJECT DETECTION =====================================================
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
# ===== ROBOT CONNECTION & CONTROL ===========================================
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
            rb = robot.Robot()
            rb.initialize(conn_type="ap")
            rb.camera.start_video_stream(display=False, resolution=r_camera.STREAM_540P)
            try:
                rb.gimbal.sub_angle(freq=50, callback=sub_angle_cb)
            except Exception as e:
                print("Gimbal sub_angle error:", e)
            self._robot = rb
            self.connected.set()
            print("✅ RoboMaster connected & camera streaming")

            try:
                rb.gimbal.recenter(pitch_speed=200, yaw_speed=200).wait_for_completed()
            except Exception as e:
                print("Recenter error:", e)

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

    def get_robot(self):
        with self._lock:
            return self._robot

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

def sub_angle_cb(angle_info):
    global gimbal_angles
    with gimbal_angle_lock:
        gimbal_angles = tuple(angle_info)

def get_compensated_target_yaw():
    """Returns the current target yaw with IMU drift compensation"""
    return CURRENT_TARGET_YAW + IMU_DRIFT_COMPENSATION_DEG

# =============================================================================
# ===== CAMERA THREADING FUNCTIONS (from Round 1) ============================
# =============================================================================

def capture_thread_func(manager: RMConnection, q: queue.Queue):
    """Camera capture thread"""
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
                if fail > 10:
                    print("⚠️ Camera capture failed multiple times")
                    time.sleep(0.5)
        except Exception as e:
            fail += 1
            if fail % 50 == 0:
                print(f"⚠️ Camera capture error: {e}")
            time.sleep(0.1)
    
    print("📹 Capture thread stopped")

def processing_thread_func(tracker: ObjectTracker, q: queue.Queue,
                          target_shape, target_color, roi_state):
    """Object detection processing thread"""
    global processed_output
    print("🧠 Processing thread started")
    processing_count = 0
    
    while not stop_event.is_set():
        if not is_tracking_mode:
            time.sleep(0.2)
            continue
            
        try:
            frame_to_process = q.get(timeout=0.3)
            processing_count += 1
            
            # Apply ROI
            ROI_X, ROI_Y, ROI_W, ROI_H = roi_state["x"], roi_state["y"], roi_state["w"], roi_state["h"]
            roi_frame = frame_to_process[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
            
            # Detect objects
            detections = tracker.get_raw_detections(roi_frame)
            
            # Find target
            target_detected = False
            target_box = None
            
            for detection in detections:
                if (detection['shape'] == target_shape and 
                    detection['color'] == target_color):
                    target_detected = True
                    target_box = detection['box']
                    break
            
            # Update processed output
            with output_lock:
                processed_output = {
                    "target_detected": target_detected,
                    "target_box": target_box,
                    "detections": detections,
                    "frame": frame_to_process,
                    "processing_count": processing_count
                }
                
        except queue.Empty:
            continue
        except Exception as e:
            if processing_count % 50 == 0:
                print(f"⚠️ Processing error: {e}")
            time.sleep(0.1)
    
    print("🧠 Processing thread stopped")

def camera_display_thread(target_shape=None, target_color=None, roi_state=None):
    """Camera display thread"""
    print("📹 Camera display thread started")
    frame_count = 0
    
    try:
        while not stop_event.is_set():
            try:
                with output_lock:
                    if processed_output.get("frame") is not None:
                        display_frame = processed_output["frame"].copy()
                        target_detected = processed_output.get("target_detected", False)
                        target_box = processed_output.get("target_box")
                        detections = processed_output.get("detections", [])
                        processing_count = processed_output.get("processing_count", 0)
                    else:
                        time.sleep(0.1)
                        continue
                
                if display_frame is None or display_frame.size == 0:
                    time.sleep(0.1)
                    continue
                
                # Apply ROI rectangle
                try:
                    if roi_state:
                        ROI_X, ROI_Y, ROI_W, ROI_H = roi_state["x"], roi_state["y"], roi_state["w"], roi_state["h"]
                        cv2.rectangle(display_frame, (ROI_X, ROI_Y), (ROI_X + ROI_W, ROI_Y + ROI_H), (0, 255, 0), 2)
                except:
                    pass
                
                # Draw detected objects
                try:
                    if roi_state and target_shape and target_color:
                        ROI_X, ROI_Y, ROI_W, ROI_H = roi_state["x"], roi_state["y"], roi_state["w"], roi_state["h"]
                        for detection in detections:
                            x, y, w, h = detection['box']
                            fx = ROI_X + x
                            fy = ROI_Y + y
                            
                            if (detection['shape'] == target_shape and detection['color'] == target_color):
                                color = (0, 255, 0)  # Green for target
                                thickness = 3
                            else:
                                color = (0, 0, 255)  # Red for other objects
                                thickness = 1
                            
                            cv2.rectangle(display_frame, (fx, fy), (fx + w, fy + h), color, thickness)
                            cv2.putText(display_frame, f"{detection['color']} {detection['shape']}", 
                                       (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                except:
                    pass
                
                # Add text overlay
                try:
                    if target_color and target_shape:
                        cv2.putText(display_frame, f"Searching: {target_color} {target_shape}", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Detection: {processing_count}", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                except:
                    pass
                
                if target_detected:
                    cv2.putText(display_frame, "TARGET DETECTED!", (10, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("Target Detection", display_frame)
                cv2.waitKey(1)
                frame_count += 1
                
            except Exception as e:
                if frame_count % 100 == 0:
                    print(f"⚠️ Display error: {e}")
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        pass
    finally:
        try:
            cv2.destroyWindow("Target Detection")
        except:
            pass
    
    print("📹 Camera display thread stopped")

# =============================================================================
# ===== MOVEMENT CONTROL =====================================================
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
        normalized_target = self.normalize_angle(target_yaw); time.sleep(0.05)
        robot_rotation = -self.normalize_angle(normalized_target - self.current_yaw)
        print(f"\n🔧 Correcting Yaw: {self.current_yaw:.1f}° -> {target_yaw:.1f}°. Rotating: {robot_rotation:.1f}°")
        
        if abs(robot_rotation) > self.yaw_tolerance:
            if abs(robot_rotation) > 45:
                chassis.move(x=0, y=0, z=robot_rotation, z_speed=80).wait_for_completed(timeout=2)
            else:
                chassis.move(x=0, y=0, z=robot_rotation, z_speed=60).wait_for_completed(timeout=1)
            time.sleep(0.1)
        
        final_error = abs(self.normalize_angle(normalized_target - self.current_yaw))
        if final_error <= self.yaw_tolerance: 
            print(f"✅ Yaw Correction Success: {self.current_yaw:.1f}°"); 
            return True
        
        print(f"⚠️ First attempt incomplete. Current: {self.current_yaw:.1f}°. Fine-tuning...")
        remaining_rotation = -self.normalize_angle(normalized_target - self.current_yaw)
        
        if abs(remaining_rotation) > 0.5 and abs(remaining_rotation) < 30:
            chassis.move(x=0, y=0, z=remaining_rotation, z_speed=30).wait_for_completed(timeout=3)
            time.sleep(0.15)
        
        final_error = abs(self.normalize_angle(normalized_target - self.current_yaw))
        if final_error <= self.yaw_tolerance: 
            print(f"✅ Yaw Fine-tuning Success: {self.current_yaw:.1f}°"); 
            return True
        else: 
            print(f"🔥🔥 Yaw Correction FAILED. Final Yaw: {self.current_yaw:.1f}°"); 
            return False

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
        KP_YAW = 0.8; MAX_YAW_SPEED = 20
        yaw_error = attitude_handler.normalize_angle(target_yaw - attitude_handler.current_yaw)
        speed = KP_YAW * yaw_error
        return max(min(speed, MAX_YAW_SPEED), -MAX_YAW_SPEED)

    def move_forward_one_grid(self, axis, attitude_handler, target_yaw):
        attitude_handler.correct_yaw_to_target(self.chassis, target_yaw)
        target_distance = 0.6
        pid = PID(Kp=0.5, Ki=0.1, Kd=20, setpoint=target_distance)
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
            yaw_correction = self._calculate_yaw_correction(attitude_handler, target_yaw)
            self.chassis.drive_speed(x=speed, y=0, z=yaw_correction, timeout=1)
            print(f"Moving... Dist: {relative_position:.3f}/{target_distance:.2f} m", end='\r')
        self.chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0); time.sleep(0.25)

    def rotate_to_direction(self, target_direction, current_direction, attitude_handler, current_yaw):
        global CURRENT_DIRECTION, CURRENT_TARGET_YAW, ROBOT_FACE
        if current_direction == target_direction: return
        
        diff = (target_direction - current_direction + 4) % 4
        if diff == 1: 
            self.rotate_90_degrees_right(current_direction, attitude_handler, current_yaw)
        elif diff == 3: 
            self.rotate_90_degrees_left(current_direction, attitude_handler, current_yaw)
        elif diff == 2: 
            self.rotate_90_degrees_right(current_direction, attitude_handler, current_yaw)
            self.rotate_90_degrees_right(CURRENT_DIRECTION, attitude_handler, CURRENT_TARGET_YAW)

    def rotate_90_degrees_right(self, current_direction, attitude_handler, current_yaw):
        global CURRENT_TARGET_YAW, CURRENT_DIRECTION, ROBOT_FACE
        print("🔄 Rotating 90° RIGHT...")
        CURRENT_TARGET_YAW = attitude_handler.normalize_angle(current_yaw + 90)
        attitude_handler.correct_yaw_to_target(self.chassis, get_compensated_target_yaw())
        CURRENT_DIRECTION = (current_direction + 1) % 4; ROBOT_FACE += 1

    def rotate_90_degrees_left(self, current_direction, attitude_handler, current_yaw):
        global CURRENT_TARGET_YAW, CURRENT_DIRECTION, ROBOT_FACE
        print("🔄 Rotating 90° LEFT...")
        CURRENT_TARGET_YAW = attitude_handler.normalize_angle(current_yaw - 90)
        attitude_handler.correct_yaw_to_target(self.chassis, get_compensated_target_yaw())
        CURRENT_DIRECTION = (current_direction - 1 + 4) % 4; ROBOT_FACE -= 1
        if ROBOT_FACE < 1: ROBOT_FACE += 4

    def cleanup(self):
        try: self.chassis.unsub_position()
        except Exception: pass

# =============================================================================
# ===== TARGETING & SHOOTING =================================================
# =============================================================================

class PIDController:
    """PID Controller สำหรับการเล็งเป้า"""
    def __init__(self, kp, ki, kd, i_clamp=1000.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.i_clamp = i_clamp
        self.prev_error = 0
        self.integral = 0
        self.prev_time = time.time()

    def compute(self, error, dt):
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with clamping
        self.integral += error * dt
        self.integral = max(min(self.integral, self.i_clamp), -self.i_clamp)
        i_term = self.ki * self.integral
        
        # Derivative term with low-pass filter
        if dt > 0:
            derivative = (error - self.prev_error) / dt
            derivative = DERIV_LPF_ALPHA * derivative + (1 - DERIV_LPF_ALPHA) * self.prev_error
        else:
            derivative = 0
        
        d_term = self.kd * derivative
        
        # Total output
        output = p_term + i_term + d_term
        
        # Update for next iteration
        self.prev_error = error
        
        return output

    def reset(self):
        self.prev_error = 0
        self.integral = 0
        self.prev_time = time.time()

def aim_and_shoot_target(manager, target_shape, target_color, roi_state, max_lock_time=10.0):
    """เล็งและยิงเป้าหมายที่กำหนด"""
    global is_tracking_mode, fired_targets, shots_fired, stop_event
    
    print(f"🎯 Starting targeting sequence for {target_color} {target_shape}")
    print(f"🔍 Starting multithreaded detection...")
    
    gimbal = manager.get_gimbal()
    blaster = manager.get_blaster()
    
    if gimbal is None or blaster is None:
        print("⚠️ Gimbal or blaster not available")
        return False
    
    # Reset stop event
    stop_event.clear()
    
    # Create detection window
    cv2.namedWindow("Target Detection", cv2.WINDOW_AUTOSIZE)
    
    # Initialize object tracker
    tracker = ObjectTracker(use_gpu=False)
    
    # Start camera threads
    cap_t = threading.Thread(target=capture_thread_func, args=(manager, frame_queue), daemon=True)
    proc_t = threading.Thread(target=processing_thread_func, 
                              args=(tracker, frame_queue, target_shape, target_color, roi_state), 
                              daemon=True)
    display_t = threading.Thread(target=camera_display_thread, 
                                args=(target_shape, target_color, roi_state), 
                                daemon=True)
    
    cap_t.start()
    proc_t.start()
    display_t.start()
    
    print("✅ All camera threads started")
    
    # Initialize PID controllers
    yaw_pid = PIDController(PID_KP, PID_KI, PID_KD, I_CLAMP)
    pitch_pid = PIDController(PID_KP, PID_KI, PID_KD, I_CLAMP)
    
    # Reset PID controllers
    yaw_pid.reset()
    pitch_pid.reset()
    
    # Tracking variables
    lock_count = 0
    start_time = time.time()
    last_detection_time = time.time()
    
    # Enable tracking mode
    is_tracking_mode = True
    
    print("🔍 Starting multithreaded target detection and tracking...")
    
    while time.time() - start_time < max_lock_time:
        try:
            # Get processed data from multithreading
            with output_lock:
                target_detected = processed_output.get("target_detected", False)
                target_box = processed_output.get("target_box")
                processing_count = processed_output.get("processing_count", 0)
            
            # Debug: Show detection count every 50 frames
            if processing_count % 50 == 0 and processing_count > 0:
                print(f"🔍 Detection attempt {processing_count}")
            
            if not target_detected or target_box is None:
                # No target detected, slowly stop gimbal
                try:
                    gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                except Exception:
                    pass
                time.sleep(0.1)
                continue
            
            # Target found
            print(f"🎯 Target detected! {target_color} {target_shape}")
            last_detection_time = time.time()
            
            # Target found, calculate errors
            x, y, w, h = target_box
            cx_roi = x + w/2.0
            cy_roi = y + h/2.0
            
            # Get ROI dimensions
            try:
                ROI_X, ROI_Y, ROI_W, ROI_H = roi_state["x"], roi_state["y"], roi_state["w"], roi_state["h"]
            except:
                ROI_X, ROI_Y, ROI_W, ROI_H = 0, 0, 640, 480
            
            # Convert to full frame coordinates
            cx = ROI_X + cx_roi
            cy = ROI_Y + cy_roi
            
            # Calculate errors (image center to target center)
            center_x = FRAME_W/2.0
            center_y = FRAME_H/2.0
            
            err_x = (center_x - cx)
            err_y = (center_y - cy) + PITCH_BIAS_PIX
            
            # Apply deadzone
            if abs(err_x) < PIX_ERR_DEADZONE: err_x = 0.0
            if abs(err_y) < PIX_ERR_DEADZONE: err_y = 0.0
            
            # Calculate PID outputs
            current_time = time.time()
            dt = current_time - yaw_pid.prev_time
            yaw_pid.prev_time = current_time
            
            u_x = yaw_pid.compute(err_x, dt)
            u_y = pitch_pid.compute(err_y, dt)
            
            # Clamp outputs
            u_x = float(np.clip(u_x, -MAX_YAW_SPEED, MAX_YAW_SPEED))
            u_y = float(np.clip(u_y, -MAX_PITCH_SPEED, MAX_PITCH_SPEED))
            
            # Apply gimbal movement
            try:
                gimbal.drive_speed(pitch_speed=-u_y, yaw_speed=u_x)
            except Exception as e:
                print(f"Gimbal drive error: {e}")
            
            # Check if locked on target
            locked = (abs(err_x) <= LOCK_TOL_X) and (abs(err_y) <= LOCK_TOL_Y)
            
            if locked:
                lock_count += 1
                print(f"🎯 Locked on target! ({lock_count}/{LOCK_STABLE_COUNT})")
                
                if lock_count >= LOCK_STABLE_COUNT:
                    # Fire at target
                    if shots_fired < FIRE_SHOTS_COUNT:
                        try:
                            blaster.fire(fire_type=r_blaster.WATER_FIRE)
                            shots_fired += 1
                            print(f"🔥 Fired shot {shots_fired}/{FIRE_SHOTS_COUNT}")
                            time.sleep(0.3)  # Wait between shots
                        except Exception as e:
                            print(f"Fire error: {e}")
                    else:
                        print("✅ Completed firing sequence")
                        is_tracking_mode = False
                        fired_targets.add(f"{target_color}_{target_shape}")
                        shots_fired = 0
                        
                        # Cleanup
                        print("🛑 Stopping camera threads...")
                        is_tracking_mode = False
                        stop_event.set()
                        
                        # Wait for threads to finish
                        try:
                            cap_t.join(timeout=2.0)
                            proc_t.join(timeout=2.0)
                            display_t.join(timeout=2.0)
                        except:
                            pass
                        
                        try:
                            cv2.destroyWindow("Target Detection")
                        except:
                            pass
                        
                        # Center gimbal after shooting (like Round 1)
                        try:
                            gimbal.moveto(pitch=0, yaw=0, yaw_speed=SPEED_ROTATE).wait_for_completed()
                            time.sleep(0.2)
                            print("✅ Gimbal centered after shooting")
                        except Exception as e:
                            print(f"⚠️ Gimbal centering error: {e}")
                        
                        print("✅ Camera threads stopped")
                        return True
            else:
                lock_count = 0
            
            time.sleep(0.01)  # Small delay for PID loop
            
        except Exception as e:
            print(f"⚠️ Targeting error: {e}")
            time.sleep(0.1)
    
    print(f"⏰ Targeting timeout reached after {max_lock_time}s - no target found")
    print(f"🔍 Total detection attempts: {processing_count}")
    
    # Close camera window
    # Cleanup
    print("🛑 Stopping camera threads...")
    is_tracking_mode = False
    stop_event.set()
    
    # Wait for threads to finish
    try:
        cap_t.join(timeout=2.0)
        proc_t.join(timeout=2.0)
        display_t.join(timeout=2.0)
    except:
        pass
    
    try:
        cv2.destroyWindow("Target Detection")
    except:
        pass
    
    # Center gimbal after timeout (like Round 1)
    try:
        gimbal = manager.get_gimbal()
        if gimbal:
            gimbal.moveto(pitch=0, yaw=0, yaw_speed=SPEED_ROTATE).wait_for_completed()
            time.sleep(0.2)
            print("✅ Gimbal centered after timeout")
    except Exception as e:
        print(f"⚠️ Gimbal centering error: {e}")
    
    print("✅ Camera threads stopped")
    return False

# =============================================================================
# ===== MISSION EXECUTION ====================================================
# =============================================================================

def find_targets_by_type(objects_data, target_shape, target_color):
    """หาเป้าหมายตามประเภทที่กำหนด"""
    matching_targets = []
    
    print(f"🔍 Debug: Looking for '{target_color}' '{target_shape}'")
    print(f"🔍 Available objects in JSON:")
    
    for i, obj in enumerate(objects_data['detected_objects']):
        print(f"  {i+1}. Color: '{obj['color']}', Shape: '{obj['shape']}', Position: {obj['cell_position']}, Detected from: {obj['detected_from_node']}")
        
        # เปรียบเทียบแบบไม่สนใจ case และ whitespace
        obj_color = obj['color'].strip().lower()
        obj_shape = obj['shape'].strip().lower()
        target_color_clean = target_color.strip().lower()
        target_shape_clean = target_shape.strip().lower()
        
        if (obj_shape == target_shape_clean and obj_color == target_color_clean):
            matching_targets.append({
                'target_position': (obj['cell_position']['row'], obj['cell_position']['col']),  # ตำแหน่งวัตถุ
                'shooting_position': (obj['detected_from_node'][0], obj['detected_from_node'][1]),  # ตำแหน่งยิง
                'shape': obj['shape'],
                'color': obj['color'],
                'zone': obj['zone']
            })
            print(f"    ✅ MATCH FOUND! Target at {obj['cell_position']}, Shoot from {obj['detected_from_node']}")
    
    print(f"🔍 Total matches found: {len(matching_targets)}")
    return matching_targets


def execute_shooting_mission(grid, width, height, target_sequence, movement_controller, 
                            attitude_handler, manager, roi_state, map_data=None):
    """Execute the complete shooting mission"""
    global CURRENT_POSITION, CURRENT_DIRECTION, CURRENT_TARGET_YAW, ROBOT_FACE, shots_fired
    
    print(f"🚀 Starting shooting mission with {len(target_sequence)} targets")
    
    for i, target_info in enumerate(target_sequence):
        shooting_pos = target_info['shooting_position']  # ตำแหน่งยิง
        target_pos = target_info['target_position']      # ตำแหน่งวัตถุ
        target_shape = target_info['shape']
        target_color = target_info['color']
        
        print(f"\n🎯 Target {i+1}/{len(target_sequence)}: {target_color} {target_shape}")
        print(f"    Target at: {target_pos}, Shoot from: {shooting_pos}")
        
        # Find path to shooting position
        path = find_path_bfs(grid, CURRENT_POSITION, shooting_pos, width, height, map_data)
        if not path:
            print(f"❌ Cannot reach shooting position at {shooting_pos}")
            continue
        
        print(f"📍 Path to shooting position: {path}")
        
        # Execute movement to shooting position
        for j in range(len(path) - 1):
            current_pos = path[j]
            next_pos = path[j + 1]
            
            # Calculate direction to next position
            dr = next_pos[0] - current_pos[0]
            dc = next_pos[1] - current_pos[1]
            
            if dr == -1: target_direction = 0  # North
            elif dr == 1: target_direction = 2  # South
            elif dc == 1: target_direction = 1  # East
            elif dc == -1: target_direction = 3  # West
            else:
                print(f"❌ Invalid direction from {current_pos} to {next_pos}")
                continue
            
            # Rotate to face the correct direction
            movement_controller.rotate_to_direction(
                target_direction, CURRENT_DIRECTION, attitude_handler, CURRENT_TARGET_YAW
            )
            
            # Move forward one grid
            axis = 'x' if ROBOT_FACE % 2 != 0 else 'y'
            movement_controller.move_forward_one_grid(axis, attitude_handler, get_compensated_target_yaw())
            
            # Update position
            CURRENT_POSITION = next_pos
            print(f"📍 Moved to {CURRENT_POSITION}")
        
        # Now we're at the shooting position, turn to face the target
        print(f"🎯 At shooting position {CURRENT_POSITION}, turning to face target at {target_pos}")
        
        # Calculate direction to target
        dr = target_pos[0] - shooting_pos[0]
        dc = target_pos[1] - shooting_pos[1]
        
        if dr == -1: target_direction = 0      # North
        elif dr == 1: target_direction = 2     # South
        elif dc == 1: target_direction = 1     # East
        elif dc == -1: target_direction = 3    # West
        else:
            print(f"❌ Invalid direction from {shooting_pos} to {target_pos}")
            continue
        
        # Turn to face the target
        movement_controller.rotate_to_direction(
            target_direction, CURRENT_DIRECTION, attitude_handler, CURRENT_TARGET_YAW
        )
        
        # Also turn gimbal to face the target (using method from Round 1)
        print(f"🎯 Adjusting gimbal to face target...")
        gimbal = manager.get_gimbal()
        if gimbal:
            try:
                # First center the gimbal to match robot direction (like Round 1)
                print("   -> Centering gimbal to match robot direction...")
                gimbal.moveto(pitch=0, yaw=0, yaw_speed=SPEED_ROTATE).wait_for_completed()
                time.sleep(0.2)
                print("   -> Gimbal centered to match robot direction")
                
                # Now calculate gimbal angle to face target
                if target_direction == 0:  # North
                    target_gimbal_yaw = 0
                elif target_direction == 1:  # East
                    target_gimbal_yaw = 90
                elif target_direction == 2:  # South
                    target_gimbal_yaw = 180
                else:  # West
                    target_gimbal_yaw = -90
                
                print(f"🔧 Target gimbal angle: {target_gimbal_yaw}°")
                
                # Move gimbal to face target
                gimbal.moveto(pitch=0, yaw=target_gimbal_yaw, yaw_speed=SPEED_ROTATE).wait_for_completed()
                time.sleep(0.2)
                print(f"✅ Gimbal adjusted to face target (yaw: {target_gimbal_yaw}°)")
                
            except Exception as e:
                print(f"⚠️ Gimbal adjustment error: {e}")
        
        print(f"🎯 Facing target, starting detection and shooting...")
        
        # Reset shooting variables
        shots_fired = 0
        is_tracking_mode = True
        
        # Start targeting and shooting
        success = aim_and_shoot_target(manager, target_shape, target_color, roi_state)
        
        if success:
            print(f"✅ Successfully completed target {i+1}")
        else:
            print(f"❌ Failed to complete target {i+1} - continuing to next target")
        
        # Small delay between targets
        time.sleep(1.0)
    
    print("🎉 Mission completed!")


# =============================================================================
# ===== MAIN FUNCTION ========================================================
# =============================================================================

def main():
    """ฟังก์ชันหลัก"""
    global CURRENT_POSITION, CURRENT_DIRECTION, CURRENT_TARGET_YAW, ROBOT_FACE
    
    print("🎯 Round 2: Complete Target Shooting System")
    print("=" * 50)
    
    # โหลดข้อมูล
    map_data, objects_data, timestamp_data = load_data()
    if not all([map_data, objects_data, timestamp_data]):
        print("❌ Failed to load required data")
        return
    
    # สร้างกริด
    grid, width, height = create_grid_from_map(map_data)
    
    # รับข้อมูลเป้าหมายจากผู้ใช้
    print("\n🎯 Target Specification:")
    print("Available shapes: Circle, Square, Rectangle_H, Rectangle_V, Uncertain")
    print("Available colors: Red, Yellow, Green, Blue")
    
    target_shape = input("Enter target shape: ").strip()
    target_color = input("Enter target color: ").strip()
    
    print(f"\n🎯 Looking for targets: {target_color} {target_shape}")
    
    # หาเป้าหมายที่ตรงกับที่กำหนด
    targets = find_targets_by_type(objects_data, target_shape, target_color)
    
    if not targets:
        print(f"❌ No targets found matching {target_color} {target_shape}")
        return
    
    print(f"✅ Found {len(targets)} matching targets:")
    for i, target in enumerate(targets):
        print(f"  {i+1}. {target['color']} {target['shape']} at {target['target_position']} (shoot from {target['shooting_position']}) ({target['zone']} zone)")
    
    print(f"\n🎯 Will eliminate all {len(targets)} {target_color} {target_shape} targets")
    
    # แก้ปัญหา TSP
    print(f"\n🧮 Solving TSP for optimal path...")
    shooting_positions = [target['shooting_position'] for target in targets]
    optimal_order = solve_tsp_greedy(grid, CURRENT_POSITION, shooting_positions, width, height, map_data)
    
    # สร้างลำดับเป้าหมาย
    target_sequence = []
    for pos in optimal_order:
        for target in targets:
            if target['shooting_position'] == pos:
                target_sequence.append(target)
                break
    
    print(f"\n🎯 Optimal target sequence:")
    for i, target in enumerate(target_sequence):
        print(f"  {i+1}. {target['color']} {target['shape']} at {target['target_position']} (shoot from {target['shooting_position']})")
    
    # เริ่มต้นการทำงานกับหุ่นยนต์
    print("\n🤖 Initializing robot connection...")
    manager = RMConnection()
    
    try:
        manager.connect()
        
        # Wait for connection
        if not manager.connected.wait(timeout=10.0):
            print("❌ Failed to connect to robot")
            return
        
        print("✅ Robot connected successfully")
        
        # Get robot components
        ep_robot = manager.get_robot()
        if ep_robot is None:
            print("❌ Failed to get robot instance")
            return
        
        ep_chassis = ep_robot.chassis
        attitude_handler = AttitudeHandler()
        movement_controller = MovementController(ep_chassis)
        attitude_handler.start_monitoring(ep_chassis)
        
        # ROI state
        roi_state = {"x": ROI_X0, "y": ROI_Y0, "w": ROI_W0, "h": ROI_H0}
        
        # Execute shooting mission
        execute_shooting_mission(grid, width, height, target_sequence, 
                               movement_controller, attitude_handler, manager, roi_state, map_data)
        
    except KeyboardInterrupt:
        print("\n⚠️ Mission interrupted by user")
    except Exception as e:
        print(f"\n❌ Mission failed: {e}")
        traceback.print_exc()
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        try:
            if 'attitude_handler' in locals() and attitude_handler.is_monitoring:
                attitude_handler.stop_monitoring(ep_chassis)
            if 'movement_controller' in locals():
                movement_controller.cleanup()
            manager.close()
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")
        
        print("✅ Mission completed")

if __name__ == '__main__':
    main()
