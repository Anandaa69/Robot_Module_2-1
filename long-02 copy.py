# -*-coding:utf-8-*-

import time
import robomaster
from robomaster import robot, camera as r_camera, blaster as r_blaster
import numpy as np
import math
import json
import matplotlib.pyplot as plt
from collections import deque
import traceback
import statistics
import os
import cv2
import threading
import queue

# =============================================================================
# ===== CONFIGURATION & PARAMETERS ============================================
# =============================================================================
SPEED_ROTATE = 480

# --- PID Target Tracking & Firing Configuration ---
TARGET_SHAPE = "Circle"  # Shape to track
TARGET_COLOR = "Red"     # Color to track
FIRE_SHOTS_COUNT = 5     # Number of shots to fire (adjustable global variable)

# PID Parameters (from fire_target.py)
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

# --- Sharp Sensor Detection Thresholds ---
SHARP_WALL_THRESHOLD_CM = 60.0  # ระยะสูงสุดที่จะถือว่าเจอผนัง
SHARP_STDEV_THRESHOLD = 0.5     # ค่าเบี่ยงเบนมาตรฐานสูงสุดที่ยอมรับได้ เพื่อกรองค่าที่แกว่ง

# --- ToF Centering Configuration (from dude_kum.py) ---
TOF_ADJUST_SPEED = 0.1             # ความเร็วในการขยับเข้า/ถอยออกเพื่อจัดตำแหน่งกลางโหนด
TOF_CALIBRATION_SLOPE = 0.0894     # ค่าจากการ Calibrate
TOF_CALIBRATION_Y_INTERCEPT = 3.8409 # ค่าจากการ Calibrate
TOF_TIME_CHECK = 0.15

GRID = 6

# --- Logical state for the grid map (from map_suay.py) ---
CURRENT_POSITION = (5,0)  # (แถว, คอลัมน์) here
CURRENT_DIRECTION =  1  # 0:North, 1:East, 2:South, 3:West here
TARGET_DESTINATION =CURRENT_POSITION #(1, 0)#here

# --- Physical state for the robot ---
CURRENT_TARGET_YAW = 0.0
ROBOT_FACE = 1  # 1,3,5.. = X axis, 2,4,6.. = Y axis

# --- Global variables for PID tracking and firing ---
is_tracking_mode = False
fired_targets = set()  # Track which targets have been fired at
current_target_id = None
shots_fired = 0
targets_found = []  # List of targets found in current detection
gimbal_angle_lock = threading.Lock()
gimbal_angles = (0.0, 0.0, 0.0, 0.0)  # (pitch, yaw, pitch_g, yaw_g)

# --- NEW: IMU Drift Compensation Parameters ---
IMU_COMPENSATION_START_NODE_COUNT = 7      # จำนวนโหนดขั้นต่ำก่อนเริ่มการชดเชย
IMU_COMPENSATION_NODE_INTERVAL = 15      # เพิ่มค่าชดเชยทุกๆ N โหนด
IMU_COMPENSATION_DEG_PER_INTERVAL = -1.0 # ค่าองศาที่ชดเชย (ลบเพื่อแก้การเบี้ยวขวา)
IMU_DRIFT_COMPENSATION_DEG = 0.0           # ตัวแปรเก็บค่าชดเชยปัจจุบัน

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

# --- Visualization Configuration ---
MAP_FIGURE_SIZE = (6, 4)  # (width, height) ปรับได้ตามต้องการ

# --- NEW: Timestamp Logging ---
POSITION_LOG = []  # เก็บข้อมูลตำแหน่งและเวลา

# --- NEW: Resume Function Variables ---
RESUME_MODE = False  # ตัวแปรบอกว่าเป็นโหมด resume หรือไม่
DATA_FOLDER = r"F:\Coder\Year2-1\Robot_Module\Assignment\dude\James_path"  # โฟลเดอร์สำหรับเก็บไฟล์ JSON

def save_all_data(occupancy_map):
    """บันทึกข้อมูลทั้งหมด (Map, Timestamps, Objects) ลง JSON"""
    try:
        print("💾 Saving map and timestamp data...")
        
        # 1. บันทึกแผนที่พร้อม objects
        final_map_data = {'nodes': []}
        for r in range(occupancy_map.height):
            for c in range(occupancy_map.width):
                cell = occupancy_map.grid[r][c]
                cell_data = {
                    "coordinate": {"row": r, "col": c},
                    "probability": round(cell.get_node_probability(), 3),
                    "is_occupied": cell.is_node_occupied(),
                    "walls": {
                        "north": cell.walls['N'].is_occupied(),
                        "south": cell.walls['S'].is_occupied(),
                        "east": cell.walls['E'].is_occupied(),
                        "west": cell.walls['W'].is_occupied()
                    },
                    "wall_probabilities": {
                        "north": round(cell.walls['N'].get_probability(), 3),
                        "south": round(cell.walls['S'].get_probability(), 3),
                        "east": round(cell.walls['E'].get_probability(), 3),
                        "west": round(cell.walls['W'].get_probability(), 3)
                    },
                    "objects": cell.objects if hasattr(cell, 'objects') else []
                }
                final_map_data["nodes"].append(cell_data)

        map_file = os.path.join(DATA_FOLDER, "Mapping_Top.json")
        with open(map_file, "w") as f:
            json.dump(final_map_data, f, indent=2)
        print(f"✅ Final Hybrid Belief Map (with objects) saved to {map_file}")
        
        # 2. บันทึกข้อมูล timestamp และตำแหน่ง
        timestamp_data = {
            "session_info": {
                "start_time": POSITION_LOG[0]["iso_timestamp"] if POSITION_LOG else "N/A",
                "end_time": POSITION_LOG[-1]["iso_timestamp"] if POSITION_LOG else "N/A",
                "total_positions_logged": len(POSITION_LOG),
                "grid_size": f"{occupancy_map.height}x{occupancy_map.width}",
                "target_destination": list(TARGET_DESTINATION),
                "interrupted": not RESUME_MODE
            },
            "position_log": POSITION_LOG
        }
        
        timestamp_file = os.path.join(DATA_FOLDER, "Robot_Position_Timestamps.json")
        with open(timestamp_file, "w") as f:
            json.dump(timestamp_data, f, indent=2)
        print(f"✅ Robot position timestamps saved to {timestamp_file}")
        
        # 3. บันทึกข้อมูลวัตถุที่ตรวจจับได้ (รวบรวมจาก map)
        all_detected_objects = []
        for r in range(occupancy_map.height):
            for c in range(occupancy_map.width):
                cell = occupancy_map.grid[r][c]
                if hasattr(cell, 'objects') and cell.objects:
                    for obj in cell.objects:
                        obj_with_pos = obj.copy()
                        obj_with_pos['cell_position'] = {'row': r, 'col': c}
                        all_detected_objects.append(obj_with_pos)
        
        objects_data = {
            "session_info": {
                "total_objects_detected": len(all_detected_objects),
                "detection_timestamp": time.time(),
                "grid_size": f"{occupancy_map.height}x{occupancy_map.width}"
            },
            "detected_objects": all_detected_objects
        }
        
        objects_file = os.path.join(DATA_FOLDER, "Detected_Objects.json")
        with open(objects_file, "w") as f:
            json.dump(objects_data, f, indent=2)
        print(f"✅ Detected objects saved to {objects_file} (Total: {len(all_detected_objects)} objects)")
        
        return True
    except Exception as save_error:
        print(f"❌ Error saving data: {save_error}")
        traceback.print_exc()
        return False

# --- CAMERA HEALTH SHARED STATE ---
last_frame_received_ts = 0.0  # อัปเดตทุกครั้งที่ได้เฟรมจากกล้อง (capture thread)

def camera_is_healthy(timeout=3.0) -> bool:
    """
    ถือว่ากล้องพร้อมใช้งานเมื่อเชื่อมต่ออยู่และมีเฟรมล่าสุดในไม่กี่วินาทีนี้
    timeout: เวลาที่ยอมให้เฟรมล่าสุดเก่าได้ (default 3.0 วินาที)
    """
    try:
        # ใช้ตัวแปร global manager ที่ถูกประกาศตอน initialize
        if not manager.connected.is_set():
            return False
    except Exception:
        return False
    return (time.time() - last_frame_received_ts) <= timeout

def wait_for_camera_recovery(pause_label="Runtime"):
    """หยุดหุ่นและรอกล้องกลับมา ถ้าเกิน 30s จะสั่ง reconnect แล้วรอต่อ"""
    print(f"🛑 {pause_label}: Camera unhealthy → locking chassis and waiting...")
    try:
        movement_controller.chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
    except Exception:
        pass
    wait_start = time.time()
    while not camera_is_healthy():
        if time.time() - wait_start > 30.0:
            print(f"⚠️ {pause_label}: Camera recovery timeout (30s). Forcing reconnect and continuing wait...")
            try:
                manager.drop_and_reconnect()
            except Exception:
                pass
            wait_start = time.time()
        time.sleep(0.2)
    print(f"✅ {pause_label}: Camera recovered. Resuming...")

# =============================================================================
# ===== OBJECT DETECTION CONFIGURATION =======================================
# =============================================================================

# Object Detection Parameters (duplicate - using values from top of file)
# TARGET_SHAPE = "Circle"  # Already defined at top
# TARGET_COLOR = "Blue"    # Already defined at top

# PID Parameters
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

# Object Detection Threading
frame_queue = queue.Queue(maxsize=1)
processed_output = {"details": []}
output_lock = threading.Lock()
stop_event = threading.Event()

# Gimbal angles
gimbal_angle_lock = threading.Lock()
gimbal_angles = (0.0, 0.0, 0.0, 0.0)

# Detection mode control
is_detecting_flag = {"v": False}  # Start with detection OFF
detection_timer = None
detection_start_time = None

# Object storage for map integration
detected_objects = []
object_lock = threading.Lock()

def sub_angle_cb(angle_info):
    global gimbal_angles
    with gimbal_angle_lock:
        gimbal_angles = angle_info
    # Debug: Print gimbal angles when in tracking mode
    if is_tracking_mode:
        print(f"🎯 Gimbal angles: pitch={angle_info[0]:.1f}°, yaw={angle_info[1]:.1f}°")
    
    # Debug: Print gimbal angles every 50 calls (about 1 second at 50Hz)
    if not hasattr(sub_angle_cb, 'call_count'):
        sub_angle_cb.call_count = 0
    sub_angle_cb.call_count += 1
    if sub_angle_cb.call_count % 50 == 0:
        print(f"🎯 Gimbal callback: pitch={angle_info[0]:.1f}°, yaw={angle_info[1]:.1f}° (call #{sub_angle_cb.call_count})")

# =============================================================================
# ===== HELPER FUNCTIONS ======================================================
# =============================================================================
def convert_adc_to_cm(adc_value):
    """Converts ADC value from Sharp sensor to centimeters."""
    if adc_value <= 0: return float('inf')
    # This formula is specific to the GP2Y0A21YK0F sensor.
    # You may need to re-calibrate for your specific sensor.
    return 30263 * (adc_value ** -1.352)

def calibrate_tof_value(raw_tof_value):
    """
    NEW: Converts raw ToF value (mm) to a calibrated distance in cm.
    From dude_kum.py.
    """
    try:
        if raw_tof_value is None or raw_tof_value <= 0:
            return float('inf')
        # The formula is: calibrated_cm = (slope * raw_mm) + y_intercept
        return (TOF_CALIBRATION_SLOPE * raw_tof_value) + TOF_CALIBRATION_Y_INTERCEPT
    except Exception:
        return float('inf')

def get_compensated_target_yaw():
    """
    NEW: Returns the current target yaw with the calculated IMU drift compensation.
    This function is now the single source of truth for the robot's target heading.
    """
    return CURRENT_TARGET_YAW + IMU_DRIFT_COMPENSATION_DEG

def log_position_timestamp(position, direction, action="arrived"):
    """
    NEW: บันทึก timestamp และตำแหน่งของหุ่นยนต์
    """
    global POSITION_LOG
    timestamp = time.time()
    direction_names = ['North', 'East', 'South', 'West']
    
    # แก้ไขการสร้าง ISO timestamp
    dt = time.gmtime(timestamp)
    microseconds = int((timestamp % 1) * 1000000)
    iso_timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", dt) + f".{microseconds:06d}Z"
    
    log_entry = {
        "timestamp": timestamp,
        "iso_timestamp": iso_timestamp,
        "position": list(position),
        "direction": direction_names[direction],
        "action": action,
        "yaw_angle": CURRENT_TARGET_YAW,
        "imu_compensation": IMU_DRIFT_COMPENSATION_DEG
    }
    
    POSITION_LOG.append(log_entry)
    print(f"📍 [{action}] Position: {position}, Direction: {direction_names[direction]}, Time: {log_entry['iso_timestamp']}")

def check_for_resume_data():
    """
    NEW: ตรวจสอบว่ามีไฟล์ JSON สำหรับ resume หรือไม่
    """
    map_file = os.path.join(DATA_FOLDER, "Mapping_Top.json")
    timestamp_file = os.path.join(DATA_FOLDER, "Robot_Position_Timestamps.json")
    
    if os.path.exists(map_file) and os.path.exists(timestamp_file):
        return True
    return False

def load_resume_data():
    """
    NEW: โหลดข้อมูลจากไฟล์ JSON เพื่อ resume การทำงาน
    """
    global CURRENT_POSITION, CURRENT_DIRECTION, CURRENT_TARGET_YAW, ROBOT_FACE, IMU_DRIFT_COMPENSATION_DEG, POSITION_LOG, RESUME_MODE
    
    try:
        print("🔄 Loading resume data...")
        
        # โหลดข้อมูล timestamp
        timestamp_file = os.path.join(DATA_FOLDER, "Robot_Position_Timestamps.json")
        with open(timestamp_file, "r", encoding="utf-8") as f:
            timestamp_data = json.load(f)
        
        # ตั้งค่า global variables จากข้อมูล timestamp
        if timestamp_data["position_log"]:
            last_log = timestamp_data["position_log"][-1]
            CURRENT_POSITION = tuple(last_log["position"])
            CURRENT_TARGET_YAW = last_log["yaw_angle"]
            IMU_DRIFT_COMPENSATION_DEG = last_log["imu_compensation"]
            POSITION_LOG = timestamp_data["position_log"]
            
            # คำนวณ direction จาก yaw angle
            yaw = last_log["yaw_angle"]
            if -45 <= yaw <= 45:
                CURRENT_DIRECTION = 0  # North
                ROBOT_FACE = 1
            elif 45 < yaw <= 135:
                CURRENT_DIRECTION = 1  # East
                ROBOT_FACE = 2
            elif 135 < yaw or yaw <= -135:
                CURRENT_DIRECTION = 2  # South
                ROBOT_FACE = 3
            else:
                CURRENT_DIRECTION = 3  # West
                ROBOT_FACE = 4
        
        print(f"✅ Resume data loaded:")
        print(f"   Position: {CURRENT_POSITION}")
        print(f"   Direction: {['North', 'East', 'South', 'West'][CURRENT_DIRECTION]}")
        print(f"   Yaw: {CURRENT_TARGET_YAW:.1f}°")
        print(f"   IMU Compensation: {IMU_DRIFT_COMPENSATION_DEG:.1f}°")
        print(f"   Previous positions logged: {len(POSITION_LOG)}")
        
        RESUME_MODE = True
        return True
        
    except Exception as e:
        print(f"❌ Error loading resume data: {e}")
        return False

def create_occupancy_map_from_json():
    """
    NEW: สร้าง OccupancyGridMap จากไฟล์ JSON
    """
    try:
        map_file = os.path.join(DATA_FOLDER, "Mapping_Top.json")
        with open(map_file, "r", encoding="utf-8") as f:
            map_data = json.load(f)
        
        # หาขนาดของกริด
        max_row = max(node['coordinate']['row'] for node in map_data['nodes'])
        max_col = max(node['coordinate']['col'] for node in map_data['nodes'])
        width = max_col + 1
        height = max_row + 1
        
        # สร้าง OccupancyGridMap
        occupancy_map = OccupancyGridMap(width, height)
        
        # โหลดข้อมูลจาก JSON กลับเข้าไปใน occupancy_map
        for node_data in map_data['nodes']:
            r = node_data['coordinate']['row']
            c = node_data['coordinate']['col']
            cell = occupancy_map.grid[r][c]
            
            # โหลด node probability
            cell.log_odds_occupied = math.log(node_data['probability'] / (1 - node_data['probability'])) if node_data['probability'] != 0.5 else 0
            
            # โหลด wall probabilities
            walls = node_data['wall_probabilities']
            for direction, prob in walls.items():
                if direction == 'north':
                    cell.walls['N'].log_odds = math.log(prob / (1 - prob)) if prob != 0.5 else 0
                elif direction == 'south':
                    cell.walls['S'].log_odds = math.log(prob / (1 - prob)) if prob != 0.5 else 0
                elif direction == 'east':
                    cell.walls['E'].log_odds = math.log(prob / (1 - prob)) if prob != 0.5 else 0
                elif direction == 'west':
                    cell.walls['W'].log_odds = math.log(prob / (1 - prob)) if prob != 0.5 else 0
        
        print(f"✅ Occupancy map loaded from JSON ({width}x{height})")
        return occupancy_map
        
    except Exception as e:
        print(f"❌ Error loading occupancy map: {e}")
        return None

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
# ===== INTEGRATED OBJECT DETECTION SYSTEM ===================================
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
            # subscribe angles
            try:
                rb.gimbal.sub_angle(freq=20, callback=sub_angle_cb)
            except Exception as e:
                print("Gimbal sub_angle error:", e)
            self._robot = rb
            self.connected.set()
            print("✅ RoboMaster connected & camera streaming")

            # recenter gimbal on start
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
    frame_count = 0
    last_success_time = time.time()
    
    while not stop_event.is_set():
        if not manager.connected.is_set():
            time.sleep(0.1)
            continue
            
        cam = manager.get_camera()
        if cam is None:
            time.sleep(0.1)
            continue
            
        try:
            frame = cam.read_cv2_image(timeout=1.5)
            if frame is not None and frame.size > 0:
                # Clear queue if it's full to prevent memory buildup
                if q.full():
                    try: 
                        q.get_nowait()
                    except queue.Empty: 
                        pass
                
                q.put(frame)
                # mark last healthy frame timestamp
                try:
                    global last_frame_received_ts
                    last_frame_received_ts = time.time()
                except Exception:
                    pass
                frame_count += 1
                last_success_time = time.time()
                fail = 0
            else:
                fail += 1
                
        except Exception as e:
            print(f"⚠️ Camera read error: {e}")
            fail += 1

        # Tolerant reconnection policy (match fire_target.py behavior)
        if fail >= 10:  # ลดจาก 30 เป็น 10 เหมือน fire_target.py
            print("⚠️ Too many camera errors → drop & reconnect")
            manager.drop_and_reconnect()
            # Clear queue to prevent memory buildup
            try:
                while True: 
                    q.get_nowait()
            except queue.Empty:
                pass
            fail = 0
            # Short sleep to allow reconnect path to proceed
            time.sleep(0.2)
            
        # Tight loop for responsiveness (as in fire_target)
        time.sleep(0.005)
    print("🛑 Capture thread stopped")

def processing_thread_func(tracker: ObjectTracker, q: queue.Queue,
                           target_shape, target_color,
                           roi_state,
                           is_detecting_func):
    global processed_output
    print("🧠 Processing thread started.")
    processing_count = 0
    last_cleanup_time = time.time()

    while not stop_event.is_set():
        if not is_detecting_func():
            time.sleep(0.2)  # Increased sleep when not detecting
            continue
            
        try:
            frame_to_process = q.get(timeout=0.3)  # Reduced timeout
            processing_count += 1

            # เลื่อน ROI ตาม pitch ปัจจุบัน (จาก fire_target.py)
            with gimbal_angle_lock:
                pitch_deg = gimbal_angles[0]  # + ขึ้น, - ลง (ตาม SDK)
            # ถ้าก้มลง (pitch < 0) => ขยับ ROI_Y ขึ้น
            roi_y_dynamic = int(ROI_Y0 - (max(0.0, -pitch_deg) * ROI_SHIFT_PER_DEG))
            roi_y_dynamic = max(ROI_Y_MIN, min(ROI_Y_MAX, roi_y_dynamic))
            
            # Debug: Print ROI adjustment when it changes significantly
            if abs(roi_y_dynamic - ROI_Y0) > 2:
                print(f"🎯 ROI adjusted: pitch={pitch_deg:.1f}°, ROI_Y: {ROI_Y0} -> {roi_y_dynamic}")
            
            # Debug: Print pitch info every 30 frames
            if processing_count % 30 == 0:
                print(f"🎯 Pitch info: pitch={pitch_deg:.1f}°, ROI_Y0={ROI_Y0}, roi_y_dynamic={roi_y_dynamic}")
            
            # Debug: Test ROI adjustment by forcing gimbal movement (TEST ONLY)
            if processing_count % 100 == 0 and is_tracking_mode:
                print(f"🎯 TEST: Forcing gimbal pitch test...")
                try:
                    # Get gimbal from manager
                    gimbal = manager.get_gimbal()
                    if gimbal:
                        # Test pitch movement
                        gimbal.moveto(pitch=-15, yaw=0, pitch_speed=100).wait_for_completed()
                        time.sleep(0.5)
                        gimbal.moveto(pitch=15, yaw=0, pitch_speed=100).wait_for_completed()
                        time.sleep(0.5)
                        gimbal.moveto(pitch=0, yaw=0, pitch_speed=100).wait_for_completed()
                        print(f"🎯 TEST: Gimbal pitch test completed")
                except Exception as e:
                    print(f"🎯 TEST: Gimbal test error: {e}")

            ROI_X, ROI_W = roi_state["x"], roi_state["w"]
            ROI_H = roi_state["h"]
            roi_state["y"] = roi_y_dynamic

            roi_frame = frame_to_process[roi_y_dynamic:roi_y_dynamic+ROI_H, ROI_X:ROI_X+ROI_W]
            detections = tracker.get_raw_detections(roi_frame)

            detailed_results = []
            divider1 = int(ROI_W*0.33)
            divider2 = int(ROI_W*0.66)

            object_id_counter = 1
            for d in detections:
                shape, color, (x,y,w,h) = d['shape'], d['color'], d['box']
                endx = x+w
                zone = "Center"
                if endx < divider1: zone = "Left"
                elif x >= divider2: zone = "Right"
                is_target = (shape == target_shape and color == target_color)

                detailed_results.append({
                    "id": object_id_counter,
                    "color": color,
                    "shape": shape,
                    "zone": zone,
                    "is_target": is_target,
                    "box": (x,y,w,h)
                })
                object_id_counter += 1

            with output_lock:
                processed_output = {"details": detailed_results}
            
            # Periodic cleanup to prevent memory buildup
            if time.time() - last_cleanup_time > 30.0:  # Every 30 seconds
                processing_count = 0
                last_cleanup_time = time.time()
                # Force garbage collection
                import gc
                gc.collect()

        except queue.Empty:
            time.sleep(0.1)  # Sleep when no frames to process
            continue
        except Exception as e:
            print(f"⚠️ Processing error: {e}")
            time.sleep(0.1)  # Increased sleep on error
            # Clear queue to prevent buildup
            try:
                while True: 
                    q.get_nowait()
            except queue.Empty:
                pass

    print("🛑 Processing thread stopped.")

def start_detection_mode():
    """Start detection mode for 1 second"""
    global is_detecting_flag, detection_start_time
    is_detecting_flag["v"] = True
    detection_start_time = time.time()
    print("🔍 Detection mode activated for 1 second")

def stop_detection_mode():
    """Stop detection mode"""
    global is_detecting_flag
    is_detecting_flag["v"] = False
    print("🔍 Detection mode deactivated")

def check_detection_timer():
    """Check if detection mode should be stopped after 1 second (unless tracking mode is active)"""
    global is_detecting_flag, detection_start_time, is_tracking_mode
    if is_detecting_flag["v"] and detection_start_time is not None:
        if time.time() - detection_start_time >= 1.0:
            # Don't stop detection if we're in tracking mode
            if not is_tracking_mode:
                stop_detection_mode()
                return True
    return False

# Removed duplicate sub_angle_cb function

def check_for_targets():
    """Check if any targets are detected and start tracking if found"""
    global is_tracking_mode, targets_found, fired_targets, current_target_id, shots_fired, is_detecting_flag
    
    try:
        with output_lock:
            objects = processed_output["details"]
    except Exception as e:
        print(f"⚠️ Error accessing processed output: {e}")
        return False
    
    print(f"🔍 DEBUG: Found {len(objects)} objects in processed_output")
    for obj in objects:
        print(f"   - Object: {obj.get('color')} {obj.get('shape')} (ID: {obj.get('id')})")
    
    # Filter targets that match our criteria and haven't been fired at
    targets_found = []
    for obj in objects:
        # Safety check: ensure object has required keys
        if (obj.get('shape') == TARGET_SHAPE and 
            obj.get('color') == TARGET_COLOR and 
            obj.get('id') not in fired_targets and
            'id' in obj and 'box' in obj):
            targets_found.append(obj)
            print(f"🎯 DEBUG: Found matching target! {obj.get('color')} {obj.get('shape')} (ID: {obj.get('id')})")
    
    if targets_found and not is_tracking_mode:
        print(f"🎯 Found {len(targets_found)} target(s) matching {TARGET_COLOR} {TARGET_SHAPE}")
        print("🎯 Starting PID tracking mode...")
        print("🔍 Keeping detection mode active for PID tracking...")
        
        # Keep detection mode active for PID tracking
        is_detecting_flag["v"] = True
        
        is_tracking_mode = True
        # Safety check: ensure targets_found has at least one element
        if len(targets_found) > 0 and 'id' in targets_found[0]:
            current_target_id = targets_found[0]['id']
            shots_fired = 0
            return True
        else:
            print("⚠️ Target found but missing ID, skipping PID tracking")
            is_tracking_mode = False
            return False
    
    print(f"🔍 DEBUG: No matching targets found. Looking for {TARGET_COLOR} {TARGET_SHAPE}")
    return False

def pid_tracking_and_firing(manager, roi_state):
    """PID tracking and firing system (from fire_target.py)"""
    global is_tracking_mode, targets_found, fired_targets, current_target_id, shots_fired
    
    if not is_tracking_mode:
        return False
    
    # PID states (simplified for real-time tracking)
    
    gimbal = manager.get_gimbal()
    blaster = manager.get_blaster()
    
    if gimbal is None or blaster is None:
        print("⚠️ Gimbal or blaster not available")
        return False
    
    # Find current target (choose largest target box from live detection)
    with output_lock:
        dets = list(processed_output["details"])
    
    target_box = None
    max_area = -1
    for det in dets:
        if det.get("is_target", False):
            x,y,w,h = det["box"]
            area = w*h
            if area > max_area:
                max_area = area
                target_box = (x,y,w,h)
    
    # Only work when there's a target
    if target_box is not None:
        x, y, w, h = target_box
        
        # Center target within ROI
        cx_roi = x + w/2.0
        cy_roi = y + h/2.0
        
        # Convert to pixels in full frame (using dynamic ROI)
        ROI_X, ROI_Y, ROI_W, ROI_H = roi_state["x"], roi_state["y"], roi_state["w"], roi_state["h"]
        cx = ROI_X + cx_roi
        cy = ROI_Y + cy_roi
        
        # Image center
        center_x = FRAME_W/2.0
        center_y = FRAME_H/2.0
        
        # Error (image → gimbal): yaw uses x, pitch uses y
        err_x = (center_x - cx)
        err_y = (center_y - cy) + PITCH_BIAS_PIX
        
        # Deadzone to reduce jitter
        if abs(err_x) < PIX_ERR_DEADZONE: err_x = 0.0
        if abs(err_y) < PIX_ERR_DEADZONE: err_y = 0.0
        
        # Simple proportional control for now
        u_x = PID_KP * err_x
        u_y = PID_KP * err_y
        
        # Clamp
        u_x = float(np.clip(u_x, -MAX_YAW_SPEED, MAX_YAW_SPEED))
        u_y = float(np.clip(u_y, -MAX_PITCH_SPEED, MAX_PITCH_SPEED))
        
        try:
            # Note: pitch_speed in SDK axis reversed with image
            gimbal.drive_speed(pitch_speed=-u_y, yaw_speed=u_x)
        except Exception as e:
            print("drive_speed error:", e)
        
        # Check lock for firing
        locked = (abs(err_x) <= LOCK_TOL_X) and (abs(err_y) <= LOCK_TOL_Y)
        
        if locked:
            if shots_fired < FIRE_SHOTS_COUNT:
                try:
                    blaster.fire(fire_type=r_blaster.WATER_FIRE)
                    shots_fired += 1
                    print(f"🔥 Fired shot {shots_fired}/{FIRE_SHOTS_COUNT} at target")
                    time.sleep(0.2)  # Prevent rapid firing
                except Exception as e:
                    print(f"fire error: {e}")
            else:
                print(f"✅ Completed firing {FIRE_SHOTS_COUNT} shots")
                is_tracking_mode = False
                fired_targets.clear()
                shots_fired = 0  # Reset for next target
                # Stop detection mode when tracking is complete
                stop_detection_mode()
                return False
    else:
        # No target → slowly stop
        try:
            gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
        except Exception:
            pass
    
    return True

def save_detected_objects_to_map(occupancy_map):
    """Save detected objects to map with position details in the next cell"""
    global processed_output, CURRENT_POSITION, CURRENT_DIRECTION
    
    try:
        with output_lock:
            objects = processed_output["details"]
    except Exception as e:
        print(f"⚠️ Error accessing processed output: {e}")
        return
    
    if objects:
        # Calculate next node position (where robot will move to)
        dir_vectors = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # N, E, S, W
        next_r, next_c = CURRENT_POSITION[0] + dir_vectors[CURRENT_DIRECTION][0], CURRENT_POSITION[1] + dir_vectors[CURRENT_DIRECTION][1]
        
        # Adjust object zones based on robot's facing direction
        adjusted_objects = []
        for obj in objects:
            adjusted_obj = obj.copy()
            
            # เพิ่มข้อมูลว่า object นี้เจอที่โหนดไหน
            adjusted_obj['detected_from_node'] = list(CURRENT_POSITION)
            
            # Adjust zone based on robot's facing direction
            # If robot faces East and sees object in Right zone, 
            # the object is actually in the Left zone of the next cell
            if CURRENT_DIRECTION == 1:  # Facing East
                if obj['zone'] == 'Right':
                    adjusted_obj['zone'] = 'Left'
                elif obj['zone'] == 'Left':
                    adjusted_obj['zone'] = 'Right'
            elif CURRENT_DIRECTION == 3:  # Facing West
                if obj['zone'] == 'Right':
                    adjusted_obj['zone'] = 'Left'
                elif obj['zone'] == 'Left':
                    adjusted_obj['zone'] = 'Right'
            elif CURRENT_DIRECTION == 0:  # Facing North
                if obj['zone'] == 'Right':
                    adjusted_obj['zone'] = 'Left'
                elif obj['zone'] == 'Left':
                    adjusted_obj['zone'] = 'Right'
            elif CURRENT_DIRECTION == 2:  # Facing South
                if obj['zone'] == 'Right':
                    adjusted_obj['zone'] = 'Left'
                elif obj['zone'] == 'Left':
                    adjusted_obj['zone'] = 'Right'
            
            adjusted_objects.append(adjusted_obj)
        
        # Save adjusted objects to map at the next position
        occupancy_map.save_objects_to_map(adjusted_objects, (next_r, next_c), CURRENT_DIRECTION)
        
        # Also save to global detected_objects
        with object_lock:
            detected_objects.extend(adjusted_objects)
        
        print(f"✅ Saved {len(adjusted_objects)} objects to map at next position ({next_r}, {next_c})")
        
        # Print detailed object information with zone positioning
        for obj in adjusted_objects:
            zone_info = f"in {obj['zone']} zone"
            if obj['zone'] == 'Left':
                zone_info += " (attached to left wall)"
            elif obj['zone'] == 'Right':
                zone_info += " (attached to right wall)"
            elif obj['zone'] == 'Center':
                zone_info += " (in center of cell)"
            
            detected_from_info = f"detected from ({obj['detected_from_node'][0]},{obj['detected_from_node'][1]})" if obj['detected_from_node'] else "detected from unknown"
            print(f"   📦 Object: {obj['color']} {obj['shape']} {zone_info} {detected_from_info} {'(TARGET!)' if obj['is_target'] else ''}")
    else:
        print("📭 No objects detected")

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

    def save_objects_to_map(self, objects, position, direction):
        """Save detected objects to the map at specified position"""
        r, c = position
        if 0 <= r < self.height and 0 <= c < self.width:
            # Add objects to the cell
            if not hasattr(self.grid[r][c], 'objects'):
                self.grid[r][c].objects = []
            
            # Add each object with zone information
            for obj in objects:
                obj_data = {
                    'color': obj.get('color', 'unknown'),
                    'shape': obj.get('shape', 'unknown'),
                    'zone': obj.get('zone', 'unknown'),
                    'is_target': obj.get('is_target', False),
                    'timestamp': time.time(),
                    'detected_from_node': obj.get('detected_from_node', [])
                }
                self.grid[r][c].objects.append(obj_data)
            
            print(f"📦 Saved {len(objects)} objects to cell ({r}, {c})")

class RealTimeVisualizer:
    def __init__(self, grid_size, target_dest=None):
        self.grid_size = grid_size
        self.target_dest = target_dest
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=MAP_FIGURE_SIZE)
        self.colors = {"robot": "#0000FF", "target": "#FFD700", "path": "#FFFF00", "wall": "#000000", "wall_prob": "#000080"}
        self.obj_color_map = {'Red': '#FF0000', 'Green': '#00FF00', 'Blue': '#0080FF', 'Yellow': '#FFFF00', 'Unknown': '#808080'}
        # เพิ่มตัวควบคุมความถี่การวาดกราฟ
        self.update_counter = 0
        self.update_interval = 3  # อัปเดตทุก 3 โหนด

    def update_plot(self, occupancy_map, robot_pos, path=None):
        # เพิ่มตัวนับและข้ามการวาดบางครั้ง
        self.update_counter += 1
        if self.update_counter % self.update_interval != 0:
            # วาดเฉพาะหุ่นยนต์และเส้นทาง
            self.ax.clear()
            self.ax.set_title("Real-time Hybrid Belief Map (Nodes & Walls)")
            self.ax.set_xticks([]); self.ax.set_yticks([])
            self.ax.set_xlim(-0.5, self.grid_size - 0.5)
            self.ax.set_ylim(self.grid_size - 0.5, -0.5)
            # วาดเฉพาะกริดพื้นฐาน
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    prob = occupancy_map.grid[r][c].get_node_probability()
                    if prob > OCCUPANCY_THRESHOLD: color = '#8B0000'
                    elif prob < FREE_THRESHOLD: color = '#D3D3D3'
                    else: color = '#90EE90'
                    self.ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=color, edgecolor='k', lw=0.5))
                    # ไม่แสดงตัวเลขความน่าจะเป็นเพื่อประหยัดเวลา
            # วาดหุ่นยนต์และเส้นทาง
            self.ax.plot(robot_pos[1], robot_pos[0], 'o', color=self.colors['robot'], markersize=12, markeredgecolor='white', markeredgewidth=2)
            if path: self.ax.plot([p[1] for p in path], [p[0] for p in path], color=self.colors['path'], linewidth=3, alpha=0.7)
            plt.draw(); plt.pause(0.01)
            return
        
        # วาดเต็มรูปแบบทุก 3 โหนด
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
                # แสดงตัวเลขเฉพาะเซลล์ที่สำคัญ
                if abs(prob - 0.5) > 0.1:  # เฉพาะเซลล์ที่มีความน่าจะเป็นชัดเจน
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
                
                # Display objects in the cell
                if hasattr(cell, 'objects') and cell.objects:
                    for i, obj in enumerate(cell.objects):
                        # Position objects in different zones
                        if obj['zone'] == 'Left':
                            obj_x, obj_y = c - 0.3, r
                        elif obj['zone'] == 'Right':
                            obj_x, obj_y = c + 0.3, r
                        elif obj['zone'] == 'Center':
                            obj_x, obj_y = c, r
                        else:
                            obj_x, obj_y = c, r
                        
                        # Shape-based marker
                        shape = obj.get('shape', 'Uncertain')
                        if shape == 'Circle':
                            marker = 'o'
                        elif shape == 'Square':
                            marker = 's'
                        elif 'Rectangle' in shape:
                            marker = 's'
                        else:
                            marker = 'D'  # Diamond for uncertain
                        
                        # Color based on detected color
                        obj_color = obj.get('color', 'Unknown')
                        color = self.obj_color_map.get(obj_color, '#808080')
                        
                        # Highlight target with red border
                        edge_color = '#FF0000' if obj.get('is_target', False) else 'black'
                        edge_width = 3 if obj.get('is_target', False) else 1
                        
                        self.ax.scatter(obj_x, obj_y, c=color, marker=marker, s=120, edgecolors=edge_color, linewidth=edge_width)
                        
                        # Display ID below the marker
                        obj_id = obj.get('id', 'Unknown')
                        self.ax.text(obj_x, obj_y + 0.35, f"ID:{obj_id}", ha="center", va="top", fontsize=7, 
                                    fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2', edgecolor='none'))
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
        
        # Collect all detected objects with their details
        all_objects = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell = occupancy_map.grid[r][c]
                if hasattr(cell, 'objects') and cell.objects:
                    for obj in cell.objects:
                        all_objects.append({
                            'id': obj.get('id', '?'),
                            'color': obj.get('color', 'Unknown'),
                            'shape': obj.get('shape', 'Uncertain'),
                            'zone': obj.get('zone', 'Center'),
                            'position': f"({r},{c})",
                            'is_target': obj.get('is_target', False),
                            'detected_from_node': obj.get('detected_from_node', [])
                        })
        
        # Display object details on the right side
        display_objects = (self.update_counter % self.update_interval == 0)  # แสดงเฉพาะทุก 3 โหนด
        if display_objects:
            # จำกัดจำนวนวัตถุที่แสดง (เฉพาะ 10 อันแรก)
            limited_objects = all_objects[:10]
            details_text = "=== DETECTED OBJECTS ===\n\n"
            for obj in sorted(limited_objects, key=lambda x: x['id']):
                target_mark = " ⭐TARGET" if obj['is_target'] else ""
                detected_from = f"detected from ({obj['detected_from_node'][0]},{obj['detected_from_node'][1]})" if obj['detected_from_node'] else "detected from unknown"
                details_text += f"ID {obj['id']}: {obj['color']} {obj['shape']}\n"
                details_text += f"  @ {obj['position']} [{obj['zone']}] {detected_from}{target_mark}\n\n"
            if len(all_objects) > 10:
                details_text += f"... and {len(all_objects) - 10} more objects"
            
            # Add text box on the right side
            self.fig.text(0.78, 0.5, details_text, fontsize=9, family='monospace',
                         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        
        legend_elements = [ 
            plt.Rectangle((0,0),1,1, facecolor='#8B0000', label=f'Node Occupied (P>{OCCUPANCY_THRESHOLD})'), 
            plt.Rectangle((0,0),1,1, facecolor='#90EE90', label=f'Node Unknown'), 
            plt.Rectangle((0,0),1,1, facecolor='#D3D3D3', label=f'Node Free (P<{FREE_THRESHOLD})'), 
            plt.Line2D([0], [0], color=self.colors['wall'], lw=4, label='Wall Occupied'), 
            plt.Rectangle((0,0),1,1, facecolor=self.colors['robot'], label='Robot'), 
            plt.Rectangle((0,0),1,1, facecolor=self.colors['target'], label='Target') 
        ]
        self.ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=8)
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
        normalized_target = self.normalize_angle(target_yaw); time.sleep(0.05)
        robot_rotation = -self.normalize_angle(normalized_target - self.current_yaw)
        print(f"\n🔧 Correcting Yaw: {self.current_yaw:.1f}° -> {target_yaw:.1f}°. Rotating: {robot_rotation:.1f}°")
        
        if abs(robot_rotation) > self.yaw_tolerance:
            # ปรับปรุงการหมุนให้แม่นยำขึ้น
            if abs(robot_rotation) > 45:  # สำหรับการหมุน 90 องศา
                chassis.move(x=0, y=0, z=robot_rotation, z_speed=80).wait_for_completed(timeout=2)
            else:  # สำหรับการปรับแต่งเล็กน้อย
                chassis.move(x=0, y=0, z=robot_rotation, z_speed=60).wait_for_completed(timeout=1)
            time.sleep(0.1)  # เพิ่มเวลารอให้การหมุนเสถียร
        
        final_error = abs(self.normalize_angle(normalized_target - self.current_yaw))
        if final_error <= self.yaw_tolerance: 
            print(f"✅ Yaw Correction Success: {self.current_yaw:.1f}°"); 
            return True
        
        print(f"⚠️ First attempt incomplete. Current: {self.current_yaw:.1f}°. Fine-tuning...")
        remaining_rotation = -self.normalize_angle(normalized_target - self.current_yaw)
        
        # ปรับปรุงการปรับแต่งให้แม่นยำขึ้น
        if abs(remaining_rotation) > 0.5 and abs(remaining_rotation) < 30:
            chassis.move(x=0, y=0, z=remaining_rotation, z_speed=30).wait_for_completed(timeout=3)
            time.sleep(0.15)  # เพิ่มเวลารอ
        
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
        KP_YAW = 1.8; MAX_YAW_SPEED = 25
        yaw_error = attitude_handler.normalize_angle(target_yaw - attitude_handler.current_yaw)
        speed = KP_YAW * yaw_error
        return max(min(speed, MAX_YAW_SPEED), -MAX_YAW_SPEED)

    def move_forward_one_grid(self, axis, attitude_handler):
        attitude_handler.correct_yaw_to_target(self.chassis, get_compensated_target_yaw()) # MODIFIED
        target_distance = 0.6
        pid = PID(Kp=1.0, Ki=0.25, Kd=8, setpoint=target_distance)
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
            yaw_correction = self._calculate_yaw_correction(attitude_handler, get_compensated_target_yaw()) # MODIFIED
            self.chassis.drive_speed(x=speed, y=0, z=yaw_correction, timeout=1)
            print(f"Moving... Dist: {relative_position:.3f}/{target_distance:.2f} m", end='\r')
        self.chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0); time.sleep(0.5)

    def adjust_position_to_wall(self, sensor_adaptor, attitude_handler, side, sensor_config, target_distance_cm, direction_multiplier):
        compensated_yaw = get_compensated_target_yaw() # MODIFIED
        print(f"\n--- Adjusting {side} Side (Yaw locked at {compensated_yaw:.2f}°) ---") # MODIFIED
        print(f"   -> Config: ID={sensor_config['sharp_id']}, Port={sensor_config['sharp_port']}, Target={target_distance_cm}cm")
        TOLERANCE_CM, MAX_EXEC_TIME, KP_SLIDE, MAX_SLIDE_SPEED = 0.8, 4, 0.045, 0.18
        start_time = time.time()
        # Reset PID controller by stopping briefly
        self.chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
        time.sleep(0.05)
        while time.time() - start_time < MAX_EXEC_TIME:
            adc_val = sensor_adaptor.get_adc(id=sensor_config["sharp_id"], port=sensor_config["sharp_port"])
            current_dist = convert_adc_to_cm(adc_val)
            dist_error = target_distance_cm - current_dist
            if abs(dist_error) <= TOLERANCE_CM:
                print(f"\n[{side}] Target distance reached! Final distance: {current_dist:.2f} cm")
                break
            slide_speed = max(min(direction_multiplier * KP_SLIDE * dist_error, MAX_SLIDE_SPEED), -MAX_SLIDE_SPEED)
            yaw_correction = self._calculate_yaw_correction(attitude_handler, compensated_yaw) # MODIFIED
            self.chassis.drive_speed(x=0, y=slide_speed, z=yaw_correction)
            print(f"Adjusting {side}... Current: {current_dist:5.2f}cm, Target: {target_distance_cm:4.1f}cm, Error: {dist_error:5.2f}cm, Speed: {slide_speed:5.3f}", end='\r')
            time.sleep(0.02)
        else:
            print(f"\n[{side}] Movement timed out!")
        self.chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
        time.sleep(0.1)

    def center_in_node_with_tof(self, scanner, attitude_handler, target_cm=17, tol_cm=1.0, max_adjust_time=6.0):
        """
        REVISED: Now respects the global activity lock from the scanner.
        It will not run if a side-scan operation is in progress.
        """
        # [CRITICAL] Guard Clause to respect the global lock
        if scanner.is_performing_full_scan:
            print("[ToF Centering] SKIPPED: A critical side-scan is in progress.")
            return

        print("\n--- Stage: Centering in Node with ToF ---")
        time.sleep(0.1)
        tof_dist = scanner.last_tof_distance_cm
        if tof_dist is None or math.isinf(tof_dist):
            print("[ToF] ❌ No valid ToF data available. Skipping centering.")
            return
        print(f"[ToF] Initial front distance: {tof_dist:.2f} cm")
        if tof_dist >= 50:
            print("[ToF] Distance >= 50cm, likely in an open space. Skipping centering.")
            return
        direction = 0
        if tof_dist > target_cm + tol_cm:
            print("[ToF] Too far from front wall. Moving forward...")
            direction = abs(TOF_ADJUST_SPEED)
        elif tof_dist < 22:
            print("[ToF] Too close to front wall. Moving backward...")
            direction = -abs(TOF_ADJUST_SPEED)
        else:
            print("[ToF] In range (22cm - target), but not centered. Moving forward...")
            direction = abs(TOF_ADJUST_SPEED)
        if direction == 0:
            print(f"[ToF] Already centered within tolerance ({tof_dist:.2f} cm). Skipping.")
            return
        start_time = time.time()
        compensated_yaw = get_compensated_target_yaw() # MODIFIED
        while time.time() - start_time < max_adjust_time:
            yaw_correction = self._calculate_yaw_correction(attitude_handler, compensated_yaw) # MODIFIED
            self.chassis.drive_speed(x=direction, y=0, z=yaw_correction, timeout=0.1)
            time.sleep(0.12)
            self.chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
            time.sleep(0.08)
            current_tof = scanner.last_tof_distance_cm
            if current_tof is None or math.isinf(current_tof):
                continue
            print(f"[ToF] Adjusting... Current Distance: {current_tof:.2f} cm", end="\r")
            if abs(current_tof - target_cm) <= tol_cm:
                print(f"\n[ToF] ✅ Centering complete. Final distance: {current_tof:.2f} cm")
                break
            if (direction > 0 and current_tof < target_cm - tol_cm) or \
            (direction < 0 and current_tof > target_cm + tol_cm):
                direction *= -1
                print("\n[ToF] 🔄 Overshot target. Reversing direction for fine-tuning.")
        else:
            print(f"\n[ToF] ⚠️ Centering timed out. Final distance: {scanner.last_tof_distance_cm:.2f} cm")
        self.chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
        time.sleep(0.1)

    def rotate_to_direction(self, target_direction, attitude_handler, scanner=None):
        global CURRENT_DIRECTION
        if CURRENT_DIRECTION == target_direction: return
        
        diff = (target_direction - CURRENT_DIRECTION + 4) % 4
        if diff == 1: 
            self.rotate_90_degrees_right(attitude_handler)
        elif diff == 3: 
            self.rotate_90_degrees_left(attitude_handler)
        elif diff == 2: 
            self.rotate_90_degrees_right(attitude_handler); 
            self.rotate_90_degrees_right(attitude_handler)
        
        # รอให้การเลี้ยวเสร็จสิ้นและเสถียร
        print("   -> Waiting for robot rotation to complete...")
        time.sleep(0.5)  # รอให้การเลี้ยวเสร็จสิ้นและเสถียร
        
        # ปรับ gimbal ให้ตรงกับทิศทางใหม่หลังจากหุ่นหมุนเสร็จ
        if scanner and scanner.gimbal:
            try:
                print("   -> Adjusting gimbal to match new robot direction...")
                scanner.gimbal.moveto(pitch=0, yaw=0, yaw_speed=SPEED_ROTATE).wait_for_completed()
                time.sleep(0.2)  # รอให้ gimbal ปรับเสร็จ
                print("   -> Gimbal adjusted to match robot direction")
            except Exception as e:
                print(f"   -> Gimbal adjustment failed: {e}")
                print("   -> Continuing without gimbal adjustment...")

    def rotate_90_degrees_right(self, attitude_handler):
        global CURRENT_TARGET_YAW, CURRENT_DIRECTION, ROBOT_FACE
        print("🔄 Rotating 90° RIGHT...")
        CURRENT_TARGET_YAW = attitude_handler.normalize_angle(CURRENT_TARGET_YAW + 90)
        
        # ปรับปรุงการหมุนให้แม่นยำขึ้น
        success = attitude_handler.correct_yaw_to_target(self.chassis, get_compensated_target_yaw())
        if not success:
            print("⚠️ Primary rotation failed, attempting alternative method...")
            # ลองใช้วิธีหมุนแบบอื่น
            attitude_handler.correct_yaw_to_target(self.chassis, get_compensated_target_yaw())
        
        CURRENT_DIRECTION = (CURRENT_DIRECTION + 1) % 4; ROBOT_FACE += 1
    def rotate_90_degrees_left(self, attitude_handler):
        global CURRENT_TARGET_YAW, CURRENT_DIRECTION, ROBOT_FACE
        print("🔄 Rotating 90° LEFT...")
        CURRENT_TARGET_YAW = attitude_handler.normalize_angle(CURRENT_TARGET_YAW - 90)
        
        # ปรับปรุงการหมุนให้แม่นยำขึ้น
        success = attitude_handler.correct_yaw_to_target(self.chassis, get_compensated_target_yaw())
        if not success:
            print("⚠️ Primary rotation failed, attempting alternative method...")
            # ลองใช้วิธีหมุนแบบอื่น
            attitude_handler.correct_yaw_to_target(self.chassis, get_compensated_target_yaw())
        
        CURRENT_DIRECTION = (CURRENT_DIRECTION - 1 + 4) % 4; ROBOT_FACE -= 1
        if ROBOT_FACE < 1: ROBOT_FACE += 4
    def cleanup(self):
        try: self.chassis.unsub_position()
        except Exception: pass

class EnvironmentScanner:
    """ 
    REVISED: Added a global activity lock 'is_performing_full_scan' to prevent
    any other function from interfering during the complex side-scanning process.
    """
    def __init__(self, sensor_adaptor, tof_sensor, gimbal, chassis):
        self.sensor_adaptor, self.tof_sensor, self.gimbal, self.chassis = sensor_adaptor, tof_sensor, gimbal, chassis
        self.tof_wall_threshold_cm = 60.0
        
        # --- State Management Variables ---
        self.last_tof_distance_cm = float('inf')  # Stores the FRONT distance
        self.side_tof_reading_cm = float('inf')   # [NEW] Temporary storage for side readings
        self.is_gimbal_centered = True            # [NEW] State flag to control the callback
        
        # --- [NEW] Global Activity Lock ---
        self.is_performing_full_scan = False

        # ลดความถี่การอ่าน ToF เพื่อลด error
        self.tof_sensor.sub_distance(freq=5, callback=self._tof_data_handler)
        
        self.side_sensors = {
            "Left":  { "sharp_id": 1, "sharp_port": 1, "ir_id": 1, "ir_port": 2 },
            "Right": { "sharp_id": 2, "sharp_port": 1, "ir_id": 2, "ir_port": 2 }
        }

    def _tof_data_handler(self, sub_info):
        """ 
        MODIFIED: This callback now respects the 'is_gimbal_centered' flag.
        It only updates the main front distance variable if the gimbal is facing forward.
        Otherwise, it updates a separate variable for side readings.
        """
        try:
            # เพิ่มการตรวจสอบข้อมูลก่อนใช้งาน
            if sub_info and len(sub_info) > 0 and sub_info[0] is not None:
                calibrated_cm = calibrate_tof_value(sub_info[0])
                if self.is_gimbal_centered:
                    self.last_tof_distance_cm = calibrated_cm
                else:
                    self.side_tof_reading_cm = calibrated_cm
        except Exception as e:
            # เงียบ error เพื่อไม่ให้รบกวนการทำงาน
            pass

    def _get_stable_reading_cm(self, side, duration=0.35):
        sensor_info = self.side_sensors.get(side)
        if not sensor_info: return None, None
        readings = []
        start_time = time.time()
        while time.time() - start_time < duration:
            adc = self.sensor_adaptor.get_adc(id=sensor_info["sharp_id"], port=sensor_info["sharp_port"])
            readings.append(convert_adc_to_cm(adc))
            time.sleep(0.05)
        if len(readings) < 5: return None, None
        return statistics.mean(readings), statistics.stdev(readings)

    def get_sensor_readings(self):
        """
        REVISED: Now uses a global lock 'is_performing_full_scan' to ensure
        this entire operation is atomic and uninterruptible.
        """
        # [CRITICAL] Set the global lock at the very beginning
        self.is_performing_full_scan = True
        try:
            self.gimbal.moveto(pitch=0, yaw=0, yaw_speed=SPEED_ROTATE).wait_for_completed(); time.sleep(0.15)
            
            readings = {}
            readings['front'] = (self.last_tof_distance_cm < self.tof_wall_threshold_cm)
            print(f"[SCAN] Front (ToF): {self.last_tof_distance_cm:.1f}cm -> {'OCCUPIED' if readings['front'] else 'FREE'}")
            
            for side in ["Left", "Right"]:
                avg_dist, std_dev = self._get_stable_reading_cm(side)
                if avg_dist is None:
                    print(f"[{side.upper()}] Wall Check Error: Not enough sensor data.")
                    readings[side.lower()] = False
                    continue
                
                is_sharp_detecting_wall = (avg_dist < SHARP_WALL_THRESHOLD_CM and std_dev < SHARP_STDEV_THRESHOLD)
                ir_value = self.sensor_adaptor.get_io(id=self.side_sensors[side]["ir_id"], port=self.side_sensors[side]["ir_port"])
                is_ir_detecting_wall = (ir_value == 0)
                
                # เพิ่มการตรวจสอบความแม่นยำของเซ็นเซอร์
                print(f"    -> Sharp: {avg_dist:.1f}cm (std: {std_dev:.3f}), IR: {ir_value}")

                print(f"\n[SCAN] {side} Side Analysis:")
                print(f"    -> Sharp -> Suggests: {'WALL' if is_sharp_detecting_wall else 'FREE'}")
                print(f"    -> IR    -> Suggests: {'WALL' if is_ir_detecting_wall else 'FREE'}")

                if is_sharp_detecting_wall == is_ir_detecting_wall:
                    is_wall = is_sharp_detecting_wall
                    print(f"    -> Decision: Sensors agree. Result is {'WALL' if is_wall else 'FREE'}.")
                else:
                    print("    -> Ambiguity detected! Confirming with ToF...")
                    print(f"    -> Sharp suggests {'WALL' if is_sharp_detecting_wall else 'FREE'}, IR suggests {'WALL' if is_ir_detecting_wall else 'FREE'}")
                    target_gimbal_yaw = -90 if side == "Left" else 90
                    
                    try:
                        self.is_gimbal_centered = False
                        self.gimbal.moveto(pitch=0, yaw=target_gimbal_yaw, yaw_speed=SPEED_ROTATE).wait_for_completed()
                        time.sleep(0.1)
                        
                        tof_confirm_dist_cm = self.side_tof_reading_cm
                        print(f"    -> ToF reading at {target_gimbal_yaw}° is {tof_confirm_dist_cm:.2f} cm.")
                        
                        is_wall = (tof_confirm_dist_cm < self.tof_wall_threshold_cm)
                        print(f"    -> ToF Confirmation: {'WALL DETECTED' if is_wall else 'NO WALL'}.")
                    
                    finally:
                        self.gimbal.moveto(pitch=0, yaw=0, yaw_speed=SPEED_ROTATE).wait_for_completed()
                        self.is_gimbal_centered = True
                        time.sleep(0.1)

                readings[side.lower()] = is_wall
                print(f"    -> Final Result for {side} side: {'WALL' if is_wall else 'FREE'}")
            
            return readings
        finally:
            # [CRITICAL] Release the global lock when the function is completely done
            self.is_performing_full_scan = False

    def get_front_tof_cm(self):
        self.gimbal.moveto(pitch=0, yaw=0, yaw_speed=SPEED_ROTATE).wait_for_completed()
        time.sleep(0.1)
        return self.last_tof_distance_cm

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

def is_dead_end(occupancy_map, position, visited_cells):
    """ตรวจสอบว่าโหนดปัจจุบันเป็นทางตันหรือไม่ - ปรับปรุงจาก debug_img_3-10.py"""
    r, c = position
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # N, E, S, W
    dir_names = ['N', 'E', 'S', 'W']
    
    accessible_neighbors = 0
    unvisited_neighbors = 0
    blocked_directions = []
    
    for i, (dr, dc) in enumerate(directions):
        nr, nc = r + dr, c + dc
        
        # ตรวจสอบว่าอยู่ในขอบเขตหรือไม่
        if 0 <= nr < occupancy_map.height and 0 <= nc < occupancy_map.width:
            # ตรวจสอบว่าเส้นทางเปิดหรือไม่
            if occupancy_map.is_path_clear(r, c, nr, nc):
                accessible_neighbors += 1
                # ตรวจสอบว่าเป็นโหนดที่ยังไม่ไปหรือไม่
                if (nr, nc) not in visited_cells:
                    unvisited_neighbors += 1
            else:
                blocked_directions.append(dir_names[i])
        else:
            blocked_directions.append(dir_names[i])
    
    # ถ้าไม่มีเพื่อนบ้านที่เข้าถึงได้ หรือมีแค่เพื่อนบ้านที่ไปแล้ว = ทางตัน
    is_dead = (accessible_neighbors == 0) or (accessible_neighbors > 0 and unvisited_neighbors == 0)
    
    if is_dead:
        print(f"   -> Dead end analysis: {accessible_neighbors} accessible neighbors, {unvisited_neighbors} unvisited")
        print(f"   -> Blocked directions: {blocked_directions}")
    
    return is_dead

def mark_cell_as_dead_end(occupancy_map, position):
    """ทำเครื่องหมายว่าโหนดนี้เป็นทางตัน - ปรับปรุงจาก debug_img_3-10.py"""
    r, c = position
    # เพิ่มข้อมูลใน occupancy map เพื่อระบุว่าโหนดนี้เป็นทางตัน
    # สามารถใช้ log_odds หรือเพิ่มฟิลด์ใหม่ได้
    print(f"   -> Marking cell {position} as dead end")
    
    # เพิ่มการตรวจสอบเพิ่มเติมว่าจริงๆ แล้วเป็นทางตันหรือไม่
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # N, E, S, W
    accessible_count = 0
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < occupancy_map.height and 0 <= nc < occupancy_map.width:
            if occupancy_map.is_path_clear(r, c, nr, nc):
                accessible_count += 1
    
    print(f"   -> Dead end verification: {accessible_count} accessible directions remaining")

def find_nearest_unvisited_path(occupancy_map, start_pos, visited_cells):
    """ใช้ multi-source BFS เพื่อหาเซลล์ที่ยังไม่ไปที่ใกล้ที่สุดใน O(N) - ปรับปรุงให้ตรวจสอบการเข้าถึงได้"""
    h, w = occupancy_map.height, occupancy_map.width
    
    # ใช้ BFS เดียวจากจุดเริ่มต้น หาเซลล์แรกที่ยังไม่ไป
    queue = [(start_pos, [start_pos])]
    visited_bfs = {start_pos}
    
    while queue:
        current_pos, path = queue.pop(0)
        
        # เช็คทุกทิศทาง
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = current_pos[0] + dr, current_pos[1] + dc
            
            # เช็คขอบเขตและไม่เคยไปใน BFS นี้
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited_bfs:
                visited_bfs.add((nr, nc))
                
                # เช็คว่าเป็นเซลล์ที่ยังไม่ไปในการสำรวจหรือไม่
                if (nr, nc) not in visited_cells and not occupancy_map.grid[nr][nc].is_node_occupied():
                    # ตรวจสอบเพิ่มเติมว่าโหนดนี้สามารถเข้าถึงได้จริงหรือไม่
                    # โดยตรวจสอบว่ามีเส้นทางที่เปิดอยู่จากโหนดปัจจุบันไปยังโหนดปลายทาง
                    if occupancy_map.is_path_clear(current_pos[0], current_pos[1], nr, nc):
                        print(f"   -> Found accessible unvisited node: ({nr},{nc})")
                        return path + [(nr, nc)]
                    else:
                        print(f"   -> Found unvisited node ({nr},{nc}) but path is blocked")
                        continue
                
                # ถ้าเป็นเซลล์ที่ไปแล้วและไม่เป็นกำแพง ให้เพิ่มในคิว
                if occupancy_map.is_path_clear(current_pos[0], current_pos[1], nr, nc):
                    new_path = list(path)
                    new_path.append((nr, nc))
                    queue.append(((nr, nc), new_path))
    
    print("   -> No accessible unvisited nodes found")
    return None

# แก้ไขฟังก์ชัน execute_path

def execute_path(path, movement_controller, attitude_handler, scanner, visualizer, occupancy_map, path_name="Backtrack"):
    global CURRENT_POSITION
    print(f"🎯 Executing {path_name} Path: {path}")
    dir_vectors_map = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}
    dir_map_abs_char = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}
    
    # บันทึกตำแหน่งเริ่มต้นของ path execution
    log_position_timestamp(CURRENT_POSITION, CURRENT_DIRECTION, f"{path_name}_start")

    # เดินไปยังโหนดก่อนสุดท้าย (ไม่ใช่โหนดสุดท้าย) - ไม่เปิดโหมด detect
    for i in range(len(path) - 2):  # หยุดที่โหนดก่อนโหนดที่ยังไม่สำรวจ
        visualizer.update_plot(occupancy_map, path[i], path)
        current_r, current_c = path[i]
        
        if i + 1 < len(path):
            next_r, next_c = path[i+1]
            dr, dc = next_r - current_r, next_c - current_c
            
            target_direction = dir_vectors_map[(dr, dc)]
            
            movement_controller.rotate_to_direction(target_direction, attitude_handler, scanner)
            
            # ตรวจสอบว่าการหมุนสำเร็จหรือไม่
            current_direction_check = CURRENT_DIRECTION
            if current_direction_check != target_direction:
                print(f"⚠️ Rotation verification failed. Expected: {target_direction}, Got: {current_direction_check}")
                print("🔄 Attempting rotation correction...")
                movement_controller.rotate_to_direction(target_direction, attitude_handler, scanner)
            
            # --- ส่วนการตรวจสอบแบบง่าย (ไม่เปิดโหมด detect) ---
            print(f"   -> [{path_name}] Quick ToF check to ({next_r},{next_c})...")
            scanner.gimbal.moveto(pitch=0, yaw=0, yaw_speed=SPEED_ROTATE).wait_for_completed()
            time.sleep(0.1)  # ลดเวลารอให้เร็วขึ้น
            
            # อ่านค่าจากเซ็นเซอร์จริง
            is_blocked = scanner.get_front_tof_cm() < scanner.tof_wall_threshold_cm
            
            # อัปเดตแผนที่
            occupancy_map.update_wall(current_r, current_c, dir_map_abs_char[CURRENT_DIRECTION], is_blocked, 'tof')
            print(f"   -> [{path_name}] Quick ToF check: Path is {'BLOCKED' if is_blocked else 'CLEAR'}.")
            visualizer.update_plot(occupancy_map, CURRENT_POSITION)

            # หยุดทันทีหากพบอุปสรรค
            if is_blocked:
                print(f"   -> 🔥 [{path_name}] IMMEDIATE STOP. Real-time sensor detected an obstacle. Aborting path.")
                break

            axis_to_monitor = 'x' if ROBOT_FACE % 2 != 0 else 'y'
            movement_controller.move_forward_one_grid(axis=axis_to_monitor, attitude_handler=attitude_handler)
            
            movement_controller.center_in_node_with_tof(scanner, attitude_handler)

            CURRENT_POSITION = (next_r, next_c)
            # บันทึกตำแหน่งใหม่ใน path execution
            log_position_timestamp(CURRENT_POSITION, CURRENT_DIRECTION, f"{path_name}_moved")
    
    # เมื่อถึงโหนดก่อนสุดท้ายแล้ว ให้เริ่มระบบปกติ (เช็ค detect, scan, etc.)
    if len(path) >= 2:
        current_r, current_c = path[-2]  # โหนดก่อนสุดท้าย (ที่หยุดอยู่)
        target_r, target_c = path[-1]    # โหนดสุดท้าย (ที่ยังไม่สำรวจ)
        
        print(f"🎯 [{path_name}] Reached pre-target node ({current_r},{current_c}). Starting normal exploration system...")
        print(f"🎯 [{path_name}] Next target: unvisited node ({target_r},{target_c})")
        
        # อัปเดตตำแหน่งปัจจุบัน
        CURRENT_POSITION = (current_r, current_c)
        log_position_timestamp(CURRENT_POSITION, CURRENT_DIRECTION, f"{path_name}_reached_pre_target")
        
        # หันหน้าไปยังทิศทางที่จะไปโหนดสุดท้าย
        dr, dc = target_r - current_r, target_c - current_c
        target_direction = dir_vectors_map[(dr, dc)]
        
        print(f"🔄 [{path_name}] Turning to face unvisited node ({target_r},{target_c})...")
        movement_controller.rotate_to_direction(target_direction, attitude_handler, scanner)
        
        # เช็ค detection ก่อนเดินไปโหนดสุดท้าย
        print("🔍 Performing object detection before moving to unvisited node...")
        try:
            start_detection_mode()
            time.sleep(1.0)
            save_detected_objects_to_map(occupancy_map)
            
            # Check for targets and start PID tracking if found
            if check_for_targets():
                print("🎯 Target detected! Starting PID tracking and firing...")
                # Use existing ROI state (don't create new one)
                
                # PID tracking loop (detection mode stays active)
                tracking_start_time = time.time()
                while is_tracking_mode and (time.time() - tracking_start_time) < 30:  # 30 second timeout
                    if pid_tracking_and_firing(manager, roi_state):
                        time.sleep(0.01)  # Small delay for PID loop
                    else:
                        break
                
                print("🎯 PID tracking completed, resuming normal exploration...")
            else:
                # No targets found, stop detection mode normally
                stop_detection_mode()
                print("🔍 Object detection completed before final move")
        except Exception as e:
            print(f"⚠️ Object detection error: {e}")
            stop_detection_mode()
            print("🔍 Object detection failed, continuing without detection...")
        
        # เช็คเส้นทางด้วย ToF ก่อนเดินไปโหนดสุดท้าย
        print(f"   -> [{path_name}] Final confirmation to unvisited node ({target_r},{target_c}) with ToF...")
        scanner.gimbal.moveto(pitch=0, yaw=0, yaw_speed=SPEED_ROTATE).wait_for_completed()
        time.sleep(0.2)
        
        is_blocked = scanner.get_front_tof_cm() < scanner.tof_wall_threshold_cm
        occupancy_map.update_wall(current_r, current_c, dir_map_abs_char[CURRENT_DIRECTION], is_blocked, 'tof')
        print(f"   -> [{path_name}] Final ToF check: Path to unvisited node is {'BLOCKED' if is_blocked else 'CLEAR'}.")
        
        if is_blocked:
            print(f"   -> 🔥 [{path_name}] FINAL STOP. Real-time sensor detected obstacle to unvisited node.")
            # ตรวจสอบว่าโหนดปลายทางเป็นทางตันหรือไม่ - ปรับปรุงจาก debug_img_3-10.py
            print(f"   -> 🔍 Analyzing if target node ({target_r},{target_c}) is a dead end...")
            if is_dead_end(occupancy_map, (target_r, target_c), set()):
                print(f"   -> 🚫 Target node ({target_r},{target_c}) is confirmed as dead end. Marking as explored.")
                mark_cell_as_dead_end(occupancy_map, (target_r, target_c))
            else:
                print(f"   -> ⚠️ Target node ({target_r},{target_c}) may still be accessible from other directions.")
            return
        
        # เดินไปโหนดสุดท้าย (ที่ยังไม่สำรวจ)
        print(f"🚀 [{path_name}] Moving to unvisited node ({target_r},{target_c})...")
        axis_to_monitor = 'x' if ROBOT_FACE % 2 != 0 else 'y'
        movement_controller.move_forward_one_grid(axis=axis_to_monitor, attitude_handler=attitude_handler)
        
        movement_controller.center_in_node_with_tof(scanner, attitude_handler)
        
        CURRENT_POSITION = (target_r, target_c)
        log_position_timestamp(CURRENT_POSITION, CURRENT_DIRECTION, f"{path_name}_reached_unvisited")
        print(f"✅ Successfully reached unvisited node ({target_r},{target_c})")
        print("✅ Backtrack complete. Resuming normal exploration...")
        visualizer.update_plot(occupancy_map, CURRENT_POSITION, path)

    print(f"✅ {path_name} complete.")

def perform_side_alignment_and_mapping(movement_controller, scanner, attitude_handler, occupancy_map, visualizer):
    print("\n--- Stage: Wall Detection & Side Alignment ---")
    r, c = CURRENT_POSITION
    dir_map_abs_char = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}
    
    side_walls_present = scanner.get_sensor_readings()

    left_dir_abs = (CURRENT_DIRECTION - 1 + 4) % 4
    occupancy_map.update_wall(r, c, dir_map_abs_char[left_dir_abs], side_walls_present['left'], 'sharp')
    visualizer.update_plot(occupancy_map, CURRENT_POSITION)
    if side_walls_present['left']:
        movement_controller.adjust_position_to_wall(
            scanner.sensor_adaptor, attitude_handler, "Left", 
            scanner.side_sensors["Left"], LEFT_TARGET_CM, direction_multiplier=1
        )
    
    right_dir_abs = (CURRENT_DIRECTION + 1) % 4
    occupancy_map.update_wall(r, c, dir_map_abs_char[right_dir_abs], side_walls_present['right'], 'sharp')
    visualizer.update_plot(occupancy_map, CURRENT_POSITION)
    if side_walls_present['right']:
        movement_controller.adjust_position_to_wall(
            scanner.sensor_adaptor, attitude_handler, "Right", 
            scanner.side_sensors["Right"], RIGHT_TARGET_CM, direction_multiplier=-1
        )

    if not side_walls_present['left'] and not side_walls_present['right']:
        print("\n⚠️  WARNING: No side walls detected. Skipping alignment.")
    
    attitude_handler.correct_yaw_to_target(movement_controller.chassis, get_compensated_target_yaw()) # MODIFIED
    time.sleep(0.05)


def explore_with_ogm(scanner, movement_controller, attitude_handler, occupancy_map, visualizer, max_steps=40):
    global CURRENT_POSITION, CURRENT_DIRECTION, IMU_DRIFT_COMPENSATION_DEG
    visited_cells = set()
    backtrack_attempts = {}  # นับจำนวนครั้งที่พยายาม backtrack ไปยังโหนดเดียวกัน
    
    # บันทึกตำแหน่งเริ่มต้น
    log_position_timestamp(CURRENT_POSITION, CURRENT_DIRECTION, "exploration_start")
    
    for step in range(max_steps):
        if not camera_is_healthy():
            wait_for_camera_recovery(pause_label=f"Step {step+1}")

        r, c = CURRENT_POSITION
        print(f"\n--- Step {step + 1} at {CURRENT_POSITION}, Facing: {['N', 'E', 'S', 'W'][CURRENT_DIRECTION]} ---")
        
        # บันทึกตำแหน่งในแต่ละ step
        log_position_timestamp(CURRENT_POSITION, CURRENT_DIRECTION, f"step_{step + 1}")
        
        print("   -> Resetting Yaw to ensure perfect alignment before new step...")
        attitude_handler.correct_yaw_to_target(movement_controller.chassis, get_compensated_target_yaw()) # MODIFIED
        
        perform_side_alignment_and_mapping(movement_controller, scanner, attitude_handler, occupancy_map, visualizer)

        print("--- Performing Scan for Mapping (Front ToF Only) ---")
        is_front_occupied = scanner.get_front_tof_cm() < scanner.tof_wall_threshold_cm
        dir_map_abs_char = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}
        occupancy_map.update_wall(r, c, dir_map_abs_char[CURRENT_DIRECTION], is_front_occupied, 'tof')
        
        occupancy_map.update_node(r, c, False, 'tof')
        visited_cells.add((r, c))
        visualizer.update_plot(occupancy_map, CURRENT_POSITION)
        
        # --- NEW: Update IMU Drift Compensation ---
        nodes_visited = len(visited_cells)
        if nodes_visited >= IMU_COMPENSATION_START_NODE_COUNT:
            # Calculate how many intervals have passed
            compensation_intervals = nodes_visited // IMU_COMPENSATION_NODE_INTERVAL
            new_compensation = compensation_intervals * IMU_COMPENSATION_DEG_PER_INTERVAL
            if new_compensation != IMU_DRIFT_COMPENSATION_DEG:
                IMU_DRIFT_COMPENSATION_DEG = new_compensation
                print(f"🔩 IMU Drift Compensation Updated: Visited {nodes_visited} nodes. New offset is {IMU_DRIFT_COMPENSATION_DEG:.1f}°")
        # --- END OF NEW CODE ---
        
        priority_dirs = [(CURRENT_DIRECTION + 1) % 4, CURRENT_DIRECTION, (CURRENT_DIRECTION - 1 + 4) % 4]
        moved = False
        dir_vectors = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        for target_dir in priority_dirs:
            target_r, target_c = r + dir_vectors[target_dir][0], c + dir_vectors[target_dir][1]
            
            if occupancy_map.is_path_clear(r, c, target_r, target_c) and (target_r, target_c) not in visited_cells:
                print(f"Path to {['N','E','S','W'][target_dir]} at ({target_r},{target_c}) seems clear. Attempting move.")
                movement_controller.rotate_to_direction(target_dir, attitude_handler, scanner)
                
                # <<< NEW CODE ADDED >>>
                # Ensure the gimbal is facing forward before checking the path and moving.
                print("    Ensuring gimbal is centered before ToF confirmation...")
                scanner.gimbal.moveto(pitch=0, yaw=0, yaw_speed=SPEED_ROTATE).wait_for_completed();
                time.sleep(0.2)  # ลดเวลารอ
                # <<< END OF NEW CODE >>>
                
                print("    Confirming path forward with ToF...")
                is_blocked = scanner.get_front_tof_cm() < scanner.tof_wall_threshold_cm
                
                occupancy_map.update_wall(r, c, dir_map_abs_char[CURRENT_DIRECTION], is_blocked, 'tof')
                print(f"    ToF confirmation: Wall belief updated. Path is {'BLOCKED' if is_blocked else 'CLEAR'}.")
                visualizer.update_plot(occupancy_map, CURRENT_POSITION)
                
                # <<< NEW: Double-check with ToF after rotation >>>
                if is_blocked:
                    print(f"    🚫 Wall detected! Turning back to original direction and recalculating path...")
                    movement_controller.rotate_to_direction(CURRENT_DIRECTION, attitude_handler, scanner)
                    print(f"    ✅ Turned back to {['N','E','S','W'][CURRENT_DIRECTION]}. Re-evaluating available paths...")
                    continue  # Skip this direction and try next one
                # <<< END OF NEW CODE >>>
                
                if occupancy_map.is_path_clear(r, c, target_r, target_c):
                    # --- OBJECT DETECTION AFTER TURNING TO NEW DIRECTION ---
                    print("🔍 Performing object detection after turning to new direction...")
                    try:
                        start_detection_mode()
                        time.sleep(1.0)
                        save_detected_objects_to_map(occupancy_map)
                        
                        # Check for targets and start PID tracking if found
                        if check_for_targets():
                            print("🎯 Target detected! Starting PID tracking and firing...")
                            # Use existing ROI state (don't create new one)
                            
                            # PID tracking loop (detection mode stays active)
                            tracking_start_time = time.time()
                            while is_tracking_mode and (time.time() - tracking_start_time) < 30:  # 30 second timeout
                                if pid_tracking_and_firing(manager, roi_state):
                                    time.sleep(0.01)  # Small delay for PID loop
                                else:
                                    break
                            
                            print("🎯 PID tracking completed, resuming normal exploration...")
                        else:
                            # No targets found, stop detection mode normally
                            stop_detection_mode()
                            print("🔍 Object detection completed after turn")
                    except Exception as e:
                        print(f"⚠️ Object detection error: {e}")
                        stop_detection_mode()
                        print("🔍 Object detection failed, continuing without detection...")
                    
                    axis_to_monitor = 'x' if ROBOT_FACE % 2 != 0 else 'y'
                    movement_controller.move_forward_one_grid(axis=axis_to_monitor, attitude_handler=attitude_handler)
                
                    movement_controller.center_in_node_with_tof(scanner, attitude_handler)
                
                    CURRENT_POSITION = (target_r, target_c)
                    # บันทึกตำแหน่งใหม่หลังจากเคลื่อนที่
                    log_position_timestamp(CURRENT_POSITION, CURRENT_DIRECTION, "moved_to_new_node")
                    moved = True
                    break
                else:
                    print(f"    Confirmation failed. Path to {['N','E','S','W'][target_dir]} is blocked. Re-evaluating.")
        
        if not moved:
            print("No immediate unvisited path. Initiating backtrack...")
            
            # ตรวจสอบว่าโหนดปัจจุบันเป็นทางตันหรือไม่ - ปรับปรุงจาก debug_img_3-10.py
            print(f"🔍 Analyzing if current position {CURRENT_POSITION} is a dead end...")
            if is_dead_end(occupancy_map, CURRENT_POSITION, visited_cells):
                print(f"🚫 Dead end confirmed at {CURRENT_POSITION}. Marking as fully explored.")
                # ทำเครื่องหมายว่าโหนดนี้เป็นทางตัน
                mark_cell_as_dead_end(occupancy_map, CURRENT_POSITION)
            else:
                print(f"⚠️ Position {CURRENT_POSITION} may still have accessible paths.")
            
            print("🔍 Searching for accessible unvisited nodes...")
            backtrack_path = find_nearest_unvisited_path(occupancy_map, CURRENT_POSITION, visited_cells)
            
            if backtrack_path and len(backtrack_path) > 1:
                target_node = backtrack_path[-1]
                print(f"🎯 Found backtrack target: {target_node}")
                
                # ตรวจสอบจำนวนครั้งที่พยายาม backtrack ไปยังโหนดเดียวกัน
                if target_node in backtrack_attempts:
                    backtrack_attempts[target_node] += 1
                    print(f"🔄 Attempt #{backtrack_attempts[target_node]} to reach {target_node}")
                    if backtrack_attempts[target_node] >= 3:  # ถ้าพยายามมากกว่า 3 ครั้ง
                        print(f"🔄 Too many attempts to reach {target_node}. Marking as dead end.")
                        mark_cell_as_dead_end(occupancy_map, target_node)
                        # ลบโหนดนี้ออกจาก backtrack path และลองหาใหม่
                        print("🔍 Searching for alternative path...")
                        backtrack_path = find_nearest_unvisited_path(occupancy_map, CURRENT_POSITION, visited_cells)
                        if not backtrack_path or len(backtrack_path) <= 1:
                            print("🎉 EXPLORATION COMPLETE! No reachable unvisited cells remain.")
                            break
                else:
                    backtrack_attempts[target_node] = 1
                    print(f"🆕 First attempt to reach {target_node}")
                
                execute_path(backtrack_path, movement_controller, attitude_handler, scanner, visualizer, occupancy_map)
                print("Backtrack to new area complete. Resuming exploration.")
                continue
            else:
                print("🎉 EXPLORATION COMPLETE! No reachable unvisited cells remain.")
                break
        # end of per-step block
    
    print("\n🎉 === EXPLORATION PHASE FINISHED ===")

# =============================================================================
# ===== MAIN EXECUTION BLOCK ==================================================
# =============================================================================
if __name__ == '__main__':
    ep_robot = None
    occupancy_map = None
    attitude_handler = AttitudeHandler()
    movement_controller = None
    scanner = None
    ep_chassis = None
    
    # --- NEW: Resume Logic ---
    if check_for_resume_data():
        print("\n🔄 Found previous session data!")
        user_input = input("Do you want to resume from previous session? (y/n): ").lower().strip()
        
        if user_input == 'y' or user_input == 'yes':
            print("🔄 Resuming from previous session...")
            if load_resume_data():
                occupancy_map = create_occupancy_map_from_json()
                if occupancy_map is None:
                    print("❌ Failed to load occupancy map. Starting fresh session.")
                    occupancy_map = OccupancyGridMap(width=GRID, height=GRID)
                    RESUME_MODE = False
            else:
                print("❌ Failed to load resume data. Starting fresh session.")
                occupancy_map = OccupancyGridMap(width=GRID, height=GRID)
                RESUME_MODE = False
        else:
            print("🆕 Starting fresh session...")
            occupancy_map = OccupancyGridMap(width=GRID, height=GRID)
            RESUME_MODE = False
    else:
        print("🆕 No previous session found. Starting fresh session...")
        occupancy_map = OccupancyGridMap(width=GRID, height=GRID)
        RESUME_MODE = False
    
    # --- INTEGRATED OBJECT DETECTION SYSTEM ---
    print("🎯 Initializing Integrated Object Detection System...")
    
    # Initialize object detection components
    tracker = ObjectTracker(use_gpu=USE_GPU)
    manager = RMConnection()
    
    # ROI state (dynamic Y)
    roi_state = {"x": ROI_X0, "y": ROI_Y0, "w": ROI_W0, "h": ROI_H0}
    
    # Start object detection threads
    reconn = threading.Thread(target=reconnector_thread, args=(manager,), daemon=True)
    reconn.start()
    
    def is_detecting(): return is_detecting_flag["v"]
    
    cap_t = threading.Thread(target=capture_thread_func, args=(manager, frame_queue), daemon=True)
    proc_t = threading.Thread(target=processing_thread_func,
                              args=(tracker, frame_queue, TARGET_SHAPE, TARGET_COLOR, roi_state, is_detecting),
                              daemon=True)
    
    cap_t.start()
    proc_t.start()
    
    print("✅ Object Detection System initialized (Camera ON, Detection OFF)")
    
    # Wait for camera to be ready - CRITICAL: Don't start exploration until camera is ready
    print("⏳ Waiting for camera to be ready...")
    camera_ready = False
    max_retries = 5
    
    for retry in range(max_retries):
        time.sleep(3.0)
        if manager.connected.is_set():
            print("✅ Camera is ready!")
            camera_ready = True
            break
        else:
            print(f"⚠️ Camera not ready (attempt {retry + 1}/{max_retries}), retrying connection...")
            manager.connect()
            time.sleep(2.0)
    
    # Final check - CRITICAL: Don't proceed without camera
    if not camera_ready:
        print("❌ Camera connection failed after all retries. Please check camera connection and restart.")
        print("🛑 Stopping program to prevent exploration without camera...")
        import sys
        sys.exit(1)
    
    print("🎯 Camera confirmed ready - Starting exploration...")
    
    # Start camera display thread (optional via SHOW_WINDOW flag)
    SHOW_WINDOW = True  # set False to disable display and reduce load on camera
    def camera_display_thread():
        print("📹 Camera display thread started")
        display_frame = None
        frame_count = 0
        last_frame_time = time.time()

        try:
            while not stop_event.is_set():
                try:
                    # Get frame with shorter timeout to prevent blocking
                    display_frame = frame_queue.get(timeout=0.5)
                    frame_count += 1
                    last_frame_time = time.time()
                    
                    # Skip processing if frame is None or corrupted
                    if display_frame is None or display_frame.size == 0:
                        continue
                        
                except queue.Empty:
                    # ลบการเช็ค 5-second timeout เพราะ capture thread จัดการเองแล้ว
                    time.sleep(0.1)
                    continue
                except Exception as e:
                    print(f"⚠️ Error getting frame: {e}")
                    time.sleep(0.1)
                    continue

                # วาด ROI และเส้นแบ่งซ้าย/กลาง/ขวา
                ROI_X, ROI_Y, ROI_W, ROI_H = roi_state["x"], roi_state["y"], roi_state["w"], roi_state["h"]
                cv2.rectangle(display_frame, (ROI_X, ROI_Y), (ROI_X+ROI_W, ROI_Y+ROI_H), (255,0,0), 2)

                if is_detecting():
                    cv2.putText(display_frame, "MODE: DETECTING", (20,40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    with output_lock:
                        details = processed_output["details"]

                    d1_abs = ROI_X + int(ROI_W*0.33)
                    d2_abs = ROI_X + int(ROI_W*0.66)
                    cv2.line(display_frame, (d1_abs, ROI_Y), (d1_abs, ROI_Y+ROI_H), (255,255,0), 1)
                    cv2.line(display_frame, (d2_abs, ROI_Y), (d2_abs, ROI_Y+ROI_H), (255,255,0), 1)

                    # กล่องวัตถุ: สี/ความหนาแบบเดิม
                    for det in details:
                        x,y,w,h = det['box']
                        abs_x, abs_y = x + ROI_X, y + ROI_Y
                        if det['is_target']:
                            box_color = (0,0,255)
                        elif det['shape'] == 'Uncertain':
                            box_color = (0,255,255)
                        else:
                            box_color = (0,255,0)
                        thickness = 4 if det['is_target'] else 2
                        cv2.rectangle(display_frame, (abs_x,abs_y), (abs_x+w, abs_y+h), box_color, thickness)
                        cv2.putText(display_frame, str(det['id']), (abs_x+5, abs_y+25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

                    # รายการซ้ายบน
                    if details:
                        y_pos = 70
                        for obj in details:
                            target_str = " (TARGET!)" if obj['is_target'] else ""
                            line = f"ID {obj['id']}: {obj['color']} {obj['shape']}{target_str}"
                            # shadow
                            cv2.putText(display_frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 4)
                            cv2.putText(display_frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                            y_pos += 25

                    # วาด crosshair กลางภาพ และเส้น bias +3°
                    cy_bias = int(FRAME_H/2 - PITCH_BIAS_PIX)
                    cv2.line(display_frame, (FRAME_W//2 - 20, FRAME_H//2), (FRAME_W//2 + 20, FRAME_H//2), (255,255,255), 1)
                    cv2.line(display_frame, (FRAME_W//2, FRAME_H//2 - 20), (FRAME_W//2, FRAME_H//2 + 20), (255,255,255), 1)
                    cv2.line(display_frame, (0, cy_bias), (FRAME_W, cy_bias), (0, 128, 255), 1)  # เส้นเป้า +3°

                else:
                    cv2.putText(display_frame, "MODE: VIEWING", (20,40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

                # สถานะ SDK
                st = "CONNECTED" if manager.connected.is_set() else "RECONNECTING..."
                cv2.putText(display_frame, f"SDK: {st}", (20, 70 if not is_detecting() else 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                # Display frame with error handling
                try:
                    cv2.imshow("Robomaster Real-time Scan + PID Track (+3°)", display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("🛑 Quit key pressed, stopping display...")
                        break
                    elif key == ord('s'):
                        is_detecting_flag["v"] = not is_detecting_flag["v"]
                        print(f"🔍 Detection {'ON' if is_detecting_flag['v'] else 'OFF'}")
                    elif key == ord('r'):
                        print("🔄 Manual reconnect requested")
                        manager.drop_and_reconnect()
                except Exception as e:
                    print(f"⚠️ Error displaying frame: {e}")
                    time.sleep(0.1)
                    try:
                        while True: frame_queue.get_nowait()
                    except queue.Empty:
                        pass

        except Exception as e:
            print(f"❌ Camera display error: {e}")
        finally:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        print("🛑 Camera display thread stopped")
    
    # Start camera display thread
    display_t = None
    if SHOW_WINDOW:
        display_t = threading.Thread(target=camera_display_thread, daemon=True)
        display_t.start()
    
    try:
        visualizer = RealTimeVisualizer(grid_size=GRID, target_dest=TARGET_DESTINATION)
        print("🤖 Connecting to robot...")
        ep_robot = robot.Robot()
        try:
            ep_robot.initialize(conn_type="ap")
            time.sleep(2.0)  # เพิ่มเวลารอให้ robot initialize เสร็จ
        except Exception as e:
            print(f"⚠️ Robot connection error: {e}")
            print("🔄 Retrying robot connection...")
            time.sleep(1.0)
            ep_robot.initialize(conn_type="ap")
            time.sleep(2.0)
        ep_chassis, ep_gimbal = ep_robot.chassis, ep_robot.gimbal
        ep_tof_sensor, ep_sensor_adaptor = ep_robot.sensor, ep_robot.sensor_adaptor
        
        print(" GIMBAL: Centering gimbal...")
        try:
            ep_gimbal.moveto(pitch=0, yaw=0, yaw_speed=SPEED_ROTATE).wait_for_completed()
            time.sleep(0.5)  # Wait for gimbal to center
        except Exception as e:
            print(f"⚠️ Gimbal centering error: {e}")
            print("🔄 Continuing without gimbal centering...")
        
        scanner = EnvironmentScanner(ep_sensor_adaptor, ep_tof_sensor, ep_gimbal, ep_chassis)
        movement_controller = MovementController(ep_chassis)
        attitude_handler.start_monitoring(ep_chassis)
        
        if RESUME_MODE:
            print("🔄 Resuming exploration from previous position...")
            log_position_timestamp(CURRENT_POSITION, CURRENT_DIRECTION, "resume_session")
        else:
            print("🆕 Starting new exploration...")
            log_position_timestamp(CURRENT_POSITION, CURRENT_DIRECTION, "new_session_start")
        
        # --- INTEGRATED EXPLORATION WITH OBJECT DETECTION ---
        print("🚀 Starting Integrated Exploration with Object Detection...")
        
        visited_cells = set()
        backtrack_attempts = {}  # นับจำนวนครั้งที่พยายาม backtrack ไปยังโหนดเดียวกัน
        
        for step in range(40):  # max_steps
            try:
                r, c = CURRENT_POSITION
                print(f"\n--- Step {step + 1} at {CURRENT_POSITION}, Facing: {['N', 'E', 'S', 'W'][CURRENT_DIRECTION]} ---")
                
                log_position_timestamp(CURRENT_POSITION, CURRENT_DIRECTION, f"step_{step + 1}")
                
                print("   -> Resetting Yaw to ensure perfect alignment before new step...")
                attitude_handler.correct_yaw_to_target(ep_chassis, get_compensated_target_yaw())
                
                # Perform side alignment and mapping
                perform_side_alignment_and_mapping(movement_controller, scanner, attitude_handler, occupancy_map, visualizer)
                
                # --- AUTOMATIC OBJECT DETECTION AFTER ALIGNMENT ---
                # ตรวจสอบว่าข้างหน้าเป็นกำแพงหรือไม่ก่อนทำ object detection
                print("--- Performing Scan for Mapping (Front ToF Only) ---")
                is_front_occupied = scanner.get_front_tof_cm() < scanner.tof_wall_threshold_cm
                dir_map_abs_char = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}
                occupancy_map.update_wall(r, c, dir_map_abs_char[CURRENT_DIRECTION], is_front_occupied, 'tof')
                
                # ถ้ากล้องไม่พร้อม ให้หยุดหุ่น ณ จุดนี้และรอให้กล้องกลับมาก่อนจึงเดินต่อ
                if not camera_is_healthy():
                    print("🛑 Camera unhealthy → pausing exploration and locking chassis until camera recovers...")
                    try:
                        movement_controller.chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
                    except Exception:
                        pass
                    # wait loop with backoff
                    wait_start = time.time()
                    while not camera_is_healthy():
                        if time.time() - wait_start > 30.0:
                            print("⚠️ Camera recovery timeout (30s). Forcing reconnect and continuing wait...")
                            manager.drop_and_reconnect()
                            wait_start = time.time()
                        time.sleep(0.2)
                    print("✅ Camera recovered. Resuming exploration...")
                
                if is_front_occupied:
                    print("🚫 Front wall detected - Skipping object detection until robot turns to new direction")
                    print("🔍 Object detection will be performed after robot turns to clear path")
                else:
                    print("🔍 Object detection will be performed after robot turns to clear path")
                
                # Check detection timer
                check_detection_timer()
                
                occupancy_map.update_node(r, c, False, 'tof')
                visited_cells.add((r, c))
                
                # อัพเดทแมพทุก 3 steps เพื่อลดการใช้ thread
                if step % 3 == 0:
                    visualizer.update_plot(occupancy_map, CURRENT_POSITION)
                
                # Update IMU Drift Compensation
                nodes_visited = len(visited_cells)
                if nodes_visited >= IMU_COMPENSATION_START_NODE_COUNT:
                    compensation_intervals = nodes_visited // IMU_COMPENSATION_NODE_INTERVAL
                    new_compensation = compensation_intervals * IMU_COMPENSATION_DEG_PER_INTERVAL
                    if new_compensation != IMU_DRIFT_COMPENSATION_DEG:
                        IMU_DRIFT_COMPENSATION_DEG = new_compensation
                        print(f"🔩 IMU Drift Compensation Updated: Visited {nodes_visited} nodes. New offset is {IMU_DRIFT_COMPENSATION_DEG:.1f}°")
                
                # Continue with normal exploration logic
                priority_dirs = [(CURRENT_DIRECTION + 1) % 4, CURRENT_DIRECTION, (CURRENT_DIRECTION - 1 + 4) % 4]
                moved = False
                dir_vectors = [(-1, 0), (0, 1), (1, 0), (0, -1)]
                
                for target_dir in priority_dirs:
                    target_r, target_c = r + dir_vectors[target_dir][0], c + dir_vectors[target_dir][1]
                    
                    if occupancy_map.is_path_clear(r, c, target_r, target_c) and (target_r, target_c) not in visited_cells:
                        print(f"Path to {['N','E','S','W'][target_dir]} at ({target_r},{target_c}) seems clear. Attempting move.")
                        movement_controller.rotate_to_direction(target_dir, attitude_handler, scanner)
                        
                        print("    Ensuring gimbal is centered before ToF confirmation...")
                        t_start = time.time()
                        scanner.gimbal.moveto(pitch=0, yaw=0, yaw_speed=SPEED_ROTATE).wait_for_completed()
                        t_gimbal = time.time() - t_start
                        if t_gimbal > 2.0:
                            print(f"    ⚠️ Gimbal center took {t_gimbal:.2f}s (unusually long!)")
                        time.sleep(0.2)  # ลดเวลารอ
                        
                        print("    Confirming path forward with ToF...")
                        t_start = time.time()
                        is_blocked = scanner.get_front_tof_cm() < scanner.tof_wall_threshold_cm
                        t_tof = time.time() - t_start
                        if t_tof > 1.0:
                            print(f"    ⚠️ ToF read took {t_tof:.2f}s (unusually long!)")
                        
                        occupancy_map.update_wall(r, c, dir_map_abs_char[CURRENT_DIRECTION], is_blocked, 'tof')
                        print(f"    ToF confirmation: Wall belief updated. Path is {'BLOCKED' if is_blocked else 'CLEAR'}.")
                        visualizer.update_plot(occupancy_map, CURRENT_POSITION)
                        
                        # <<< NEW: Double-check with ToF after rotation >>>
                        if is_blocked:
                            print(f"    🚫 Wall detected! Turning back to original direction and recalculating path...")
                            movement_controller.rotate_to_direction(CURRENT_DIRECTION, attitude_handler, scanner)
                            print(f"    ✅ Turned back to {['N','E','S','W'][CURRENT_DIRECTION]}. Re-evaluating available paths...")
                            continue  # Skip this direction and try next one
                        # <<< END OF NEW CODE >>>
                        
                        if occupancy_map.is_path_clear(r, c, target_r, target_c):
                            # --- OBJECT DETECTION AFTER TURNING TO NEW DIRECTION ---
                            print("🔍 Performing object detection after turning to new direction...")
                            try:
                                start_detection_mode()
                                time.sleep(1.0)
                                save_detected_objects_to_map(occupancy_map)
                                
                                # Check for targets and start PID tracking if found
                                if check_for_targets():
                                    print("🎯 Target detected! Starting PID tracking and firing...")
                                    # Use existing ROI state (don't create new one)
                                    
                                    # PID tracking loop (detection mode stays active)
                                    tracking_start_time = time.time()
                                    while is_tracking_mode and (time.time() - tracking_start_time) < 30:  # 30 second timeout
                                        if pid_tracking_and_firing(manager, roi_state):
                                            time.sleep(0.01)  # Small delay for PID loop
                                        else:
                                            break
                                    
                                    print("🎯 PID tracking completed, resuming normal exploration...")
                                else:
                                    # No targets found, stop detection mode normally
                                    stop_detection_mode()
                                    print("🔍 Object detection completed after turn")
                            except Exception as e:
                                print(f"⚠️ Object detection error: {e}")
                                stop_detection_mode()
                                print("🔍 Object detection failed, continuing without detection...")
                            
                            axis_to_monitor = 'x' if ROBOT_FACE % 2 != 0 else 'y'
                            t_start = time.time()
                            movement_controller.move_forward_one_grid(axis=axis_to_monitor, attitude_handler=attitude_handler)
                            t_move = time.time() - t_start
                            if t_move > 15.0:
                                print(f"    ⚠️ Movement took {t_move:.2f}s (unusually long!)")
                            
                            movement_controller.center_in_node_with_tof(scanner, attitude_handler)
                            
                            CURRENT_POSITION = (target_r, target_c)
                            log_position_timestamp(CURRENT_POSITION, CURRENT_DIRECTION, "moved_to_new_node")
                            
                            moved = True
                            break
                        else:
                            print(f"    Confirmation failed. Path to {['N','E','S','W'][target_dir]} is blocked. Re-evaluating.")
                
                if not moved:
                    print("No immediate unvisited path. Initiating backtrack...")
                    
                    # ตรวจสอบว่าโหนดปัจจุบันเป็นทางตันหรือไม่
                    print(f"🔍 Analyzing if current position {CURRENT_POSITION} is a dead end...")
                    if is_dead_end(occupancy_map, CURRENT_POSITION, visited_cells):
                        print(f"🚫 Dead end confirmed at {CURRENT_POSITION}. Marking as fully explored.")
                        mark_cell_as_dead_end(occupancy_map, CURRENT_POSITION)
                    else:
                        print(f"⚠️ Position {CURRENT_POSITION} may still have accessible paths.")
                    
                    print("🔍 Searching for accessible unvisited nodes...")
                    backtrack_path = find_nearest_unvisited_path(occupancy_map, CURRENT_POSITION, visited_cells)
                    
                    if backtrack_path and len(backtrack_path) > 1:
                        target_node = backtrack_path[-1]
                        print(f"🎯 Found backtrack target: {target_node}")
                        
                        # ตรวจสอบจำนวนครั้งที่พยายาม backtrack ไปยังโหนดเดียวกัน
                        if target_node in backtrack_attempts:
                            backtrack_attempts[target_node] += 1
                            print(f"🔄 Attempt #{backtrack_attempts[target_node]} to reach {target_node}")
                            if backtrack_attempts[target_node] >= 3:  # ถ้าพยายามมากกว่า 3 ครั้ง
                                print(f"🔄 Too many attempts to reach {target_node}. Marking as dead end.")
                                mark_cell_as_dead_end(occupancy_map, target_node)
                                # ลบโหนดนี้ออกจาก backtrack path และลองหาใหม่
                                print("🔍 Searching for alternative path...")
                                backtrack_path = find_nearest_unvisited_path(occupancy_map, CURRENT_POSITION, visited_cells)
                                if not backtrack_path or len(backtrack_path) <= 1:
                                    print("🎉 EXPLORATION COMPLETE! No reachable unvisited cells remain.")
                                    break
                        else:
                            backtrack_attempts[target_node] = 1
                            print(f"🆕 First attempt to reach {target_node}")
                        
                        execute_path(backtrack_path, movement_controller, attitude_handler, scanner, visualizer, occupancy_map)
                        print("Backtrack to new area complete. Resuming exploration.")
                        continue
                    else:
                        print("🎉 EXPLORATION COMPLETE! No reachable unvisited cells remain.")
                        break
            
            except Exception as e:
                print(f"\n❌ Error during step {step+1}: {e}")
                print("🛑 Stopping robot and waiting for camera recovery...")
                try:
                    movement_controller.chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
                except Exception:
                    pass
                wait_for_camera_recovery(pause_label=f"Step {step+1} Recovery")
                print("✅ Recovery complete. Resuming from current position...")
                continue
        
        
        print("\n🎉 === INTEGRATED EXPLORATION PHASE FINISHED ===")
        
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
        
    except KeyboardInterrupt: 
        print("\n⚠️ User interrupted exploration.")
        print("💾 Saving data before exit...")
        if occupancy_map:
            save_all_data(occupancy_map)
    except Exception as e: 
        print(f"\n⚌ An error occurred: {e}")
        traceback.print_exc()
        print("💾 Saving data before exit...")
        if occupancy_map:
            save_all_data(occupancy_map)
    finally:
        # Stop object detection threads
        stop_event.set()
        
        # Wait for threads to finish
        try:
            cap_t.join(timeout=2.0)
            proc_t.join(timeout=2.0)
            display_t.join(timeout=2.0)
        except Exception:
            pass
        
        # Close camera display
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        
        # บันทึกข้อมูลแม้จะมีการ interrupt (ใช้ฟังก์ชัน save_all_data)
        if occupancy_map:
            save_all_data(occupancy_map)
        
        # ทำความสะอาดการเชื่อมต่อ
        if ep_robot:
            print("🔌 Cleaning up and closing connection...")
            try:
                if scanner: scanner.cleanup()
                if attitude_handler and attitude_handler.is_monitoring: attitude_handler.stop_monitoring(ep_chassis)
                if movement_controller: movement_controller.cleanup()
                manager.close()
                ep_robot.close()
                print("🔌 Connection closed.")
            except Exception as cleanup_error:
                print(f"⚠️ Error during cleanup: {cleanup_error}")
        
        print("... You can close the plot window now ...")
        plt.ioff()
        plt.show()