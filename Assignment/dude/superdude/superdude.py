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

# --- NEW: Resume Function Variables ---
RESUME_MODE = False  # ตัวแปรบอกว่าเป็นโหมด resume หรือไม่
DATA_FOLDER = r"D:\downsyndrome\year2_1\Robot_Module_2-1\Assignment\dude\superdude"  # โฟลเดอร์สำหรับเก็บไฟล์ JSON

# เปลี่ยนจากพาธแบบเต็มเป็นพาธสัมพันธ์
MAP_FILE = os.path.join(DATA_FOLDER, "Mapping_Top.json")
POS_FILE = os.path.join(DATA_FOLDER, "Robot_Position_Timestamps.json")
OBJ_FILE = os.path.join(DATA_FOLDER, "Detected_Objects.json")

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
SHARP_STDEV_THRESHOLD = 0.3    # ค่าเบี่ยงเบนมาตรฐานสูงสุดที่ยอมรับได้ เพื่อกรองค่าที่แกว่ง

# --- ToF Centering Configuration (from dude_kum.py) ---
TOF_ADJUST_SPEED = 0.1             # ความเร็วในการขยับเข้า/ถอยออกเพื่อจัดตำแหน่งกลางโหนด
TOF_CALIBRATION_SLOPE = 0.0894     # ค่าจากการ Calibrate
TOF_CALIBRATION_Y_INTERCEPT = 3.8409 # ค่าจากการ Calibrate
TOF_TIME_CHECK = 0.15

GRID = 5

# --- Logical state for the grid map (from map_suay.py) ---
CURRENT_POSITION = (4,0)  # (แถว, คอลัมน์) here
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

        with open(MAP_FILE, "w") as f:
            json.dump(final_map_data, f, indent=2)
        print(f"✅ Final Hybrid Belief Map (with objects) saved to {MAP_FILE}")
        
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
        
        with open(POS_FILE, "w") as f:
            json.dump(timestamp_data, f, indent=2)
        print(f"✅ Robot position timestamps saved to {POS_FILE}")
        
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
        
        with open(OBJ_FILE, "w") as f:
            json.dump(objects_data, f, indent=2)
        print(f"✅ Detected objects saved to {OBJ_FILE} (Total: {len(all_detected_objects)} objects)")
        
        return True
    except Exception as save_error:
        print(f"❌ Error saving data: {save_error}")
        traceback.print_exc()
        return False

# =============================================================================
# ===== NEW: FAST RESUME SYSTEM ===============================================
# =============================================================================

def check_for_resume_data():
    """ตรวจสอบว่ามีไฟล์ JSON สำหรับ resume หรือไม่"""
    return os.path.exists(MAP_FILE) and os.path.exists(POS_FILE)

def load_resume_data():
    """โหลดข้อมูลจากไฟล์ JSON เพื่อ resume การทำงาน"""
    global CURRENT_POSITION, CURRENT_DIRECTION, CURRENT_TARGET_YAW, ROBOT_FACE, IMU_DRIFT_COMPENSATION_DEG, POSITION_LOG, RESUME_MODE
    
    try:
        print("🔄 Loading resume data...")
        
        # โหลดข้อมูล timestamp
        with open(POS_FILE, "r", encoding="utf-8") as f:
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
    """สร้าง OccupancyGridMap จากไฟล์ JSON"""
    try:
        with open(MAP_FILE, "r", encoding="utf-8") as f:
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
            
            # โหลด objects
            if 'objects' in node_data and node_data['objects']:
                cell.objects = node_data['objects']
        
        print(f"✅ Occupancy map loaded from JSON ({width}x{height})")
        return occupancy_map
        
    except Exception as e:
        print(f"❌ Error loading occupancy map: {e}")
        return None

def fast_navigate_to_last_position(movement_controller, attitude_handler, scanner, occupancy_map, visualizer):
    """นำทางไปยังตำแหน่งล่าสุดอย่างรวดเร็วโดยใช้ข้อมูลจากแผนที่"""
    global CURRENT_POSITION, CURRENT_DIRECTION
    
    if not RESUME_MODE:
        return True
    
    print("\n🚀 FAST NAVIGATION: Moving to last known position...")
    
    # หาเส้นทางจากจุดเริ่มต้นไปยังตำแหน่งล่าสุด
    start_pos = (4, 0)  # จุดเริ่มต้นคงที่
    target_pos = CURRENT_POSITION
    
    if start_pos == target_pos:
        print("✅ Already at target position")
        return True
    
    path = find_path_bfs(occupancy_map, start_pos, target_pos)
    if not path:
        print("❌ No path found to last position")
        return False
    
    print(f"📍 Fast path: {path}")
    
    # เดินทางตามเส้นทางโดยไม่ต้องตรวจสอบกำแพงซ้ำ
    for i in range(len(path) - 1):
        current = path[i]
        next_pos = path[i + 1]
        
        # คำนวณทิศทางที่ต้องเดิน
        dr = next_pos[0] - current[0]
        dc = next_pos[1] - current[1]
        
        dir_vectors_map = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}
        target_direction = dir_vectors_map[(dr, dc)]
        
        # หมุนหุ่นไปยังทิศทางที่ต้องการ
        movement_controller.rotate_to_direction(target_direction, attitude_handler, scanner)
        
        # เดินหน้าไปยังโหนดถัดไป (ไม่ต้องตรวจสอบกำแพง)
        axis_to_monitor = 'x' if ROBOT_FACE % 2 != 0 else 'y'
        movement_controller.move_forward_one_grid(axis=axis_to_monitor, attitude_handler=attitude_handler)
        
        # จัดตำแหน่งกลางโหนด
        movement_controller.center_in_node_with_tof(scanner, attitude_handler)
        
        # อัพเดทตำแหน่งปัจจุบัน
        CURRENT_POSITION = next_pos
        
        # บันทึกตำแหน่ง
        log_position_timestamp(CURRENT_POSITION, CURRENT_DIRECTION, "fast_navigation")
        
        # อัพเดท visualization
        visualizer.update_plot(occupancy_map, CURRENT_POSITION, path)
        
        print(f"✅ Moved to {CURRENT_POSITION}")
    
    print("🎉 FAST NAVIGATION COMPLETE!")
    return True

# =============================================================================
# ===== MODIFIED HELPER FUNCTIONS =============================================
# =============================================================================

def log_position_timestamp(position, direction, action="arrived"):
    """บันทึก timestamp และตำแหน่งของหุ่นยนต์"""
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

# =============================================================================
# ===== MODIFIED MAIN EXECUTION BLOCK =========================================
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
                    # สำเร็จในการโหลดข้อมูล resume
                    RESUME_MODE = True
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
    
    # ... (ส่วนที่เหลือของโค้ดเดิม)

    try:
        visualizer = RealTimeVisualizer(grid_size=GRID, target_dest=TARGET_DESTINATION)
        
        # +++ NEW CODE: รอให้ manager เชื่อมต่อสำเร็จก่อน +++
        print("🤖 Waiting for robot connection from manager...")
        # รอให้ manager ทำการเชื่อมต่อให้สำเร็จก่อน (timeout 15 วินาที)
        if not manager.connected.wait(timeout=15.0):
            print("❌ CRITICAL: Robot connection failed to establish via manager. Exiting.")
            # ทำการ cleanup ก่อนออกจากโปรแกรม
            stop_event.set()
            if ep_robot: ep_robot.close()
            exit()

        print("✅ Robot connected via manager.")
        ep_robot = manager.get_robot() # <-- ดึง instance ของ robot มาจาก manager
        if ep_robot is None:
            print("❌ CRITICAL: Failed to get robot instance from manager. Exiting.")
            stop_event.set()
            exit()
        # +++ END OF NEW CODE +++

        # ไม่ต้อง initialize อีกแล้ว เพราะ manager ทำให้แล้ว
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
        
        # --- NEW: FAST NAVIGATION TO LAST POSITION ---
        if RESUME_MODE:
            print("🔄 Fast navigating to last position...")
            success = fast_navigate_to_last_position(movement_controller, attitude_handler, scanner, occupancy_map, visualizer)
            if success:
                print("✅ Successfully reached last position. Resuming exploration...")
                log_position_timestamp(CURRENT_POSITION, CURRENT_DIRECTION, "resume_session")
            else:
                print("❌ Fast navigation failed. Starting from initial position.")
                CURRENT_POSITION = (4, 0)
                CURRENT_DIRECTION = 1
                log_position_timestamp(CURRENT_POSITION, CURRENT_DIRECTION, "fresh_start_after_failed_resume")
        else:
            print("🆕 Starting new exploration...")
            log_position_timestamp(CURRENT_POSITION, CURRENT_DIRECTION, "new_session_start")
        
        # ... (ส่วนที่เหลือของโค้ดเดิมสำหรับการสำรวจ)
        
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