# -*-coding:utf-8-*-
"""
โปรแกรมสำหรับอ่านข้อมูล objects ที่ตรวจพบจาก JSON files
แล้วเดินไปยิง objects ทั้งหมดโดยหลบกำแพงที่เจอ
"""

import time
import robomaster
from robomaster import robot, camera as r_camera, blaster as r_blaster
import numpy as np
import math
import json
import traceback
import os
import cv2
import threading
from collections import deque
import heapq

# =============================================================================
# ===== CONFIGURATION & PARAMETERS ============================================
# =============================================================================

# --- Data Files Configuration ---
DATA_FOLDER = r"D:\downsyndrome\year2_1\Robot_Module_2-1\Assignment\dude\James_path\test_thread\walk2"
MAPPING_FILE = os.path.join(DATA_FOLDER, "Mapping_Top.json")
TIMESTAMP_FILE = os.path.join(DATA_FOLDER, "Robot_Position_Timestamps.json")
OBJECTS_FILE = os.path.join(DATA_FOLDER, "Detected_Objects.json")

# --- PID Target Tracking & Firing Configuration ---
TARGET_SHAPE = "Circle"  # Shape to track
TARGET_COLOR = "Red"     # Color to track
FIRE_SHOTS_COUNT = 5     # Number of shots to fire per target

# PID Parameters
PID_KP = -0.15
PID_KI = -0.005
PID_KD = -0.02
MAX_YAW_SPEED = 120
MAX_PITCH_SPEED = 100
PIX_ERR_DEADZONE = 8
LOCK_TOL_X = 12
LOCK_TOL_Y = 12

# Camera Configuration
FRAME_W, FRAME_H = 960, 540
VERTICAL_FOV_DEG = 54.0
PIXELS_PER_DEG_V = FRAME_H / VERTICAL_FOV_DEG
PITCH_BIAS_DEG = 2.5
PITCH_BIAS_PIX = +PITCH_BIAS_DEG * PIXELS_PER_DEG_V

# ROI Configuration
ROI_Y0, ROI_H0, ROI_X0, ROI_W0 = 264, 270, 10, 911

# Movement Configuration
SPEED_ROTATE = 480
GRID_SIZE = 5

# --- Global Variables ---
CURRENT_POSITION = (4, 0)  # (row, col)
CURRENT_DIRECTION = 1  # 0:North, 1:East, 2:South, 3:West
CURRENT_TARGET_YAW = 0.0

# Object detection tracking
output_lock = threading.Lock()
processed_output = {"count": 0, "details": []}
is_tracking_mode = False
fired_targets = set()
current_target_id = None
shots_fired = 0

# =============================================================================
# ===== DATA LOADING FUNCTIONS ================================================
# =============================================================================

def load_json_data():
    """โหลดข้อมูลจาก 3 JSON files"""
    print("📂 Loading JSON data files...")
    
    try:
        # Load mapping data
        with open(MAPPING_FILE, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)
        print(f"✅ Loaded mapping data: {len(mapping_data.get('nodes', []))} nodes")
        
        # Load timestamp data
        with open(TIMESTAMP_FILE, "r", encoding="utf-8") as f:
            timestamp_data = json.load(f)
        print(f"✅ Loaded timestamp data: {len(timestamp_data.get('position_log', []))} positions")
        
        # Load objects data
        with open(OBJECTS_FILE, "r", encoding="utf-8") as f:
            objects_data = json.load(f)
        print(f"✅ Loaded objects data: {objects_data['session_info']['total_objects_detected']} objects")
        
        return mapping_data, timestamp_data, objects_data
    
    except Exception as e:
        print(f"❌ Error loading JSON files: {e}")
        traceback.print_exc()
        return None, None, None

def extract_objects_and_positions(objects_data):
    """ดึงข้อมูล objects และตำแหน่งที่ต้องไปยิง"""
    objects_to_fire = []
    
    detected_objects = objects_data.get('detected_objects', [])
    
    for obj in detected_objects:
        if obj.get('is_target', False):  # เฉพาะ target ที่ต้องยิง
            cell_pos = obj.get('cell_position', {})
            row, col = cell_pos.get('row'), cell_pos.get('col')
            
            if row is not None and col is not None:
                # ดึงข้อมูลโหนดที่ตรวจพบ object
                detected_from = obj.get('detected_from_node', [])
                
                obj_info = {
                    'color': obj.get('color', 'unknown'),
                    'shape': obj.get('shape', 'unknown'),
                    'zone': obj.get('zone', 'unknown'),
                    'object_cell': (row, col),
                    'shoot_from_cells': detected_from if detected_from else [(row, col)]
                }
                objects_to_fire.append(obj_info)
    
    print(f"\n🎯 Found {len(objects_to_fire)} targets to fire at:")
    for i, obj in enumerate(objects_to_fire, 1):
        print(f"   {i}. {obj['color']} {obj['shape']} at cell {obj['object_cell']}, "
            f"shoot from: {obj['shoot_from_cells']}")
    
    return objects_to_fire

def build_occupancy_map(mapping_data, grid_size=5):
    """สร้าง occupancy map จากข้อมูล JSON"""
    print(f"\n🗺️  Building occupancy map ({grid_size}x{grid_size})...")
    
    # สร้าง grid map
    occupancy_map = [[{'occupied': False, 'walls': {}} for _ in range(grid_size)] for _ in range(grid_size)]
    
    nodes = mapping_data.get('nodes', [])
    
    for node in nodes:
        coord = node.get('coordinate', {})
        r, c = coord.get('row'), coord.get('col')
        
        if r is not None and c is not None and 0 <= r < grid_size and 0 <= c < grid_size:
            # Check if node is occupied
            occupancy_map[r][c]['occupied'] = node.get('is_occupied', False)
            
            # Check walls
            walls = node.get('walls', {})
            occupancy_map[r][c]['walls'] = {
                'N': walls.get('north', False),
                'S': walls.get('south', False),
                'E': walls.get('east', False),
                'W': walls.get('west', False)
            }
    
    print("✅ Occupancy map built successfully")
    return occupancy_map

# =============================================================================
# ===== PATHFINDING (A* ALGORITHM) ============================================
# =============================================================================

def get_neighbors(pos, occupancy_map, grid_size):
    """หาโหนดข้างเคียงที่สามารถเดินไปได้ (ไม่มีกำแพงกั้น)"""
    r, c = pos
    neighbors = []
    
    # Directions: North, East, South, West
    directions = [
        ((-1, 0), 'N', 'S'),  # North
        ((0, 1), 'E', 'W'),   # East
        ((1, 0), 'S', 'N'),   # South
        ((0, -1), 'W', 'E')   # West
    ]
    
    for (dr, dc), wall_check, opposite_wall in directions:
        new_r, new_c = r + dr, c + dc
        
        # Check if within bounds
        if 0 <= new_r < grid_size and 0 <= new_c < grid_size:
            # Check if there's a wall blocking
            current_cell = occupancy_map[r][c]
            next_cell = occupancy_map[new_r][new_c]
            
            # ตรวจสอบกำแพงของ cell ปัจจุบันและ cell ถัดไป
            has_wall = current_cell['walls'].get(wall_check, False)
            
            # ตรวจสอบว่า cell ถัดไปไม่ถูก occupied
            is_next_occupied = next_cell.get('occupied', False)
            
            if not has_wall and not is_next_occupied:
                neighbors.append((new_r, new_c))
    
    return neighbors

def heuristic(pos1, pos2):
    """Manhattan distance heuristic"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def a_star_pathfinding(start, goal, occupancy_map, grid_size):
    """A* pathfinding algorithm เพื่อหาเส้นทางที่หลบกำแพง"""
    print(f"🔍 Finding path from {start} to {goal}...")
    
    if start == goal:
        return [start]
    
    # Priority queue: (f_score, counter, position, path)
    counter = 0
    open_set = [(0, counter, start, [start])]
    closed_set = set()
    g_scores = {start: 0}
    
    while open_set:
        f_score, _, current, path = heapq.heappop(open_set)
        
        if current in closed_set:
            continue
        
        if current == goal:
            print(f"✅ Found path with {len(path)} steps: {path}")
            return path
        
        closed_set.add(current)
        
        for neighbor in get_neighbors(current, occupancy_map, grid_size):
            if neighbor in closed_set:
                continue
            
            tentative_g = g_scores[current] + 1
            
            if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                g_scores[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                counter += 1
                new_path = path + [neighbor]
                heapq.heappush(open_set, (f, counter, neighbor, new_path))
    
    print(f"❌ No path found from {start} to {goal}")
    return None

# =============================================================================
# ===== ROBOT MOVEMENT FUNCTIONS ==============================================
# =============================================================================

def get_direction_between_nodes(from_pos, to_pos):
    """หาทิศทางระหว่าง 2 โหนด"""
    dr = to_pos[0] - from_pos[0]
    dc = to_pos[1] - from_pos[1]
    
    if dr == -1 and dc == 0:
        return 0  # North
    elif dr == 0 and dc == 1:
        return 1  # East
    elif dr == 1 and dc == 0:
        return 2  # South
    elif dr == 0 and dc == -1:
        return 3  # West
    else:
        return None

def rotate_to_direction(chassis, target_direction):
    """หมุนหุ่นยนต์ไปยังทิศทางที่ต้องการ"""
    global CURRENT_DIRECTION, CURRENT_TARGET_YAW
    
    if CURRENT_DIRECTION == target_direction:
        return
    
    diff = (target_direction - CURRENT_DIRECTION + 4) % 4
    
    if diff == 1:  # Turn right 90°
        angle = -90
    elif diff == 2:  # Turn 180°
        angle = 180
    elif diff == 3:  # Turn left 90°
        angle = 90
    else:
        return
    
    print(f"🔄 Rotating {angle}° (from direction {CURRENT_DIRECTION} to {target_direction})")
    
    try:
        chassis.move(x=0, y=0, z=angle, xy_speed=0, z_speed=SPEED_ROTATE).wait_for_completed()
        time.sleep(0.3)
        
        CURRENT_DIRECTION = target_direction
        CURRENT_TARGET_YAW = (CURRENT_TARGET_YAW + angle) % 360
        
        print(f"✅ Rotated to direction {CURRENT_DIRECTION}")
    except Exception as e:
        print(f"❌ Rotation error: {e}")

def move_forward_one_grid(chassis):
    """เดินหน้า 1 grid"""
    print("➡️  Moving forward one grid...")
    
    try:
        chassis.move(x=0.6, y=0, z=0, xy_speed=0.3, z_speed=0).wait_for_completed()
        time.sleep(0.3)
        print("✅ Moved forward one grid")
        return True
    except Exception as e:
        print(f"❌ Movement error: {e}")
        return False

def execute_path(chassis, path):
    """เดินตามเส้นทางที่กำหนด"""
    global CURRENT_POSITION
    
    print(f"\n🚶 Executing path: {path}")
    
    for i in range(len(path) - 1):
        current = path[i]
        next_pos = path[i + 1]
        
        # หาทิศทางที่ต้องหันหน้าไป
        target_direction = get_direction_between_nodes(current, next_pos)
        
        if target_direction is None:
            print(f"❌ Invalid path segment: {current} -> {next_pos}")
            continue
        
        # หมุนไปยังทิศทางที่ถูกต้อง
        rotate_to_direction(chassis, target_direction)
        
        # เดินหน้า 1 grid
        if move_forward_one_grid(chassis):
            CURRENT_POSITION = next_pos
            print(f"📍 Current position: {CURRENT_POSITION}")
        else:
            print(f"❌ Failed to move to {next_pos}")
            return False
    
    print("✅ Path execution completed")
    return True

# =============================================================================
# ===== OBJECT DETECTION & TRACKING ===========================================
# =============================================================================

def processing_thread_func(tracker, q, target_shape, target_color, roi_state, is_detecting_func):
    """Thread สำหรับประมวลผลภาพและตรวจจับวัตถุ"""
    global processed_output
    
    while True:
        try:
            if not is_detecting_func():
                time.sleep(0.05)
                continue
            
            frame = q.get(timeout=1.0)
            
            # Extract ROI
            ROI_X, ROI_Y, ROI_W, ROI_H = roi_state["x"], roi_state["y"], roi_state["w"], roi_state["h"]
            roi = frame[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
            
            # Track objects
            tracker.track(roi)
            tracked = tracker.get_tracked_objects()
            
            # Filter targets
            target_dets = []
            for det in tracked:
                if det.get("shape") == target_shape and det.get("color") == target_color:
                    det["is_target"] = True
                    target_dets.append(det)
            
            with output_lock:
                processed_output = {"count": len(target_dets), "details": target_dets}
        
        except Exception as e:
            if "Empty" not in str(e):
                print(f"Processing thread error: {e}")
            time.sleep(0.05)

# =============================================================================
# ===== FIRING SYSTEM =========================================================
# =============================================================================

def fire_at_target(manager, roi_state, target_info):
    """ยิงเป้าหมายด้วย PID tracking"""
    global is_tracking_mode, fired_targets, current_target_id, shots_fired
    
    print(f"\n🎯 Starting to fire at {target_info['color']} {target_info['shape']} at {target_info['object_cell']}")
    
    gimbal = manager.get_gimbal()
    blaster = manager.get_blaster()
    
    if gimbal is None or blaster is None:
        print("⚠️ Gimbal or blaster not available")
        return False
    
    # Reset gimbal to center
    try:
        gimbal.recenter().wait_for_completed()
        time.sleep(0.5)
    except:
        pass
    
    # Start tracking mode
    is_tracking_mode = True
    shots_fired = 0
    
    max_attempts = 100  # Maximum tracking attempts
    attempts = 0
    lock_count = 0
    
    while shots_fired < FIRE_SHOTS_COUNT and attempts < max_attempts:
        attempts += 1
        
        # Get current detections
        with output_lock:
            dets = list(processed_output["details"])
        
        if not dets:
            print("🔍 No target detected, searching...")
            time.sleep(0.1)
            continue
        
        # Find largest target
        target_box = None
        max_area = -1
        
        for det in dets:
            if det.get("is_target", False):
                x, y, w, h = det["box"]
                area = w * h
                if area > max_area:
                    max_area = area
                    target_box = (x, y, w, h)
                    current_target_id = det.get("id")
        
        if target_box is None:
            print("🔍 No valid target found")
            time.sleep(0.1)
            continue
        
        # Calculate error
        x, y, w, h = target_box
        cx_roi = x + w/2.0
        cy_roi = y + h/2.0
        
        ROI_X, ROI_Y, ROI_W, ROI_H = roi_state["x"], roi_state["y"], roi_state["w"], roi_state["h"]
        cx = ROI_X + cx_roi
        cy = ROI_Y + cy_roi
        
        center_x = FRAME_W / 2.0
        center_y = FRAME_H / 2.0
        
        err_x = center_x - cx
        err_y = (center_y - cy) + PITCH_BIAS_PIX
        
        # Deadzone
        if abs(err_x) < PIX_ERR_DEADZONE:
            err_x = 0.0
        if abs(err_y) < PIX_ERR_DEADZONE:
            err_y = 0.0
        
        # PID control
        u_x = PID_KP * err_x
        u_y = PID_KP * err_y
        
        u_x = float(np.clip(u_x, -MAX_YAW_SPEED, MAX_YAW_SPEED))
        u_y = float(np.clip(u_y, -MAX_PITCH_SPEED, MAX_PITCH_SPEED))
        
        try:
            gimbal.drive_speed(pitch_speed=-u_y, yaw_speed=u_x)
        except Exception as e:
            print(f"Gimbal control error: {e}")
        
        # Check if locked on target
        locked = (abs(err_x) <= LOCK_TOL_X) and (abs(err_y) <= LOCK_TOL_Y)
        
        if locked:
            lock_count += 1
            if lock_count >= 3:  # Require stable lock
                try:
                    blaster.fire(fire_type=r_blaster.WATER_FIRE)
                    shots_fired += 1
                    print(f"🔥 Fired shot {shots_fired}/{FIRE_SHOTS_COUNT}")
                    time.sleep(0.3)
                    lock_count = 0
                except Exception as e:
                    print(f"Fire error: {e}")
        else:
            lock_count = 0
            print(f"🎯 Tracking... err_x={err_x:.1f}, err_y={err_y:.1f}")
        
        time.sleep(0.05)
    
    # Stop tracking
    is_tracking_mode = False
    
    try:
        gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
        gimbal.recenter().wait_for_completed()
        time.sleep(0.5)
    except:
        pass
    
    if shots_fired >= FIRE_SHOTS_COUNT:
        print(f"✅ Successfully fired {shots_fired} shots at target")
        return True
    else:
        print(f"⚠️ Only fired {shots_fired}/{FIRE_SHOTS_COUNT} shots")
        return False

# =============================================================================
# ===== MAIN PROGRAM ==========================================================
# =============================================================================

class ObjectTracker:
    """Simple object tracker for color and shape detection"""
    def __init__(self):
        self.tracked_objects = []
        self.next_id = 0
    
    def track(self, frame):
        """Track objects in frame (simplified version)"""
        # This is a simplified version - in real implementation,
        # you would use actual computer vision algorithms
        self.tracked_objects = []
        # For now, return empty list - will be filled by actual detection
    
    def get_tracked_objects(self):
        return self.tracked_objects

def main():
    """Main program"""
    global CURRENT_POSITION, CURRENT_DIRECTION, is_tracking_mode
    
    print("=" * 70)
    print("  🎯 FIRE ALL DETECTED OBJECTS - ROBOMASTER")
    print("=" * 70)
    
    # 1. Load JSON data
    mapping_data, timestamp_data, objects_data = load_json_data()
    
    if mapping_data is None or objects_data is None:
        print("❌ Failed to load data files. Exiting...")
        return
    
    # 2. Extract objects and build map
    objects_to_fire = extract_objects_and_positions(objects_data)
    occupancy_map = build_occupancy_map(mapping_data, grid_size=GRID_SIZE)
    
    if not objects_to_fire:
        print("\n⚠️ No objects to fire at. Exiting...")
        return
    
    # 3. Connect to robot
    print("\n🤖 Connecting to RoboMaster...")
    ep_robot = robot.Robot()
    
    try:
        ep_robot.initialize(conn_type="ap")
        print("✅ Connected to RoboMaster")
        
        chassis = ep_robot.chassis
        gimbal = ep_robot.gimbal
        blaster = ep_robot.blaster
        camera = ep_robot.camera
        
        # Initialize camera
        camera.start_video_stream(display=False, resolution=r_camera.STREAM_360P)
        time.sleep(2)
        
        # Setup object detection
        tracker = ObjectTracker()
        frame_queue = deque(maxlen=2)
        roi_state = {"x": ROI_X0, "y": ROI_Y0, "w": ROI_W0, "h": ROI_H0}
        
        is_detecting = lambda: True
        
        # Start processing thread
        proc_thread = threading.Thread(
            target=processing_thread_func,
            args=(tracker, frame_queue, TARGET_SHAPE, TARGET_COLOR, roi_state, is_detecting),
            daemon=True
        )
        proc_thread.start()
        
        # Camera frame capture (simplified - in real implementation, use actual camera feed)
        
        # 4. Visit each object and fire
        print(f"\n{'='*70}")
        print(f"  🎯 STARTING OBJECT FIRING SEQUENCE - {len(objects_to_fire)} TARGETS")
        print(f"{'='*70}\n")
        
        for idx, obj_info in enumerate(objects_to_fire, 1):
            print(f"\n{'='*70}")
            print(f"  TARGET {idx}/{len(objects_to_fire)}: {obj_info['color']} {obj_info['shape']}")
            print(f"{'='*70}")
            
            # Get best shooting position
            shoot_from_positions = obj_info['shoot_from_cells']
            target_pos = shoot_from_positions[0] if shoot_from_positions else obj_info['object_cell']
            
            print(f"📍 Target at: {obj_info['object_cell']}")
            print(f"📍 Will shoot from: {target_pos}")
            print(f"📍 Current position: {CURRENT_POSITION}")
            
            # Find path to shooting position
            path = a_star_pathfinding(CURRENT_POSITION, target_pos, occupancy_map, GRID_SIZE)
            
            if path is None:
                print(f"⚠️ Cannot reach target {idx}. Skipping...")
                continue
            
            # Execute path
            if len(path) > 1:
                if not execute_path(chassis, path):
                    print(f"⚠️ Failed to reach target {idx}. Skipping...")
                    continue
            
            # Fire at target
            print(f"\n🔫 Preparing to fire at target {idx}...")
            time.sleep(1)  # Give time to stabilize
            
            fire_at_target(ep_robot, roi_state, obj_info)
            
            print(f"\n✅ Completed target {idx}/{len(objects_to_fire)}")
            time.sleep(2)
        
        print(f"\n{'='*70}")
        print(f"  ✅ ALL TARGETS COMPLETED!")
        print(f"{'='*70}\n")
    
    except KeyboardInterrupt:
        print("\n⚠️ Program interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
    
    finally:
        print("\n🔌 Closing connection...")
        try:
            ep_robot.camera.stop_video_stream()
            ep_robot.close()
        except:
            pass
        print("✅ Program ended")

if __name__ == "__main__":
    main()