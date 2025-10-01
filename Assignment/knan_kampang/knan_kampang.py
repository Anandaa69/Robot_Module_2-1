# -*-coding:utf-8-*-

import time
import robomaster
from robomaster import robot
import numpy as np
from scipy.ndimage import median_filter
from datetime import datetime
import json
import os
from collections import deque
import math

# Global variables
ROBOT_FACE = 1 
CURRENT_TARGET_YAW = 0.0

# =======================================================================
# ===== CONFIGURATION CONSTANTS =========================================
# =======================================================================
TARGET_WALL_DISTANCE_CM = 17.0
CENTERING_THRESHOLD_CM = 2.0
# =======================================================================

def convert_to_json_serializable(obj):
    # ... (No changes) ...
    if isinstance(obj, np.bool_): return bool(obj)
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_to_json_serializable(i) for i in obj]
    return obj
    

class MovementTracker:
    # ... (No changes) ...
    def __init__(self):
        self.consecutive_forward_moves = 0
        self.consecutive_backward_moves = 0
        self.last_movement_type = None
    def record_movement(self, movement_type):
        if movement_type == 'forward':
            self.consecutive_forward_moves = self.consecutive_forward_moves + 1 if self.last_movement_type == 'forward' else 1
            self.consecutive_backward_moves = 0
        elif movement_type == 'backward':
            self.consecutive_backward_moves = self.consecutive_backward_moves + 1 if self.last_movement_type == 'backward' else 1
            self.consecutive_forward_moves = 0
        elif movement_type == 'rotation':
            self.consecutive_forward_moves = 0
            self.consecutive_backward_moves = 0
        self.last_movement_type = movement_type
        print(f"📊 Movement: {movement_type}, Fwd Streak: {self.consecutive_forward_moves}, Bwd Streak: {self.consecutive_backward_moves}")
    def has_consecutive_forward_moves(self, threshold=2):
        return self.consecutive_forward_moves >= threshold
    def has_consecutive_backward_moves(self, threshold=2):
        return self.consecutive_backward_moves >= threshold

class AttitudeHandler:
    def __init__(self):
        self.current_yaw = 0.0
        self.yaw_tolerance = 1.5  # <--- CHANGED
        self.is_monitoring = False
    def attitude_handler(self, attitude_info):
        # ... (No changes) ...
        if self.is_monitoring:
            self.current_yaw = attitude_info[0]
            print(f"\r🧭 Current Yaw: {self.current_yaw:.1f}°", end="", flush=True)
    def start_monitoring(self, chassis):
        # ... (No changes) ...
        self.is_monitoring = True
        chassis.sub_attitude(freq=20, callback=self.attitude_handler)
    def stop_monitoring(self, chassis):
        # ... (No changes) ...
        self.is_monitoring = False
        try: chassis.unsub_attitude()
        except Exception: pass
    def normalize_angle(self, angle):
        # ... (No changes) ...
        while angle > 180: angle -= 360
        while angle <= -180: angle += 360
        return angle
    def is_at_target_yaw(self, target_yaw=0.0):
        # ... (No changes) ...
        normalized_current = self.normalize_angle(self.current_yaw)
        normalized_target = self.normalize_angle(target_yaw)
        diff = abs(self.normalize_angle(normalized_current - normalized_target))
        if abs(normalized_target) == 180:
            return 180 - abs(normalized_current) <= self.yaw_tolerance
        return diff <= self.yaw_tolerance
    def correct_yaw_to_target(self, chassis, target_yaw=0.0):
        # ... (No changes) ...
        if self.is_at_target_yaw(target_yaw):
            print(f"\n✅ Yaw OK: {self.current_yaw:.1f}° (Target: {target_yaw}°)")
            return True
        try:
            robot_rotation = -self.normalize_angle(target_yaw - self.current_yaw)
            print(f"\n🔧 Correcting Yaw: {self.current_yaw:.1f}° -> {target_yaw}°. Rotating: {robot_rotation:.1f}°")
            if abs(robot_rotation) > self.yaw_tolerance:
                chassis.move(x=0, y=0, z=robot_rotation, z_speed=60).wait_for_completed(timeout=3)
                time.sleep(0.2)
            if self.is_at_target_yaw(target_yaw):
                print(f"✅ Yaw Correction Success: {self.current_yaw:.1f}°")
                return True
            print(f"⚠️ First attempt incomplete. Current: {self.current_yaw:.1f}°. Fine-tuning...")
            remaining_rotation = -self.normalize_angle(target_yaw - self.current_yaw)
            if abs(remaining_rotation) > 0.5 and abs(remaining_rotation) < 20:
                print(f"   🔧 Fine-tuning by: {remaining_rotation:.1f}°")
                chassis.move(x=0, y=0, z=remaining_rotation, z_speed=40).wait_for_completed(timeout=2)
                time.sleep(0.2)
            if self.is_at_target_yaw(target_yaw):
                print(f"✅ Yaw Fine-tuning Success: {self.current_yaw:.1f}°")
                return True
            else:
                print(f"🔥🔥 Yaw Correction FAILED. Final Yaw: {self.current_yaw:.1f}° (Target: {target_yaw}°)")
                return False
        except Exception as e:
            print(f"❌ Exception during Yaw Correction: {e}")
            return False

class PID:
    # ... (No changes) ...
    def __init__(self, Kp, Ki, Kd, setpoint=0):
        self.Kp, self.Ki, self.Kd, self.setpoint = Kp, Ki, Kd, setpoint
        self.prev_error, self.integral, self.integral_max = 0, 0, 1.0
    def compute(self, current, dt):
        error = self.setpoint - current
        self.integral += error * dt
        if self.integral > self.integral_max: self.integral = self.integral_max
        elif self.integral < -self.integral_max: self.integral = -self.integral_max
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

class MovementController:
    def __init__(self, chassis):
        # ... (No changes) ...
        self.chassis = chassis
        self.current_x, self.current_y = 0.0, 0.0
        self.KP, self.KI, self.KD = 1.9, 0.25, 10
        self.RAMP_UP_TIME, self.MOVE_TIMEOUT = 0.8, 4.0 
        self.movement_tracker = MovementTracker()
        self.nodes_visited_count, self.total_drift_corrections, self.last_correction_at = 0, 0, 0
        self.DRIFT_CORRECTION_INTERVAL, self.DRIFT_CORRECTION_ANGLE = 7, 2
        self.chassis.sub_position(freq=20, callback=self.position_handler)
        time.sleep(0.1)

    def position_handler(self, position_info):
        self.current_x, self.current_y = position_info[0], position_info[1]

    def increment_node_visit_for_backtrack_with_correction(self, attitude_handler):
        self.nodes_visited_count += 1
        print(f"📊 Backtrack Node Count: {self.nodes_visited_count}")
        if (self.nodes_visited_count % self.DRIFT_CORRECTION_INTERVAL == 0 and self.nodes_visited_count != self.last_correction_at):
            return self.perform_attitude_drift_correction(attitude_handler)
        return True

    def increment_node_visit_main_exploration(self, attitude_handler):
        self.nodes_visited_count += 1
        print(f"📊 Main Exploration Node Count: {self.nodes_visited_count}")
        if (self.nodes_visited_count % self.DRIFT_CORRECTION_INTERVAL == 0 and self.nodes_visited_count != self.last_correction_at):
            return self.perform_attitude_drift_correction(attitude_handler)
        return True

    def perform_attitude_drift_correction(self, attitude_handler):
        global CURRENT_TARGET_YAW
        print(f"⚙️ Performing Attitude Drift Correction (+{self.DRIFT_CORRECTION_ANGLE}°)...")
        CURRENT_TARGET_YAW = attitude_handler.normalize_angle(CURRENT_TARGET_YAW + self.DRIFT_CORRECTION_ANGLE)
        success = attitude_handler.correct_yaw_to_target(self.chassis, CURRENT_TARGET_YAW)
        if success:
            self.total_drift_corrections += 1
            self.last_correction_at = self.nodes_visited_count
            self.movement_tracker.record_movement('rotation')
            print(f"✅ Drift Correction Completed! Total Corrections: {self.total_drift_corrections}")
        else:
            print(f"⚠️ Attitude drift correction FAILED!")
        time.sleep(0.2)
        return success

    def move_forward_with_pid(self, target_distance, axis, direction=1, allow_yaw_correction=True):
        self.movement_tracker.record_movement('forward' if direction == 1 else 'backward')
        
        # --- DISABLED AUTOMATIC YAW CORRECTION ---
        # if allow_yaw_correction and (self.movement_tracker.has_consecutive_forward_moves(2) or self.movement_tracker.has_consecutive_backward_moves(2)):
        #     attitude_handler.correct_yaw_to_target(self.chassis, attitude_handler.normalize_angle(CURRENT_TARGET_YAW))
        
        pid = PID(Kp=self.KP, Ki=self.KI, Kd=self.KD, setpoint=target_distance)
        start_time, last_time = time.time(), time.time()
        start_position = self.current_x if axis == 'x' else self.current_y
        max_speed = 1.5
        print(f"🚀 Moving {'FORWARD' if direction == 1 else 'BACKWARD'} {target_distance}m on {axis}-axis")
        target_reached = False
        while not target_reached:
            if time.time() - start_time > self.MOVE_TIMEOUT:
                print(f"🔥🔥 MOVEMENT TIMEOUT!")
                break
            now = time.time(); dt = now - last_time; last_time = now
            current_position = self.current_x if axis == 'x' else self.current_y
            relative_position = abs(current_position - start_position)
            if abs(relative_position - target_distance) < 0.02:
                target_reached = True; break
            output = pid.compute(relative_position, dt)
            ramp_multiplier = min(1.0, 0.1 + ((now - start_time) / self.RAMP_UP_TIME) * 0.9)
            speed = max(-max_speed, min(max_speed, output * ramp_multiplier))
            self.chassis.drive_speed(x=speed * direction, y=0, z=0, timeout=1)
        self.chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
        print(f"✅ Target reached!" if target_reached else f"⚠️ Target possibly not reached.")

    def rotate_90_degrees_right(self, attitude_handler):
        # ... (No changes) ...
        global CURRENT_TARGET_YAW
        print("🔄 Rotating 90° RIGHT...")
        self.movement_tracker.record_movement('rotation'); time.sleep(0.2)
        CURRENT_TARGET_YAW += 90
        target_angle = attitude_handler.normalize_angle(CURRENT_TARGET_YAW)
        attitude_handler.correct_yaw_to_target(self.chassis, target_angle); time.sleep(0.2)

    def rotate_90_degrees_left(self, attitude_handler):
        # ... (No changes) ...
        global CURRENT_TARGET_YAW
        print("🔄 Rotating 90° LEFT...")
        self.movement_tracker.record_movement('rotation'); time.sleep(0.2)
        CURRENT_TARGET_YAW -= 90
        target_angle = attitude_handler.normalize_angle(CURRENT_TARGET_YAW)
        attitude_handler.correct_yaw_to_target(self.chassis, target_angle); time.sleep(0.2)
        
    def reverse_from_dead_end(self):
        # ... (No changes) ...
        global ROBOT_FACE
        print("🔙 DEAD END DETECTED - Reversing...")
        axis_test = 'y' if ROBOT_FACE % 2 == 0 else 'x'
        self.move_forward_with_pid(0.6, axis_test, direction=-1)
        print("✅ Reverse from dead end completed!")
        
    def reverse_to_previous_node(self):
        # ... (No changes) ...
        global ROBOT_FACE
        print("🔙 BACKTRACKING - Reversing to previous node...")
        axis_test = 'y' if ROBOT_FACE % 2 == 0 else 'x'
        self.move_forward_with_pid(0.6, axis_test, direction=-1, allow_yaw_correction=False)
        print("✅ Reverse backtrack completed!")
        
    def center_using_left_wall(self, chassis, sharp_handler, current_robot_direction):
        """
        ปรับตำแหน่งหุ่นยนต์ด้านข้างเพื่อรักษาระยะห่างจากกำแพงด้านซ้าย
        """
        print(f"📐 Targeting wall distance: {TARGET_WALL_DISTANCE_CM} cm...")
        s1, s2 = sharp_handler.get_distances()
        
        current_distance = (s1 + s2) / 2.0

        if current_distance > 50: 
            print(" Bypassing centering: Wall is too far.")
            return

        error = current_distance - TARGET_WALL_DISTANCE_CM
        print(f"   Current distance: {current_distance:.1f} cm, Error: {error:.1f} cm")

        if abs(error) > CENTERING_THRESHOLD_CM:
            dir_map_left = {'north': 'west', 'east': 'north', 'south': 'east', 'west': 'south'}
            dir_map_right = {'north': 'east', 'east': 'south', 'south': 'west', 'west': 'north'}
            
            left_dir_world = dir_map_left[current_robot_direction]
            right_dir_world = dir_map_right[current_robot_direction]

            if error > 0:
                print(f"   Too far from wall. Nudging LEFT (towards {left_dir_world}).")
                nudge_robot(chassis, direction=left_dir_world, distance_m=0.05)
            else:
                print(f"   Too close to wall. Nudging RIGHT (towards {right_dir_world}).")
                nudge_robot(chassis, direction=right_dir_world, distance_m=0.05)
        else:
            print(" Centering OK.")

    def align_with_left_wall(self, sharp_handler, attitude_handler, sensor_distance_cm=20.0, max_align_dist_cm=45.0, align_threshold_deg=1.5):
        # ... (No changes) ...
        global CURRENT_TARGET_YAW
        print("📐 Attempting to align with left wall...")
        s1, s2 = sharp_handler.get_distances()
        if s1 > max_align_dist_cm or s2 > max_align_dist_cm:
            print(f" Bypassing alignment: Wall is too far (s1={s1:.1f}, s2={s2:.1f}).")
            return
        delta_s = s2 - s1
        angle_rad = np.arctan(delta_s / sensor_distance_cm)
        angle_deg = np.degrees(angle_rad)
        print(f"📐 Calculated deviation angle: {angle_deg:.2f}°")
        if abs(angle_deg) > align_threshold_deg:
            print(f"🔧 Alignment needed. Current Target Yaw: {CURRENT_TARGET_YAW:.2f}°")
            correction_angle = -angle_deg
            new_target_yaw = attitude_handler.normalize_angle(CURRENT_TARGET_YAW + correction_angle)
            print(f"   Correcting by: {correction_angle:.2f}°. New Target Yaw: {new_target_yaw:.2f}°")
            if attitude_handler.correct_yaw_to_target(self.chassis, new_target_yaw):
                CURRENT_TARGET_YAW = new_target_yaw
                print("✅ Wall alignment successful.")
            else:
                print("⚠️ Wall alignment failed.")
        else:
            print(" Alignment OK.")
            
    def cleanup(self):
        try: self.chassis.unsub_position()
        except: pass

    def get_drift_correction_status(self):
        return {'nodes_visited': self.nodes_visited_count,'next_correction_at': ((self.nodes_visited_count // 7) + 1) * 7,'total_corrections': self.total_drift_corrections,'last_correction_at': self.last_correction_at}

class GraphNode:
    # ... (No changes) ...
    def __init__(self, node_id, position):
        self.id, self.position = node_id, position
        self.walls = {'north': False, 'south': False, 'east': False, 'west': False}
        self.neighbors = {'north': None, 'south': None, 'east': None, 'west': None}
        self.visited, self.fullyScanned, self.isDeadEnd = True, False, False
        self.exploredDirections, self.unexploredExits = [], []
        self.sensorReadings, self.initialScanDirection = {}, None
        self.outOfBoundsExits = [] 

class GraphMapper:
    def __init__(self, min_x=-100, max_x=100, min_y=-100, max_y=100):
        # ... (No changes) ...
        self.nodes, self.currentPosition, self.currentDirection = {}, (0, 0), 'north'
        self.frontierQueue, self.previous_node = [], None
        self.find_next_exploration_direction = self.find_next_exploration_direction_with_priority
        self.min_x, self.min_y, self.max_x, self.max_y = min_x, min_y, max_x, max_y
    def get_node_id(self, position): return f"{position[0]}_{position[1]}"
    def create_node(self, position):
        node_id = self.get_node_id(position)
        if node_id not in self.nodes:
            node = GraphNode(node_id, position)
            node.initialScanDirection = self.currentDirection
            self.nodes[node_id] = node
        return self.nodes[node_id]
    def get_current_node(self): return self.nodes.get(self.get_node_id(self.currentPosition))
    def update_current_node_walls_absolute(self, left_wall, right_wall, front_wall, back_wall_scan=None):
        # ... (No changes) ...
        current_node = self.get_current_node()
        if not current_node: return
        dir_map = {'north': {'front': 'north', 'left': 'west', 'right': 'east', 'back': 'south'},'south': {'front': 'south', 'left': 'east', 'right': 'west', 'back': 'north'},'east': {'front': 'east', 'left': 'north', 'right': 'south', 'back': 'west'},'west': {'front': 'west', 'left': 'south', 'right': 'north', 'back': 'east'}}[self.currentDirection]
        current_node.walls[dir_map['front']] = front_wall
        current_node.walls[dir_map['left']] = left_wall
        if right_wall is not None: current_node.walls[dir_map['right']] = right_wall
        if back_wall_scan is not None: current_node.walls[dir_map['back']] = back_wall_scan
        current_node.fullyScanned = True
        self.update_unexplored_exits_with_priority(current_node)
        self.build_connections()
    def update_unexplored_exits_with_priority(self, node):
        # ... (No changes) ...
        node.unexploredExits = []
        x, y = node.position
        dir_map = {'north': {'front': 'north', 'left': 'west', 'right': 'east'},'south': {'front': 'south', 'left': 'east', 'right': 'west'},'east': {'front': 'east', 'left': 'north', 'right': 'south'},'west': {'front': 'west', 'left': 'south', 'right': 'north'}}[self.currentDirection]
        priority_order = ['left', 'front', 'right']
        possible_directions = {'north': (x, y + 1), 'south': (x, y - 1), 'east': (x + 1, y), 'west': (x - 1, y)}
        for rel_dir in priority_order:
            abs_dir = dir_map.get(rel_dir)
            if not abs_dir: continue
            target_pos = possible_directions.get(abs_dir)
            if not target_pos: continue
            tx, ty = target_pos
            if tx < self.min_x or tx > self.max_x or ty < self.min_y or ty > self.max_y:
                if abs_dir not in node.outOfBoundsExits: node.outOfBoundsExits.append(abs_dir)
                continue
            target_node = self.nodes.get(self.get_node_id(target_pos))
            if not node.walls.get(abs_dir, True) and abs_dir not in node.exploredDirections and not (target_node and target_node.fullyScanned):
                node.unexploredExits.append(abs_dir)
        if node.unexploredExits:
            if node.id not in self.frontierQueue: self.frontierQueue.append(node.id)
        elif node.id in self.frontierQueue:
            self.frontierQueue.remove(node.id)
        node.isDeadEnd = sum(1 for w in node.walls.values() if w) >= 3
        if node.isDeadEnd and node.id in self.frontierQueue:
            self.frontierQueue.remove(node.id)
    def build_connections(self):
        # ... (No changes) ...
        for node_id, node in self.nodes.items():
            x, y = node.position
            for d, n_pos in {'north':(x,y+1),'south':(x,y-1),'east':(x+1,y),'west':(x-1,y)}.items():
                if self.get_node_id(n_pos) in self.nodes: node.neighbors[d] = self.nodes[self.get_node_id(n_pos)]
    def get_next_position(self, direction):
        # ... (No changes) ...
        x, y = self.currentPosition
        if direction == 'north': return (x, y + 1)
        if direction == 'south': return (x, y - 1)
        if direction == 'east': return (x + 1, y)
        if direction == 'west': return (x - 1, y)
        return self.currentPosition
    def rotate_to_absolute_direction(self, target_direction, movement_controller, attitude_handler):
        # ... (No changes) ...
        global ROBOT_FACE
        if self.currentDirection == target_direction: return
        print(f"🎯 Rotating from {self.currentDirection} to {target_direction}")
        order = ['north', 'east', 'south', 'west']
        diff = (order.index(target_direction) - order.index(self.currentDirection)) % 4
        if diff == 1: movement_controller.rotate_90_degrees_right(attitude_handler); ROBOT_FACE += 1
        elif diff == 3: movement_controller.rotate_90_degrees_left(attitude_handler); ROBOT_FACE += 1
        elif diff == 2:
            movement_controller.rotate_90_degrees_right(attitude_handler); ROBOT_FACE += 1
            movement_controller.rotate_90_degrees_right(attitude_handler); ROBOT_FACE += 1
        self.currentDirection = target_direction
    def handle_dead_end(self, movement_controller):
        # ... (No changes) ...
        print(f"🚫 DEAD END HANDLER ACTIVATED at {self.currentPosition}")
        dead_end_facing = self.currentDirection
        movement_controller.reverse_from_dead_end()
        rev_map = {'north': 'south', 'south': 'north', 'east': 'west', 'west': 'east'}
        self.currentPosition = self.get_next_position(rev_map[dead_end_facing])
        previous_node = self.get_current_node()
        if previous_node:
            if dead_end_facing not in previous_node.exploredDirections:
                previous_node.exploredDirections.append(dead_end_facing)
            if dead_end_facing in previous_node.unexploredExits:
                previous_node.unexploredExits.remove(dead_end_facing)
            self.update_unexplored_exits_with_priority(previous_node)
        return True
    def move_to_absolute_direction(self, target_direction, movement_controller, attitude_handler, sharp_handler):
        global ROBOT_FACE
        print(f"🎯 Moving to ABSOLUTE direction: {target_direction}")
        self.rotate_to_absolute_direction(target_direction, movement_controller, attitude_handler)
        axis_test = 'y' if ROBOT_FACE % 2 == 0 else 'x'
        movement_controller.move_forward_with_pid(0.6, axis_test, direction=1)
        dir_map = {'north': 'west', 'south': 'east', 'east': 'north', 'west': 'south'}
        left_wall_abs_dir = dir_map[self.currentDirection]
        if self.previous_node and self.previous_node.walls.get(left_wall_abs_dir, False):
            print(f"🧱 Left wall detected ({left_wall_abs_dir}). Initiating correction sequence.")
            time.sleep(0.3)
            # --- MODIFIED CALL ---
            movement_controller.center_using_left_wall(movement_controller.chassis, sharp_handler, self.currentDirection)
            time.sleep(0.2)
            movement_controller.align_with_left_wall(sharp_handler, attitude_handler)
        else:
            print(f" Bypassing correction (No left wall from previous node).")
        self.currentPosition = self.get_next_position(target_direction)
        if self.previous_node and target_direction not in self.previous_node.exploredDirections:
            self.previous_node.exploredDirections.append(target_direction)
            self.update_unexplored_exits_with_priority(self.previous_node)
        print(f"✅ Successfully moved to {self.currentPosition}")
        return True
    def reverse_to_absolute_direction(self, target_direction, movement_controller, attitude_handler):
        # ... (No changes) ...
        print(f"🔙 BACKTRACK: Reversing towards ABSOLUTE direction: {target_direction}")
        rev_map = {'north': 'south', 'south': 'north', 'east': 'west', 'west': 'east'}
        required_facing = rev_map[target_direction]
        self.rotate_to_absolute_direction(required_facing, movement_controller, attitude_handler)
        movement_controller.reverse_to_previous_node()
        self.currentPosition = self.get_next_position(target_direction)
        print(f"✅ Successfully reversed to {self.currentPosition}, NOW FACING {self.currentDirection}")
        return True
    def find_next_exploration_direction_with_priority(self):
        # ... (No changes) ...
        current_node = self.get_current_node()
        if not current_node or not current_node.unexploredExits: return None
        print(f"🔍 Finding next move from {current_node.position}. Unexplored: {current_node.unexploredExits}")
        dir_map = {'north':{'left':'west','right':'east','front':'north'},'south':{'left':'east','right':'west','front':'south'},'east':{'left':'north','right':'south','front':'east'},'west':{'left':'south','right':'north','front':'west'}}[self.currentDirection]
        for rel_dir in ['left', 'front', 'right']:
            abs_dir = dir_map.get(rel_dir)
            if abs_dir and abs_dir in current_node.unexploredExits:
                print(f"✅ Selected priority direction: {rel_dir} ({abs_dir})")
                return abs_dir
        print(f"⚠️ No priority direction. Using fallback: {current_node.unexploredExits[0]}")
        return current_node.unexploredExits[0]
    def find_path_to_frontier(self, target_node_id):
        # ... (No changes) ...
        queue = deque([(self.currentPosition, [])]); visited = {self.currentPosition}
        while queue:
            current_pos, path = queue.popleft()
            if self.get_node_id(current_pos) == target_node_id: return path
            current_node = self.nodes.get(self.get_node_id(current_pos))
            if not current_node: continue
            x, y = current_pos
            for d, n_pos in {'north':(x,y+1),'south':(x,y-1),'east':(x+1,y),'west':(x-1,y)}.items():
                if n_pos not in visited and not current_node.walls.get(d, True):
                    visited.add(n_pos); queue.append((n_pos, path + [d]))
        return None
    def execute_path_to_frontier_with_reverse(self, path, movement_controller, attitude_handler):
        # ... (No changes) ...
        print(f"🗺️ Executing REVERSE path to frontier: {path}")
        for i, step_direction in enumerate(path):
            if not self.reverse_to_absolute_direction(step_direction, movement_controller, attitude_handler): return False
            if not movement_controller.increment_node_visit_for_backtrack_with_correction(attitude_handler):
                print("🔥🔥 Critical failure during backtrack drift correction. Aborting."); return False
            time.sleep(0.2)
        print(f"✅ Successfully reached frontier at {self.currentPosition}")
        return True
    def find_nearest_frontier(self):
        # ... (No changes) ...
        print("🔍 Finding nearest frontier...")
        valid_frontiers = [fid for fid in self.frontierQueue if self.nodes.get(fid) and self.nodes.get(fid).unexploredExits]
        if not valid_frontiers:
            print(f"❌ No valid frontiers found!")
            return None, None, None
        best_frontier, shortest_path, min_distance = None, None, float('inf')
        for frontier_id in valid_frontiers:
            path = self.find_path_to_frontier(frontier_id)
            if path is not None and len(path) < min_distance:
                min_distance, best_frontier, shortest_path = len(path), frontier_id, path
        if best_frontier: 
            print(f"🏆 SELECTED FRONTIER: {best_frontier} at distance {min_distance} via path {shortest_path}")
        else: 
            print(f"❌ No reachable frontiers found!")
        return best_frontier, None, shortest_path
    def print_graph_summary(self):
        # ... (No changes) ...
        print("\n" + "="*40 + " GRAPH SUMMARY " + "="*40)
        print(f"🤖 Pos: {self.currentPosition}, Dir: {self.currentDirection}, Nodes: {len(self.nodes)}, Frontiers: {len(self.frontierQueue)}")
        for node_id, node in sorted(self.nodes.items()):
            walls_str = ', '.join([d for d, w in sorted(node.walls.items()) if w])
            print(f"  📍 {node.id:<5} @{str(node.position):<8} | Walls: [{walls_str:<25}] | Unexplored: {node.unexploredExits} | DeadEnd: {node.isDeadEnd}")
        print("="*95 + "\n")

class ToFSensorHandler:
    # ... (No changes) ...
    def __init__(self):
        self.CALIBRATION_SLOPE, self.CALIBRATION_Y_INTERCEPT = 0.0894, 3.8409
        self.WINDOW_SIZE, self.WALL_THRESHOLD = 5, 50.0
        self.tof_buffer, self.current_readings, self.collecting_data = [], [], False
    def tof_data_handler(self, sub_info):
        if not self.collecting_data: return
        raw_tof_mm = sub_info[0]
        if 0 < raw_tof_mm <= 4000:
            calibrated_cm = (self.CALIBRATION_SLOPE * raw_tof_mm) + self.CALIBRATION_Y_INTERCEPT
            self.tof_buffer.append(calibrated_cm)
            if len(self.tof_buffer) >= self.WINDOW_SIZE:
                self.current_readings.append(median_filter(self.tof_buffer[-self.WINDOW_SIZE:], size=self.WINDOW_SIZE)[-1])
    def scan_direction(self, sensor, direction):
        self.tof_buffer.clear(); self.current_readings.clear(); self.collecting_data = True
        sensor.sub_distance(freq=25, callback=self.tof_data_handler); time.sleep(0.3)
        self.collecting_data = False; 
        try: sensor.unsub_distance() 
        except: pass
        if not self.current_readings: 
            print(f"[{direction.upper():<5}] NO READINGS -> WALL (FAIL-SAFE)")
            return 0.0, True
        avg_distance = np.mean(self.current_readings)
        is_wall = 0 < avg_distance <= self.WALL_THRESHOLD
        print(f"[{direction.upper():<5}] {avg_distance:.2f}cm -> {'WALL' if is_wall else 'OPEN'}")
        return avg_distance, is_wall

class SharpSensorHandler:
    # ... (No changes) ...
    def __init__(self, sensor_adaptor, front_id, front_port, rear_id, rear_port):
        self.sensor_adaptor = sensor_adaptor
        self.front_id, self.front_port = front_id, front_port
        self.rear_id, self.rear_port = rear_id, rear_port
        print("🔩 Sharp Sensor Handler (Adaptor Version) initialized.")
        print(f"   - Front Sensor (s1): ID={front_id}, Port={front_port}")
        print(f"   - Rear Sensor  (s2): ID={rear_id}, Port={rear_port}")
    def _adc_to_cm(self, adc_value):
        if adc_value <= 50: return float('inf')
        A, B = 30263, -1.352
        return A * (adc_value ** B)
    def get_distances(self, num_samples=5, delay=0.02):
        front_readings, rear_readings = [], []
        for _ in range(num_samples):
            front_adc = self.sensor_adaptor.get_adc(id=self.front_id, port=self.front_port)
            rear_adc = self.sensor_adaptor.get_adc(id=self.rear_id, port=self.rear_port)
            front_readings.append(self._adc_to_cm(front_adc))
            rear_readings.append(self._adc_to_cm(rear_adc))
            time.sleep(delay)
        s1_distance = np.median(front_readings)
        s2_distance = np.median(rear_readings)
        print(f"📏 Sharp Readings: s1(Front)={s1_distance:6.2f} cm | s2(Rear)={s2_distance:6.2f} cm")
        return s1_distance, s2_distance

def nudge_robot(chassis, direction, distance_m=0.05):
    """
    ขยับหุ่นยนต์ในทิศทางของ World Frame (north, south, east, west)
    """
    print(f"🔩 Nudging robot towards {direction} by {distance_m * 100} cm...")
    x_move, y_move = 0, 0
    if direction == 'north': y_move = distance_m
    elif direction == 'south': y_move = -distance_m
    elif direction == 'east': x_move = distance_m
    elif direction == 'west': x_move = -distance_m
    else:
        print(f"⚠️ Invalid nudge direction: {direction}")
        return
    chassis.move(x=x_move, y=y_move, z=0, xy_speed=0.3).wait_for_completed()
    time.sleep(0.2)

def scan_current_node_simplified(sensor, tof_handler, sharp_handler, graph_mapper):
    # ... (No changes) ...
    print(f"\n🗺️ === Simplified Scanning Node at {graph_mapper.currentPosition} ===")
    current_node = graph_mapper.create_node(graph_mapper.currentPosition)
    if current_node.fullyScanned:
        print("🔄 Node already scanned. Using cached data."); return

    front_dist, front_wall = tof_handler.scan_direction(sensor, 'front')
    
    s1, s2 = sharp_handler.get_distances()
    avg_left_dist = (s1 + s2) / 2.0
    SHARP_WALL_THRESHOLD_CM = 40.0 
    left_wall = avg_left_dist <= SHARP_WALL_THRESHOLD_CM
    print(f"[LEFT_SHARP] {avg_left_dist:.2f}cm -> {'WALL' if left_wall else 'OPEN'}")

    graph_mapper.update_current_node_walls_absolute(
        left_wall=left_wall, right_wall=None, front_wall=front_wall, back_wall_scan=False
    )
    current_node.sensorReadings = {'front': front_dist, 'left': avg_left_dist}
    print(f"✅ Simplified Scan complete. Walls Stored (Abs): {current_node.walls}")

def explore_autonomously_with_absolute_directions(gimbal, chassis, sensor, tof_handler, graph_mapper, movement_controller, attitude_handler, sharp_handler, max_nodes=49):
    # ... (No changes) ...
    print("\n🚀 === STARTING AUTONOMOUS EXPLORATION (Sensor Adaptor Version) ===")
    nodes_explored = 0
    gimbal.moveto(pitch=0, yaw=0, pitch_speed=150, yaw_speed=150).wait_for_completed()
    
    while nodes_explored < max_nodes:
        print(f"\n{'='*25} STEP {nodes_explored + 1} {'='*25}")
        print(f"🤖 At: {graph_mapper.currentPosition}, Facing: {graph_mapper.currentDirection}")
        
        if not movement_controller.increment_node_visit_main_exploration(attitude_handler):
            print("🔥🔥 Critical failure during main drift correction. Aborting."); break
        
        current_node = graph_mapper.get_current_node()
        if not current_node or not current_node.fullyScanned:
             scan_current_node_simplified(sensor, tof_handler, sharp_handler, graph_mapper)
        else:
             print("🔄 Node already scanned. Using cached data.")
        
        nodes_explored += 1
        current_node = graph_mapper.get_current_node()
        graph_mapper.previous_node = current_node
        graph_mapper.print_graph_summary()
        
        next_direction = graph_mapper.find_next_exploration_direction()
        
        if next_direction:
            graph_mapper.move_to_absolute_direction(next_direction, movement_controller, attitude_handler, sharp_handler)
            time.sleep(0.2)
            continue 

        if current_node.isDeadEnd:
            graph_mapper.handle_dead_end(movement_controller)
            time.sleep(0.2)
            continue
            
        print("🛑 No direct path. Backtracking to nearest frontier...")
        frontier_id, _, path = graph_mapper.find_nearest_frontier()
        
        if frontier_id and path is not None:
            if not graph_mapper.execute_path_to_frontier_with_reverse(path, movement_controller, attitude_handler): break
            continue
        else:
            print("🎉 EXPLORATION COMPLETE! No more frontiers found.")
            break
    
    print("\n🎉 === EXPLORATION FINISHED ===")
    generate_exploration_report(graph_mapper, movement_controller)

def generate_exploration_report(graph_mapper, movement_controller):
    # ... (No changes) ...
    print(f"\n{'='*30} FINAL EXPLORATION REPORT {'='*30}")
    if not movement_controller: print("Movement controller not initialized."); return
    total_nodes, dead_ends = len(graph_mapper.nodes), sum(1 for n in graph_mapper.nodes.values() if n.isDeadEnd)
    drift = movement_controller.get_drift_correction_status()
    print(f"📊 Total Nodes Explored: {total_nodes}\n🚫 Dead Ends: {dead_ends}\n🔧 Drift Corrections: {drift['total_corrections']}\n📈 Total Moves: {drift['nodes_visited']}")
    if not graph_mapper.nodes: print("No maze data to export."); return
    pos = [n.position for n in graph_mapper.nodes.values()]
    bounds = {"min_x": int(min(p[0] for p in pos)), "max_x": int(max(p[0] for p in pos)), "min_y": int(min(p[1] for p in pos)), "max_y": int(max(p[1] for p in pos))}
    maze_data = {"metadata": {"timestamp": datetime.now().isoformat(), "total_nodes": total_nodes, "boundaries": bounds, "drift_corrections": drift['total_corrections']},"nodes": {n.id: {"position": n.position, "walls": n.walls, "is_dead_end": n.isDeadEnd, "explored_directions": n.exploredDirections, "unexplored_exits": n.unexploredExits, "out_of_bounds_exits": n.outOfBoundsExits} for n in graph_mapper.nodes.values()}}
    maze_data = convert_to_json_serializable(maze_data)
    output_dir = "Assignment/data"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, "maze_data_final.json")
    with open(filename, 'w', encoding='utf-8') as f: json.dump(maze_data, f, indent=2)
    print(f"✅ Maze data exported to: {filename}\n{'='*82}")

if __name__ == '__main__':
    # ... (No changes) ...
    ep_robot = None
    
    FRONT_SENSOR_ADAPTOR_ID = 3
    FRONT_SENSOR_PORT = 1
    REAR_SENSOR_ADAPTOR_ID = 1
    REAR_SENSOR_PORT = 1
    
    print("🗺️  Defining map boundaries...")
    graph_mapper = GraphMapper(min_x=0, max_x=5, min_y=0, max_y=3)
    print(f"   - X-axis range: [{graph_mapper.min_x}, {graph_mapper.max_x}]")
    print(f"   - Y-axis range: [{graph_mapper.min_y}, {graph_mapper.max_y}]")

    attitude_handler = AttitudeHandler()
    movement_controller = None

    try:
        print("🤖 Connecting to robot...")
        ep_robot = robot.Robot(); ep_robot.initialize(conn_type="ap")
        
        ep_gimbal, ep_chassis = ep_robot.gimbal, ep_robot.chassis
        ep_sensor, ep_sensor_adaptor = ep_robot.sensor, ep_robot.sensor_adaptor
        
        tof_handler = ToFSensorHandler()
        movement_controller = MovementController(ep_chassis)
        attitude_handler.start_monitoring(ep_chassis)
        
        sharp_sensor_handler = SharpSensorHandler(
            sensor_adaptor=ep_sensor_adaptor, 
            front_id=FRONT_SENSOR_ADAPTOR_ID, 
            front_port=FRONT_SENSOR_PORT, 
            rear_id=REAR_SENSOR_ADAPTOR_ID, 
            rear_port=REAR_SENSOR_PORT
        )
        
        explore_autonomously_with_absolute_directions(
            ep_gimbal, ep_chassis, ep_sensor, 
            tof_handler, graph_mapper, movement_controller, 
            attitude_handler, sharp_sensor_handler, 
            max_nodes=49
        )

    except KeyboardInterrupt: print("\n⚠️ User interrupted exploration.")
    except Exception as e: print(f"\n❌ An error occurred: {e}"); import traceback; traceback.print_exc()
    finally:
        if ep_robot:
            print("🔌 Cleaning up and closing connection...")
            try:
                if movement_controller and graph_mapper:
                    generate_exploration_report(graph_mapper, movement_controller)
                if attitude_handler.is_monitoring:
                    attitude_handler.stop_monitoring(ep_chassis)
                if movement_controller:
                    movement_controller.cleanup()
            except Exception as report_err: 
                print(f"Error during final reporting/cleanup: {report_err}")
            finally: 
                ep_robot.close()
                print("🔌 Connection closed.")