import time
import robomaster
from robomaster import robot
import numpy as np
from scipy.ndimage import median_filter
from datetime import datetime
import json
import os
from collections import deque

# Global variables representing the robot's state in a fixed world frame
ROBOT_FACE = 1 
CURRENT_TARGET_YAW = 0.0

# Helper function to convert NumPy types to native Python types for JSON serialization
def convert_to_json_serializable(obj):
    if isinstance(obj, np.bool_): return bool(obj)
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_to_json_serializable(i) for i in obj]
    return obj

class ExplorationLogger:
    """Logs exploration history for animation"""
    def __init__(self):
        self.history = []
        self.step_counter = 0
        
    def log_step(self, position, direction, action_type, walls=None, sensor_readings=None, 
                 robot_face=None, target_yaw=None, frontier_queue=None):
        """Log a single exploration step"""
        step_data = {
            'step': self.step_counter,
            'timestamp': datetime.now().isoformat(),
            'position': position,
            'direction': direction,
            'action_type': action_type,  # 'scan', 'move', 'rotate', 'backtrack', 'dead_end'
            'walls': walls.copy() if walls else None,
            'sensor_readings': sensor_readings.copy() if sensor_readings else None,
            'robot_face': robot_face,
            'target_yaw': target_yaw,
            'frontier_queue': frontier_queue.copy() if frontier_queue else []
        }
        self.history.append(step_data)
        self.step_counter += 1
        
    def get_history(self):
        return self.history

class MovementTracker:
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
        self.yaw_tolerance = 3
        self.is_monitoring = False

    def attitude_handler(self, attitude_info):
        if self.is_monitoring:
            self.current_yaw = attitude_info[0]
            print(f"\r🧭 Current Yaw: {self.current_yaw:.1f}°", end="", flush=True)

    def start_monitoring(self, chassis):
        self.is_monitoring = True
        chassis.sub_attitude(freq=20, callback=self.attitude_handler)

    def stop_monitoring(self, chassis):
        self.is_monitoring = False
        try: chassis.unsub_attitude()
        except Exception: pass

    def normalize_angle(self, angle):
        while angle > 180: angle -= 360
        while angle <= -180: angle += 360
        return angle

    def is_at_target_yaw(self, target_yaw=0.0):
        normalized_current = self.normalize_angle(self.current_yaw)
        normalized_target = self.normalize_angle(target_yaw)
        diff = abs(self.normalize_angle(normalized_current - normalized_target))
        if abs(normalized_target) == 180:
            return 180 - abs(normalized_current) <= self.yaw_tolerance
        return diff <= self.yaw_tolerance

    def correct_yaw_to_target(self, chassis, target_yaw=0.0):
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
        self.chassis = chassis
        self.current_x, self.current_y = 0.0, 0.0
        self.KP, self.KI, self.KD = 2.08, 0.25, 10
        self.RAMP_UP_TIME = 0.7
        self.MOVE_TIMEOUT = 4.0 
        self.movement_tracker = MovementTracker()
        self.nodes_visited_count, self.total_drift_corrections, self.last_correction_at = 0, 0, 0
        self.DRIFT_CORRECTION_INTERVAL, self.DRIFT_CORRECTION_ANGLE = 10, 2
        self.chassis.sub_position(freq=20, callback=self.position_handler)
        time.sleep(0.1)
    
    def position_handler(self, position_info):
        self.current_x, self.current_y = position_info[0], position_info[1]

    def increment_node_visit_for_backtrack_with_correction(self, attitude_handler):
        self.nodes_visited_count += 1
        print(f"📊 Backtrack Node Count: {self.nodes_visited_count}")
        if (self.nodes_visited_count % self.DRIFT_CORRECTION_INTERVAL == 0 and 
            self.nodes_visited_count != self.last_correction_at):
            print(f"🔧 BACKTRACK DRIFT CORRECTION TRIGGERED!")
            return self.perform_attitude_drift_correction(attitude_handler)
        return True

    def increment_node_visit_main_exploration(self, attitude_handler):
        self.nodes_visited_count += 1
        print(f"📊 Main Exploration Node Count: {self.nodes_visited_count}")
        if (self.nodes_visited_count % self.DRIFT_CORRECTION_INTERVAL == 0 and 
            self.nodes_visited_count != self.last_correction_at):
            print(f"🔧 MAIN EXPLORATION DRIFT CORRECTION TRIGGERED!")
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
        if allow_yaw_correction and (self.movement_tracker.has_consecutive_forward_moves(2) or self.movement_tracker.has_consecutive_backward_moves(2)):
            print("⚠️ Consecutive moves detected! Re-checking yaw...")
            attitude_handler.correct_yaw_to_target(self.chassis, attitude_handler.normalize_angle(CURRENT_TARGET_YAW))
        pid = PID(Kp=self.KP, Ki=self.KI, Kd=self.KD, setpoint=target_distance)
        start_time, last_time = time.time(), time.time()
        start_position = self.current_x if axis == 'x' else self.current_y
        max_speed = 1.5
        print(f"🚀 Moving {'FORWARD' if direction == 1 else 'BACKWARD'} {target_distance}m on {axis}-axis")
        target_reached = False
        while not target_reached:
            if time.time() - start_time > self.MOVE_TIMEOUT:
                print(f"🔥🔥 MOVEMENT TIMEOUT! Unable to reach target in {self.MOVE_TIMEOUT} seconds. Stopping.")
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
        if target_reached: print(f"✅ Target reached!")
        else: print(f"⚠️ Target possibly not reached due to timeout.")

    def rotate_90_degrees_right(self, attitude_handler):
        global CURRENT_TARGET_YAW
        print("🔄 Rotating 90° RIGHT...")
        self.movement_tracker.record_movement('rotation'); time.sleep(0.2)
        CURRENT_TARGET_YAW += 90
        target_angle = attitude_handler.normalize_angle(CURRENT_TARGET_YAW)
        attitude_handler.correct_yaw_to_target(self.chassis, target_angle); time.sleep(0.2)

    def rotate_90_degrees_left(self, attitude_handler):
        global CURRENT_TARGET_YAW
        print("🔄 Rotating 90° LEFT...")
        self.movement_tracker.record_movement('rotation'); time.sleep(0.2)
        CURRENT_TARGET_YAW -= 90
        target_angle = attitude_handler.normalize_angle(CURRENT_TARGET_YAW)
        attitude_handler.correct_yaw_to_target(self.chassis, target_angle); time.sleep(0.2)

    def reverse_from_dead_end(self):
        global ROBOT_FACE
        print("🔙 DEAD END DETECTED - Reversing...")
        axis_test = 'y' if ROBOT_FACE % 2 == 0 else 'x'
        self.move_forward_with_pid(0.6, axis_test, direction=-1)
        print("✅ Reverse from dead end completed!")

    def reverse_to_previous_node(self):
        global ROBOT_FACE
        print("🔙 BACKTRACKING - Reversing to previous node...")
        axis_test = 'y' if ROBOT_FACE % 2 == 0 else 'x'
        self.move_forward_with_pid(0.6, axis_test, direction=-1, allow_yaw_correction=False)
        print("✅ Reverse backtrack completed!")
    
    def cleanup(self):
        try: self.chassis.unsub_position()
        except: pass
        
    def get_drift_correction_status(self):
        return {'nodes_visited': self.nodes_visited_count,'next_correction_at': ((self.nodes_visited_count // self.DRIFT_CORRECTION_INTERVAL) + 1) * self.DRIFT_CORRECTION_INTERVAL,'total_corrections': self.total_drift_corrections,'last_correction_at': self.last_correction_at}

class GraphNode:
    def __init__(self, node_id, position):
        self.id, self.position = node_id, position
        self.walls = {'north': False, 'south': False, 'east': False, 'west': False}
        self.neighbors = {'north': None, 'south': None, 'east': None, 'west': None}
        self.visited, self.fullyScanned, self.isDeadEnd = True, False, False
        self.exploredDirections, self.unexploredExits = [], []
        self.sensorReadings, self.initialScanDirection = {}, None
        
        # Animation properties
        self.discovery_step = 0  # Step number when this node was first discovered
        self.scan_step = 0       # Step number when this node was fully scanned
        self.last_visited_step = 0  # Last step when robot was at this node
        self.visit_count = 0     # Number of times robot visited this node
        self.path_history = []   # History of paths taken from this node

class GraphMapper:
    def __init__(self):
        self.nodes, self.currentPosition, self.currentDirection = {}, (0, 0), 'north'
        self.frontierQueue, self.previous_node = [], None
        self.find_next_exploration_direction = self.find_next_exploration_direction_with_priority
        
        # Animation properties
        self.exploration_path = []  # Complete path taken during exploration
        self.decision_points = []   # Points where robot made important decisions
        self.backtrack_segments = []  # Segments where robot backtracked
        self.current_step = 0

    def get_node_id(self, position): return f"{position[0]}_{position[1]}"
    
    def create_node(self, position):
        node_id = self.get_node_id(position)
        if node_id not in self.nodes:
            node = GraphNode(node_id, position)
            node.initialScanDirection = self.currentDirection
            node.discovery_step = self.current_step
            self.nodes[node_id] = node
        return self.nodes[node_id]

    def get_current_node(self): return self.nodes.get(self.get_node_id(self.currentPosition))
    
    def update_current_node_walls_absolute(self, left_wall, right_wall, front_wall, back_wall=None):
        current_node = self.get_current_node()
        if not current_node: return
        dir_map = {
            'north': {'front': 'north', 'left': 'west', 'right': 'east', 'back': 'south'},
            'south': {'front': 'south', 'left': 'east', 'right': 'west', 'back': 'north'},
            'east': {'front': 'east', 'left': 'north', 'right': 'south', 'back': 'west'},
            'west': {'front': 'west', 'left': 'south', 'right': 'north', 'back': 'east'}
        }[self.currentDirection]
        current_node.walls[dir_map['front']] = front_wall
        current_node.walls[dir_map['left']] = left_wall
        current_node.walls[dir_map['right']] = right_wall
        if back_wall is not None: current_node.walls[dir_map['back']] = back_wall
        current_node.fullyScanned = True
        current_node.scan_step = self.current_step
        self.update_unexplored_exits_with_priority(current_node)
        self.build_connections()

    def update_unexplored_exits_with_priority(self, node):
        node.unexploredExits = []
        x, y = node.position
        dir_map = {
            'north': {'front': 'north', 'left': 'west', 'right': 'east', 'back': 'south'},
            'south': {'front': 'south', 'left': 'east', 'right': 'west', 'back': 'north'},
            'east': {'front': 'east', 'left': 'north', 'right': 'south', 'back': 'west'},
            'west': {'front': 'west', 'left': 'south', 'right': 'north', 'back': 'east'}
        }[self.currentDirection]
        priority_order = ['left', 'front', 'right', 'back']
        possible_directions = {'north': (x, y + 1), 'south': (x, y - 1), 'east': (x + 1, y), 'west': (x - 1, y)}
        for rel_dir in priority_order:
            abs_dir = dir_map.get(rel_dir)
            if not abs_dir: continue
            target_pos = possible_directions.get(abs_dir)
            if not target_pos: continue
            target_node = self.nodes.get(self.get_node_id(target_pos))
            is_blocked = node.walls.get(abs_dir, True)
            already_explored = abs_dir in node.exploredDirections
            target_fully_explored = target_node.fullyScanned if target_node else False
            if not is_blocked and not already_explored and not target_fully_explored:
                node.unexploredExits.append(abs_dir)
        
        # CRITICAL FIX: Update frontier queue properly
        if node.unexploredExits:
            if node.id not in self.frontierQueue:
                self.frontierQueue.append(node.id)
        else:
            # Remove from frontier if no unexplored exits
            if node.id in self.frontierQueue:
                self.frontierQueue.remove(node.id)
                print(f"🗑️ Removed {node.id} from frontier queue (no unexplored exits)")
        
        node.isDeadEnd = sum(1 for w in node.walls.values() if w) >= 3
        if node.isDeadEnd and node.id in self.frontierQueue:
            self.frontierQueue.remove(node.id)
            print(f"🗑️ Removed {node.id} from frontier queue (dead end)")

    def build_connections(self):
        for node_id, node in self.nodes.items():
            x, y = node.position
            for d, n_pos in {'north':(x,y+1),'south':(x,y-1),'east':(x+1,y),'west':(x-1,y)}.items():
                if self.get_node_id(n_pos) in self.nodes: node.neighbors[d] = self.nodes[self.get_node_id(n_pos)]

    def get_next_position(self, direction):
        x, y = self.currentPosition
        if direction == 'north': return (x, y + 1)
        elif direction == 'south': return (x, y - 1)
        elif direction == 'east': return (x + 1, y)
        elif direction == 'west': return (x - 1, y)
        return self.currentPosition
    
    def rotate_to_absolute_direction(self, target_direction, movement_controller, attitude_handler):
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
        print(f"🚫 DEAD END HANDLER ACTIVATED at {self.currentPosition}")
        dead_end_facing = self.currentDirection
        
        # Log decision point
        self.decision_points.append({
            'step': self.current_step,
            'position': self.currentPosition,
            'decision_type': 'dead_end',
            'action': 'reverse'
        })
        
        movement_controller.reverse_from_dead_end()
        rev_map = {'north': 'south', 'south': 'north', 'east': 'west', 'west': 'east'}
        self.currentPosition = self.get_next_position(rev_map[dead_end_facing])
        
        # Add to exploration path
        self.exploration_path.append({
            'step': self.current_step,
            'from_position': self.get_next_position(dead_end_facing),
            'to_position': self.currentPosition,
            'action_type': 'dead_end_reverse',
            'direction': rev_map[dead_end_facing]
        })
        
        previous_node = self.get_current_node()
        if previous_node:
            direction_to_dead_end = dead_end_facing
            print(f"🧠 Updating node {previous_node.id} to mark path '{direction_to_dead_end}' as explored dead end.")
            if direction_to_dead_end not in previous_node.exploredDirections:
                previous_node.exploredDirections.append(direction_to_dead_end)
            if direction_to_dead_end in previous_node.unexploredExits:
                previous_node.unexploredExits.remove(direction_to_dead_end)
            # CRITICAL FIX: Update frontier status after marking explored
            self.update_unexplored_exits_with_priority(previous_node)
        return True
    
    def move_to_absolute_direction(self, target_direction, movement_controller, attitude_handler):
        global ROBOT_FACE
        print(f"🎯 Moving to ABSOLUTE direction: {target_direction}")
        
        old_position = self.currentPosition
        self.rotate_to_absolute_direction(target_direction, movement_controller, attitude_handler)
        axis_test = 'y' if ROBOT_FACE % 2 == 0 else 'x'
        movement_controller.move_forward_with_pid(0.6, axis_test, direction=1)
        self.currentPosition = self.get_next_position(target_direction)
        
        # Add to exploration path
        self.exploration_path.append({
            'step': self.current_step,
            'from_position': old_position,
            'to_position': self.currentPosition,
            'action_type': 'explore_move',
            'direction': target_direction
        })
        
        if self.previous_node and target_direction not in self.previous_node.exploredDirections:
            self.previous_node.exploredDirections.append(target_direction)
            # Add to path history
            self.previous_node.path_history.append({
                'step': self.current_step,
                'direction': target_direction,
                'destination': self.currentPosition
            })
            # CRITICAL FIX: Update frontier status after marking explored
            self.update_unexplored_exits_with_priority(self.previous_node)
        print(f"✅ Successfully moved to {self.currentPosition}")
        return True

    def reverse_to_absolute_direction(self, target_direction, movement_controller, attitude_handler):
        print(f"🔙 BACKTRACK: Reversing towards ABSOLUTE direction: {target_direction}")
        
        old_position = self.currentPosition
        rev_map = {'north': 'south', 'south': 'north', 'east': 'west', 'west': 'east'}
        required_facing = rev_map[target_direction]
        self.rotate_to_absolute_direction(required_facing, movement_controller, attitude_handler)
        movement_controller.reverse_to_previous_node()
        self.currentPosition = self.get_next_position(target_direction)
        
        # Add to backtrack segments
        self.backtrack_segments.append({
            'step': self.current_step,
            'from_position': old_position,
            'to_position': self.currentPosition,
            'direction': target_direction
        })
        
        # Add to exploration path
        self.exploration_path.append({
            'step': self.current_step,
            'from_position': old_position,
            'to_position': self.currentPosition,
            'action_type': 'backtrack',
            'direction': target_direction
        })
        
        print(f"✅ Successfully reversed to {self.currentPosition}, NOW FACING {self.currentDirection}")
        return True

    def find_next_exploration_direction_with_priority(self):
        current_node = self.get_current_node()
        if not current_node or not current_node.unexploredExits: return None
        print(f"🔍 Finding next move from {current_node.position}. Facing: {self.currentDirection}. Unexplored: {current_node.unexploredExits}")
        dir_map = {'north':{'left':'west','right':'east','front':'north'},'south':{'left':'east','right':'west','front':'south'},'east':{'left':'north','right':'south','front':'east'},'west':{'left':'south','right':'north','front':'west'}}[self.currentDirection]
        priority_order = ['left', 'right', 'front']
        for rel_dir in priority_order:
            abs_dir = dir_map.get(rel_dir)
            if abs_dir and abs_dir in current_node.unexploredExits:
                print(f"✅ Selected priority direction: {rel_dir} ({abs_dir})")
                
                # Log decision point
                self.decision_points.append({
                    'step': self.current_step,
                    'position': self.currentPosition,
                    'decision_type': 'exploration_choice',
                    'chosen_direction': abs_dir,
                    'available_options': current_node.unexploredExits.copy()
                })
                
                return abs_dir
        fallback_dir = current_node.unexploredExits[0]
        print(f"⚠️ No priority direction. Using fallback: {fallback_dir}")
        return fallback_dir

    def find_path_to_frontier(self, target_node_id):
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
        print(f"🗺️ Executing REVERSE path to frontier: {path}")
        
        # Log decision point for backtracking
        self.decision_points.append({
            'step': self.current_step,
            'position': self.currentPosition,
            'decision_type': 'backtrack_to_frontier',
            'path': path.copy()
        })
        
        for i, step_direction in enumerate(path):
            print(f"🔍 Step {i+1}/{len(path)}: Current pos: {self.currentPosition}, reversing towards {step_direction}")
            if not self.reverse_to_absolute_direction(step_direction, movement_controller, attitude_handler): return False
            if not movement_controller.increment_node_visit_for_backtrack_with_correction(attitude_handler):
                print("🔥🔥 Critical failure during backtrack drift correction. Aborting exploration."); return False
            
            # Update current node visit info
            current_node = self.get_current_node()
            if current_node:
                current_node.last_visited_step = self.current_step
                current_node.visit_count += 1
                
            time.sleep(0.2)
        print(f"✅ Successfully reached frontier at {self.currentPosition}")
        return True
    
    def find_nearest_frontier(self):
        print("🔍 Finding nearest frontier...")
        print(f"🔍 Current frontier queue: {self.frontierQueue}")
        
        # CRITICAL FIX: Clean up frontier queue before searching
        valid_frontiers = []
        for frontier_id in self.frontierQueue[:]:  # Copy to avoid modification during iteration
            frontier_node = self.nodes.get(frontier_id)
            if frontier_node and frontier_node.unexploredExits:
                valid_frontiers.append(frontier_id)
            else:
                self.frontierQueue.remove(frontier_id)
                print(f"🗑️ Cleaned up invalid frontier: {frontier_id}")
        
        if not valid_frontiers:
            print(f"❌ No valid frontiers found after cleanup!")
            return None, None, None
            
        best_frontier, best_direction, shortest_path, min_distance = None, None, None, float('inf')
        for frontier_id in valid_frontiers:
            path = self.find_path_to_frontier(frontier_id)
            if path is not None and len(path) < min_distance:
                frontier_node = self.nodes.get(frontier_id)
                if frontier_node and frontier_node.unexploredExits:
                    min_distance, best_frontier, best_direction, shortest_path = len(path), frontier_id, frontier_node.unexploredExits[0], path
        
        if best_frontier: 
            print(f"🏆 SELECTED FRONTIER: {best_frontier} at distance {min_distance} via path {shortest_path}")
        else: 
            print(f"❌ No reachable frontiers found!")
        return best_frontier, best_direction, shortest_path

    def print_graph_summary(self):
        print("\n" + "="*40 + " GRAPH SUMMARY " + "="*40)
        print(f"🤖 Pos: {self.currentPosition}, Dir: {self.currentDirection}, Nodes: {len(self.nodes)}, Frontiers: {self.frontierQueue}")
        for node_id, node in sorted(self.nodes.items()):
            walls_str = ', '.join([d for d, w in sorted(node.walls.items()) if w])
            print(f"  📍 {node.id:<5} @{str(node.position):<8} | Walls: [{walls_str:<25}] | Unexplored: {node.unexploredExits} | DeadEnd: {node.isDeadEnd}")
        print("="*95 + "\n")
        
    def update_step_counter(self):
        """Update step counter and node visit info"""
        self.current_step += 1
        current_node = self.get_current_node()
        if current_node:
            current_node.last_visited_step = self.current_step
            current_node.visit_count += 1

class ToFSensorHandler:
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

def scan_current_node_absolute(gimbal, chassis, sensor, tof_handler, graph_mapper, exploration_logger):
    print(f"\n🗺️ === Scanning Node at {graph_mapper.currentPosition} ===")
    current_node = graph_mapper.create_node(graph_mapper.currentPosition)
    graph_mapper.update_step_counter()
    
    if current_node.fullyScanned:
        print("📄 Node already scanned. Using cached data.")
        exploration_logger.log_step(
            position=graph_mapper.currentPosition, direction=graph_mapper.currentDirection,
            action_type='cached_scan', walls=current_node.walls, sensor_readings=current_node.sensorReadings,
            robot_face=ROBOT_FACE, target_yaw=CURRENT_TARGET_YAW, frontier_queue=graph_mapper.frontierQueue
        )
        return
        
    chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0); time.sleep(0.2)
    speed, scan_results = 480, {}
    
    # --- START OF IMPROVEMENTS ---
    # 1. Subscribe to the sensor ONCE at the beginning of the scan.
    tof_handler.tof_buffer.clear()
    tof_handler.current_readings.clear()
    tof_handler.collecting_data = True
    sensor.sub_distance(freq=25, callback=tof_handler.tof_data_handler)
    time.sleep(0.1) # Give a moment for the subscription to start

    try:
        # Helper function to perform a single scan direction robustly
        def perform_scan(direction_name, yaw_angle):
            try:
                # 2. Add a timeout to wait_for_completed() to prevent infinite hangs.
                gimbal.moveto(pitch=0, yaw=yaw_angle, pitch_speed=speed, yaw_speed=speed).wait_for_completed(timeout=3)
                time.sleep(0.3) # Wait for readings to stabilize
                
                if not tof_handler.current_readings:
                    print(f"[{direction_name.upper():<5}] NO READINGS -> WALL (FAIL-SAFE)")
                    return 0.0, True
                    
                avg_distance = np.mean(tof_handler.current_readings)
                is_wall = 0 < avg_distance <= tof_handler.WALL_THRESHOLD
                print(f"[{direction_name.upper():<5}] {avg_distance:.2f}cm -> {'WALL' if is_wall else 'OPEN'}")
                return avg_distance, is_wall
            except Exception as e:
                print(f"❌ Error during '{direction_name}' scan: {e}. Assuming it's a wall.")
                return 0.0, True # Fail-safe: if gimbal move fails, assume it's a wall.
            finally:
                tof_handler.current_readings.clear() # Clear readings for the next direction

        # Perform scans for all directions
        front_dist, front_wall = perform_scan('front', 0); scan_results['front'] = front_dist
        if 0 < front_dist <= 19.0: chassis.move(x=-(23 - front_dist)/100, y=0, xy_speed=0.2).wait_for_completed(timeout=2)

        left_dist, left_wall = perform_scan('left', -90); scan_results['left'] = left_dist
        if 0 < left_dist < 15: chassis.move(x=0.01, y=(20 - left_dist)/100, xy_speed=0.5).wait_for_completed(timeout=2)

        right_dist, right_wall = perform_scan('right', 90); scan_results['right'] = right_dist
        if 0 < right_dist < 15: chassis.move(x=0.01, y=-(21 - right_dist)/100, xy_speed=0.5).wait_for_completed(timeout=2)

        back_wall_scan = None
        if graph_mapper.currentPosition == (0, 0):
            back_dist, back_wall_scan = perform_scan('back', 180); scan_results['back'] = back_dist

    finally:
        # 3. Unsubscribe from the sensor ONCE at the very end, inside a 'finally' block.
        # This ensures it always runs, even if an error occurs.
        tof_handler.collecting_data = False
        try:
            sensor.unsub_distance()
        except Exception:
            pass # Ignore errors on unsubscribing
        
        # Recenter gimbal
        try:
            gimbal.moveto(pitch=0, yaw=0, pitch_speed=speed, yaw_speed=speed).wait_for_completed(timeout=3)
        except Exception as e:
            print(f"⚠️ Could not recenter gimbal: {e}")
    # --- END OF IMPROVEMENTS ---

    graph_mapper.update_current_node_walls_absolute(left_wall, right_wall, front_wall, back_wall_scan)
    current_node.sensorReadings = scan_results
    
    exploration_logger.log_step(
        position=graph_mapper.currentPosition, direction=graph_mapper.currentDirection,
        action_type='scan', walls=current_node.walls, sensor_readings=scan_results,
        robot_face=ROBOT_FACE, target_yaw=CURRENT_TARGET_YAW, frontier_queue=graph_mapper.frontierQueue
    )
    
    print(f"✅ Scan complete. Walls Stored (Abs): {current_node.walls}")

def explore_autonomously_with_absolute_directions(gimbal, chassis, sensor, tof_handler, graph_mapper, movement_controller, attitude_handler, max_nodes=49):
    print("\n🚀 === STARTING AUTONOMOUS EXPLORATION (ENHANCED WITH ANIMATION DATA) ===")
    exploration_logger = ExplorationLogger()
    nodes_explored = 0
    
    while nodes_explored < max_nodes:
        print(f"\n{'='*25} STEP {nodes_explored + 1} {'='*25}")
        print(f"🤖 At: {graph_mapper.currentPosition}, Facing: {graph_mapper.currentDirection} (ROBOT_FACE: {ROBOT_FACE})")
        drift = movement_controller.get_drift_correction_status()
        print(f"🔧 Drift Status: {drift['total_corrections']} corrections. Next check at node {drift['next_correction_at']}.")
        
        if not movement_controller.increment_node_visit_main_exploration(attitude_handler):
            print("🔥🔥 Critical failure during main drift correction. Aborting exploration."); break
        
        current_node = graph_mapper.get_current_node()
        if not current_node or not current_node.fullyScanned:
             scan_current_node_absolute(gimbal, chassis, sensor, tof_handler, graph_mapper, exploration_logger)
        else:
             print("📄 Node already scanned. Using cached data.")
             # Log cached scan
             exploration_logger.log_step(
                position=graph_mapper.currentPosition,
                direction=graph_mapper.currentDirection,
                action_type='cached_scan',
                walls=current_node.walls,
                sensor_readings=current_node.sensorReadings,
                robot_face=ROBOT_FACE,
                target_yaw=CURRENT_TARGET_YAW,
                frontier_queue=graph_mapper.frontierQueue
            )
        
        nodes_explored += 1
        current_node = graph_mapper.get_current_node()
        graph_mapper.previous_node = current_node
        graph_mapper.print_graph_summary()
        
        # Find next exploration direction
        next_direction = graph_mapper.find_next_exploration_direction()
        
        if next_direction:
            # Log movement step
            exploration_logger.log_step(
                position=graph_mapper.currentPosition,
                direction=graph_mapper.currentDirection,
                action_type='move',
                walls=current_node.walls if current_node else None,
                robot_face=ROBOT_FACE,
                target_yaw=CURRENT_TARGET_YAW,
                frontier_queue=graph_mapper.frontierQueue
            )
            
            graph_mapper.move_to_absolute_direction(next_direction, movement_controller, attitude_handler)
            graph_mapper.update_step_counter()
            time.sleep(0.2)
            continue 

        if current_node.isDeadEnd:
            # Log dead end step
            exploration_logger.log_step(
                position=graph_mapper.currentPosition,
                direction=graph_mapper.currentDirection,
                action_type='dead_end',
                walls=current_node.walls,
                robot_face=ROBOT_FACE,
                target_yaw=CURRENT_TARGET_YAW,
                frontier_queue=graph_mapper.frontierQueue
            )
            
            graph_mapper.handle_dead_end(movement_controller)
            graph_mapper.update_step_counter()
            time.sleep(0.2)
            continue
            
        print("🛑 No direct path from here. Initiating backtracking to nearest frontier...")
        frontier_id, _, path = graph_mapper.find_nearest_frontier()
        
        if frontier_id and path is not None:
            # Log backtrack initiation
            exploration_logger.log_step(
                position=graph_mapper.currentPosition,
                direction=graph_mapper.currentDirection,
                action_type='backtrack',
                walls=current_node.walls if current_node else None,
                robot_face=ROBOT_FACE,
                target_yaw=CURRENT_TARGET_YAW,
                frontier_queue=graph_mapper.frontierQueue
            )
            
            if not graph_mapper.execute_path_to_frontier_with_reverse(path, movement_controller, attitude_handler):
                break
            graph_mapper.update_step_counter()
            continue
        else:
            print("🎉 EXPLORATION COMPLETE! No more frontiers found.")
            
            # Log completion
            exploration_logger.log_step(
                position=graph_mapper.currentPosition,
                direction=graph_mapper.currentDirection,
                action_type='complete',
                walls=current_node.walls if current_node else None,
                robot_face=ROBOT_FACE,
                target_yaw=CURRENT_TARGET_YAW,
                frontier_queue=graph_mapper.frontierQueue
            )
            break
    
    print("\n🎉 === EXPLORATION FINISHED ===")
    generate_exploration_report(graph_mapper, movement_controller, exploration_logger)

def generate_exploration_report(graph_mapper, movement_controller, exploration_logger):
    print(f"\n{'='*30} FINAL EXPLORATION REPORT {'='*30}")
    if not movement_controller:
        print("Movement controller not initialized. Cannot generate report."); return
    total_nodes = len(graph_mapper.nodes)
    dead_ends = sum(1 for node in graph_mapper.nodes.values() if node.isDeadEnd)
    drift = movement_controller.get_drift_correction_status()
    print(f"📊 Total Nodes Explored: {total_nodes}\n🚫 Dead Ends Found: {dead_ends}\n🔧 Total Drift Corrections: {drift['total_corrections']}\n📈 Total Nodes Visited (incl. backtrack): {drift['nodes_visited']}")
    if not graph_mapper.nodes: print("No maze data to export."); return
    pos = [n.position for n in graph_mapper.nodes.values()]
    bounds = {"min_x": int(min(p[0] for p in pos)), "max_x": int(max(p[0] for p in pos)), "min_y": int(min(p[1] for p in pos)), "max_y": int(max(p[1] for p in pos))}
    
    # Enhanced maze data with animation properties
    maze_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(), 
            "total_nodes": total_nodes, 
            "boundaries": bounds, 
            "drift_corrections": drift['total_corrections'],
            "total_steps": exploration_logger.step_counter,
            "total_backtrack_segments": len(graph_mapper.backtrack_segments),
            "total_decision_points": len(graph_mapper.decision_points)
        },
        "nodes": {
            node.id: {
                "position": node.position, 
                "walls": node.walls,
                "is_dead_end": node.isDeadEnd, 
                "explored_directions": node.exploredDirections,
                "unexplored_exits": node.unexploredExits,
                "discovery_step": node.discovery_step,
                "scan_step": node.scan_step,
                "last_visited_step": node.last_visited_step,
                "visit_count": node.visit_count,
                "path_history": node.path_history,
                "sensor_readings": node.sensorReadings,
                "initial_scan_direction": node.initialScanDirection
            } for node in graph_mapper.nodes.values()
        },
        "exploration_path": graph_mapper.exploration_path,
        "decision_points": graph_mapper.decision_points,
        "backtrack_segments": graph_mapper.backtrack_segments,
        "exploration_history": exploration_logger.get_history()
    }
    
    maze_data = convert_to_json_serializable(maze_data)
    filename = "Assignment/data/maze_data_enhanced.json"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w', encoding='utf-8') as f: 
        json.dump(maze_data, f, indent=2)
    print(f"✅ Enhanced maze data successfully exported to: {filename}\n{'='*82}")

if __name__ == '__main__':
    ep_robot = None
    graph_mapper, movement_controller, attitude_handler = GraphMapper(), None, AttitudeHandler()
    try:
        print("🤖 Connecting to robot...")
        ep_robot = robot.Robot(); ep_robot.initialize(conn_type="ap")
        ep_gimbal, ep_chassis, ep_sensor = ep_robot.gimbal, ep_robot.chassis, ep_robot.sensor
        tof_handler = ToFSensorHandler()
        movement_controller = MovementController(ep_chassis)
        attitude_handler.start_monitoring(ep_chassis)
        ep_gimbal.recenter(pitch_speed=100, yaw_speed=100).wait_for_completed()
        explore_autonomously_with_absolute_directions(ep_gimbal, ep_chassis, ep_sensor, tof_handler, graph_mapper, movement_controller, attitude_handler, max_nodes=49)
    except KeyboardInterrupt: print("\nâš ï¸ User interrupted exploration.")
    except Exception as e: print(f"\nâŒ An error occurred: {e}"); import traceback; traceback.print_exc()
    finally:
        if ep_robot:
            print("Cleaning up and closing connection...")
            try:
                attitude_handler.stop_monitoring(ep_chassis)
                if movement_controller:
                    movement_controller.cleanup()
                    # Note: generate_exploration_report will be called in the main exploration function
            except Exception as report_err: print(f"Error during final reporting: {report_err}")
            finally: ep_robot.close(); print("Connection closed.")