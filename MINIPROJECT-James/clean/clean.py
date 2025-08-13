import time
import robomaster
from robomaster import robot
import numpy as np
from scipy.ndimage import median_filter
from datetime import datetime
from collections import deque

ROBOT_FACE = 1
CURRENT_TARGET_YAW = 0.0

class MovementTracker:
    def __init__(self):
        self.consecutive_forward_moves = 0
        self.consecutive_backward_moves = 0
        self.last_movement_type = None
        
    def record_movement(self, movement_type):
        if movement_type == 'forward':
            if self.last_movement_type == 'forward':
                self.consecutive_forward_moves += 1
            else:
                self.consecutive_forward_moves = 1
            self.consecutive_backward_moves = 0
        elif movement_type == 'backward':
            if self.last_movement_type == 'backward':
                self.consecutive_backward_moves += 1
            else:
                self.consecutive_backward_moves = 1
            self.consecutive_forward_moves = 0
        elif movement_type == 'rotation':
            self.consecutive_forward_moves = 0
            self.consecutive_backward_moves = 0
        
        self.last_movement_type = movement_type
    
    def has_consecutive_forward_moves(self, threshold=2):
        return self.consecutive_forward_moves >= threshold
    
    def has_consecutive_backward_moves(self, threshold=2):
        return self.consecutive_backward_moves >= threshold

class AttitudeHandler:
    def __init__(self):
        self.current_yaw = 0.0
        self.yaw_tolerance = 3.0
        self.max_correction_attempts = 3
        self.min_rotation_speed = 30
        self.max_rotation_speed = 90
        self.is_monitoring = False
        
    def attitude_handler(self, attitude_info):
        if not self.is_monitoring:
            return
        yaw, pitch, roll = attitude_info
        self.current_yaw = yaw
        
    def start_monitoring(self, chassis):
        self.is_monitoring = True
        chassis.sub_attitude(freq=20, callback=self.attitude_handler)
        
    def stop_monitoring(self, chassis):
        self.is_monitoring = False
        try:
            chassis.unsub_attitude()
        except:
            pass
            
    def normalize_angle(self, angle):
        angle = angle % 360
        if angle > 180:
            angle -= 360
        elif angle <= -180:
            angle += 360
        return angle

    def calculate_angular_difference(self, current, target):
        return self.normalize_angle(target - current)

    def is_at_target_yaw(self, target_yaw=0.0):
        if abs(target_yaw) == 180:
            diff_180 = abs(self.normalize_angle(self.current_yaw - 180))
            diff_neg180 = abs(self.normalize_angle(self.current_yaw - (-180)))
            diff = min(diff_180, diff_neg180)
        else:
            diff = abs(self.calculate_angular_difference(self.current_yaw, target_yaw))
            
        return diff <= self.yaw_tolerance

    def get_adaptive_rotation_speed(self, angle_diff):
        abs_diff = abs(angle_diff)
        if abs_diff <= 5:
            return self.min_rotation_speed
        elif abs_diff >= 45:
            return self.max_rotation_speed
        else:
            ratio = (abs_diff - 5) / (45 - 5)
            speed = self.min_rotation_speed + ratio * (self.max_rotation_speed - self.min_rotation_speed)
            return int(speed)

    def correct_yaw_to_target(self, chassis, target_yaw=0.0):
        if self.is_at_target_yaw(target_yaw):
            return True
        
        correction_attempts = 0
        while correction_attempts < self.max_correction_attempts:
            correction_attempts += 1
            angle_to_rotate = self.calculate_angular_difference(self.current_yaw, target_yaw)
            
            if abs(angle_to_rotate) <= self.yaw_tolerance:
                break
                
            rotation_speed = self.get_adaptive_rotation_speed(angle_to_rotate)
            
            try:
                chassis.move(x=0, y=0, z=angle_to_rotate, z_speed=rotation_speed).wait_for_completed()
                time.sleep(0.4)
                
                final_diff = abs(self.calculate_angular_difference(self.current_yaw, target_yaw))
                if final_diff <= self.yaw_tolerance:
                    return True
                    
            except Exception as e:
                print(f"Error during rotation: {e}")
                break
        
        return True

class PID:
    def __init__(self, Kp, Ki, Kd, setpoint=0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.prev_error = 0
        self.integral = 0
        self.integral_max = 1.0

    def compute(self, current, dt):
        error = self.setpoint - current
        self.integral += error * dt
        
        if self.integral > self.integral_max:
            self.integral = self.integral_max
        elif self.integral < -self.integral_max:
            self.integral = -self.integral_max
            
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

class MovementController:
    def __init__(self, chassis):
        self.chassis = chassis
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        
        # PID Parameters
        self.KP = 2.08
        self.KI = 0.25
        self.KD = 10
        self.RAMP_UP_TIME = 0.7

        self.movement_tracker = MovementTracker()
        self.nodes_visited_count = 0
        self.DRIFT_CORRECTION_INTERVAL = 10
        self.DRIFT_CORRECTION_ANGLE = 2
        self.total_drift_corrections = 0
        self.last_correction_at = 0

        # Subscribe to position updates
        self.chassis.sub_position(freq=20, callback=self.position_handler)
        time.sleep(0.25)
    
    def position_handler(self, position_info):
        self.current_x = position_info[0]
        self.current_y = position_info[1]
        self.current_z = position_info[2]

    def increment_node_visit_for_backtrack_with_correction(self, attitude_handler):
        self.nodes_visited_count += 1
        
        if (self.nodes_visited_count % self.DRIFT_CORRECTION_INTERVAL == 0 and 
            self.nodes_visited_count != self.last_correction_at):
            
            success = self.perform_attitude_drift_correction(attitude_handler)
            self.last_correction_at = self.nodes_visited_count
            return success
        return False

    def increment_node_visit_main_exploration(self, attitude_handler):
        self.nodes_visited_count += 1
        
        if (self.nodes_visited_count % self.DRIFT_CORRECTION_INTERVAL == 0 and 
            self.nodes_visited_count != self.last_correction_at):
            
            success = self.perform_attitude_drift_correction(attitude_handler)
            self.last_correction_at = self.nodes_visited_count
            return success
        return False

    def perform_attitude_drift_correction(self, attitude_handler):
        global CURRENT_TARGET_YAW
        
        CURRENT_TARGET_YAW += self.DRIFT_CORRECTION_ANGLE
        target_after_correction = attitude_handler.normalize_angle(CURRENT_TARGET_YAW)
        
        try:
            success = attitude_handler.correct_yaw_to_target(self.chassis, target_after_correction)
            if success:
                self.total_drift_corrections += 1
                self.movement_tracker.record_movement('rotation')
                return True
            return False
        except Exception as e:
            print(f"Error during attitude drift correction: {e}")
            return False

    def move_forward_with_pid(self, target_distance, axis, direction=1):
        movement_type = 'forward' if direction == 1 else 'backward'
        self.movement_tracker.record_movement(movement_type)
        
        # Check for consecutive movements and correct if needed
        if self.movement_tracker.has_consecutive_forward_moves(2):
            target_angle = attitude_handler.normalize_angle(CURRENT_TARGET_YAW)
            attitude_handler.correct_yaw_to_target(self.chassis, target_angle)
        
        if self.movement_tracker.has_consecutive_backward_moves(2):
            target_angle = attitude_handler.normalize_angle(CURRENT_TARGET_YAW)
            attitude_handler.correct_yaw_to_target(self.chassis, target_angle)
        
        pid = PID(Kp=self.KP, Ki=self.KI, Kd=self.KD, setpoint=target_distance)
        
        start_time = time.time()
        last_time = start_time
        target_reached = False
        
        min_speed = 0.1
        max_speed = 1.5
        
        if axis == 'x':
            start_position = self.current_x
        else:
            start_position = self.current_y

        try:
            while not target_reached:
                now = time.time()
                dt = now - last_time
                last_time = now
                elapsed_time = now - start_time
                
                if axis == 'x':
                    current_position = self.current_x
                else:
                    current_position = self.current_y
                
                relative_position = abs(current_position - start_position)
                output = pid.compute(relative_position, dt)
                
                # Ramp-up logic
                if elapsed_time < self.RAMP_UP_TIME:
                    ramp_multiplier = min_speed + (elapsed_time / self.RAMP_UP_TIME) * (1.0 - min_speed)
                else:
                    ramp_multiplier = 1.0
                
                ramped_output = output * ramp_multiplier
                speed = max(min(ramped_output, max_speed), -max_speed)
                
                self.chassis.drive_speed(x=speed * direction, y=0, z=0, timeout=1)

                if abs(relative_position - target_distance) < 0.02:
                    self.chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
                    target_reached = True
                    break
                    
        except KeyboardInterrupt:
            print("Movement interrupted by user.")
    
    def rotate_90_degrees_right(self, attitude_handler=None):
        global CURRENT_TARGET_YAW
        self.movement_tracker.record_movement('rotation')
        time.sleep(0.2)
        
        CURRENT_TARGET_YAW += 90
        target_angle = attitude_handler.normalize_angle(CURRENT_TARGET_YAW)
        
        success = attitude_handler.correct_yaw_to_target(self.chassis, target_angle)
        time.sleep(0.2)

    def rotate_90_degrees_left(self, attitude_handler=None):
        global CURRENT_TARGET_YAW
        self.movement_tracker.record_movement('rotation')
        time.sleep(0.2)
        
        CURRENT_TARGET_YAW -= 90
        target_angle = attitude_handler.normalize_angle(CURRENT_TARGET_YAW)
        
        success = attitude_handler.correct_yaw_to_target(self.chassis, target_angle)
        time.sleep(0.2)
    
    def reverse_from_dead_end(self):
        global ROBOT_FACE
        self.movement_tracker.record_movement('backward')
        
        axis_test = 'x' if ROBOT_FACE % 2 == 1 else 'y'
        self.move_forward_with_pid(0.6, axis_test, direction=-1)

    def reverse_to_previous_node(self):
        global ROBOT_FACE
        self.movement_tracker.record_movement('backward')
        
        axis_test = 'x' if ROBOT_FACE % 2 == 1 else 'y'
        self.move_forward_with_pid(0.6, axis_test, direction=-1)
    
    def cleanup(self):
        try:
            self.chassis.unsub_position()
        except:
            pass

    def get_drift_correction_status(self):
        return {
            'nodes_visited': self.nodes_visited_count,
            'next_correction_at': ((self.nodes_visited_count // self.DRIFT_CORRECTION_INTERVAL) + 1) * self.DRIFT_CORRECTION_INTERVAL,
            'nodes_until_correction': self.DRIFT_CORRECTION_INTERVAL - (self.nodes_visited_count % self.DRIFT_CORRECTION_INTERVAL),
            'total_corrections': self.total_drift_corrections,
            'correction_interval': self.DRIFT_CORRECTION_INTERVAL,
            'correction_angle': self.DRIFT_CORRECTION_ANGLE,
            'last_correction_at': self.last_correction_at
        }

class GraphNode:
    def __init__(self, node_id, position):
        self.id = node_id
        self.position = position
        self.walls = {'north': False, 'south': False, 'east': False, 'west': False}
        self.neighbors = {'north': None, 'south': None, 'east': None, 'west': None}
        self.visited = True
        self.exploredDirections = []
        self.unexploredExits = []
        self.isDeadEnd = False
        self.fullyScanned = False
        self.sensorReadings = {}
        self.initialScanDirection = None

class GraphMapper:
    def __init__(self):
        self.nodes = {}
        self.currentPosition = (0, 0)
        self.currentDirection = 'north'
        self.frontierQueue = []
        self.visitedNodes = set()
        self.previous_node = None

    def get_node_id(self, position):
        return f"{position[0]}_{position[1]}"
    
    def create_node(self, position):
        node_id = self.get_node_id(position)
        if node_id not in self.nodes:
            node = GraphNode(node_id, position)
            node.initialScanDirection = self.currentDirection
            self.nodes[node_id] = node
            self.visitedNodes.add(node_id)
        return self.nodes[node_id]

    def get_current_node(self):
        node_id = self.get_node_id(self.currentPosition)
        return self.nodes.get(node_id)
    
    def update_current_node_walls_absolute(self, left_wall, right_wall, front_wall):
        current_node = self.get_current_node()
        if current_node:
            direction_map = {
                'north': {'front': 'north', 'left': 'west', 'right': 'east'},
                'south': {'front': 'south', 'left': 'east', 'right': 'west'},
                'east': {'front': 'east', 'left': 'north', 'right': 'south'},
                'west': {'front': 'west', 'left': 'south', 'right': 'north'}
            }
            
            current_mapping = direction_map[self.currentDirection]
            
            current_node.walls[current_mapping['front']] = front_wall
            current_node.walls[current_mapping['left']] = left_wall
            current_node.walls[current_mapping['right']] = right_wall
            
            current_node.fullyScanned = True
            current_node.scanTimestamp = datetime.now().isoformat()
            
            self.update_unexplored_exits_absolute(current_node)
            self.build_connections()

    def update_unexplored_exits_absolute(self, node):
        node.unexploredExits = []
        x, y = node.position
        
        possible_directions = {
            'north': (x, y + 1),
            'south': (x, y - 1),
            'east': (x + 1, y),
            'west': (x - 1, y)
        }
        
        for direction, target_pos in possible_directions.items():
            target_node_id = self.get_node_id(target_pos)
            is_blocked = node.walls.get(direction, True)
            already_explored = direction in node.exploredDirections
            target_exists = target_node_id in self.nodes
            target_fully_explored = target_exists and self.nodes[target_node_id].fullyScanned
            
            should_explore = (not is_blocked and 
                            not already_explored and 
                            (not target_exists or not target_fully_explored))
            
            if should_explore:
                node.unexploredExits.append(direction)
        
        has_unexplored = len(node.unexploredExits) > 0
        
        if has_unexplored and node.id not in self.frontierQueue:
            self.frontierQueue.append(node.id)
        elif not has_unexplored and node.id in self.frontierQueue:
            self.frontierQueue.remove(node.id)
        
        blocked_count = sum(1 for blocked in node.walls.values() if blocked)
        node.isDeadEnd = blocked_count >= 3
        
        if node.isDeadEnd and node.id in self.frontierQueue:
            self.frontierQueue.remove(node.id)
    
    def build_connections(self):
        for node_id, node in self.nodes.items():
            x, y = node.position
            
            directions = {
                'north': (x, y + 1),
                'south': (x, y - 1),
                'east': (x + 1, y),
                'west': (x - 1, y)
            }
            
            for direction, neighbor_pos in directions.items():
                neighbor_id = self.get_node_id(neighbor_pos)
                if neighbor_id in self.nodes:
                    node.neighbors[direction] = self.nodes[neighbor_id]
    
    def get_next_position(self, direction):
        x, y = self.currentPosition
        if direction == 'north':
            return (x, y + 1)
        elif direction == 'south':
            return (x, y - 1)
        elif direction == 'east':
            return (x + 1, y)
        elif direction == 'west':
            return (x - 1, y)
        return self.currentPosition
    
    def get_next_position_from(self, position, direction):
        x, y = position
        if direction == 'north':
            return (x, y + 1)
        elif direction == 'south':
            return (x, y - 1)
        elif direction == 'east':
            return (x + 1, y)
        elif direction == 'west':
            return (x - 1, y)
        return position
    
    def is_dead_end(self, node=None):
        if node is None:
            node = self.get_current_node()
        return node.isDeadEnd if node else False
    
    def can_move_to_direction_absolute(self, target_direction):
        current_node = self.get_current_node()
        if not current_node:
            return False
        
        is_blocked = current_node.walls.get(target_direction, True)
        return not is_blocked
    
    def rotate_to_absolute_direction(self, target_direction, movement_controller, attitude_handler):
        global ROBOT_FACE
        
        if self.currentDirection == target_direction:
            return
        
        direction_order = ['north', 'east', 'south', 'west']
        current_idx = direction_order.index(self.currentDirection)
        target_idx = direction_order.index(target_direction)
        
        diff = (target_idx - current_idx) % 4
        
        if diff == 1:  # Turn right
            movement_controller.rotate_90_degrees_right(attitude_handler)
            ROBOT_FACE += 1
        elif diff == 3:  # Turn left
            movement_controller.rotate_90_degrees_left(attitude_handler)
            ROBOT_FACE += 1
        elif diff == 2:  # Turn around (180°)
            movement_controller.rotate_90_degrees_right(attitude_handler)
            movement_controller.rotate_90_degrees_right(attitude_handler)
            ROBOT_FACE += 2
        
        self.currentDirection = target_direction

    def handle_dead_end(self, movement_controller):
        movement_controller.reverse_from_dead_end()
        
        reverse_direction_map = {
            'north': 'south',
            'south': 'north',
            'east': 'west',
            'west': 'east'
        }

        reverse_direction = reverse_direction_map[self.currentDirection]
        self.currentPosition = self.get_next_position(reverse_direction)
        return True
    
    def move_to_absolute_direction(self, target_direction, movement_controller, attitude_handler):
        global ROBOT_FACE
        
        if not self.can_move_to_direction_absolute(target_direction):
            return False
        
        self.rotate_to_absolute_direction(target_direction, movement_controller, attitude_handler)

        axis_test = 'x' if ROBOT_FACE % 2 == 1 else 'y'
        movement_controller.move_forward_with_pid(0.6, axis_test, direction=1)
        
        self.currentPosition = self.get_next_position(target_direction)
        
        if hasattr(self, 'previous_node') and self.previous_node:
            if target_direction not in self.previous_node.exploredDirections:
                self.previous_node.exploredDirections.append(target_direction)
            
            if target_direction in self.previous_node.unexploredExits:
                self.previous_node.unexploredExits.remove(target_direction)
        
        return True

    def reverse_to_absolute_direction(self, target_direction, movement_controller, attitude_handler):
        reverse_direction_map = {
            'north': 'south',
            'south': 'north',
            'east': 'west',
            'west': 'east'
        }
        
        required_facing_direction = reverse_direction_map[target_direction]
        self.rotate_to_absolute_direction(required_facing_direction, movement_controller, attitude_handler)
        
        movement_controller.reverse_to_previous_node()
        self.currentPosition = self.get_next_position(target_direction)
        
        return True

    def find_next_exploration_direction(self):
        current_node = self.get_current_node()
        if not current_node or self.is_dead_end(current_node):
            return None
        
        # LEFT-FIRST priority strategy
        direction_map = {
            'north': {'left': 'west', 'front': 'north', 'right': 'east', 'back': 'south'},
            'south': {'left': 'east', 'front': 'south', 'right': 'west', 'back': 'north'},
            'east': {'left': 'north', 'front': 'east', 'right': 'south', 'back': 'west'},
            'west': {'left': 'south', 'front': 'west', 'right': 'north', 'back': 'east'}
        }
        
        current_mapping = direction_map[self.currentDirection]
        priority_order = ['left', 'front', 'right', 'back']
        
        for relative_direction in priority_order:
            absolute_direction = current_mapping.get(relative_direction)
            
            if absolute_direction and absolute_direction in current_node.unexploredExits:
                if self.can_move_to_direction_absolute(absolute_direction):
                    return absolute_direction
                else:
                    current_node.unexploredExits.remove(absolute_direction)
        
        return None
    
    def find_path_to_frontier(self, target_node_id):
        if target_node_id not in self.nodes:
            return None
        
        queue = deque([(self.currentPosition, [])])
        visited = set()
        visited.add(self.currentPosition)
        
        while queue:
            current_pos, path = queue.popleft()
            current_node_id = self.get_node_id(current_pos)
            
            if current_node_id == target_node_id:
                return path
            
            if current_node_id in self.nodes:
                current_node = self.nodes[current_node_id]
                x, y = current_pos
                
                directions = {
                    'north': (x, y + 1),
                    'south': (x, y - 1), 
                    'east': (x + 1, y),
                    'west': (x - 1, y)
                }
                
                for direction, neighbor_pos in directions.items():
                    neighbor_id = self.get_node_id(neighbor_pos)
                    
                    if (neighbor_pos not in visited and 
                        neighbor_id in self.nodes and
                        not current_node.walls.get(direction, True)):
                        
                        visited.add(neighbor_pos)
                        new_path = path + [direction]
                        queue.append((neighbor_pos, new_path))
        
        return None
    
    def execute_path_to_frontier_with_reverse(self, path, movement_controller, attitude_handler):
        for i, step_direction in enumerate(path):
            success = self.reverse_to_absolute_direction(step_direction, movement_controller, attitude_handler)
            
            if not success:
                return False
            
            movement_controller.increment_node_visit_for_backtrack_with_correction(attitude_handler)
            time.sleep(0.2)
        
        return True
    
    def find_nearest_frontier(self):
        if not self.frontierQueue:
            self.rebuild_frontier_queue()
            
            if not self.frontierQueue:
                return None, None, None
        
        valid_frontiers = []
        
        for frontier_id in self.frontierQueue[:]:
            if frontier_id not in self.nodes:
                continue
                
            frontier_node = self.nodes[frontier_id]
            
            valid_exits = []
            for exit_direction in frontier_node.unexploredExits[:]:
                target_pos = self.get_next_position_from(frontier_node.position, exit_direction)
                target_node_id = self.get_node_id(target_pos)
                
                target_exists = target_node_id in self.nodes
                if target_exists:
                    target_node = self.nodes[target_node_id]
                    target_fully_explored = target_node.fullyScanned
                    
                    if not target_fully_explored:
                        valid_exits.append(exit_direction)
                else:
                    valid_exits.append(exit_direction)
            
            frontier_node.unexploredExits = valid_exits
            
            if valid_exits:
                valid_frontiers.append(frontier_id)
        
        self.frontierQueue = valid_frontiers
        
        if not valid_frontiers:
            return None, None, None
        
        # Find the nearest valid frontier
        best_frontier = None
        best_direction = None
        shortest_path = None
        min_distance = float('inf')
        
        for frontier_id in valid_frontiers:
            frontier_node = self.nodes[frontier_id]
            path = self.find_path_to_frontier(frontier_id)
            
            if path is not None:
                distance = len(path)
                
                if distance < min_distance:
                    min_distance = distance
                    best_frontier = frontier_id
                    best_direction = frontier_node.unexploredExits[0]
                    shortest_path = path
        
        return best_frontier, best_direction, shortest_path
    
    def rebuild_frontier_queue(self):
        self.frontierQueue = []
        
        for node_id, node in self.nodes.items():
            valid_exits = []
            
            if hasattr(node, 'unexploredExits'):
                for exit_direction in node.unexploredExits:
                    target_pos = self.get_next_position_from(node.position, exit_direction)
                    target_node_id = self.get_node_id(target_pos)
                    
                    target_exists = target_node_id in self.nodes
                    if target_exists:
                        target_node = self.nodes[target_node_id]
                        if not target_node.fullyScanned:
                            valid_exits.append(exit_direction)
                    else:
                        valid_exits.append(exit_direction)
            
            node.unexploredExits = valid_exits
            
            if valid_exits:
                self.frontierQueue.append(node_id)

class ToFSensorHandler:
    def __init__(self):
        self.CALIBRATION_SLOPE = 0.0894 
        self.CALIBRATION_Y_INTERCEPT = 3.8409
        self.WINDOW_SIZE = 5
        self.tof_buffer = []
        self.WALL_THRESHOLD = 50.00
        
        self.readings = {'front': [], 'left': [], 'right': []}
        self.current_scan_direction = None
        self.collecting_data = False
        
    def calibrate_tof_value(self, raw_tof_mm):
        return (self.CALIBRATION_SLOPE * raw_tof_mm) + self.CALIBRATION_Y_INTERCEPT
    
    def apply_median_filter(self, data, window_size):
        if len(data) == 0:
            return 0.0 
        if len(data) < window_size:
            return data[-1] 
        else:
            filtered = median_filter(data[-window_size:], size=window_size)
            return filtered[-1]
    
    def tof_data_handler(self, sub_info):
        if not self.collecting_data or not self.current_scan_direction:
            return
            
        raw_tof_mm = sub_info[0]
        
        if raw_tof_mm <= 0 or raw_tof_mm > 4000:
            return
            
        calibrated_tof_cm = self.calibrate_tof_value(raw_tof_mm)
        self.tof_buffer.append(calibrated_tof_cm)
        filtered_tof_cm = self.apply_median_filter(self.tof_buffer, self.WINDOW_SIZE)
        
        if len(self.tof_buffer) <= 20:
            self.readings[self.current_scan_direction].append({
                'filtered_cm': filtered_tof_cm,
                'timestamp': datetime.now().isoformat()
            })
    
    def start_scanning(self, direction):
        self.current_scan_direction = direction
        self.tof_buffer.clear()
        if direction not in self.readings:
            self.readings[direction] = []
        else:
            self.readings[direction].clear()
        self.collecting_data = True
        
    def stop_scanning(self, unsub_distance_func):
        self.collecting_data = False
        try:
            unsub_distance_func()
        except:
            pass
    
    def get_average_distance(self, direction):
        if direction not in self.readings or len(self.readings[direction]) == 0:
            return 0.0
        
        filtered_values = [reading['filtered_cm'] for reading in self.readings[direction]]
        
        if len(filtered_values) > 4:
            q1 = np.percentile(filtered_values, 25)
            q3 = np.percentile(filtered_values, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            filtered_values = [x for x in filtered_values if lower_bound <= x <= upper_bound]
        
        return np.mean(filtered_values) if filtered_values else 0.0
    
    def is_wall_detected(self, direction):
        avg_distance = self.get_average_distance(direction)
        return avg_distance <= self.WALL_THRESHOLD and avg_distance > 0

def scan_current_node_absolute(gimbal, chassis, sensor, tof_handler, graph_mapper):
    current_node = graph_mapper.create_node(graph_mapper.currentPosition)
    
    if current_node.fullyScanned:
        return current_node.sensorReadings
    
    chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
    time.sleep(0.2)
    
    speed = 480
    scan_results = {}
    
    # Scan front (0°)
    gimbal.moveto(pitch=0, yaw=0, pitch_speed=speed, yaw_speed=speed).wait_for_completed()
    time.sleep(0.2)
    
    tof_handler.start_scanning('front')
    sensor.sub_distance(freq=25, callback=tof_handler.tof_data_handler)
    time.sleep(0.2)
    tof_handler.stop_scanning(sensor.unsub_distance)
    
    front_distance = tof_handler.get_average_distance('front')
    front_wall = tof_handler.is_wall_detected('front')
    scan_results['front'] = front_distance

    if front_distance <= 19.0:
        move_distance = -(23 - front_distance)
        chassis.move(x=move_distance/100, y=0, xy_speed=0.2).wait_for_completed()
        time.sleep(0.2)

    # Scan left (-90°)
    gimbal.moveto(pitch=0, yaw=-90, pitch_speed=speed, yaw_speed=speed).wait_for_completed()
    time.sleep(0.2)
    
    tof_handler.start_scanning('left')
    sensor.sub_distance(freq=25, callback=tof_handler.tof_data_handler)
    time.sleep(0.2)
    tof_handler.stop_scanning(sensor.unsub_distance)
    
    left_distance = tof_handler.get_average_distance('left')
    left_wall = tof_handler.is_wall_detected('left')
    scan_results['left'] = left_distance
    
    if left_distance < 15:
        move_distance = 20 - left_distance
        chassis.move(x=0.01, y=move_distance/100, xy_speed=0.5).wait_for_completed()
        time.sleep(0.3)

    # Scan right (90°)
    gimbal.moveto(pitch=0, yaw=90, pitch_speed=speed, yaw_speed=speed).wait_for_completed()
    time.sleep(0.2)
    
    tof_handler.start_scanning('right')
    sensor.sub_distance(freq=25, callback=tof_handler.tof_data_handler)
    time.sleep(0.2)
    tof_handler.stop_scanning(sensor.unsub_distance)
    
    right_distance = tof_handler.get_average_distance('right')
    right_wall = tof_handler.is_wall_detected('right')
    scan_results['right'] = right_distance

    if right_distance < 15:
        move_distance = -(21 - right_distance)
        chassis.move(x=0.01, y=move_distance/100, xy_speed=0.5).wait_for_completed()
        time.sleep(0.3)

    # Return to center
    gimbal.moveto(pitch=0, yaw=0, pitch_speed=speed, yaw_speed=speed).wait_for_completed()
    time.sleep(0.2)
    
    chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0, timeout=0.1)
    time.sleep(0.2)
    
    graph_mapper.update_current_node_walls_absolute(left_wall, right_wall, front_wall)
    current_node.sensorReadings = scan_results
    
    return scan_results

def explore_autonomously_with_absolute_directions(gimbal, chassis, sensor, tof_handler, graph_mapper, movement_controller, attitude_handler, max_nodes=20):
    print("\n🚀 === STARTING AUTONOMOUS EXPLORATION ===")
    
    nodes_explored = 0
    dead_end_reversals = 0
    reverse_backtracks = 0
    
    while nodes_explored < max_nodes:
        print(f"\n{'='*50}")
        print(f"--- EXPLORATION STEP {nodes_explored + 1} ---")
        print(f"🤖 Current position: {graph_mapper.currentPosition}")
        print(f"🧭 Current direction: {graph_mapper.currentDirection}")
        
        # Drift correction check
        movement_controller.increment_node_visit_main_exploration(attitude_handler)
        
        # Check if current node needs scanning
        current_node = graph_mapper.create_node(graph_mapper.currentPosition)
        
        if not current_node.fullyScanned:
            scan_results = scan_current_node_absolute(gimbal, chassis, sensor, tof_handler, graph_mapper)
            
            # Check if dead end
            if graph_mapper.is_dead_end(current_node):
                success = graph_mapper.handle_dead_end(movement_controller)
                if success:
                    dead_end_reversals += 1
                    nodes_explored += 1
                    continue
                else:
                    break
        else:
            graph_mapper.update_unexplored_exits_absolute(current_node)
            graph_mapper.build_connections()
        
        nodes_explored += 1
        graph_mapper.previous_node = current_node
        
        # Try to find unexplored direction from current node
        next_direction = graph_mapper.find_next_exploration_direction()
        
        if next_direction:
            if graph_mapper.can_move_to_direction_absolute(next_direction):
                try:
                    success = graph_mapper.move_to_absolute_direction(next_direction, movement_controller, attitude_handler)
                    if success:
                        time.sleep(0.2)
                        continue
                    else:
                        if current_node and next_direction in current_node.unexploredExits:
                            current_node.unexploredExits.remove(next_direction)
                        continue
                except Exception as e:
                    print(f"❌ Error during movement: {e}")
                    break
            else:
                if current_node and next_direction in current_node.unexploredExits:
                    current_node.unexploredExits.remove(next_direction)
                continue
        
        # Backtracking logic
        frontier_id, frontier_direction, path = graph_mapper.find_nearest_frontier()
        
        if frontier_id and path is not None and frontier_direction:
            try:
                success = graph_mapper.execute_path_to_frontier_with_reverse(path, movement_controller, attitude_handler)
                
                if success:
                    reverse_backtracks += 1
                    time.sleep(0.2)
                    continue
                else:
                    break
            except Exception as e:
                print(f"❌ Error during backtracking: {e}")
                break
        else:
            # Final check
            graph_mapper.rebuild_frontier_queue()
            
            if graph_mapper.frontierQueue:
                continue
            else:
                print("🎉 EXPLORATION COMPLETE!")
                break
        
        if nodes_explored >= max_nodes:
            print(f"⚠️ Reached maximum nodes limit ({max_nodes})")
            break
    
    # Final statistics
    final_drift_status = movement_controller.get_drift_correction_status()
    
    print(f"\n🎉 === EXPLORATION COMPLETED ===")
    print(f"📊 PERFORMANCE SUMMARY:")
    print(f"   🗺️ Total exploration steps: {nodes_explored}")
    print(f"   📊 Total nodes visited: {final_drift_status['nodes_visited']}")
    print(f"   🔙 Dead end reversals: {dead_end_reversals}")
    print(f"   🔙 Reverse backtracks: {reverse_backtracks}")
    print(f"   🔧 Total drift corrections: {final_drift_status['total_corrections']}")

if __name__ == '__main__':
    print("🤖 Connecting to robot...")
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    
    ep_gimbal = ep_robot.gimbal
    ep_chassis = ep_robot.chassis
    ep_sensor = ep_robot.sensor
    
    # Initialize components
    tof_handler = ToFSensorHandler()
    graph_mapper = GraphMapper()
    movement_controller = MovementController(ep_chassis)
    attitude_handler = AttitudeHandler()
    attitude_handler.start_monitoring(ep_chassis)
    
    try:
        print("✅ Recalibrating gimbal...")
        ep_gimbal.recenter(pitch_speed=100, yaw_speed=100).wait_for_completed()
        ep_gimbal.moveto(pitch=0, yaw=0, pitch_speed=50, yaw_speed=50).wait_for_completed()
        time.sleep(0.3)
        
        explore_autonomously_with_absolute_directions(ep_gimbal, ep_chassis, ep_sensor, tof_handler, 
                           graph_mapper, movement_controller, attitude_handler, max_nodes=49)
            
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            ep_sensor.unsub_distance()
            movement_controller.cleanup()
            attitude_handler.stop_monitoring(ep_chassis)
        except:
            pass
        ep_robot.close()
        print("🔌 Connection closed")