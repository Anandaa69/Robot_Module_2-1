import time
import robomaster
from robomaster import robot
import numpy as np
from scipy.ndimage import median_filter
from datetime import datetime
import json
from collections import deque

ROBOT_FACE = 1 
CURRENT_TARGET_YAW = 0.0

class AttitudeHandler:
    def __init__(self):
        self.current_yaw = 0.0
        self.current_pitch = 0.0
        self.current_roll = 0.0
        self.target_yaw = 0.0
        self.yaw_tolerance = 0.5
        self.is_monitoring = False
        
    def attitude_handler(self, attitude_info):
        if not self.is_monitoring:
            return
            
        yaw, pitch, roll = attitude_info
        self.current_yaw = yaw
        self.current_pitch = pitch
        self.current_roll = roll
        print(f"\r🧭 Current chassis attitude: yaw={yaw:.1f}°, pitch={pitch:.1f}°, roll={roll:.1f}°", end="", flush=True)
        
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
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle
        
    def is_at_target_yaw(self, target_yaw=0.0):
        if abs(target_yaw) == 180:
            diff_180 = abs(self.normalize_angle(self.current_yaw - 180))
            diff_neg180 = abs(self.normalize_angle(self.current_yaw - (-180)))
            diff = min(diff_180, diff_neg180)
            target_display = f"±180"
        else:
            diff = abs(self.normalize_angle(self.current_yaw - target_yaw))
            target_display = f"{target_yaw}"
            
        is_correct = diff <= self.yaw_tolerance
        print(f"\n🎯 Yaw check: current={self.current_yaw:.1f}°, target={target_display}°, diff={diff:.1f}°, correct={is_correct}")
        return is_correct
        
    def correct_yaw_to_target(self, chassis, target_yaw=0.0):
        if self.is_at_target_yaw(target_yaw):
            print(f"✅ Chassis already at correct yaw: {self.current_yaw:.1f}° (target: {target_yaw}°)")
            return True
            
        gimbal_to_target = target_yaw - self.current_yaw
        gimbal_diff = self.normalize_angle(gimbal_to_target)
        robot_rotation = -gimbal_diff
        
        print(f"🔧 Correcting chassis yaw: from {self.current_yaw:.1f}° to {target_yaw}°")
        print(f"📐 Gimbal needs to change: {gimbal_diff:.1f}°")
        print(f"📐 Robot will rotate: {robot_rotation:.1f}°")
        
        try:
            if abs(robot_rotation) > self.yaw_tolerance:
                correction_speed = 30
                
                print(f"🔄 Rotating robot {robot_rotation:.1f}°")
                chassis.move(x=0, y=0, z=robot_rotation, z_speed=correction_speed).wait_for_completed()
                time.sleep(1.0)
            
            final_check = self.is_at_target_yaw(target_yaw)
            
            if final_check:
                print(f"✅ Successfully corrected chassis yaw to {self.current_yaw:.1f}°")
                return True
            else:
                print(f"⚠️ Chassis yaw correction incomplete: {self.current_yaw:.1f}° (target: {target_yaw}°)")
                
                remaining_gimbal = target_yaw - self.current_yaw
                remaining_diff = self.normalize_angle(remaining_gimbal)
                remaining_robot = -remaining_diff
                print(f"📐 Remaining gimbal difference: {remaining_diff:.1f}°")
                print(f"📐 Additional robot rotation needed: {remaining_robot:.1f}°")
                
                if abs(remaining_robot) > self.yaw_tolerance and abs(remaining_robot) < 45:
                    print(f"🔧 Fine-tuning robot with additional {remaining_robot:.1f}°")
                    chassis.move(x=0, y=0, z=remaining_robot, z_speed=20).wait_for_completed()
                    time.sleep(0.5)
                    return self.is_at_target_yaw(target_yaw)
                else:
                    print(f"⚠️ Remaining rotation too large ({remaining_robot:.1f}°), may need multiple corrections")
                return False
                
        except Exception as e:
            print(f"❌ Failed to correct chassis yaw: {e}")
            return False

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
        
        self.KP = 1.5
        self.KI = 0.3
        self.KD = 4
        self.RAMP_UP_TIME = 0.7
        self.ROTATE_TIME = 2.11
        self.ROTATE_LEFT_TIME = 1.9
        
        self.chassis.sub_position(freq=20, callback=self.position_handler)
        time.sleep(0.25)
    
    def position_handler(self, position_info):
        self.current_x = position_info[0]
        self.current_y = position_info[1]
        self.current_z = position_info[2]
    
    def move_forward_with_pid(self, target_distance, axis, direction=1, gimbal=None, tof_handler=None, sensor=None):
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

        direction_text = "FORWARD" if direction == 1 else "BACKWARD"
        print(f"🚀 Moving {direction_text} {target_distance}m on {axis}-axis, direction: {direction}")

        if gimbal:
            yaw_angle = 0 if direction == 1 else 180
            gimbal.moveto(pitch=0, yaw=yaw_angle, pitch_speed=300, yaw_speed=300).wait_for_completed()
            time.sleep(0.1)

        if tof_handler and sensor:
            tof_handler.start_scanning('front')
            sensor.sub_distance(freq=25, callback=tof_handler.tof_data_handler)

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

                if tof_handler and tof_handler.is_wall_detected('front'):
                    print("🛑 Obstacle detected ahead! Stopping movement.")
                    self.chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
                    break

                output = pid.compute(relative_position, dt)

                if elapsed_time < self.RAMP_UP_TIME:
                    ramp_multiplier = min_speed + (elapsed_time / self.RAMP_UP_TIME) * (1.0 - min_speed)
                else:
                    ramp_multiplier = 1.0

                ramped_output = output * ramp_multiplier
                speed = max(min(ramped_output, max_speed), -max_speed)

                self.chassis.drive_speed(x=speed * direction, y=0, z=0, timeout=1)

                if abs(relative_position - target_distance) < 0.02:
                    print(f"✅ Target reached! Final position: {current_position:.3f}")
                    self.chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
                    target_reached = True
                    break

        except KeyboardInterrupt:
            print("Movement interrupted by user.")
        finally:
            if tof_handler and sensor:
                tof_handler.stop_scanning(sensor.unsub_distance)
                
    def rotate_90_degrees_right(self, movement_controller=None, attitude_handler=None):
        global CURRENT_TARGET_YAW
        print("🔄 Rotating 90° RIGHT...")
        time.sleep(0.2)
        
        CURRENT_TARGET_YAW += 90
        target_angle = attitude_handler.normalize_angle(CURRENT_TARGET_YAW)
        
        print(f"🎯 Target yaw: {target_angle}°")
        success = attitude_handler.correct_yaw_to_target(self.chassis, target_angle)
        
        if success:
            print("✅ Right rotation completed!")
        else:
            print("⚠️ Right rotation may be incomplete")
            
        time.sleep(0.2)
        return success

    def rotate_90_degrees_left(self, movement_controller=None, attitude_handler=None):
        global CURRENT_TARGET_YAW
        print("🔄 Rotating 90° LEFT...")
        time.sleep(0.2)
        
        CURRENT_TARGET_YAW -= 90
        target_angle = attitude_handler.normalize_angle(CURRENT_TARGET_YAW)
        
        print(f"🎯 Target yaw: {target_angle}°")
        success = attitude_handler.correct_yaw_to_target(self.chassis, target_angle)
        
        if success:
            print("✅ Left rotation completed!")
        else:
            print("⚠️ Left rotation may be incomplete")
            
        time.sleep(0.2)
        return success
    
    def reverse_from_dead_end(self):
        global ROBOT_FACE
        print("🔙 DEAD END DETECTED - Reversing...")
        
        axis_test = 'x'
        if ROBOT_FACE % 2 == 0:
            axis_test = 'y'
        elif ROBOT_FACE % 2 == 1:
            axis_test = 'x'
        
        print(f"🔙 Reversing 0.6m on {axis_test}-axis")
        self.move_forward_with_pid(0.6, axis_test, direction=-1)
        print("✅ Reverse from dead end completed!")

    def reverse_to_previous_node(self):
        global ROBOT_FACE
        print("🔙 BACKTRACKING - Reversing to previous node...")
        
        axis_test = 'x'
        if ROBOT_FACE % 2 == 0:
            axis_test = 'y'
        elif ROBOT_FACE % 2 == 1:
            axis_test = 'x'
        
        print(f"🔙 Reversing 0.6m on {axis_test}-axis for backtrack")
        self.move_forward_with_pid(0.6, axis_test, direction=-1)
        print("✅ Reverse backtrack completed!")
    
    def cleanup(self):
        try:
            self.chassis.unsub_position()
        except:
            pass

class GraphNode:
    def __init__(self, node_id, position):
        self.id = node_id
        self.position = position

        self.walls = {
            'north': False,
            'south': False,
            'east': False,
            'west': False
        }

        self.neighbors = {
            'north': None,
            'south': None,
            'east': None,
            'west': None
        }

        self.visited = True
        self.visitCount = 1
        self.exploredDirections = []
        self.unexploredExits = []
        self.isDeadEnd = False

        self.fullyScanned = False
        self.scanTimestamp = None

        self.marker = False
        self.lastVisited = datetime.now().isoformat()
        self.sensorReadings = {}
        self.initialScanDirection = None

class GraphMapper:
    def __init__(self):
        self.nodes = {}
        self.currentPosition = (0, 0)
        self.currentDirection = 'north'
        self.frontierQueue = []
        self.pathStack = []
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
            current_node.lastVisited = datetime.now().isoformat()
            current_node.fullyScanned = True
            current_node.scanTimestamp = datetime.now().isoformat()
            
            # 🛠️ FIX: Use updated unexplored exits logic
            self.update_unexplored_exits_absolute(current_node)
            self.build_connections()

    # 🛠️ FIXED: Correct dead end detection and frontier management
    def update_unexplored_exits_absolute(self, node):
        """FIXED: Better dead end detection and frontier queue management"""
        print(f"\n🔄 Updating unexplored exits for node {node.id} at {node.position}")
        print(f"🧱 Wall status (absolute): {node.walls}")
        
        node.unexploredExits = []
        x, y = node.position
        
        possible_directions = {
            'north': (x, y + 1),
            'south': (x, y - 1),
            'east': (x + 1, y),
            'west': (x - 1, y)
        }
        
        # Count actual blocking walls
        wall_count = 0
        
        for direction, target_pos in possible_directions.items():
            target_node_id = self.get_node_id(target_pos)
            is_blocked = node.walls.get(direction, True)
            
            print(f"   📍 {direction}: wall={is_blocked}")
            
            if is_blocked:
                wall_count += 1
                continue
                
            already_explored = direction in node.exploredDirections
            target_exists = target_node_id in self.nodes
            target_fully_explored = False
            
            if target_exists:
                target_node = self.nodes[target_node_id]
                target_fully_explored = target_node.fullyScanned
                print(f"      🎯 Target exists: {target_exists}, fully explored: {target_fully_explored}")
            
            should_explore = (not is_blocked and 
                             not already_explored and 
                             (not target_exists or not target_fully_explored))
            
            if should_explore:
                node.unexploredExits.append(direction)
                print(f"      ✅ Added {direction} to unexplored exits")
            else:
                print(f"      ❌ {direction} not added (blocked={is_blocked}, explored={already_explored})")
        
        # 🛠️ FIXED: Correct dead end detection
        # Dead end = all 4 directions are either walls OR fully explored
        accessible_exits = []
        for direction in ['north', 'south', 'east', 'west']:
            if not node.walls.get(direction, True):  # Not a wall
                target_pos = possible_directions[direction]
                target_node_id = self.get_node_id(target_pos)
                
                if target_node_id not in self.nodes:
                    # Unexplored area - definitely not blocked
                    accessible_exits.append(direction)
                else:
                    target_node = self.nodes[target_node_id]
                    if not target_node.fullyScanned:
                        # Target exists but not fully explored
                        accessible_exits.append(direction)
        
        # Dead end only if NO accessible exits remain
        node.isDeadEnd = len(accessible_exits) == 0
        
        print(f"   🔍 Wall count: {wall_count}/4")
        print(f"   🚪 Accessible exits: {accessible_exits}")
        print(f"   🔍 Unexplored exits: {node.unexploredExits}")
        print(f"   🚫 Is dead end: {node.isDeadEnd}")
        
        # 🛠️ FIXED: Better frontier queue management
        has_unexplored = len(node.unexploredExits) > 0
        is_in_queue = node.id in self.frontierQueue
        
        if has_unexplored and not is_in_queue:
            self.frontierQueue.append(node.id)
            print(f"   🚀 ADDED to frontier queue")
        elif not has_unexplored and is_in_queue:
            self.frontierQueue.remove(node.id)
            print(f"   🧹 REMOVED from frontier queue")
        
        # Only remove from frontier if it's truly a dead end
        if node.isDeadEnd and is_in_queue:
            self.frontierQueue.remove(node.id)
            print(f"   💀 DEAD END - removed from frontier queue")

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
        if not node:
            return False
        return node.isDeadEnd

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

    def get_previous_position(self, direction):
        x, y = self.currentPosition
        if direction == 'north':
            return (x, y - 1)
        elif direction == 'south':
            return (x, y + 1)
        elif direction == 'east':
            return (x - 1, y)
        elif direction == 'west':
            return (x + 1, y)
        return self.currentPosition

    def can_move_to_direction_absolute(self, target_direction):
        current_node = self.get_current_node()
        if not current_node:
            return False
        
        is_blocked = current_node.walls.get(target_direction, True)
        return not is_blocked
    
    def rotate_to_absolute_direction(self, target_direction, movement_controller, attitude_handler=None):
        global ROBOT_FACE
        global CURRENT_TARGET_YAW
        print(f"🎯 Rotating from {self.currentDirection} to {target_direction}")
        
        if self.currentDirection == target_direction:
            print(f"✅ Already facing {target_direction}")
            return True
        
        direction_order = ['north', 'east', 'south', 'west']
        current_idx = direction_order.index(self.currentDirection)
        target_idx = direction_order.index(target_direction)
        
        diff = (target_idx - current_idx) % 4
        success = True
        
        if diff == 1:
            print(f"🔄 Need to turn RIGHT (90°)")
            success = movement_controller.rotate_90_degrees_right(movement_controller, attitude_handler)
            if success:
                ROBOT_FACE += 1
        elif diff == 3:
            print(f"🔄 Need to turn LEFT (-90°)")
            success = movement_controller.rotate_90_degrees_left(movement_controller, attitude_handler)
            if success:
                ROBOT_FACE += 1
        elif diff == 2:
            print(f"🔄 Need to turn AROUND (180°)")
            success1 = movement_controller.rotate_90_degrees_right(movement_controller, attitude_handler)
            success2 = movement_controller.rotate_90_degrees_right(movement_controller, attitude_handler)
            success = success1 and success2
            if success:
                ROBOT_FACE += 2
        
        if success:
            self.currentDirection = target_direction
            print(f"✅ Successfully rotated to face {self.currentDirection}")
        else:
            print(f"❌ Failed to rotate to {target_direction}")
            
        return success

    def handle_dead_end(self, movement_controller):
        print(f"🚫 === DEAD END HANDLER ACTIVATED ===")
        current_node = self.get_current_node()

        if current_node:
            print(f"📍 Dead end at position: {current_node.position}")
            print(f"🧱 Walls: {current_node.walls}")

        movement_controller.reverse_from_dead_end()

        reverse_direction_map = {
            'north': 'south',
            'south': 'north',
            'east': 'west',
            'west': 'east'
        }

        reverse_direction = reverse_direction_map[self.currentDirection]
        self.currentPosition = self.get_next_position(reverse_direction)

        print(f"🔙 Reversed to position: {self.currentPosition}")
        print(f"🧭 Still facing: {self.currentDirection}")

        return True
    
    def move_to_absolute_direction(self, target_direction, movement_controller, attitude_handler=None):
        global ROBOT_FACE
        print(f"🎯 Moving to ABSOLUTE direction: {target_direction}")
        
        if not self.can_move_to_direction_absolute(target_direction):
            print(f"❌ BLOCKED! Cannot move to {target_direction} - wall detected!")
            return False
        
        self.rotate_to_absolute_direction(target_direction, movement_controller, attitude_handler)
        
        axis_test = 'x'
        if ROBOT_FACE % 2 == 0:
            axis_test = 'y'
        elif ROBOT_FACE % 2 == 1:
            axis_test = 'x'
        
        print(f"🚀 Moving forward on {axis_test}-axis")
        
        # 🛠️ FIX: Pass global variables properly
        movement_controller.move_forward_with_pid(0.6, axis_test, direction=1)
        
        self.currentPosition = self.get_next_position(target_direction)
        
        if hasattr(self, 'previous_node') and self.previous_node:
            if target_direction not in self.previous_node.exploredDirections:
                self.previous_node.exploredDirections.append(target_direction)
            
            if target_direction in self.previous_node.unexploredExits:
                self.previous_node.unexploredExits.remove(target_direction)
                print(f"🔄 Removed {target_direction} from unexplored exits of {self.previous_node.id}")
        
        print(f"✅ Successfully moved to {self.currentPosition}")
        return True

    def reverse_to_absolute_direction(self, target_direction, movement_controller):
        global ROBOT_FACE
        print(f"🔙 BACKTRACK: Reversing to ABSOLUTE direction: {target_direction}")
        
        reverse_direction_map = {
            'north': 'south',
            'south': 'north',
            'east': 'west',
            'west': 'east'
        }
        
        required_facing_direction = reverse_direction_map[target_direction]
        self.rotate_to_absolute_direction(required_facing_direction, movement_controller)
        movement_controller.reverse_to_previous_node()
        self.currentPosition = self.get_next_position(target_direction)
        
        print(f"✅ Successfully reversed to {self.currentPosition}, still facing {self.currentDirection}")
        return True

    # 🛠️ FIXED: Better frontier detection with priority
    def find_next_exploration_direction_with_priority(self):
        current_node = self.get_current_node()
        if not current_node:
            print("❌ No current node found!")
            return None
        
        # 🛠️ FIX: Check if node is actually a dead end properly
        if self.is_dead_end(current_node):
            print(f"🚫 Current node {current_node.id} is confirmed dead end")
            return None
        
        print(f"🧭 Current robot facing: {self.currentDirection}")
        print(f"🔍 Available unexplored exits: {current_node.unexploredExits}")
        
        if not current_node.unexploredExits:
            print("❌ No unexplored exits available")
            return None
        
        direction_map = {
            'north': {'front': 'north', 'left': 'west', 'right': 'east', 'back': 'south'},
            'south': {'front': 'south', 'left': 'east', 'right': 'west', 'back': 'north'},
            'east': {'front': 'east', 'left': 'north', 'right': 'south', 'back': 'west'},
            'west': {'front': 'west', 'left': 'south', 'right': 'north', 'back': 'east'}
        }
        
        current_mapping = direction_map[self.currentDirection]
        priority_order = ['left', 'front', 'right', 'back']
        
        print(f"🎯 Checking exploration priority order: {priority_order}")
        
        for relative_direction in priority_order:
            absolute_direction = current_mapping.get(relative_direction)
            
            if absolute_direction and absolute_direction in current_node.unexploredExits:
                if self.can_move_to_direction_absolute(absolute_direction):
                    print(f"✅ Selected direction: {relative_direction} ({absolute_direction})")
                    return absolute_direction
                else:
                    print(f"❌ {relative_direction} ({absolute_direction}) is blocked by wall!")
                    current_node.unexploredExits.remove(absolute_direction)
        
        print(f"❌ No valid exploration direction found")
        return None

    def find_next_exploration_direction(self):
        """Find the next ABSOLUTE direction to explore"""
        current_node = self.get_current_node()
        if not current_node:
            return None
        
        if self.is_dead_end(current_node):
            print(f"🚫 Current node is a dead end - no exploration directions available")
            return None
        
        if current_node.unexploredExits:
            for unexplored_dir in current_node.unexploredExits:
                if self.can_move_to_direction_absolute(unexplored_dir):
                    return unexplored_dir
        
        return None
    
    def find_path_to_frontier(self, target_node_id):
        """Find shortest path to frontier node using BFS"""
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
                        self.is_path_clear_absolute(current_pos, neighbor_pos, direction)):
                        
                        visited.add(neighbor_pos)
                        new_path = path + [direction]
                        queue.append((neighbor_pos, new_path))
        
        return None
    
    def is_path_clear_absolute(self, from_pos, to_pos, direction):
        from_node_id = self.get_node_id(from_pos)
        to_node_id = self.get_node_id(to_pos)
        
        if from_node_id not in self.nodes or to_node_id not in self.nodes:
            return False
        
        from_node = self.nodes[from_node_id]
        is_blocked = from_node.walls.get(direction, True)
        return not is_blocked
    
    def execute_path_to_frontier_with_reverse(self, path, movement_controller):
        """Execute path using reverse movements for backtracking"""
        print(f"🗺️ Executing REVERSE path to frontier: {path}")
        
        for i, step_direction in enumerate(path):
            print(f"📍 Step {i+1}/{len(path)}: Current position: {self.currentPosition}, moving {step_direction}")
            
            success = self.reverse_to_absolute_direction(step_direction, movement_controller)
            
            if not success:
                print(f"❌ Failed to reverse {step_direction} during backtracking!")
                return False
            
            time.sleep(0.2)
        
        print(f"✅ Successfully reached frontier at {self.currentPosition}")
        return True
    
    # 🛠️ FIXED: Better frontier validation and error handling
    def find_nearest_frontier(self):
        """FIXED: Find frontier with better validation and error handling"""
        print("🔍 === FINDING NEAREST FRONTIER ===")
        
        if not self.frontierQueue:
            print("🔄 No frontiers in queue - rebuilding...")
            self.rebuild_frontier_queue()
            
            if not self.frontierQueue:
                print("🎉 No unexplored areas found - exploration complete!")
                return None, None, None
        
        print(f"🔍 Checking {len(self.frontierQueue)} frontier candidates...")
        
        # 🛠️ FIX: Validate each frontier more carefully
        valid_frontiers = []
        
        for frontier_id in self.frontierQueue[:]:  # Use slice to avoid modification during iteration
            if frontier_id not in self.nodes:
                print(f"❌ Removing non-existent frontier {frontier_id}")
                continue
                
            frontier_node = self.nodes[frontier_id]
            
            print(f"\n🔍 Validating frontier {frontier_id} at {frontier_node.position}:")
            print(f"   📋 Current unexplored exits: {frontier_node.unexploredExits}")
            
            # Re-validate unexplored exits
            valid_exits = []
            for exit_direction in frontier_node.unexploredExits[:]:
                target_pos = self.get_next_position_from(frontier_node.position, exit_direction)
                target_node_id = self.get_node_id(target_pos)
                
                # Check wall status
                is_wall_blocked = frontier_node.walls.get(exit_direction, True)
                if is_wall_blocked:
                    print(f"      ❌ {exit_direction}: Blocked by wall!")
                    continue
                
                # Check target exploration status
                target_exists = target_node_id in self.nodes
                if target_exists:
                    target_node = self.nodes[target_node_id]
                    target_fully_explored = target_node.fullyScanned
                    if target_fully_explored:
                        print(f"      ❌ {exit_direction}: Target already fully explored")
                        continue
                    else:
                        print(f"      ✅ {exit_direction}: Target exists but not fully explored")
                        valid_exits.append(exit_direction)
                else:
                    print(f"      ✅ {exit_direction}: NEW AREA - valid for exploration")
                    valid_exits.append(exit_direction)
            
            # Update frontier node with validated exits
            frontier_node.unexploredExits = valid_exits
            
            if valid_exits:
                valid_frontiers.append(frontier_id)
                print(f"   ✅ Frontier {frontier_id} is VALID with {len(valid_exits)} exits: {valid_exits}")
            else:
                print(f"   ❌ Frontier {frontier_id} has NO valid unexplored exits")
        
        # Update frontier queue
        self.frontierQueue = valid_frontiers
        
        if not valid_frontiers:
            print("🎉 No valid frontiers remaining - exploration complete!")
            return None, None, None
        
        print(f"\n🎯 Found {len(valid_frontiers)} valid frontiers")
        
        # Find nearest frontier
        best_frontier = None
        best_direction = None
        shortest_path = None
        min_distance = float('inf')
        
        for frontier_id in valid_frontiers:
            frontier_node = self.nodes[frontier_id]
            path = self.find_path_to_frontier(frontier_id)
            
            if path is not None:
                distance = len(path)
                print(f"   📍 {frontier_id}: distance={distance}, exits={frontier_node.unexploredExits}")
                
                if distance < min_distance:
                    min_distance = distance
                    best_frontier = frontier_id
                    best_direction = frontier_node.unexploredExits[0]
                    shortest_path = path
            else:
                print(f"   ❌ {frontier_id}: No path found!")
        
        if best_frontier:
            print(f"\n🏆 SELECTED: {best_frontier} with direction {best_direction} (distance: {min_distance})")
            print(f"🗺️ Path: {shortest_path}")
        else:
            print(f"\n❌ No reachable frontiers found!")
        
        return best_frontier, best_direction, shortest_path
    
    # 🛠️ FIXED: More thorough frontier rebuilding
    def rebuild_frontier_queue(self):
        """FIXED: More thorough frontier queue rebuilding"""
        print("🔄 === REBUILDING FRONTIER QUEUE ===")
        self.frontierQueue = []
        
        total_nodes_checked = 0
        added_to_frontier = 0
        
        for node_id, node in self.nodes.items():
            total_nodes_checked += 1
            print(f"\n🔍 Checking node {node_id} at {node.position}:")
            print(f"   🔍 Currently fully scanned: {node.fullyScanned}")
            print(f"   🧱 Wall status: {node.walls}")
            
            # Re-validate unexplored exits for this node
            valid_exits = []
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
                
                if is_blocked:
                    print(f"      🧱 {direction}: Blocked by wall")
                    continue
                
                already_explored = direction in node.exploredDirections
                target_exists = target_node_id in self.nodes
                target_fully_explored = False
                
                if target_exists:
                    target_node = self.nodes[target_node_id]
                    target_fully_explored = target_node.fullyScanned
                    print(f"      🎯 {direction} -> target exists, fully explored: {target_fully_explored}")
                else:
                    print(f"      🆕 {direction} -> NEW AREA")
                
                should_explore = (not is_blocked and 
                                not already_explored and 
                                (not target_exists or not target_fully_explored))
                
                if should_explore:
                    valid_exits.append(direction)
                    print(f"         ✅ VALID for exploration")
                else:
                    print(f"         ❌ Not valid (explored={already_explored})")
            
            # Update node's unexplored exits
            old_exits = node.unexploredExits[:]
            node.unexploredExits = valid_exits
            
            if valid_exits != old_exits:
                print(f"   🔄 Updated unexplored exits: {old_exits} -> {valid_exits}")
            
            # Add to frontier if has valid exits
            if valid_exits:
                self.frontierQueue.append(node_id)
                added_to_frontier += 1
                print(f"   🚀 ADDED to frontier queue with exits: {valid_exits}")
            else:
                print(f"   ❌ No valid unexplored exits - not added to frontier")
        
        print(f"\n✅ Frontier queue rebuilt:")
        print(f"   📊 Nodes checked: {total_nodes_checked}")
        print(f"   🚀 Added to frontier: {added_to_frontier}")
        print(f"   🎯 Active frontiers: {self.frontierQueue}")
        
        return len(self.frontierQueue) > 0
    
    def print_graph_summary(self):
        print("\n" + "="*60)
        print("📊 GRAPH MAPPING SUMMARY")
        print("="*60)
        print(f"🤖 Current Position: {self.currentPosition}")
        print(f"🧭 Current Direction: {self.currentDirection}")
        print(f"🗺️  Total Nodes: {len(self.nodes)}")
        print(f"🚀 Frontier Queue: {len(self.frontierQueue)} nodes")
        
        if self.frontierQueue:
            print(f"🎯 Active frontiers: {self.frontierQueue}")
        
        print("-"*60)
        
        for node_id, node in self.nodes.items():
            print(f"\n📍 Node: {node.id} at {node.position}")
            print(f"   🔍 Fully Scanned: {node.fullyScanned}")
            print(f"   🧱 Walls (absolute): {node.walls}")
            print(f"   🔍 Unexplored exits: {node.unexploredExits}")
            print(f"   ✅ Explored directions: {node.exploredDirections}")
            print(f"   🚫 Is dead end: {node.isDeadEnd}")
            
            if node.sensorReadings:
                print(f"   📡 Sensor readings:")
                for direction, reading in node.sensorReadings.items():
                    print(f"      {direction}: {reading:.2f}cm")
        
        print("-"*60)
        if self.frontierQueue:
            print(f"🚀 Next exploration targets: {self.frontierQueue}")
        else:
            print("🎉 EXPLORATION COMPLETE - No more frontiers!")
        print("="*60)

class ToFSensorHandler:
    def __init__(self):
        self.CALIBRATION_SLOPE = 0.0894 
        self.CALIBRATION_Y_INTERCEPT = 3.8409
        self.WINDOW_SIZE = 5
        self.tof_buffer = []
        self.WALL_THRESHOLD = 50.00
        
        self.readings = {
            'front': [],
            'left': [],
            'right': []
        }
        
        self.current_scan_direction = None
        self.collecting_data = False
        
    def calibrate_tof_value(self, raw_tof_mm):
        calibrated_cm = (self.CALIBRATION_SLOPE * raw_tof_mm) + self.CALIBRATION_Y_INTERCEPT
        return calibrated_cm
    
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
        
        wall_status = "🧱 WALL" if filtered_tof_cm <= self.WALL_THRESHOLD else "🚪 OPEN"
        print(f"[{self.current_scan_direction.upper()}] {filtered_tof_cm:.2f}cm {wall_status}")
    
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
        is_wall = avg_distance <= self.WALL_THRESHOLD and avg_distance > 0
        
        print(f"🔍 Wall check [{direction.upper()}]: {avg_distance:.2f}cm -> {'WALL' if is_wall else 'OPEN'}")
        
        return is_wall

def scan_current_node_absolute(gimbal, chassis, sensor, tof_handler, graph_mapper):
    print(f"\n🗺️ === Scanning Node at {graph_mapper.currentPosition} ===")
    current_node = graph_mapper.create_node(graph_mapper.currentPosition)
    
    if current_node.fullyScanned:
        print(f"🔄 Node {current_node.id} already fully scanned - using cached data!")
        print(f"   🧱 Cached walls (absolute): {current_node.walls}")
        print(f"   🔍 Cached unexplored exits: {current_node.unexploredExits}")
        if current_node.sensorReadings:
            print(f"   📡 Cached sensor readings:")
            for direction, reading in current_node.sensorReadings.items():
                print(f"      {direction}: {reading:.2f}cm")
        print("⚡ Skipping physical scan - using cached data")
        return current_node.sensorReadings
    
    print(f"🆕 First time visiting node {current_node.id} - performing full scan")
    print(f"🧭 Robot currently facing: {graph_mapper.currentDirection}")
    chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
    ep_chassis.move(x=0, y=0, z=0, z_speed=45).wait_for_completed()
    time.sleep(0.2)
    speed = 480
    scan_results = {}
    
    # Scan front (0°)
    print("🔍 Scanning FRONT (0°)...")
    gimbal.moveto(pitch=0, yaw=0, pitch_speed=speed, yaw_speed=speed).wait_for_completed()
    time.sleep(0.2)
    tof_handler.start_scanning('front')
    sensor.sub_distance(freq=25, callback=tof_handler.tof_data_handler)
    time.sleep(0.2)
    tof_handler.stop_scanning(sensor.unsub_distance)
    front_distance = tof_handler.get_average_distance('front')
    front_wall = tof_handler.is_wall_detected('front')
    scan_results['front'] = front_distance
    print(f"📏 FRONT scan result: {front_distance:.2f}cm - {'WALL' if front_wall else 'OPEN'}")
    
    # Scan left (-90°)
    print("🔍 Scanning LEFT (physical: -90°)...")
    gimbal.moveto(pitch=0, yaw=-90, pitch_speed=speed, yaw_speed=speed).wait_for_completed()
    time.sleep(0.2)
    tof_handler.start_scanning('left')
    sensor.sub_distance(freq=25, callback=tof_handler.tof_data_handler)
    time.sleep(0.2)
    tof_handler.stop_scanning(sensor.unsub_distance)
    left_distance = tof_handler.get_average_distance('left')
    left_wall = tof_handler.is_wall_detected('left')
    scan_results['left'] = left_distance
    print(f"📏 LEFT scan result: {left_distance:.2f}cm - {'WALL' if left_wall else 'OPEN'}")
    
    # Scan right (90°)
    print("🔍 Scanning RIGHT (physical: 90°)...")
    gimbal.moveto(pitch=0, yaw=90, pitch_speed=speed, yaw_speed=speed).wait_for_completed()
    time.sleep(0.2)
    tof_handler.start_scanning('right')
    sensor.sub_distance(freq=25, callback=tof_handler.tof_data_handler)
    time.sleep(0.2)
    tof_handler.stop_scanning(sensor.unsub_distance)
    right_distance = tof_handler.get_average_distance('right')
    right_wall = tof_handler.is_wall_detected('right')
    scan_results['right'] = right_distance
    print(f"📏 RIGHT scan result: {right_distance:.2f}cm - {'WALL' if right_wall else 'OPEN'}")
    
    # Scan back (180°) ONLY for (0,0)
    if graph_mapper.currentPosition == (0, 0):
        print("🔍 Scanning BACK (physical: 180°) at (0,0)...")
        gimbal.moveto(pitch=0, yaw=180, pitch_speed=speed, yaw_speed=speed).wait_for_completed()
        time.sleep(0.2)
        tof_handler.start_scanning('back')
        sensor.sub_distance(freq=25, callback=tof_handler.tof_data_handler)
        time.sleep(0.2)
        tof_handler.stop_scanning(sensor.unsub_distance)
        back_distance = tof_handler.get_average_distance('back')
        back_wall = tof_handler.is_wall_detected('back')
        scan_results['back'] = back_distance
        print(f"📏 BACK scan result: {back_distance:.2f}cm - {'WALL' if back_wall else 'OPEN'}")
    
    # Return to center
    gimbal.moveto(pitch=0, yaw=0, pitch_speed=speed, yaw_speed=speed).wait_for_completed()
    time.sleep(0.2)
    chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0, timeout=0.1)
    ep_chassis.move(x=0, y=0, z=0, z_speed=45).wait_for_completed()
    time.sleep(0.2)
    
    graph_mapper.update_current_node_walls_absolute(left_wall, right_wall, front_wall)
    current_node.sensorReadings = scan_results
    
    print(f"✅ Node {current_node.id} scan complete:")
    print(f"   🧱 Walls detected (relative): Left={left_wall}, Right={right_wall}, Front={front_wall}")
    print(f"   🧱 Walls stored (absolute): {current_node.walls}")
    print(f"   📏 Distances: Left={left_distance:.1f}cm, Right={right_distance:.1f}cm, Front={front_distance:.1f}cm")
    if graph_mapper.currentPosition == (0, 0):
        print(f"   📏 Back: {scan_results['back']:.1f}cm")
    
    return scan_results

# 🛠️ FIXED: Main exploration function with proper error handling
def explore_autonomously_with_absolute_directions(gimbal, chassis, sensor, tof_handler, graph_mapper, movement_controller, attitude_handler, max_nodes=49):
    """FIXED: Enhanced autonomous exploration with better frontier handling"""
    global CURRENT_TARGET_YAW  # 🛠️ FIX: Add global declaration
    
    print("\n🚀 === STARTING AUTONOMOUS EXPLORATION WITH ABSOLUTE DIRECTIONS ===")
    print(f"🎯 Wall Detection Threshold: {tof_handler.WALL_THRESHOLD}cm")
    print("⚡ OPTIMIZATION: Previously scanned nodes will NOT be re-scanned!")
    print("🔙 NEW: Backtracking uses REVERSE movement (no 180° turns)!")
    print("🧭 ENHANCED: Uses ABSOLUTE directions (North is always North)!")
    
    nodes_explored = 0
    scanning_iterations = 0
    dead_end_reversals = 0
    backtrack_attempts = 0
    reverse_backtracks = 0
    
    # 🛠️ FIX: Add safety counter to prevent infinite loops
    safety_counter = 0
    max_safety_iterations = max_nodes * 3  # Allow some backtracking
    
    while nodes_explored < max_nodes and safety_counter < max_safety_iterations:
        safety_counter += 1
        
        print(f"\n{'='*50}")
        print(f"--- EXPLORATION STEP {nodes_explored + 1} (Safety: {safety_counter}/{max_safety_iterations}) ---")
        print(f"🤖 Current position: {graph_mapper.currentPosition}")
        print(f"🧭 Current direction (absolute): {graph_mapper.currentDirection}")
        print(f"{'='*50}")
        
        # 🛠️ FIX: Better node creation and validation
        try:
            current_node = graph_mapper.create_node(graph_mapper.currentPosition)
        except Exception as e:
            print(f"❌ Error creating node: {e}")
            break
        
        # Check if current node needs scanning
        if not current_node.fullyScanned:
            print("🔍 NEW NODE - Performing full scan...")
            try:
                scan_results = scan_current_node_absolute(gimbal, chassis, sensor, tof_handler, graph_mapper)
                scanning_iterations += 1
                
                # 🛠️ FIX: Check for dead end AFTER updating unexplored exits
                if graph_mapper.is_dead_end(current_node):
                    print(f"🚫 DEAD END DETECTED after scanning!")
                    print(f"🔙 Initiating reverse maneuver...")
                    
                    success = graph_mapper.handle_dead_end(movement_controller)
                    if success:
                        dead_end_reversals += 1
                        print(f"✅ Successfully reversed from dead end (Total reversals: {dead_end_reversals})")
                        nodes_explored += 1
                        continue
                    else:
                        print(f"❌ Failed to reverse from dead end!")
                        break
                        
            except Exception as e:
                print(f"❌ Error during scanning: {e}")
                break
        else:
            print("⚡ REVISITED NODE - Using cached scan data (no physical scanning)")
            # 🛠️ FIX: Always update unexplored exits for revisited nodes
            try:
                graph_mapper.update_unexplored_exits_absolute(current_node)
                graph_mapper.build_connections()
            except Exception as e:
                print(f"❌ Error updating node data: {e}")
                break
        
        nodes_explored += 1
        
        # Print current graph state
        graph_mapper.print_graph_summary()
        
        # 🛠️ FIX: Better next direction finding
        graph_mapper.previous_node = current_node
        
        # STEP 1: Try local exploration
        try:
            next_direction = graph_mapper.find_next_exploration_direction()
        except Exception as e:
            print(f"❌ Error finding next direction: {e}")
            next_direction = None
        
        if next_direction:
            print(f"\n🎯 Next exploration direction (absolute): {next_direction}")
            
            can_move = graph_mapper.can_move_to_direction_absolute(next_direction)
            print(f"🚦 Movement check: {'ALLOWED' if can_move else 'BLOCKED'}")
            
            if can_move:
                try:
                    success = graph_mapper.move_to_absolute_direction(next_direction, movement_controller, attitude_handler)
                    if success:
                        print(f"✅ Successfully moved to {graph_mapper.currentPosition}")
                        time.sleep(0.2)
                        continue
                    else:
                        print(f"❌ Movement failed - wall detected!")
                        if current_node and next_direction in current_node.unexploredExits:
                            current_node.unexploredExits.remove(next_direction)
                        continue
                    
                except Exception as e:
                    print(f"❌ Error during movement: {e}")
                    break
            else:
                print(f"🚫 Cannot move to {next_direction} - blocked by wall!")
                if current_node and next_direction in current_node.unexploredExits:
                    current_node.unexploredExits.remove(next_direction)
                continue
        
        # STEP 2: Backtracking to frontier
        print(f"\n🔍 No unexplored directions from current node")
        print(f"🔙 Attempting REVERSE BACKTRACK to nearest frontier...")
        backtrack_attempts += 1
        
        try:
            frontier_id, frontier_direction, path = graph_mapper.find_nearest_frontier()
        except Exception as e:
            print(f"❌ Error finding frontier: {e}")
            break
        
        if frontier_id and path is not None and frontier_direction:
            print(f"🎯 Found frontier node {frontier_id} with unexplored direction: {frontier_direction}")
            print(f"🗺️ Path to frontier: {path} (distance: {len(path)} steps)")
            print("🔙 REVERSE BACKTRACK: Using reverse movements (no 180° turns)!")
            
            try:
                success = graph_mapper.execute_path_to_frontier_with_reverse(path, movement_controller)
                
                if success:
                    reverse_backtracks += 1
                    print(f"✅ Successfully REVERSE backtracked to frontier at {graph_mapper.currentPosition}")
                    print(f"   📊 Total reverse backtracks: {reverse_backtracks}")
                    time.sleep(0.2)
                    continue
                else:
                    print(f"❌ Failed to execute reverse backtracking path!")
                    # 🛠️ FIX: Try to continue exploration instead of breaking
                    print("🔄 Trying to rebuild frontier queue...")
                    if graph_mapper.rebuild_frontier_queue():
                        print("✅ Found more frontiers after rebuilding")
                        continue
                    else:
                        print("🎉 No more frontiers available - exploration complete!")

                        # Return to start position (0,0) before stopping
                        if graph_mapper.currentPosition != (0, 0):
                            print(f"🏁 Returning to start (0,0) from {graph_mapper.currentPosition}...")
                            start_node_id = graph_mapper.get_node_id((0, 0))
                            if start_node_id is not None:
                                path_home = graph_mapper.find_path(graph_mapper.currentNodeID, start_node_id)
                                if path_home:
                                    graph_mapper.execute_path_to_frontier_with_reverse(path_home, movement_controller)
                                else:
                                    print("⚠️ No path found to (0,0)! Staying in place.")
                            else:
                                print("⚠️ Start node (0,0) not found in graph!")

                        break
                    
            except Exception as e:
                print(f"❌ Error during reverse backtracking: {e}")
                # 🛠️ FIX: Try to recover from backtracking errors
                print("🔄 Attempting to recover...")
                try:
                    if graph_mapper.rebuild_frontier_queue():
                        print("✅ Recovery successful - continuing exploration")
                        continue
                    else:
                        print("❌ Recovery failed - ending exploration")
                        break
                except Exception as recovery_error:
                    print(f"❌ Recovery attempt failed: {recovery_error}")
                    break
        else:
            # STEP 3: No frontiers found - try final rebuild
            print("🎉 No frontiers found from initial search!")
            
            print("🔄 Performing COMPREHENSIVE final frontier scan...")
            try:
                rebuild_success = graph_mapper.rebuild_frontier_queue()
                
                if rebuild_success and graph_mapper.frontierQueue:
                    print(f"🚀 Found {len(graph_mapper.frontierQueue)} missed frontiers after rebuild!")
                    print(f"🎯 Continuing exploration with: {graph_mapper.frontierQueue}")
                    continue
                else:
                    print("🎉 EXPLORATION DEFINITELY COMPLETE - No more areas to explore!")
                    break
                    
            except Exception as e:
                print(f"❌ Error during final frontier rebuild: {e}")
                print("🎉 Ending exploration due to errors")
                break
        
        # 🛠️ FIX: Safety check to prevent infinite loops
        if safety_counter >= max_safety_iterations:
            print(f"⚠️ Safety limit reached ({max_safety_iterations} iterations)")
            print("🛑 Ending exploration to prevent infinite loops")
            break
    
    print(f"\n🎉 === EXPLORATION COMPLETED ===")
    print(f"📊 PERFORMANCE SUMMARY:")
    print(f"   🗺️ Total nodes visited: {nodes_explored}")
    print(f"   🔍 Physical scans performed: {scanning_iterations}")
    print(f"   🔙 Dead end reversals: {dead_end_reversals}")
    print(f"   🔄 Backtrack attempts: {backtrack_attempts}")
    print(f"   🔙 Reverse backtracks: {reverse_backtracks}")
    print(f"   🛡️ Safety iterations used: {safety_counter}/{max_safety_iterations}")
    print(f"   ⚡ Scans saved by caching: {nodes_explored - scanning_iterations}")
    if nodes_explored > 0:
        print(f"   📈 Efficiency gain: {((nodes_explored - scanning_iterations) / nodes_explored * 100):.1f}% less scanning")
    print(f"   🎯 Reverse movement efficiency: {reverse_backtracks} efficient backtracks")
    
    graph_mapper.print_graph_summary()
    generate_exploration_report_absolute(graph_mapper, nodes_explored, dead_end_reversals, reverse_backtracks)

def generate_exploration_report_absolute(graph_mapper, nodes_explored, dead_end_reversals=0, reverse_backtracks=0):
    """Generate comprehensive exploration report with absolute direction info"""
    print(f"\n{'='*60}")
    print("📋 FINAL EXPLORATION REPORT (ABSOLUTE DIRECTIONS)")
    print(f"{'='*60}")
    
    total_nodes = len(graph_mapper.nodes)
    dead_ends = sum(1 for node in graph_mapper.nodes.values() if node.isDeadEnd)
    frontier_nodes = len(graph_mapper.frontierQueue)
    fully_scanned_nodes = sum(1 for node in graph_mapper.nodes.values() if node.fullyScanned)
    
    print(f"📊 STATISTICS:")
    print(f"   🏁 Total nodes explored: {total_nodes}")
    print(f"   🎯 Node visits: {nodes_explored}")
    print(f"   🔍 Fully scanned nodes: {fully_scanned_nodes}")
    print(f"   🚫 Dead ends found: {dead_ends}")
    print(f"   🔙 Dead end reversals performed: {dead_end_reversals}")
    print(f"   🔙 Reverse backtracks performed: {reverse_backtracks}")
    print(f"   🚀 Remaining frontiers: {frontier_nodes}")
    
    revisited_nodes = nodes_explored - total_nodes
    if revisited_nodes > 0:
        print(f"   🔄 Node revisits (backtracking): {revisited_nodes}")
        print(f"   ⚡ Scans saved by caching: {revisited_nodes}")
        print(f"   📈 Scanning efficiency: {(revisited_nodes / nodes_explored * 100):.1f}% improvement")
    
    if graph_mapper.nodes:
        positions = [node.position for node in graph_mapper.nodes.values()]
        min_x = min(pos[0] for pos in positions)
        max_x = max(pos[0] for pos in positions)
        min_y = min(pos[1] for pos in positions)
        max_y = max(pos[1] for pos in positions)
        
        print(f"\n🗺️ MAP BOUNDARIES:")
        print(f"   X range: {min_x} to {max_x} (width: {max_x - min_x + 1})")
        print(f"   Y range: {min_y} to {max_y} (height: {max_y - min_y + 1})")
    
    wall_stats = {'north': 0, 'south': 0, 'east': 0, 'west': 0}
    opening_stats = {'north': 0, 'south': 0, 'east': 0, 'west': 0}
    
    for node in graph_mapper.nodes.values():
        if hasattr(node, 'walls') and node.walls:
            for direction, is_wall in node.walls.items():
                if is_wall:
                    wall_stats[direction] += 1
                else:
                    opening_stats[direction] += 1
    
    print(f"\n🧱 WALL ANALYSIS (ABSOLUTE DIRECTIONS):")
    total_walls = sum(wall_stats.values())
    total_openings = sum(opening_stats.values())
    
    print(f"   Total walls detected: {total_walls}")
    print(f"   Total openings detected: {total_openings}")
    if (total_walls + total_openings) > 0:
        print(f"   Wall density: {total_walls/(total_walls+total_openings)*100:.1f}%")
    
    print(f"   Walls by direction:")
    for direction in ['north', 'south', 'east', 'west']:
        total_dir = wall_stats[direction] + opening_stats[direction]
        if total_dir > 0:
            wall_pct = wall_stats[direction] / total_dir * 100
            print(f"      {direction.upper()}: {wall_stats[direction]} walls, {opening_stats[direction]} openings ({wall_pct:.1f}% walls)")
    
    print(f"\n🔙 MOVEMENT EFFICIENCY:")
    print(f"   Dead end reversals: {dead_end_reversals}")
    print(f"   Reverse backtracks: {reverse_backtracks}")
    print(f"   Total reverse movements: {dead_end_reversals + reverse_backtracks}")
    print(f"   Time saved vs 180° turns: ~{(dead_end_reversals + reverse_backtracks) * 2:.1f} seconds")
    
    if graph_mapper.frontierQueue:
        print(f"\n🔍 UNEXPLORED AREAS:")
        for frontier_id in graph_mapper.frontierQueue:
            if frontier_id in graph_mapper.nodes:
                node = graph_mapper.nodes[frontier_id]
                print(f"   📍 {node.position}: {len(node.unexploredExits)} unexplored exits {node.unexploredExits}")
    
    print(f"\n⭐ ABSOLUTE DIRECTION BENEFITS:")
    print(f"   🧭 Consistent navigation regardless of robot orientation")
    print(f"   🔙 Efficient reverse movements for backtracking")
    print(f"   📍 Accurate wall mapping using global coordinates")
    print(f"   🎯 Reliable frontier detection and path planning")
    
    print(f"\n{'='*60}")
    print("✅ ABSOLUTE DIRECTION EXPLORATION REPORT COMPLETE")
    print(f"{'='*60}")

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
        time.sleep(1)
        
        print(f"🎯 Wall Detection Threshold: {tof_handler.WALL_THRESHOLD}cm")
        print("🧭 ABSOLUTE DIRECTIONS: North is always North, regardless of robot facing!")
        print("🔙 REVERSE BACKTRACKING: Efficient movement without 180° turns!")
        print("⚡ SMART CACHING: Previously scanned nodes reuse cached data!")
        
        # 🛠️ FIX: Pass all required parameters
        explore_autonomously_with_absolute_directions(
            ep_gimbal, ep_chassis, ep_sensor, tof_handler, 
            graph_mapper, movement_controller, attitude_handler, 
            max_nodes=49
        )
            
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