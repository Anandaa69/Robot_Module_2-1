import time
import robomaster
from robomaster import robot
import numpy as np
from scipy.ndimage import median_filter
from datetime import datetime
import json

# ===== PID Controller =====
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

# ===== Movement Controller =====
class MovementController:
    def __init__(self, chassis):
        self.chassis = chassis
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        
        # PID Parameters
        self.KP = 2.1
        self.KI = 0.3
        self.KD = 10
        self.RAMP_UP_TIME = 0.5
        self.ROTATE_TIME = 2.11  # Right turn
        self.ROTATE_LEFT_TIME = 1.9  # Left turn
        
        # Subscribe to position updates
        self.chassis.sub_position(freq=20, callback=self.position_handler)
        time.sleep(0.25)
    
    def position_handler(self, position_info):
        self.current_x = position_info[0]
        self.current_y = position_info[1]
        self.current_z = position_info[2]
        # print(f"Position: x:{self.current_x:.3f}, y:{self.current_y:.3f}, z:{self.current_z:.3f}")
    
    def move_forward_with_pid(self, target_distance, axis='x', direction=1):
        """Move forward using PID control"""
        pid = PID(Kp=self.KP, Ki=self.KI, Kd=self.KD, setpoint=target_distance)
        
        start_time = time.time()
        last_time = start_time
        target_reached = False
        
        # Ramp-up parameters
        min_speed = 0.1
        max_speed = 1.5
        
        if axis == 'x':
            start_position = self.current_x
        else:
            start_position = self.current_y

        print(f"🚀 Moving {target_distance}m on {axis}-axis, direction: {direction}")
        
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

                # PID calculation
                output = pid.compute(relative_position, dt)
                
                # Ramp-up logic
                if elapsed_time < self.RAMP_UP_TIME:
                    ramp_multiplier = min_speed + (elapsed_time / self.RAMP_UP_TIME) * (1.0 - min_speed)
                else:
                    ramp_multiplier = 1.0
                
                ramped_output = output * ramp_multiplier
                speed = max(min(ramped_output, max_speed), -max_speed)
                
                if axis == 'x':
                    self.chassis.drive_speed(x=speed * direction, y=0, z=0, timeout=1)
                else:
                    self.chassis.drive_speed(x=0, y=speed * direction, z=0, timeout=1)

                # Stop condition
                if abs(relative_position - target_distance) < 0.02:
                    print(f"✅ Target reached! Final position: {current_position:.3f}")
                    self.chassis.drive_speed(x=0, y=0, z=0, timeout=1)
                    target_reached = True
                    break
                    
        except KeyboardInterrupt:
            print("Movement interrupted by user.")
            self.chassis.drive_speed(x=0, y=0, z=0, timeout=1)
    
    def rotate_90_degrees_right(self):
        """Rotate 90 degrees clockwise"""
        print("🔄 Rotating 90° RIGHT...")
        time.sleep(0.25)
        self.chassis.drive_speed(x=0, y=0, z=45, timeout=self.ROTATE_TIME)
        time.sleep(self.ROTATE_TIME + 0.2)
        self.chassis.drive_speed(x=0, y=0, z=0, timeout=0.5)
        time.sleep(0.25)
        print("✅ Right rotation completed!")

    def rotate_90_degrees_left(self):
        """Rotate 90 degrees counter-clockwise"""
        print("🔄 Rotating 90° LEFT...")
        time.sleep(0.25)
        self.chassis.drive_speed(x=0, y=0, z=-45, timeout=self.ROTATE_LEFT_TIME)
        time.sleep(self.ROTATE_LEFT_TIME + 0.2)
        self.chassis.drive_speed(x=0, y=0, z=0, timeout=0.5)
        time.sleep(0.25)
        print("✅ Left rotation completed!")
    
    def cleanup(self):
        """Clean up position subscription"""
        try:
            self.chassis.unsub_position()
        except:
            pass

# ===== Graph Node =====
class GraphNode:
    def __init__(self, node_id, position):
        self.id = node_id
        self.position = position  # (x, y)
        
        # Wall detection
        self.wallLeft = False
        self.wallRight = False
        self.wallFront = False
        self.wallBack = False
        
        # Neighbors (connected nodes)
        self.neighbors = {
            'north': None,
            'south': None,
            'east': None,
            'west': None
        }
        
        # Exploration state
        self.visited = True
        self.visitCount = 1
        self.exploredDirections = []
        self.unexploredExits = []
        self.isDeadEnd = False
        
        # Additional info
        self.marker = False
        self.lastVisited = datetime.now().isoformat()
        self.sensorReadings = {}

# ===== Graph Mapper =====
class GraphMapper:
    def __init__(self):
        self.nodes = {}
        self.currentPosition = (0, 0)
        self.currentDirection = 'north'
        self.frontierQueue = []
        self.pathStack = []
        self.visitedNodes = set()
        
    def get_node_id(self, position):
        return f"{position[0]}_{position[1]}"
    
    def create_node(self, position):
        node_id = self.get_node_id(position)
        if node_id not in self.nodes:
            self.nodes[node_id] = GraphNode(node_id, position)
            self.visitedNodes.add(node_id)
        return self.nodes[node_id]
    
    def get_current_node(self):
        node_id = self.get_node_id(self.currentPosition)
        return self.nodes.get(node_id)
    
    def update_current_node_walls(self, left_wall, right_wall, front_wall):
        current_node = self.get_current_node()
        if current_node:
            current_node.wallLeft = left_wall
            current_node.wallRight = right_wall
            current_node.wallFront = front_wall
            current_node.lastVisited = datetime.now().isoformat()
            
            self.update_unexplored_exits(current_node)
    
    def update_unexplored_exits(self, node):
        node.unexploredExits = []
        
        direction_map = {
            'north': {'front': 'north', 'left': 'west', 'right': 'east'},
            'south': {'front': 'south', 'left': 'east', 'right': 'west'},
            'east': {'front': 'east', 'left': 'north', 'right': 'south'},
            'west': {'front': 'west', 'left': 'south', 'right': 'north'}
        }
        
        directions = direction_map[self.currentDirection]
        
        # Check unexplored exits (no walls and not yet explored)
        if not node.wallFront:
            exit_direction = directions['front']
            if exit_direction not in node.exploredDirections:
                node.unexploredExits.append(exit_direction)
                
        if not node.wallLeft:
            exit_direction = directions['left']
            if exit_direction not in node.exploredDirections:
                node.unexploredExits.append(exit_direction)
                
        if not node.wallRight:
            exit_direction = directions['right']
            if exit_direction not in node.exploredDirections:
                node.unexploredExits.append(exit_direction)
        
        # Update frontier queue
        if node.unexploredExits and node.id not in self.frontierQueue:
            self.frontierQueue.append(node.id)
        elif not node.unexploredExits and node.id in self.frontierQueue:
            self.frontierQueue.remove(node.id)
            
        node.isDeadEnd = (node.wallFront and node.wallLeft and node.wallRight)
    
    def get_next_position(self, direction):
        """Calculate next position based on current position and direction"""
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
    
    def can_move_to_direction(self, target_direction):
        """Check if robot can move to target direction (no wall blocking)"""
        current_node = self.get_current_node()
        if not current_node:
            return False
        
        direction_map = {
            'north': {'front': 'north', 'left': 'west', 'right': 'east'},
            'south': {'front': 'south', 'left': 'east', 'right': 'west'},
            'east': {'front': 'east', 'left': 'north', 'right': 'south'},
            'west': {'front': 'west', 'left': 'south', 'right': 'north'}
        }
        
        directions = direction_map[self.currentDirection]
        
        # Map target direction to sensor reading
        if target_direction == directions['front']:
            return not current_node.wallFront
        elif target_direction == directions['left']:
            return not current_node.wallLeft
        elif target_direction == directions['right']:
            return not current_node.wallRight
        else:
            # For back direction or complex cases, assume it's possible for now
            # In a full implementation, you'd check the back wall too
            return True
    
    def move_to_direction(self, target_direction, movement_controller):
        """Turn robot to face target direction and move forward"""
        print(f"🎯 Attempting to move from {self.currentDirection} to {target_direction}")
        
        # CRITICAL FIX: Check if movement is possible BEFORE moving
        if not self.can_move_to_direction(target_direction):
            print(f"❌ BLOCKED! Cannot move to {target_direction} - wall detected!")
            return False
        
        # Calculate rotation needed
        direction_order = ['north', 'east', 'south', 'west']
        current_idx = direction_order.index(self.currentDirection)
        target_idx = direction_order.index(target_direction)
        
        # Calculate shortest rotation
        diff = (target_idx - current_idx) % 4
        
        if diff == 1:  # Turn right
            movement_controller.rotate_90_degrees_right()
        elif diff == 3:  # Turn left (3 rights = 1 left)
            movement_controller.rotate_90_degrees_left()
        elif diff == 2:  # Turn around (180°)
            movement_controller.rotate_90_degrees_right()
            movement_controller.rotate_90_degrees_right()
        # diff == 0 means no rotation needed
        
        # Update current direction
        self.currentDirection = target_direction
        
        # Move forward
        movement_controller.move_forward_with_pid(0.6, 'x', direction=1)
        
        # Update position
        self.currentPosition = self.get_next_position(target_direction)
        
        # Mark this direction as explored from the previous node
        if hasattr(self, 'previous_node') and self.previous_node:
            self.previous_node.exploredDirections.append(target_direction)
        
        print(f"✅ Successfully moved to {self.currentPosition}")
        return True
    
    def find_next_exploration_direction(self):
        """Find the next direction to explore based on priority"""
        current_node = self.get_current_node()
        if not current_node:
            return None
        
        # Priority: front, left, right (to encourage forward exploration)
        direction_priority = ['front', 'left', 'right']
        direction_map = {
            'north': {'front': 'north', 'left': 'west', 'right': 'east'},
            'south': {'front': 'south', 'left': 'east', 'right': 'west'},
            'east': {'front': 'east', 'left': 'north', 'right': 'south'},
            'west': {'front': 'west', 'left': 'south', 'right': 'north'}
        }
        
        directions = direction_map[self.currentDirection]
        
        for priority_dir in direction_priority:
            actual_direction = directions[priority_dir]
            
            # Check if this direction is unexplored and not blocked by wall
            if (actual_direction in current_node.unexploredExits and 
                self.can_move_to_direction(actual_direction)):
                return actual_direction
        
        return None
    
    def find_nearest_frontier(self):
        """Find the nearest frontier node to explore"""
        if not self.frontierQueue:
            return None, None
        
        # For now, return the first frontier (can be improved with pathfinding)
        for frontier_id in self.frontierQueue:
            frontier_node = self.nodes[frontier_id]
            if frontier_node.unexploredExits:
                return frontier_id, frontier_node.unexploredExits[0]
        
        return None, None
    
    def print_graph_summary(self):
        print("\n" + "="*60)
        print("📊 GRAPH MAPPING SUMMARY")
        print("="*60)
        print(f"🤖 Current Position: {self.currentPosition}")
        print(f"🧭 Current Direction: {self.currentDirection}")
        print(f"🗺️  Total Nodes: {len(self.nodes)}")
        print(f"🚀 Frontier Queue: {len(self.frontierQueue)} nodes")
        print("-"*60)
        
        for node_id, node in self.nodes.items():
            print(f"\n📍 Node: {node.id} at {node.position}")
            print(f"   🧱 Walls: L:{node.wallLeft} R:{node.wallRight} F:{node.wallFront}")
            print(f"   🔍 Unexplored exits: {node.unexploredExits}")
            print(f"   ✅ Explored directions: {node.exploredDirections}")
            print(f"   🎯 Is dead end: {node.isDeadEnd}")
            
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

# ===== ToF Sensor Handler =====
class ToFSensorHandler:
    def __init__(self):
        self.CALIBRATION_SLOPE = 0.0894 
        self.CALIBRATION_Y_INTERCEPT = 3.8409
        self.WINDOW_SIZE = 5
        self.tof_buffer = []
        self.WALL_THRESHOLD = 30.0  # INCREASED from 25cm to 30cm for better detection
        
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
        
        # ADDED: Filter out obviously bad readings
        if raw_tof_mm <= 0 or raw_tof_mm > 4000:  # 4m max range check
            return
            
        calibrated_tof_cm = self.calibrate_tof_value(raw_tof_mm)
        self.tof_buffer.append(calibrated_tof_cm)
        filtered_tof_cm = self.apply_median_filter(self.tof_buffer, self.WINDOW_SIZE)
        
        # INCREASED sample count for more reliable readings
        if len(self.tof_buffer) <= 20:  # Changed from 15 to 20
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
        
        # ADDED: Remove outliers using IQR method for more robust average
        if len(filtered_values) > 4:
            q1 = np.percentile(filtered_values, 25)
            q3 = np.percentile(filtered_values, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Filter outliers
            filtered_values = [x for x in filtered_values if lower_bound <= x <= upper_bound]
        
        return np.mean(filtered_values) if filtered_values else 0.0
    
    def is_wall_detected(self, direction):
        avg_distance = self.get_average_distance(direction)
        is_wall = avg_distance <= self.WALL_THRESHOLD and avg_distance > 0
        
        # ADDED: Extra logging for debugging
        print(f"🔍 Wall check [{direction.upper()}]: {avg_distance:.2f}cm -> {'WALL' if is_wall else 'OPEN'}")
        
        return is_wall

# ===== Main Exploration Functions =====
def scan_current_node(gimbal, chassis, sensor, tof_handler, graph_mapper):
    """Scan current node and update graph"""
    print(f"\n🗺️ === Scanning Node at {graph_mapper.currentPosition} ===")
    
    current_node = graph_mapper.create_node(graph_mapper.currentPosition)
    
    # Lock wheels
    chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
    time.sleep(0.5)
    
    speed = 540
    scan_results = {}
    
    # Scan front (0°)
    print("🔍 Scanning FRONT (0°)...")
    gimbal.moveto(pitch=0, yaw=0, pitch_speed=speed, yaw_speed=speed).wait_for_completed()
    time.sleep(0.5)
    
    tof_handler.start_scanning('front')
    sensor.sub_distance(freq=25, callback=tof_handler.tof_data_handler)
    time.sleep(0.8)
    tof_handler.stop_scanning(sensor.unsub_distance)
    
    front_distance = tof_handler.get_average_distance('front')
    front_wall = tof_handler.is_wall_detected('front')
    scan_results['front'] = front_distance
    
    print(f"📏 FRONT scan result: {front_distance:.2f}cm - {'WALL' if front_wall else 'OPEN'}")
    
    # ===== แก้ไข: สลับการสแกนซ้าย-ขวา =====
    # Scan right (-90°) - แต่เก็บข้อมูลเป็น 'left' เพราะ physical sensor อาจจะสลับ
    print("🔍 Scanning LEFT (physical: -90°)...")
    gimbal.moveto(pitch=0, yaw=-90, pitch_speed=speed, yaw_speed=speed).wait_for_completed()
    time.sleep(0.5)
    
    tof_handler.start_scanning('left')
    sensor.sub_distance(freq=25, callback=tof_handler.tof_data_handler)
    time.sleep(0.8)
    tof_handler.stop_scanning(sensor.unsub_distance)
    
    left_distance = tof_handler.get_average_distance('left')
    left_wall = tof_handler.is_wall_detected('left')
    scan_results['left'] = left_distance
    
    print(f"📏 LEFT scan result: {left_distance:.2f}cm - {'WALL' if left_wall else 'OPEN'}")
    
    # Scan left (90°) - แต่เก็บข้อมูลเป็น 'right' เพราะ physical sensor อาจจะสลับ
    print("🔍 Scanning RIGHT (physical: 90°)...")
    gimbal.moveto(pitch=0, yaw=90, pitch_speed=speed, yaw_speed=speed).wait_for_completed()
    time.sleep(0.5)
    
    tof_handler.start_scanning('right')
    sensor.sub_distance(freq=25, callback=tof_handler.tof_data_handler)
    time.sleep(0.8)
    tof_handler.stop_scanning(sensor.unsub_distance)
    
    right_distance = tof_handler.get_average_distance('right')
    right_wall = tof_handler.is_wall_detected('right')
    scan_results['right'] = right_distance
    
    print(f"📏 RIGHT scan result: {right_distance:.2f}cm - {'WALL' if right_wall else 'OPEN'}")
    
    # Return to center
    gimbal.moveto(pitch=0, yaw=0, pitch_speed=speed, yaw_speed=speed).wait_for_completed()
    time.sleep(0.5)
    
    # Unlock wheels
    chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0, timeout=0.1)
    time.sleep(0.2)
    
    # Update graph with wall information
    graph_mapper.update_current_node_walls(left_wall, right_wall, front_wall)
    current_node.sensorReadings = scan_results
    
    print(f"✅ Node {current_node.id} scan complete:")
    print(f"   🧱 Walls detected: Left={left_wall}, Right={right_wall}, Front={front_wall}")
    print(f"   📏 Distances: Left={left_distance:.1f}cm, Right={right_distance:.1f}cm, Front={front_distance:.1f}cm")
    
    return scan_results

def explore_autonomously(gimbal, chassis, sensor, tof_handler, graph_mapper, movement_controller, max_nodes=10):
    """Main autonomous exploration algorithm"""
    print("\n🚀 === STARTING AUTONOMOUS EXPLORATION ===")
    print(f"🎯 Wall Detection Threshold: {tof_handler.WALL_THRESHOLD}cm")
    
    nodes_explored = 0
    
    while nodes_explored < max_nodes:
        print(f"\n{'='*50}")
        print(f"--- EXPLORATION STEP {nodes_explored + 1} ---")
        print(f"🤖 Current position: {graph_mapper.currentPosition}")
        print(f"🧭 Current direction: {graph_mapper.currentDirection}")
        print(f"{'='*50}")
        
        # Scan current node
        scan_results = scan_current_node(gimbal, chassis, sensor, tof_handler, graph_mapper)
        nodes_explored += 1
        
        # Print current graph state
        graph_mapper.print_graph_summary()
        
        # Find next direction to explore
        current_node = graph_mapper.get_current_node()
        graph_mapper.previous_node = current_node
        
        # IMPROVED: Use the new prioritized direction finding
        next_direction = graph_mapper.find_next_exploration_direction()
        
        if next_direction:
            print(f"\n🎯 Next exploration direction: {next_direction}")
            
            # CRITICAL: Double-check wall detection before moving
            can_move = graph_mapper.can_move_to_direction(next_direction)
            print(f"🚦 Movement check: {'ALLOWED' if can_move else 'BLOCKED'}")
            
            if can_move:
                try:
                    # Move to next direction
                    success = graph_mapper.move_to_direction(next_direction, movement_controller)
                    if success:
                        print(f"✅ Successfully moved to {graph_mapper.currentPosition}")
                        time.sleep(1)  # Brief pause between moves
                    else:
                        print(f"❌ Movement failed - wall detected!")
                        break
                    
                except Exception as e:
                    print(f"❌ Error during movement: {e}")
                    break
            else:
                print(f"🚫 Cannot move to {next_direction} - blocked by wall!")
                # Remove this direction from unexplored exits
                if current_node:
                    if next_direction in current_node.unexploredExits:
                        current_node.unexploredExits.remove(next_direction)
                continue
        else:
            # Try to find a frontier node to backtrack to
            frontier_id, frontier_direction = graph_mapper.find_nearest_frontier()
            if frontier_id and frontier_direction:
                print(f"🔍 Need to backtrack to frontier node {frontier_id}")
                print("⚠️ Backtracking not yet implemented - stopping exploration")
                break
            else:
                print("🎉 No more frontiers found - exploration complete!")
                break
    
    print(f"\n🎉 === EXPLORATION COMPLETED ({nodes_explored} nodes explored) ===")
    graph_mapper.print_graph_summary()

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
    
    try:
        print("✅ Recalibrating gimbal...")
        ep_gimbal.recenter(pitch_speed=100, yaw_speed=100).wait_for_completed()
        ep_gimbal.moveto(pitch=0, yaw=0, pitch_speed=50, yaw_speed=50).wait_for_completed()
        time.sleep(0.5)
        
        print(f"🎯 Wall Detection Threshold: {tof_handler.WALL_THRESHOLD}cm")
        
        # Start autonomous exploration
        explore_autonomously(ep_gimbal, ep_chassis, ep_sensor, tof_handler, 
                           graph_mapper, movement_controller, max_nodes=49)
            
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
        except:
            pass
        ep_robot.close()
        print("🔌 Connection closed")