import time
import robomaster
from robomaster import robot, vision
import numpy as np
from scipy.ndimage import median_filter
from datetime import datetime
import json
from collections import deque

ROBOT_FACE = 1 # 0 1

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
        
        # PID Parameters - Reduced for gentler movement
        self.KP = 0.8  # Reduced from 1.6
        self.KI = 0.15  # Reduced from 0.3
        self.KD = 5  # Reduced from 10
        self.RAMP_UP_TIME = 1.0  # Increased for smoother start
        self.ROTATE_TIME = 2.11  # Right turn
        self.ROTATE_LEFT_TIME = 1.9  # Left turn
        
        # Subscribe to position updates
        self.chassis.sub_position(freq=20, callback=self.position_handler)
        time.sleep(0.25)
    
    def position_handler(self, position_info):
        self.current_x = position_info[0]
        self.current_y = position_info[1]
        self.current_z = position_info[2]
    
    def move_forward_with_pid(self, target_distance, axis, direction=1):
        """Move forward using PID control"""
        pid = PID(Kp=self.KP, Ki=self.KI, Kd=self.KD, setpoint=target_distance)
        
        start_time = time.time()
        last_time = start_time
        target_reached = False
        
        # Ramp-up parameters - Reduced for gentler movement
        min_speed = 0.05  # Reduced from 0.1
        max_speed = 0.8   # Reduced from 1.5
        
        if axis == 'x':
            start_position = self.current_x
        else:
            start_position = self.current_y

        direction_text = "FORWARD" if direction == 1 else "BACKWARD"
        print(f"🚀 Moving {direction_text} {target_distance}m on {axis}-axis, direction: {direction}")
        
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
                    self.chassis.drive_speed(x=speed * direction, y=0, z=0, timeout=1.5)
                else:
                    self.chassis.drive_speed(x=speed * direction, y=0, z=0, timeout=1.5)

                # Stop condition
                if abs(relative_position - target_distance) < 0.02:
                    print(f"✅ Target reached! Final position: {current_position:.3f}")
                    self.chassis.drive_speed(x=0, y=0, z=0, timeout=1)
                    target_reached = True
                    time.sleep(0.2)
                    break
                    
        except KeyboardInterrupt:
            print("Movement interrupted by user.")
            self.chassis.drive_speed(x=0, y=0, z=0, timeout=1)
    
    def rotate_90_degrees_right(self):
        """Rotate 90 degrees clockwise"""
        print("🔄 Rotating 90° RIGHT...")
        time.sleep(0.25)
        self.chassis.drive_speed(x=0, y=0, z=30, timeout=self.ROTATE_TIME)  # Reduced from 45 to 30
        time.sleep(self.ROTATE_TIME + 0.5)  # Extra time to settle
        time.sleep(0.5)  # Additional settling time
        print("✅ Right rotation completed!")

    def rotate_90_degrees_left(self):
        """Rotate 90 degrees counter-clockwise"""
        print("🔄 Rotating 90° LEFT...")
        time.sleep(0.25)
        self.chassis.drive_speed(x=0, y=0, z=-30, timeout=self.ROTATE_LEFT_TIME)  # Reduced from -45 to -30
        time.sleep(self.ROTATE_LEFT_TIME + 0.5)  # Extra time to settle
        time.sleep(0.5)  # Additional settling time
        print("✅ Left rotation completed!")
    
    def reverse_from_dead_end(self):
        """Reverse robot from dead end position"""
        global ROBOT_FACE
        print("🔙 DEAD END DETECTED - Reversing...")
        
        # Determine current axis based on robot face
        axis_test = 'x'
        if ROBOT_FACE % 2 == 0:
            axis_test = 'y'
        elif ROBOT_FACE % 2 == 1:
            axis_test = 'x'
        
        print(f"🔙 Reversing 0.6m on {axis_test}-axis")
        
        # Move backward using negative direction
        self.move_forward_with_pid(0.6, axis_test, direction=-1)
        
        print("✅ Reverse from dead end completed!")
    
    def cleanup(self):
        """Clean up position subscription"""
        try:
            self.chassis.unsub_position()
        except:
            pass

# ===== Marker Detection Classes =====
class MarkerInfo:
    """ข้อมูล Marker ที่ตรวจพบ"""
    def __init__(self, x, y, w, h, marker_id):
        self._x = x
        self._y = y
        self._w = w
        self._h = h
        self._id = marker_id

    @property
    def id(self):
        return self._id

class MarkerVisionHandler:
    def __init__(self, graph_mapper):
        self.graph_mapper = graph_mapper
        self.markers = []  # เก็บ marker ที่ detect ล่าสุด
        self.marker_detected = False  # ตัวแปรสถานะการเจอ marker
        self.detection_count = 0  # นับจำนวนครั้งที่เจอ marker
        self.first_detection = True  # ตรวจสอบการเจอครั้งแรก
        self.MAX_DETECTION_DISTANCE = 30.0  # ระยะห่างสูงสุดที่จะนับว่าเจอ marker (cm)
        self.is_active = False  # สถานะการทำงานของ vision system
        self.detection_timeout = 3.0  # เวลารอ marker detection (วินาที)
    
    def calculate_marker_distance(self, marker_width, marker_height):
        """คำนวณระยะห่างของ marker จากขนาดที่ตรวจพบ (ประมาณการ)"""
        # ใช้ขนาดเฉลี่ยของ marker เป็นตัวประมาณ
        # สมมติว่า marker มีขนาดจริง 10cm และระยะห่างแปรผกผันกับขนาดที่เห็น
        REAL_MARKER_SIZE_CM = 10.0  # ขนาดจริงของ marker (cm)
        CAMERA_FOCAL_LENGTH = 700   # ค่าประมาณของ focal length
        
        # ใช้ขนาดเฉลี่ย (width + height) / 2 เป็นตัวคำนวณ
        apparent_size = (marker_width + marker_height) / 2
        
        if apparent_size <= 0:
            return float('inf')  # ถ้าขนาด marker ผิดปกติ ให้ถือว่าอยู่ไกลมาก
        
        # คำนวณระยะห่างประมาณ (cm)
        estimated_distance = (REAL_MARKER_SIZE_CM * CAMERA_FOCAL_LENGTH) / (apparent_size * 10)
        return estimated_distance

    def on_detect_marker(self, marker_info):
        """Callback function สำหรับ marker detection (ตรวจจับตลอดเวลา)"""
        if not self.is_active:
            return
            
        number = len(marker_info)
        valid_markers = []  # เก็บ marker ที่ตรวจพบ
        
        if number > 0:
            # ตรวจสอบแต่ละ marker
            for i in range(number):
                x, y, w, h, marker_id = marker_info[i]
                
                # Accept all detected markers regardless of estimated distance
                # Let the distance sensor handle the distance filtering
                marker = MarkerInfo(x, y, w, h, marker_id)
                valid_markers.append(marker)
            
            # จัดการ marker ที่ตรวจพบ
            if valid_markers:
                self.marker_detected = True
                self.detection_count += 1
                self.markers = valid_markers
        
        # ไม่เจอ marker เลย - ไม่ต้องเซ็ต false ทันที เพราะอาจจะเจอใน callback ครั้งถัดไป
    
    def wait_for_markers(self, timeout=None):
        """รอให้ระบบ marker detection ทำงานและรวบรวมข้อมูล"""
        if timeout is None:
            timeout = self.detection_timeout
        
        print(f"⏱️ Waiting {timeout} seconds for marker detection...")
        
        # รีเซ็ต detection state
        self.marker_detected = False
        self.markers.clear()
        
        start_time = time.time()
        last_detection_time = start_time
        
        while (time.time() - start_time) < timeout:
            # ถ้าเจอ marker แล้ว รอเพิ่มอีกนิดเพื่อให้แน่ใจว่าได้ข้อมูลครบ
            if self.marker_detected and (time.time() - last_detection_time) > 0.5:  # Reduced wait time
                print(f"✅ Marker detection stable after {time.time() - start_time:.1f}s")
                break
            
            if self.marker_detected:
                last_detection_time = time.time()
            
            time.sleep(0.05)  # Faster polling
        
        # ถ้าไม่เจอ marker เลย ให้เซ็ตสถานะ node (แต่ไม่ overwrite ถ้าเจอแล้ว)
        if not self.marker_detected:
            current_node = self.graph_mapper.get_current_node()
            if current_node and not hasattr(current_node, 'marker_checked') and not current_node.marker:
                # Only set to False if marker hasn't been detected yet in this node
                current_node.marker = False
                current_node.marker_checked = True
                print(f"❌ No markers found at node {current_node.id}")
        
        return self.marker_detected
    
    def start_continuous_detection(self, vision):
        """เริ่มการตรวจจับ marker อย่างต่อเนื่อง"""
        print("🔍 Starting continuous marker detection...")
        try:
            # หยุด detection ก่อน (ถ้ามี)
            self.stop_continuous_detection(vision)
            time.sleep(0.3)  # Reduced delay
            
            # เริ่ม marker detection
            result = vision.sub_detect_info(name="marker", callback=self.on_detect_marker)
            if result:
                self.is_active = True
                print("✅ Marker detection system activated")
                time.sleep(0.5)  # Reduced initialization time
                return True
            else:
                print("❌ Failed to start marker detection")
                return False
        except Exception as e:
            print(f"❌ Error starting marker detection: {e}")
            return False
    
    def stop_continuous_detection(self, vision):
        """หยุดการตรวจจับ marker อย่างต่อเนื่อง"""
        try:
            self.is_active = False
            vision.unsub_detect_info(name="marker")
            print("🛑 Marker detection stopped")
        except Exception as e:
            print(f"⚠️ Error stopping marker detection: {e}")
    
    def reset_detection(self):
        """รีเซ็ตสถานะการตรวจจับสำหรับการสแกนครั้งใหม่"""
        self.marker_detected = False
        self.markers.clear()
        self.detection_count = 0
        self.first_detection = True
        
        # DON'T reset node marker status - let the scanning logic handle it
        # This prevents overwriting markers detected in previous directions
    
    def get_detection_summary(self):
        """ข้อมูลสรุปการตรวจจับ marker"""
        return {
            'detected': self.marker_detected,
            'count': len(self.markers),
            'total_detections': self.detection_count,
            'marker_ids': [m.id for m in self.markers] if self.markers else []
        }

# ===== Enhanced Graph Node =====
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
        
        # NEW: Add flag to track if node has been fully scanned
        self.fullyScanned = False
        self.scanTimestamp = None
        
        # ENHANCED: Marker detection features
        self.marker = False
        self.detected_marker_ids = []  # เก็บ id marker ที่เจอใน node นี้
        self.markers_per_direction = {'front': [], 'left': [], 'right': []}  # NEW: Markers per angle
        
        # Additional info
        self.lastVisited = datetime.now().isoformat()
        self.sensorReadings = {}

# ===== Enhanced Graph Mapper =====
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
            
            # NEW: Mark node as fully scanned
            current_node.fullyScanned = True
            current_node.scanTimestamp = datetime.now().isoformat()
            
            self.update_unexplored_exits(current_node)
            self.build_connections()
    
    def build_connections(self):
        """Build connections between adjacent nodes"""
        for node_id, node in self.nodes.items():
            x, y = node.position
            
            # Check all four directions
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
    
    def update_unexplored_exits(self, node):
        """Update unexplored exits based on ALL possible directions, not just current facing"""
        node.unexploredExits = []
        
        # Check all four absolute directions regardless of current robot facing
        x, y = node.position
        
        # Define all possible directions from this node
        possible_directions = {
            'north': (x, y + 1),
            'south': (x, y - 1), 
            'east': (x + 1, y),
            'west': (x - 1, y)
        }
        
        # Map current sensor readings to absolute directions based on robot's facing
        direction_map = {
            'north': {'front': 'north', 'left': 'west', 'right': 'east'},
            'south': {'front': 'south', 'left': 'east', 'right': 'west'},
            'east': {'front': 'east', 'left': 'north', 'right': 'south'},
            'west': {'front': 'west', 'left': 'south', 'right': 'north'}
        }
        
        current_mapping = direction_map[self.currentDirection]
        
        # Create wall status for all directions
        wall_status = {}
        
        # Map sensor readings to absolute directions
        wall_status[current_mapping['front']] = node.wallFront
        wall_status[current_mapping['left']] = node.wallLeft  
        wall_status[current_mapping['right']] = node.wallRight
        
        # For the back direction (opposite of front), assume no wall if we came from there
        back_direction = None
        if current_mapping['front'] == 'north': back_direction = 'south'
        elif current_mapping['front'] == 'south': back_direction = 'north'
        elif current_mapping['front'] == 'east': back_direction = 'west' 
        elif current_mapping['front'] == 'west': back_direction = 'east'
        
        if back_direction:
            # Check if we have a neighbor in the back direction (meaning we came from there)
            back_neighbor_pos = possible_directions[back_direction] 
            back_neighbor_id = self.get_node_id(back_neighbor_pos)
            if back_neighbor_id in self.nodes:
                wall_status[back_direction] = False  # No wall if we came from there
            else:
                wall_status[back_direction] = True   # Assume wall if we haven't been there
        
        print(f"🧭 Robot facing: {self.currentDirection}")
        print(f"🔍 Wall status mapping: {wall_status}")
        
        # Check each direction for unexplored exits
        for direction, target_pos in possible_directions.items():
            target_node_id = self.get_node_id(target_pos)
            
            # Check if this direction is blocked by wall
            is_blocked = wall_status.get(direction, True)
            
            # Check if we've already explored this direction
            already_explored = direction in node.exploredDirections
            
            # Check if target node already exists (meaning we've been there)
            target_exists = target_node_id in self.nodes
            
            print(f"   📍 Direction {direction}: blocked={is_blocked}, explored={already_explored}, target_exists={target_exists}")
            
            # Add to unexplored exits if: not blocked AND not explored AND target doesn't exist
            if not is_blocked and not already_explored and not target_exists:
                node.unexploredExits.append(direction)
                print(f"   ✅ Added {direction} to unexplored exits")
        
        print(f"🎯 Final unexplored exits for {node.id}: {node.unexploredExits}")
        
        # Update frontier queue
        if node.unexploredExits and node.id not in self.frontierQueue:
            self.frontierQueue.append(node.id)
            print(f"🚀 Added {node.id} to frontier queue")
        elif not node.unexploredExits and node.id in self.frontierQueue:
            self.frontierQueue.remove(node.id)
            print(f"🧹 Removed {node.id} from frontier queue")
            
        # Update dead end status
        all_blocked = all(wall_status.get(direction, True) for direction in ['north', 'south', 'east', 'west'])
        node.isDeadEnd = all_blocked
        
        # NEW: Check if this is a dead end (all directions blocked)
        if all_blocked:
            print(f"🚫 DEAD END DETECTED at {node.id} - all directions blocked!")
    
    def is_dead_end(self, node=None):
        """Check if current node or given node is a dead end"""
        if node is None:
            node = self.get_current_node()
        
        if not node:
            return False
        
        return node.isDeadEnd
    
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
    
    def get_previous_position(self, direction):
        """Calculate previous position based on current position and direction robot came from"""
        x, y = self.currentPosition
        if direction == 'north':
            return (x, y - 1)  # came from south
        elif direction == 'south':
            return (x, y + 1)  # came from north
        elif direction == 'east':
            return (x - 1, y)  # came from west
        elif direction == 'west':
            return (x + 1, y)  # came from east
        return self.currentPosition
    
    def can_move_to_direction_absolute(self, target_direction):
        """Check if robot can move to target direction based on absolute direction"""
        current_node = self.get_current_node()
        if not current_node:
            return False
        
        # Get wall status for the target direction
        x, y = current_node.position
        possible_directions = {
            'north': (x, y + 1),
            'south': (x, y - 1),
            'east': (x + 1, y),
            'west': (x - 1, y)
        }
        
        # Map current sensor readings to absolute directions
        direction_map = {
            'north': {'front': 'north', 'left': 'west', 'right': 'east'},
            'south': {'front': 'south', 'left': 'east', 'right': 'west'},
            'east': {'front': 'east', 'left': 'north', 'right': 'south'},
            'west': {'front': 'west', 'left': 'south', 'right': 'north'}
        }
        
        current_mapping = direction_map[self.currentDirection]
        
        # Check if target direction has a wall
        if target_direction == current_mapping['front']:
            return not current_node.wallFront
        elif target_direction == current_mapping['left']:
            return not current_node.wallLeft
        elif target_direction == current_mapping['right']:
            return not current_node.wallRight
        else:
            # For back direction or other directions, check if target node exists
            target_pos = possible_directions[target_direction]
            target_node_id = self.get_node_id(target_pos)
            return target_node_id in self.nodes  # Can go if we've been there before
    
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
            # For back direction, assume it's possible (we came from there)
            return True
    
    def handle_dead_end(self, movement_controller):
        """Handle dead end situation by reversing instead of turning around"""
        global ROBOT_FACE
        
        print(f"🚫 === DEAD END HANDLER ACTIVATED ===")
        current_node = self.get_current_node()
        
        if current_node:
            print(f"📍 Dead end at position: {current_node.position}")
            print(f"🧱 Walls: Front={current_node.wallFront}, Left={current_node.wallLeft}, Right={current_node.wallRight}")
        
        # Use the new reverse method instead of turning around
        movement_controller.reverse_from_dead_end()
        
        # Update position after reversing (move back in opposite direction)
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
        
        # Robot face doesn't change since we only moved backward
        print(f"🤖 Robot face unchanged: {ROBOT_FACE}")
        
        return True
    
    def move_to_direction(self, target_direction, movement_controller):
        global ROBOT_FACE
        """Turn robot to face target direction and move forward"""
        print(f"🎯 Attempting to move from {self.currentDirection} to {target_direction}")
        
        # Check if movement is possible BEFORE moving
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
            ROBOT_FACE += 1
        elif diff == 3:  # Turn left (3 rights = 1 left)
            movement_controller.rotate_90_degrees_left()
            ROBOT_FACE += 1
        elif diff == 2:  # Turn around (180°) - MODIFIED FOR DEAD END
            # Check if this is a dead end situation
            current_node = self.get_current_node()
            if current_node and self.is_dead_end(current_node):
                print("🚫 Dead end detected - using reverse instead of turn around")
                return self.handle_dead_end(movement_controller)
            else:
                # Normal turn around for non-dead-end situations
                movement_controller.rotate_90_degrees_right()
                movement_controller.rotate_90_degrees_right()
                ROBOT_FACE += 2
        # diff == 0 means no rotation needed
        
        # Update current direction
        self.currentDirection = target_direction
        
        axis_test = 'x'
        if ROBOT_FACE % 2 == 0:
            axis_test = 'y'
        elif ROBOT_FACE % 2 == 1:
            axis_test = 'x'
        print(f'-------------------------{axis_test}-------------------------')
        
        # Move forward
        movement_controller.move_forward_with_pid(0.6, axis_test, direction=1)
        
        # Update position
        self.currentPosition = self.get_next_position(target_direction)
        
        # CRITICAL FIX: Mark this direction as explored from the previous node
        if hasattr(self, 'previous_node') and self.previous_node:
            if target_direction not in self.previous_node.exploredDirections:
                self.previous_node.exploredDirections.append(target_direction)
            
            # Remove from unexplored exits
            if target_direction in self.previous_node.unexploredExits:
                self.previous_node.unexploredExits.remove(target_direction)
                print(f"🔄 Removed {target_direction} from unexplored exits of {self.previous_node.id}")
        
        print(f"✅ Successfully moved to {self.currentPosition}")
        return True
    
    def find_next_exploration_direction(self):
        """Find the next direction to explore based on priority"""
        current_node = self.get_current_node()
        if not current_node:
            return None
        
        # Check if this is a dead end first
        if self.is_dead_end(current_node):
            print(f"🚫 Current node is a dead end - no exploration directions available")
            return None
        
        # ENHANCED: First check if there are any unexplored exits at current node
        if current_node.unexploredExits:
            # Prioritize based on current robot facing direction
            direction_priority = ['front', 'left', 'right']
            direction_map = {
                'north': {'front': 'north', 'left': 'west', 'right': 'east'},
                'south': {'front': 'south', 'left': 'east', 'right': 'west'},
                'east': {'front': 'east', 'left': 'north', 'right': 'south'},
                'west': {'front': 'west', 'left': 'south', 'right': 'north'}
            }
            
            directions = direction_map[self.currentDirection]
            
            # Try to find a direction that matches our priority AND is in unexploredExits
            for priority_dir in direction_priority:
                actual_direction = directions[priority_dir]
                
                # Check if this direction is unexplored and not blocked by wall
                if (actual_direction in current_node.unexploredExits and 
                    self.can_move_to_direction(actual_direction)):
                    return actual_direction
            
            # If no priority direction works, try any unexplored exit
            for unexplored_dir in current_node.unexploredExits:
                if self.can_move_to_direction_absolute(unexplored_dir):
                    return unexplored_dir
        
        return None
    
    def find_path_to_frontier(self, target_node_id):
        """Find shortest path to frontier node using BFS"""
        if target_node_id not in self.nodes:
            return None
        
        # BFS to find shortest path
        queue = deque([(self.currentPosition, [])])
        visited = set()
        visited.add(self.currentPosition)
        
        while queue:
            current_pos, path = queue.popleft()
            current_node_id = self.get_node_id(current_pos)
            
            # Check if we reached the target
            if current_node_id == target_node_id:
                return path
            
            # Explore neighbors
            if current_node_id in self.nodes:
                current_node = self.nodes[current_node_id]
                x, y = current_pos
                
                # Check all four directions
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
                        self.is_path_clear(current_pos, neighbor_pos, direction)):
                        
                        visited.add(neighbor_pos)
                        new_path = path + [direction]
                        queue.append((neighbor_pos, new_path))
        
        return None  # No path found
    
    def is_path_clear(self, from_pos, to_pos, direction):
        """Check if path between two adjacent nodes is clear"""
        from_node_id = self.get_node_id(from_pos)
        to_node_id = self.get_node_id(to_pos)
        
        if from_node_id not in self.nodes or to_node_id not in self.nodes:
            return False
        
        from_node = self.nodes[from_node_id]
        
        # Check if there's no wall blocking the path from source node
        # This is simplified - in reality you'd need to check walls properly
        # based on robot's orientation when it scanned that node
        return True  # For now, assume paths between explored nodes are clear
    
    def execute_path_to_frontier(self, path, movement_controller):
        """Execute a sequence of moves to reach frontier node"""
        print(f"🗺️ Executing path to frontier: {path}")
        
        for step_direction in path:
            print(f"📍 Current position: {self.currentPosition}, moving {step_direction}")
            
            success = self.move_to_direction(step_direction, movement_controller)
            if not success:
                print(f"❌ Failed to move {step_direction} during backtracking!")
                return False
            
            time.sleep(0.5)  # Brief pause between moves
        
        print(f"✅ Successfully reached frontier at {self.currentPosition}")
        return True
    
    def find_nearest_frontier(self):
        """Find the nearest frontier node to explore"""
        if not self.frontierQueue:
            return None, None, None
        
        # Clean up frontier queue first
        valid_frontiers = []
        for frontier_id in self.frontierQueue[:]:  # Copy list to avoid modification during iteration
            frontier_node = self.nodes[frontier_id]
            
            if frontier_node.unexploredExits:
                valid_frontiers.append(frontier_id)
            else:
                # Remove invalid frontiers
                print(f"🧹 Removing invalid frontier {frontier_id} - no unexplored exits")
        
        # Update frontier queue with valid ones only
        self.frontierQueue = valid_frontiers
        
        if not self.frontierQueue:
            return None, None, None
        
        print(f"🔍 Checking {len(self.frontierQueue)} valid frontier(s): {self.frontierQueue}")
        
        best_frontier = None
        best_direction = None
        shortest_path = None
        min_distance = float('inf')
        
        for frontier_id in self.frontierQueue:
            frontier_node = self.nodes[frontier_id]
            
            # Find path to this frontier
            path = self.find_path_to_frontier(frontier_id)
            
            if path is not None and len(path) < min_distance:
                min_distance = len(path)
                best_frontier = frontier_id
                best_direction = frontier_node.unexploredExits[0]  # Take first unexplored direction
                shortest_path = path
        
        if best_frontier:
            print(f"🎯 Selected frontier {best_frontier} with direction {best_direction} (distance: {min_distance})")
        
        return best_frontier, best_direction, shortest_path
    
    def print_graph_summary(self):
        print("\n" + "="*60)
        print("📊 ENHANCED GRAPH MAPPING SUMMARY WITH MARKERS")
        print("="*60)
        print(f"🤖 Current Position: {self.currentPosition}")
        print(f"🧭 Current Direction: {self.currentDirection}")
        print(f"🗺️  Total Nodes: {len(self.nodes)}")
        print(f"🚀 Frontier Queue: {len(self.frontierQueue)} nodes")
        
        # Count markers
        marker_nodes = sum(1 for node in self.nodes.values() if node.marker)
        total_markers = sum(len(node.detected_marker_ids) for node in self.nodes.values())
        
        print(f"🔖 Nodes with Markers: {marker_nodes}")
        print(f"🏷️  Total Markers Found: {total_markers}")
        print("-"*60)
        
        for node_id, node in self.nodes.items():
            print(f"\n📍 Node: {node.id} at {node.position}")
            print(f"   🔍 Fully Scanned: {node.fullyScanned}")
            print(f"   🧱 Walls: L:{node.wallLeft} R:{node.wallRight} F:{node.wallFront}")
            print(f"   🔍 Unexplored exits: {node.unexploredExits}")
            print(f"   ✅ Explored directions: {node.exploredDirections}")
            print(f"   🎯 Is dead end: {node.isDeadEnd}")
            print(f"   🔖 Has Markers: {node.marker}")
            
            if node.detected_marker_ids:
                print(f"   🆔 Marker IDs: {node.detected_marker_ids}")
            
            if any(node.markers_per_direction.values()):
                print(f"   📐 Markers per direction:")
                for dir, ids in node.markers_per_direction.items():
                    if ids:
                        print(f"      {dir}: {ids} (count: {len(ids)})")
            
            # Display detailed marker information if available
            if hasattr(node, 'marker_details') and node.marker_details:
                print(f"   🎯 Detailed marker info:")
                for dir, details in node.marker_details.items():
                    if 'marker_types' in details and details['marker_types']:
                        print(f"      {dir.upper()}: {details['direction_type']} at {details['angle']}° ({details['distance']:.1f}cm)")
                        print(f"         Markers: {', '.join(details['marker_types'])}")
                    else:
                        print(f"      {dir.upper()}: {details.get('direction_type', 'UNKNOWN')} at {details['angle']}° ({details['distance']:.1f}cm)")
            
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
        self.WALL_THRESHOLD = 35.0
        
        self.readings = {
            'front': [],
            'left': [],
            'right': [],
            'front_marker': [],
            'left_marker': [],
            'right_marker': []
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

# ===== Main Exploration Functions =====
def scan_current_node_with_markers(gimbal, chassis, sensor, tof_handler, graph_mapper, marker_handler):
    """Enhanced scan function with integrated marker detection"""
    print(f"\n🗺️ === ENHANCED SCANNING: Node at {graph_mapper.currentPosition} ===")
    
    current_node = graph_mapper.create_node(graph_mapper.currentPosition)
    
    # Initialize marker detection state for this node (only if not already set)
    if not hasattr(current_node, 'marker_initialized'):
        current_node.marker = False
        current_node.detected_marker_ids = []
        current_node.marker_initialized = True
    
    # Only scan if node hasn't been fully scanned before
    print(f"🆕 First time visiting node {current_node.id} - performing full scan with marker detection")
    
    # Lock wheels
    chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
    time.sleep(0.5)
    
    speed = 200  # Reduced gimbal speed for smoother movement
    pitch_down = -45  # Tilt down significantly for better marker detection
    scan_results = {}
    directions = ['front', 'left', 'right']
    yaw_angles = {'front': 0, 'left': -90, 'right': 90}
    
    for direction in directions:
        # Scan distance at pitch=0
        print(f"🔍 Scanning {direction.upper()} ({yaw_angles[direction]}°)...")
        gimbal.moveto(pitch=0, yaw=yaw_angles[direction], pitch_speed=speed, yaw_speed=speed).wait_for_completed()
        time.sleep(0.5)
        
        tof_handler.start_scanning(direction)
        sensor.sub_distance(freq=25, callback=tof_handler.tof_data_handler)
        time.sleep(0.8)
        tof_handler.stop_scanning(sensor.unsub_distance)
        
        distance = tof_handler.get_average_distance(direction)
        wall = tof_handler.is_wall_detected(direction)
        scan_results[direction] = distance
        
        print(f"📏 {direction.upper()} scan result: {distance:.2f}cm - {'WALL' if wall else 'OPEN'}")
        
        # Tilt down 45 degrees for marker detection
        print(f"🔖 Tilting down 45 degrees for marker detection in {direction.upper()}...")
        gimbal.moveto(pitch=pitch_down, yaw=yaw_angles[direction], pitch_speed=speed, yaw_speed=speed).wait_for_completed()
        time.sleep(0.3)  # Reduced delay
        
        # Quick distance check first
        tof_handler.start_scanning(f"{direction}_marker")
        sensor.sub_distance(freq=50, callback=tof_handler.tof_data_handler)  # Increased frequency
        time.sleep(0.5)  # Reduced scan time
        tof_handler.stop_scanning(sensor.unsub_distance)
        
        marker_distance = tof_handler.get_average_distance(f"{direction}_marker")
        
        # Determine direction type based on yaw angle
        direction_type = ""
        marker_angle = yaw_angles[direction]
        if marker_angle == -90:
            direction_type = "LEFT_SIDE"
        elif marker_angle == 90:
            direction_type = "RIGHT_SIDE"
        elif marker_angle == 0:
            direction_type = "CENTER"
        
        # Try vision detection for markers
        print(f"🎯 Scanning for markers in {direction_type} at {marker_angle}°...")
        
        # Reset and start fresh marker detection
        marker_handler.reset_detection()
        
        # Ensure vision system is active
        if not marker_handler.is_active:
            print("🔄 Reactivating marker detection...")
            try:
                vision.enable_detection(name="marker")
                marker_handler.start_continuous_detection(vision)
                time.sleep(0.3)
            except Exception as e:
                print(f"❌ Failed to reactivate: {e}")
        
        # Wait for marker detection with longer timeout for better detection
        detected = marker_handler.wait_for_markers(timeout=2.0)
        
        if detected and marker_handler.markers:
            # Get actual marker information from vision system
            detected_markers = marker_handler.markers
            marker_ids = [m.id for m in detected_markers]
            
            # Apply distance filter - only accept if within 30cm
            if marker_distance > 0 and marker_distance < 30.0:
                # Update node with real marker information
                current_node.marker = True
                
                for marker_id in marker_ids:
                    full_marker_id = f"ID{marker_id}_{direction_type}_{marker_angle}deg_{marker_distance:.1f}cm"
                    current_node.detected_marker_ids.append(full_marker_id)
                
                current_node.markers_per_direction[direction] = [f"ID{mid}_{direction_type}" for mid in marker_ids]
                
                # Store detailed marker information
                if not hasattr(current_node, 'marker_details'):
                    current_node.marker_details = {}
                current_node.marker_details[direction] = {
                    'angle': marker_angle,
                    'distance': marker_distance,
                    'direction_type': direction_type,
                    'marker_ids': marker_ids,
                    'marker_types': [f"Marker ID {mid}" for mid in marker_ids]
                }
                
                # Force update node properties
                current_node.lastVisited = datetime.now().isoformat()
                
                print(f"✅ MARKERS DETECTED!")
                print(f"   📐 Angle: {marker_angle}°")
                print(f"   📏 Distance: {marker_distance:.2f}cm")
                print(f"   🏷️ Direction: {direction_type}")
                print(f"   🆔 Marker Types: {[f'ID {mid}' for mid in marker_ids]}")
                print(f"✅ Node {current_node.id} updated with {len(marker_ids)} marker(s) on {direction_type}")
            else:
                print(f"❌ Markers detected by vision but outside 30cm range: {marker_distance:.2f}cm")
                print(f"   🔍 Detected marker IDs: {marker_ids} (not counted)")
        else:
            print(f"❌ No markers detected in {direction_type} (distance: {marker_distance:.2f}cm)")
        
        # Turn back to pitch=0 with faster speed
        gimbal.moveto(pitch=0, yaw=yaw_angles[direction], pitch_speed=speed, yaw_speed=speed).wait_for_completed()
        time.sleep(0.2)  # Reduced delay
    
    # Return to center
    print("🔍 Returning to center...")
    gimbal.moveto(pitch=0, yaw=0, pitch_speed=speed, yaw_speed=speed).wait_for_completed()
    time.sleep(0.5)
    
    # Unlock wheels
    chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0, timeout=0.1)
    time.sleep(0.2)
    
    # Update graph with wall information
    graph_mapper.update_current_node_walls(
        left_wall=tof_handler.is_wall_detected('left'),
        right_wall=tof_handler.is_wall_detected('right'),
        front_wall=tof_handler.is_wall_detected('front')
    )
    current_node.sensorReadings = scan_results
    
    # Final check: ensure marker flag is correct based on detected markers
    if current_node.detected_marker_ids and len(current_node.detected_marker_ids) > 0:
        current_node.marker = True
    
    # Report marker detection results
    print(f"🔖 MARKER DETECTION RESULTS:")
    print(f"   📍 Node {current_node.id}: Markers detected = {current_node.marker}")
    if current_node.detected_marker_ids:
        print(f"   🆔 Total Marker IDs: {current_node.detected_marker_ids}")
    for dir, ids in current_node.markers_per_direction.items():
        if ids:
            print(f"   📐 {dir.upper()}: {len(ids)} markers {ids}")
    
    # Display detailed marker information
    if hasattr(current_node, 'marker_details') and current_node.marker_details:
        print(f"   🎯 DETAILED MARKER INFO:")
        for dir, details in current_node.marker_details.items():
            if 'marker_types' in details and details['marker_types']:
                print(f"      {dir.upper()}: {details['direction_type']} at {details['angle']}° (distance: {details['distance']:.1f}cm)")
                print(f"         Markers: {', '.join(details['marker_types'])}")
            else:
                print(f"      {dir.upper()}: {details.get('direction_type', 'UNKNOWN')} at {details['angle']}° (distance: {details['distance']:.1f}cm)")
    
    print(f"✅ Node {current_node.id} scan complete:")
    print(f"   🧱 Walls detected: Left={tof_handler.is_wall_detected('left')}, Right={tof_handler.is_wall_detected('right')}, Front={tof_handler.is_wall_detected('front')}")
    print(f"   📏 Distances: Left={scan_results['left']:.1f}cm, Right={scan_results['right']:.1f}cm, Front={scan_results['front']:.1f}cm")
    
    return scan_results

def explore_autonomously_with_markers(gimbal, chassis, sensor, tof_handler, graph_mapper, movement_controller, marker_handler, vision, max_nodes=20):
    """Main exploration function with marker detection"""
    nodes_explored = 0
    scanning_iterations = 0
    dead_end_reversals = 0
    
    # Initialize marker detection system for vision-based marker identification
    print("🔍 Initializing marker detection system...")
    
    # Enable marker detection in vision system
    try:
        vision.enable_detection(name="marker")
        print("✅ Vision marker detection enabled")
    except Exception as e:
        print(f"⚠️ Could not enable marker detection: {e}")
    
    marker_detection_success = marker_handler.start_continuous_detection(vision)
    
    if not marker_detection_success:
        print("⚠️ Warning: Marker detection failed to initialize - continuing with distance-only detection")
    else:
        print("✅ Marker detection system ready!")
    
    try:
        while nodes_explored < max_nodes:
            print(f"\n{'='*50}")
            print(f"--- EXPLORATION STEP {nodes_explored + 1} ---")
            print(f"🤖 Current position: {graph_mapper.currentPosition}")
            print(f"🧭 Current direction: {graph_mapper.currentDirection}")
            print(f"{'='*50}")
            
            # Check if current node needs scanning
            current_node = graph_mapper.create_node(graph_mapper.currentPosition)
            
            if not current_node.fullyScanned:
                print("🔍 NEW NODE - Performing full scan with marker detection...")
                scan_results = scan_current_node_with_markers(gimbal, chassis, sensor, tof_handler, 
                                                            graph_mapper, marker_handler)
                scanning_iterations += 1
                
                # Check if this scan revealed a dead end
                if graph_mapper.is_dead_end(current_node):
                    print(f"🚫 DEAD END DETECTED after scanning!")
                    print(f"🔙 Initiating reverse maneuver...")
                    
                    success = graph_mapper.handle_dead_end(movement_controller)
                    if success:
                        dead_end_reversals += 1
                        print(f"✅ Successfully reversed from dead end (Total reversals: {dead_end_reversals})")
                        nodes_explored += 1  # Count the dead end node
                        continue
                    else:
                        print(f"❌ Failed to reverse from dead end!")
                        break
            else:
                print("⚡ REVISITED NODE - Using cached scan data (no physical scanning)")
                # Just update the graph structure without scanning
                graph_mapper.update_unexplored_exits(current_node)
                graph_mapper.build_connections()
            
            nodes_explored += 1
            
            # Print current graph state
            graph_mapper.print_graph_summary()
            
            # Show marker detection status
            print(f"\n🔖 CURRENT MARKER STATUS:")
            current_node = graph_mapper.get_current_node()
            if current_node and current_node.marker:
                print(f"   📍 Current node has markers: {current_node.detected_marker_ids}")
                if hasattr(current_node, 'marker_details') and current_node.marker_details:
                    for dir, details in current_node.marker_details.items():
                        print(f"      {dir.upper()}: {details['direction_type']} at {details['angle']}° ({details['distance']:.1f}cm)")
            else:
                print(f"   📍 Current node: No markers detected")
            
            # Find next direction to explore
            graph_mapper.previous_node = current_node
            
            # Try to find unexplored direction from current node
            next_direction = graph_mapper.find_next_exploration_direction()
            
            if next_direction:
                print(f"\n🎯 Next exploration direction from current node: {next_direction}")
                
                # Double-check wall detection before moving
                can_move = graph_mapper.can_move_to_direction(next_direction)
                print(f"🚦 Movement check: {'ALLOWED' if can_move else 'BLOCKED'}")
                
                if can_move:
                    try:
                        # Move to next direction
                        success = graph_mapper.move_to_direction(next_direction, movement_controller)
                        if success:
                            print(f"✅ Successfully moved to {graph_mapper.currentPosition}")
                            time.sleep(1)
                        else:
                            print(f"❌ Movement failed - wall detected!")
                            # Remove this direction from unexplored exits
                            if current_node and next_direction in current_node.unexploredExits:
                                current_node.unexploredExits.remove(next_direction)
                            continue
                        
                    except Exception as e:
                        print(f"❌ Error during movement: {e}")
                        break
                else:
                    print(f"🚫 Cannot move to {next_direction} - blocked by wall!")
                    # Remove this direction from unexplored exits
                    if current_node and next_direction in current_node.unexploredExits:
                        current_node.unexploredExits.remove(next_direction)
                    continue
            else:
                # No local unexplored directions - try backtracking to frontier
                print(f"\n🔍 No unexplored directions from current node")
                print(f"🔄 Attempting SMART backtrack to nearest frontier...")
                
                frontier_id, frontier_direction, path = graph_mapper.find_nearest_frontier()
                
                if frontier_id and path is not None and frontier_direction:
                    print(f"🎯 Found frontier node {frontier_id} with unexplored direction: {frontier_direction}")
                    print(f"🗺️ Path to frontier: {path} (distance: {len(path)} steps)")
                    print("⚡ SMART BACKTRACK: Will NOT re-scan nodes along the path!")
                    
                    try:
                        # Execute backtracking path (no scanning during backtrack)
                        success = graph_mapper.execute_path_to_frontier(path, movement_controller)
                        
                        if success:
                            print(f"✅ Successfully backtracked to frontier at {graph_mapper.currentPosition}")
                            time.sleep(1)
                            
                            # IMPORTANT: The next iteration will handle the frontier node
                            # It will either scan if new, or use cached data if already scanned
                            print(f"🚀 Ready to process frontier node on next iteration...")
                            continue
                            
                        else:
                            print(f"❌ Failed to execute backtracking path!")
                            break
                            
                    except Exception as e:
                        print(f"❌ Error during backtracking: {e}")
                        break
                else:
                    print("🎉 No more frontiers found - exploration complete!")
                    break
            
            # Safety check - prevent infinite loops
            if nodes_explored >= max_nodes:
                print(f"⚠️ Reached maximum nodes limit ({max_nodes})")
                break
    
    finally:
        # Stop marker detection
        marker_handler.stop_continuous_detection(vision)
    
    print(f"\n🎉 === ENHANCED EXPLORATION COMPLETED ===")
    print(f"📊 PERFORMANCE SUMMARY:")
    print(f"   🗺️ Total nodes visited: {nodes_explored}")
    print(f"   🔍 Physical scans performed: {scanning_iterations}")
    print(f"   🔙 Dead end reversals: {dead_end_reversals}")
    print(f"   ⚡ Scans saved by caching: {nodes_explored - scanning_iterations}")
    print(f"   📈 Efficiency gain: {((nodes_explored - scanning_iterations) / nodes_explored * 100):.1f}% less scanning")
    print(f"   🎯 Dead end handling: {dead_end_reversals} reversals performed")
    
    # Marker detection summary
    total_marker_nodes = sum(1 for node in graph_mapper.nodes.values() if node.marker)
    total_markers_found = sum(len(node.detected_marker_ids) for node in graph_mapper.nodes.values())
    
    # Collect marker type statistics
    marker_types = {}
    direction_stats = {}
    for node in graph_mapper.nodes.values():
        if hasattr(node, 'marker_details') and node.marker_details:
            for dir, details in node.marker_details.items():
                # Count by direction type
                direction_type = details.get('direction_type', 'UNKNOWN')
                if direction_type not in direction_stats:
                    direction_stats[direction_type] = 0
                direction_stats[direction_type] += 1
                
                # Count by actual marker IDs
                if 'marker_types' in details and details['marker_types']:
                    for marker_type in details['marker_types']:
                        if marker_type not in marker_types:
                            marker_types[marker_type] = 0
                        marker_types[marker_type] += 1
    
    print(f"\n🔖 MARKER DETECTION SUMMARY:")
    print(f"   🎯 Nodes with markers: {total_marker_nodes}")
    print(f"   🏷️ Total markers found: {total_markers_found}")
    if marker_types:
        print(f"   📊 Marker types found:")
        for marker_type, count in marker_types.items():
            print(f"      {marker_type}: {count} markers")
    if direction_stats:
        print(f"   🧭 Markers by direction:")
        for direction_type, count in direction_stats.items():
            print(f"      {direction_type}: {count} markers")
    
    graph_mapper.print_graph_summary()
    
    # Generate final exploration report
    generate_enhanced_exploration_report(graph_mapper, nodes_explored, dead_end_reversals)

def generate_enhanced_exploration_report(graph_mapper, nodes_explored, dead_end_reversals=0):
    """Generate comprehensive exploration report with marker data"""
    print(f"\n{'='*60}")
    print("📋 ENHANCED EXPLORATION REPORT WITH MARKERS")
    print(f"{'='*60}")
    
    # Basic statistics
    total_nodes = len(graph_mapper.nodes)
    dead_ends = sum(1 for node in graph_mapper.nodes.values() if node.isDeadEnd)
    frontier_nodes = len(graph_mapper.frontierQueue)
    fully_scanned_nodes = sum(1 for node in graph_mapper.nodes.values() if node.fullyScanned)
    
    # Marker statistics
    marker_nodes = sum(1 for node in graph_mapper.nodes.values() if node.marker)
    total_markers = sum(len(node.detected_marker_ids) for node in graph_mapper.nodes.values())
    unique_marker_ids = set()
    for node in graph_mapper.nodes.values():
        unique_marker_ids.update(node.detected_marker_ids)
    
    print(f"📊 EXPLORATION STATISTICS:")
    print(f"   🏁 Total nodes explored: {total_nodes}")
    print(f"   🎯 Node visits: {nodes_explored}")
    print(f"   🔍 Fully scanned nodes: {fully_scanned_nodes}")
    print(f"   🚫 Dead ends found: {dead_ends}")
    print(f"   🔙 Dead end reversals performed: {dead_end_reversals}")
    print(f"   🚀 Remaining frontiers: {frontier_nodes}")
    
    print(f"\n🔖 MARKER DETECTION STATISTICS:")
    print(f"   🎯 Nodes with markers: {marker_nodes}")
    print(f"   🏷️ Total marker instances: {total_markers}")
    print(f"   🆔 Unique marker IDs: {len(unique_marker_ids)}")
    if unique_marker_ids:
        print(f"   📋 Marker ID list: {sorted(list(unique_marker_ids))}")
    
    # Efficiency metrics
    revisited_nodes = nodes_explored - total_nodes
    if revisited_nodes > 0:
        print(f"\n⚡ EFFICIENCY METRICS:")
        print(f"   🔄 Node revisits (backtracking): {revisited_nodes}")
        print(f"   ⚡ Scans saved by caching: {revisited_nodes}")
        print(f"   📈 Scanning efficiency: {(revisited_nodes / nodes_explored * 100):.1f}% improvement")
    
    # Map boundaries
    if graph_mapper.nodes:
        positions = [node.position for node in graph_mapper.nodes.values()]
        min_x = min(pos[0] for pos in positions)
        max_x = max(pos[0] for pos in positions)
        min_y = min(pos[1] for pos in positions)
        max_y = max(pos[1] for pos in positions)
        
        print(f"\n🗺️ MAP BOUNDARIES:")
        print(f"   X range: {min_x} to {max_x} (width: {max_x - min_x + 1})")
        print(f"   Y range: {min_y} to {max_y} (height: {max_y - min_y + 1})")
        print(f"   📍 Starting position: (0, 0)")
        print(f"   🏁 Final position: {graph_mapper.currentPosition}")
    
    # Detailed marker locations
    if marker_nodes > 0:
        print(f"\n🔖 DETAILED MARKER LOCATIONS:")
        for node_id, node in graph_mapper.nodes.items():
            if node.marker and node.detected_marker_ids:
                print(f"   📍 Node {node.id} at {node.position}: Total Markers {node.detected_marker_ids}")
                for dir, ids in node.markers_per_direction.items():
                    if ids:
                        print(f"      {dir.upper()}: {ids} (count: {len(ids)})")
                
                # Display detailed marker information
                if hasattr(node, 'marker_details') and node.marker_details:
                    print(f"      🎯 Marker Details:")
                    for dir, details in node.marker_details.items():
                        if 'marker_types' in details and details['marker_types']:
                            print(f"         {dir.upper()}: {details['direction_type']} at {details['angle']}° ({details['distance']:.1f}cm)")
                            print(f"            Markers: {', '.join(details['marker_types'])}")
                        else:
                            print(f"         {dir.upper()}: {details.get('direction_type', 'UNKNOWN')} at {details['angle']}° ({details['distance']:.1f}cm)")

    
    # Unexplored areas
    if graph_mapper.frontierQueue:
        print(f"\n🔍 UNEXPLORED AREAS:")
        for frontier_id in graph_mapper.frontierQueue:
            node = graph_mapper.nodes[frontier_id]
            print(f"   📍 {node.position}: {len(node.unexploredExits)} unexplored exits {node.unexploredExits}")
    
    # Wall statistics
    total_walls = 0
    total_openings = 0
    
    for node in graph_mapper.nodes.values():
        if hasattr(node, 'sensorReadings') and node.sensorReadings:
            for direction, distance in node.sensorReadings.items():
                if distance <= 35.0:  # Wall threshold
                    total_walls += 1
                else:
                    total_openings += 1

if __name__ == '__main__':
    print("🤖 Connecting to robot...")
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    
    ep_gimbal = ep_robot.gimbal
    ep_chassis = ep_robot.chassis
    ep_sensor = ep_robot.sensor
    ep_vision = ep_robot.vision  # Add vision system
    
    # Initialize components
    tof_handler = ToFSensorHandler()
    graph_mapper = GraphMapper()
    movement_controller = MovementController(ep_chassis)
    marker_handler = MarkerVisionHandler(graph_mapper)  # Add marker handler
    
    try:
        print("✅ Recalibrating gimbal...")
        ep_gimbal.recenter(pitch_speed=100, yaw_speed=100).wait_for_completed()
        ep_gimbal.moveto(pitch=0, yaw=0, pitch_speed=50, yaw_speed=50).wait_for_completed()
        time.sleep(0.5)
        
        # Start enhanced autonomous exploration with marker detection
        explore_autonomously_with_markers(ep_gimbal, ep_chassis, ep_sensor, tof_handler, 
                                        graph_mapper, movement_controller, marker_handler,
                                        ep_vision, max_nodes=49)
            
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
            marker_handler.stop_continuous_detection(ep_vision)
        except:
            pass
        ep_robot.close()
        print("🔌 Connection closed")