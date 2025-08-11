import time
import robomaster
from robomaster import robot
import numpy as np
from scipy.ndimage import median_filter
from datetime import datetime, timedelta
import json
import heapq
from collections import deque
import math

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

# ===== Enhanced Localization System =====
class HybridLocalizer:
    def __init__(self, chassis):
        self.chassis = chassis
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0  # heading
        
        # Dead reckoning state
        self.odometry_x = 0.0
        self.odometry_y = 0.0
        self.odometry_heading = 0.0
        
        # Drift correction
        self.drift_correction_x = 0.0
        self.drift_correction_y = 0.0
        self.last_correction_time = time.time()
        
        # Position history for filtering
        self.position_history = deque(maxlen=10)
        
        # Subscribe to position updates
        self.chassis.sub_position(freq=20, callback=self.position_handler)
        time.sleep(0.25)
    
    def position_handler(self, position_info):
        raw_x, raw_y, raw_z = position_info
        
        # Apply median filter
        self.position_history.append((raw_x, raw_y, raw_z))
        
        if len(self.position_history) >= 3:
            positions = list(self.position_history)
            filtered_x = np.median([pos[0] for pos in positions[-3:]])
            filtered_y = np.median([pos[1] for pos in positions[-3:]])
            filtered_z = np.median([pos[2] for pos in positions[-3:]])
        else:
            filtered_x, filtered_y, filtered_z = raw_x, raw_y, raw_z
        
        # Apply drift correction
        self.current_x = filtered_x + self.drift_correction_x
        self.current_y = filtered_y + self.drift_correction_y
        self.current_z = filtered_z
    
    def apply_landmark_correction(self, expected_pos, actual_pos):
        """Apply correction based on landmark detection (e.g., known walls)"""
        error_x = expected_pos[0] - actual_pos[0]
        error_y = expected_pos[1] - actual_pos[1]
        
        # Gradual drift correction to avoid jumps
        correction_factor = 0.3
        self.drift_correction_x += error_x * correction_factor
        self.drift_correction_y += error_y * correction_factor
        
        print(f"🎯 Drift correction applied: ({error_x:.3f}, {error_y:.3f})")
    
    def get_position(self):
        return (self.current_x, self.current_y, self.current_z)
    
    def cleanup(self):
        try:
            self.chassis.unsub_position()
        except:
            pass

# ===== Enhanced Movement Controller =====
class MovementController:
    def __init__(self, chassis, localizer):
        self.chassis = chassis
        self.localizer = localizer
        
        # PID Parameters
        self.KP = 2.1
        self.KI = 0.3
        self.KD = 10
        self.RAMP_UP_TIME = 0.5
        self.ROTATE_TIME = 2.11
        self.ROTATE_LEFT_TIME = 1.9
        
        # Auto-centering parameters
        self.TARGET_WALL_DISTANCE = 40.0  # 40cm from walls
        self.CENTERING_TOLERANCE = 5.0    # 5cm tolerance
    
    def auto_center_in_corridor(self, left_distance, right_distance):
        """Auto-center robot in corridor when too close to walls"""
        center_needed = False
        move_direction = None
        
        if left_distance < self.TARGET_WALL_DISTANCE and left_distance > 5:
            # Too close to left wall - move right
            center_needed = True
            move_direction = 'right'
            move_distance = (self.TARGET_WALL_DISTANCE - left_distance) / 100.0  # Convert to meters
        elif right_distance < self.TARGET_WALL_DISTANCE and right_distance > 5:
            # Too close to right wall - move left
            center_needed = True
            move_direction = 'left'
            move_distance = (self.TARGET_WALL_DISTANCE - right_distance) / 100.0
        
        if center_needed:
            print(f"🎯 Auto-centering: Moving {move_direction} by {move_distance:.2f}m")
            if move_direction == 'left':
                self.chassis.drive_speed(x=0, y=0.3, z=0, timeout=move_distance/0.3)
                time.sleep(move_distance/0.3 + 0.2)
            else:  # right
                self.chassis.drive_speed(x=0, y=-0.3, z=0, timeout=move_distance/0.3)
                time.sleep(move_distance/0.3 + 0.2)
            
            self.chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
            time.sleep(0.3)
            return True
        
        return False
    
    def move_forward_with_pid(self, target_distance, axis='x', direction=1):
        """Move forward using PID control"""
        pid = PID(Kp=self.KP, Ki=self.KI, Kd=self.KD, setpoint=target_distance)
        
        start_time = time.time()
        last_time = start_time
        target_reached = False
        
        min_speed = 0.1
        max_speed = 1.5
        
        start_x, start_y, _ = self.localizer.get_position()
        if axis == 'x':
            start_position = start_x
        else:
            start_position = start_y

        print(f"🚀 Moving {target_distance}m on {axis}-axis, direction: {direction}")
        
        try:
            while not target_reached:
                now = time.time()
                dt = now - last_time
                last_time = now
                elapsed_time = now - start_time
                
                current_x, current_y, _ = self.localizer.get_position()
                if axis == 'x':
                    current_position = current_x
                else:
                    current_position = current_y
                
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
    
    def move_backward_with_pid(self, target_distance, axis='x'):
        """Move backward using PID control"""
        return self.move_forward_with_pid(target_distance, axis, direction=-1)
    
    def rotate_90_degrees_right(self):
        print("🔄 Rotating 90° RIGHT...")
        time.sleep(0.25)
        self.chassis.drive_speed(x=0, y=0, z=45, timeout=self.ROTATE_TIME)
        time.sleep(self.ROTATE_TIME + 0.2)
        self.chassis.drive_speed(x=0, y=0, z=0, timeout=0.5)
        time.sleep(0.25)
        print("✅ Right rotation completed!")

    def rotate_90_degrees_left(self):
        print("🔄 Rotating 90° LEFT...")
        time.sleep(0.25)
        self.chassis.drive_speed(x=0, y=0, z=-45, timeout=self.ROTATE_LEFT_TIME)
        time.sleep(self.ROTATE_LEFT_TIME + 0.2)
        self.chassis.drive_speed(x=0, y=0, z=0, timeout=0.5)
        time.sleep(0.25)
        print("✅ Left rotation completed!")

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
        
        # Distance readings
        self.leftDistance = 0.0
        self.rightDistance = 0.0
        self.frontDistance = 0.0
        
        # Neighbors and connections
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
        self.isFrontier = False
        
        # Gap information
        self.detectedGaps = []  # List of gaps detected from this node
        
        # Path information
        self.parent = None  # For pathfinding
        self.g_cost = float('inf')  # Cost from start
        self.h_cost = 0  # Heuristic cost
        self.f_cost = float('inf')  # Total cost
        
        # Timing
        self.lastVisited = datetime.now().isoformat()
        self.firstVisited = datetime.now().isoformat()
        
        # Phase information
        self.discoveredInPhase = 1

# ===== A* Pathfinding Implementation =====
class AStarPathfinder:
    def __init__(self, graph_mapper):
        self.graph = graph_mapper
    
    def heuristic(self, pos1, pos2):
        """Manhattan distance heuristic"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def get_neighbors_for_pathfinding(self, node):
        """Get valid neighbors for pathfinding (no walls blocking)"""
        neighbors = []
        x, y = node.position
        
        # Check all four directions
        directions = [
            ('north', (x, y+1)),
            ('south', (x, y-1)),
            ('east', (x+1, y)),
            ('west', (x-1, y))
        ]
        
        for direction, next_pos in directions:
            node_id = self.graph.get_node_id(next_pos)
            if node_id in self.graph.nodes:
                next_node = self.graph.nodes[node_id]
                # Check if path is not blocked by walls
                if not self.is_path_blocked(node, next_node, direction):
                    neighbors.append(next_node)
        
        return neighbors
    
    def is_path_blocked(self, from_node, to_node, direction):
        """Check if path between nodes is blocked by walls"""
        # This is a simplified check - in reality you might need more complex logic
        if direction == 'north' and from_node.wallFront:
            return True
        elif direction == 'south' and from_node.wallBack:
            return True
        elif direction == 'east' and from_node.wallRight:
            return True
        elif direction == 'west' and from_node.wallLeft:
            return True
        return False
    
    def find_path(self, start_pos, goal_pos):
        """Find shortest path using A* algorithm"""
        start_id = self.graph.get_node_id(start_pos)
        goal_id = self.graph.get_node_id(goal_pos)
        
        if start_id not in self.graph.nodes or goal_id not in self.graph.nodes:
            return []
        
        start_node = self.graph.nodes[start_id]
        goal_node = self.graph.nodes[goal_id]
        
        # Initialize
        open_set = []
        heapq.heappush(open_set, (0, start_id))
        
        # Reset costs for all nodes
        for node in self.graph.nodes.values():
            node.g_cost = float('inf')
            node.f_cost = float('inf')
            node.parent = None
        
        start_node.g_cost = 0
        start_node.h_cost = self.heuristic(start_pos, goal_pos)
        start_node.f_cost = start_node.h_cost
        
        closed_set = set()
        
        while open_set:
            current_f, current_id = heapq.heappop(open_set)
            
            if current_id in closed_set:
                continue
                
            current_node = self.graph.nodes[current_id]
            closed_set.add(current_id)
            
            if current_id == goal_id:
                # Reconstruct path
                path = []
                node = current_node
                while node:
                    path.append(node.position)
                    node = node.parent
                path.reverse()
                return path
            
            for neighbor in self.get_neighbors_for_pathfinding(current_node):
                if neighbor.id in closed_set:
                    continue
                
                tentative_g = current_node.g_cost + 1  # Assume cost of 1 between adjacent nodes
                
                if tentative_g < neighbor.g_cost:
                    neighbor.parent = current_node
                    neighbor.g_cost = tentative_g
                    neighbor.h_cost = self.heuristic(neighbor.position, goal_pos)
                    neighbor.f_cost = neighbor.g_cost + neighbor.h_cost
                    
                    heapq.heappush(open_set, (neighbor.f_cost, neighbor.id))
        
        return []  # No path found

# ===== Time Management System =====
class TimeManager:
    def __init__(self, total_time=600):  # 10 minutes = 600 seconds
        self.total_time = total_time
        self.start_time = time.time()
        self.phase_durations = {
            1: 180,  # Phase 1: 3 minutes
            2: 240,  # Phase 2: 4 minutes  
            3: 180   # Phase 3: 3 minutes
        }
        self.current_phase = 1
        self.phase_start_time = self.start_time
    
    def get_elapsed_time(self):
        return time.time() - self.start_time
    
    def get_remaining_time(self):
        return self.total_time - self.get_elapsed_time()
    
    def get_phase_elapsed_time(self):
        return time.time() - self.phase_start_time
    
    def get_phase_remaining_time(self):
        return self.phase_durations[self.current_phase] - self.get_phase_elapsed_time()
    
    def should_advance_phase(self):
        return self.get_phase_elapsed_time() >= self.phase_durations[self.current_phase]
    
    def advance_phase(self):
        if self.current_phase < 3:
            self.current_phase += 1
            self.phase_start_time = time.time()
            print(f"⏰ ADVANCING TO PHASE {self.current_phase}")
            return True
        return False
    
    def is_exploration_complete(self):
        return self.get_elapsed_time() >= self.total_time
    
    def get_status(self):
        return {
            'current_phase': self.current_phase,
            'total_elapsed': self.get_elapsed_time(),
            'total_remaining': self.get_remaining_time(),
            'phase_elapsed': self.get_phase_elapsed_time(),
            'phase_remaining': self.get_phase_remaining_time()
        }

# ===== Enhanced Graph Mapper with Hybrid Strategy =====
class HybridGraphMapper:
    def __init__(self, time_manager):
        self.nodes = {}
        self.currentPosition = (0, 0)
        self.currentDirection = 'north'
        
        # Path tracking for backtracking
        self.path_stack = []  # Stack for backtracking
        self.visited_path = []  # Full path history
        
        # Frontier management
        self.frontiers = []
        self.gaps = []  # Detected gaps for phase 3
        
        # Phase management
        self.time_manager = time_manager
        self.pathfinder = AStarPathfinder(self)
        
        # Phase-specific states
        self.wall_following_direction = 'left'  # Follow left wall
        self.last_junction = None
        
        # Coverage tracking
        self.total_area_explored = 0
        self.nodes_per_phase = {1: 0, 2: 0, 3: 0}
    
    def get_node_id(self, position):
        return f"{position[0]}_{position[1]}"
    
    def create_or_get_node(self, position):
        node_id = self.get_node_id(position)
        if node_id not in self.nodes:
            self.nodes[node_id] = GraphNode(node_id, position)
            self.nodes[node_id].discoveredInPhase = self.time_manager.current_phase
            self.nodes_per_phase[self.time_manager.current_phase] += 1
        return self.nodes[node_id]
    
    def get_current_node(self):
        return self.create_or_get_node(self.currentPosition)
    
    def update_current_node_walls(self, left_wall, right_wall, front_wall, left_dist, right_dist, front_dist):
        """Update current node with wall and distance information"""
        current_node = self.get_current_node()
        
        # Update wall information
        current_node.wallLeft = left_wall
        current_node.wallRight = right_wall  
        current_node.wallFront = front_wall
        
        # Update distance information
        current_node.leftDistance = left_dist
        current_node.rightDistance = right_dist
        current_node.frontDistance = front_dist
        
        # Update timing
        current_node.lastVisited = datetime.now().isoformat()
        current_node.visitCount += 1
        
        # Detect gaps for Phase 3
        self.detect_gaps(current_node)
        
        # Update exploration status
        self.update_exploration_status(current_node)
    
    def detect_gaps(self, node):
        """Detect potential gaps (unexplored areas) from current node"""
        gaps_found = []
        
        # Check for gaps in each direction
        directions = ['north', 'south', 'east', 'west']
        
        for direction in directions:
            if not self.is_direction_blocked(node, direction):
                next_pos = self.get_next_position(direction)
                next_node_id = self.get_node_id(next_pos)
                
                if next_node_id not in self.nodes:
                    # This is a potential gap
                    gaps_found.append({
                        'position': next_pos,
                        'direction': direction,
                        'distance_from_origin': abs(next_pos[0]) + abs(next_pos[1]),
                        'discovered_time': datetime.now().isoformat(),
                        'priority': self.calculate_gap_priority(next_pos)
                    })
        
        # Add new gaps to global list
        for gap in gaps_found:
            if gap not in self.gaps:
                self.gaps.append(gap)
                print(f"🔍 Gap detected at {gap['position']} via {gap['direction']}")
    
    def calculate_gap_priority(self, gap_position):
        """Calculate priority for gap filling (closer = higher priority)"""
        distance = math.sqrt((gap_position[0] - self.currentPosition[0])**2 + 
                        (gap_position[1] - self.currentPosition[1])**2)
        return 1.0 / (distance + 1.0)  # Higher value = higher priority
    
    def is_direction_blocked(self, node, direction):
        """Check if a direction is blocked by walls"""
        direction_map = {
            'north': {'front': 'north', 'left': 'west', 'right': 'east', 'back': 'south'},
            'south': {'front': 'south', 'left': 'east', 'right': 'west', 'back': 'north'},
            'east': {'front': 'east', 'left': 'north', 'right': 'south', 'back': 'west'},
            'west': {'front': 'west', 'left': 'south', 'right': 'north', 'back': 'east'}
        }
        
        current_directions = direction_map[self.currentDirection]
        
        if direction == current_directions['front']:
            return node.wallFront
        elif direction == current_directions['left']:
            return node.wallLeft
        elif direction == current_directions['right']:
            return node.wallRight
        elif direction == current_directions['back']:
            return node.wallBack  # Assume we can check back wall
        
        return False
    
    def update_exploration_status(self, node):
        """Update node exploration status and frontier list"""
        # Find unexplored exits
        node.unexploredExits = []
        
        directions = ['north', 'south', 'east', 'west']
        for direction in directions:
            if not self.is_direction_blocked(node, direction):
                if direction not in node.exploredDirections:
                    node.unexploredExits.append(direction)
        
        # Update frontier status
        node.isFrontier = len(node.unexploredExits) > 0
        
        # Update global frontier list
        if node.isFrontier and node.id not in [f.id for f in self.frontiers]:
            self.frontiers.append(node)
        elif not node.isFrontier and node.id in [f.id for f in self.frontiers]:
            self.frontiers = [f for f in self.frontiers if f.id != node.id]
        
        # Check if dead end
        node.isDeadEnd = (node.wallFront and node.wallLeft and node.wallRight)
    
    def get_next_position(self, direction):
        """Calculate next position based on direction"""
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

    # ===== PHASE 1: LEFT WALL FOLLOWING =====
    def phase1_wall_following_strategy(self):
        """Phase 1: Left wall following strategy"""
        current_node = self.get_current_node()
        
        print("🧱 PHASE 1: Wall Following (Left Priority)")
        
        # Priority: Left > Front > Right > Back
        if not current_node.wallLeft:
            return 'left'
        elif not current_node.wallFront:
            return 'front'
        elif not current_node.wallRight:
            return 'right'
        else:
            return 'back'  # Need to backtrack
    
    # ===== PHASE 2: FRONTIER EXPLORATION =====
    def phase2_frontier_strategy(self):
        """Phase 2: Strategic Frontier Exploration"""
        print("🚀 PHASE 2: Frontier Exploration")
        
        if not self.frontiers:
            return None
        
        # Hybrid frontier selection
        best_frontier = self.select_best_frontier()
        
        if best_frontier:
            # Find path to frontier
            path = self.pathfinder.find_path(self.currentPosition, best_frontier.position)
            if path and len(path) > 1:
                next_pos = path[1]  # Next step in path
                return self.get_direction_to_position(next_pos)
        
        return None
    
    def select_best_frontier(self):
        """Select best frontier using hybrid criteria"""
        if not self.frontiers:
            return None
        
        scored_frontiers = []
        
        for frontier in self.frontiers:
            # Distance score (closer = better)
            distance = math.sqrt((frontier.position[0] - self.currentPosition[0])**2 + 
                            (frontier.position[1] - self.currentPosition[1])**2)
            distance_score = 1.0 / (distance + 1.0)
            
            # Unexplored exits score (more exits = better)
            exits_score = len(frontier.unexploredExits)
            
            # Systematic exploration bonus (prefer unvisited areas)
            systematic_score = 1.0 / (frontier.visitCount + 1.0)
            
            # Combined score
            total_score = distance_score * 0.4 + exits_score * 0.4 + systematic_score * 0.2
            
            scored_frontiers.append((total_score, frontier))
        
        # Return highest scoring frontier
        scored_frontiers.sort(reverse=True)
        return scored_frontiers[0][1]
    
    # ===== PHASE 3: GAP FILLING =====
    def phase3_gap_filling_strategy(self):
        """Phase 3: Fill remaining gaps"""
        print("🔍 PHASE 3: Gap Filling")
        
        if not self.gaps:
            return None
        
        # Sort gaps by priority (distance from current position)
        current_gaps = []
        for gap in self.gaps:
            gap_id = self.get_node_id(gap['position'])
            if gap_id not in self.nodes:  # Still unexplored
                distance = math.sqrt((gap['position'][0] - self.currentPosition[0])**2 + 
                                (gap['position'][1] - self.currentPosition[1])**2)
                current_gaps.append((distance, gap))
        
        if current_gaps:
            current_gaps.sort()  # Sort by distance
            nearest_gap = current_gaps[0][1]
            
            # Find path to gap
            path = self.pathfinder.find_path(self.currentPosition, nearest_gap['position'])
            if path and len(path) > 1:
                next_pos = path[1]
                return self.get_direction_to_position(next_pos)
        
        return None
    
    def get_direction_to_position(self, target_pos):
        """Get direction to reach target position from current position"""
        dx = target_pos[0] - self.currentPosition[0]
        dy = target_pos[1] - self.currentPosition[1]
        
        if dx > 0:
            return 'east'
        elif dx < 0:
            return 'west'
        elif dy > 0:
            return 'north'
        elif dy < 0:
            return 'south'
        
        return None
    
    def get_next_exploration_direction(self):
        """Get next exploration direction based on current phase"""
        phase = self.time_manager.current_phase
        
        if phase == 1:
            return self.phase1_wall_following_strategy()
        elif phase == 2:
            return self.phase2_frontier_strategy()
        elif phase == 3:
            return self.phase3_gap_filling_strategy()
        
        return None
    
    def move_to_direction(self, target_direction, movement_controller):
        """Execute movement to target direction"""
        if not target_direction:
            return False
        
        # Check if movement is possible
        if not self.can_move_to_direction(target_direction):
            print(f"❌ BLOCKED! Cannot move to {target_direction}")
            return False
        
        # Record current position for backtracking
        self.path_stack.append(self.currentPosition)
        self.visited_path.append(self.currentPosition)
        
        print(f"🎯 Moving from {self.currentDirection} to {target_direction}")
        
        # Calculate and execute rotation
        self.rotate_to_direction(target_direction, movement_controller)
        
        # Move forward
        axis, direction = self.get_movement_axis_and_direction(target_direction)
        movement_controller.move_forward_with_pid(0.6, axis, direction=direction)
        
        # Update position and direction
        self.currentPosition = self.get_next_position(target_direction)
        self.currentDirection = target_direction
        
        # Mark direction as explored from previous node
        if self.path__