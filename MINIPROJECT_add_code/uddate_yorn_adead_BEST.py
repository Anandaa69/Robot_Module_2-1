import time
import robomaster
from robomaster import robot
import numpy as np
from scipy.ndimage import median_filter
from datetime import datetime
import json
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import threading
import os

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

# ===== Map Visualizer =====
class MapVisualizer:
    def __init__(self, graph_mapper):
        self.graph_mapper = graph_mapper
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.fig.suptitle('Robot Autonomous Exploration Map', fontsize=16, fontweight='bold')
        
        # Path tracking
        self.movement_path = [graph_mapper.currentPosition]
        self.path_timestamps = [datetime.now()]
        self.movement_counter = 0
        
        # Colors and styles
        self.colors = {
            'robot': '#FF4444',
            'visited': '#4CAF50',
            'dead_end': '#F44336',
            'frontier': '#FF9800',
            'wall': '#333333',
            'path': '#2196F3',
            'unexplored': '#E0E0E0'
        }
        
        # Animation setup
        self.animation_active = False
        self.update_interval = 500  # ms
        
    def setup_plot(self):
        """Setup the initial plot configuration"""
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X Position (Grid Units)', fontweight='bold')
        self.ax.set_ylabel('Y Position (Grid Units)', fontweight='bold')
        
        # Create legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors['robot'], 
                    markersize=12, label='Current Robot Position'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=self.colors['visited'], 
                    markersize=10, label='Visited Node'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=self.colors['dead_end'], 
                    markersize=10, label='Dead End'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=self.colors['frontier'], 
                    markersize=10, label='Frontier Node'),
            plt.Line2D([0], [0], color=self.colors['wall'], linewidth=4, label='Wall'),
            plt.Line2D([0], [0], color=self.colors['path'], linewidth=2, label='Robot Path'),
        ]
        self.ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
        
        plt.tight_layout()
    
    def add_movement(self, new_position):
        """Add a new position to the movement path"""
        self.movement_path.append(new_position)
        self.path_timestamps.append(datetime.now())
        self.movement_counter += 1
        
        print(f"📍 Movement #{self.movement_counter}: {new_position}")
    
    def update_plot(self, save_image=False):
        """Update the plot with current graph state"""
        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        
        if not self.graph_mapper.nodes:
            return
        
        # Get map boundaries
        positions = [node.position for node in self.graph_mapper.nodes.values()]
        if self.movement_path:
            positions.extend(self.movement_path)
        
        min_x = min(pos[0] for pos in positions) - 1
        max_x = max(pos[0] for pos in positions) + 1
        min_y = min(pos[1] for pos in positions) - 1
        max_y = max(pos[1] for pos in positions) + 1
        
        self.ax.set_xlim(min_x - 0.5, max_x + 0.5)
        self.ax.set_ylim(min_y - 0.5, max_y + 0.5)
        
        # Draw grid background
        for x in range(int(min_x), int(max_x) + 1):
            self.ax.axvline(x, color='lightgray', alpha=0.3, linewidth=0.5)
        for y in range(int(min_y), int(max_y) + 1):
            self.ax.axhline(y, color='lightgray', alpha=0.3, linewidth=0.5)
        
        # Draw walls
        self.draw_walls()
        
        # Draw movement path
        self.draw_movement_path()
        
        # Draw nodes
        self.draw_nodes()
        
        # Draw robot current position
        self.draw_robot()
        
        # Update title with statistics
        self.update_title()
        
        # Recreate legend
        self.setup_plot()
        
        if save_image:
            self.save_map_image()
        
        plt.draw()
    
    def draw_walls(self):
        """Draw walls between nodes"""
        wall_segments = []
        
        for node in self.graph_mapper.nodes.values():
            x, y = node.position
            
            # Check each direction for walls
            if node.walls.get('north', False):
                wall_segments.append([(x-0.4, y+0.5), (x+0.4, y+0.5)])
            if node.walls.get('south', False):
                wall_segments.append([(x-0.4, y-0.5), (x+0.4, y-0.5)])
            if node.walls.get('east', False):
                wall_segments.append([(x+0.5, y-0.4), (x+0.5, y+0.4)])
            if node.walls.get('west', False):
                wall_segments.append([(x-0.5, y-0.4), (x-0.5, y+0.4)])
        
        # Draw all wall segments
        for segment in wall_segments:
            self.ax.plot([segment[0][0], segment[1][0]], 
                        [segment[0][1], segment[1][1]], 
                        color=self.colors['wall'], linewidth=4, solid_capstyle='round')
    
    def draw_movement_path(self):
        """Draw the robot's movement path"""
        if len(self.movement_path) > 1:
            path_x = [pos[0] for pos in self.movement_path]
            path_y = [pos[1] for pos in self.movement_path]
            
            # Draw path with arrows showing direction
            for i in range(len(path_x) - 1):
                dx = path_x[i+1] - path_x[i]
                dy = path_y[i+1] - path_y[i]
                
                # Draw line segment
                self.ax.plot([path_x[i], path_x[i+1]], [path_y[i], path_y[i+1]], 
                        color=self.colors['path'], linewidth=2, alpha=0.7)
                
                # Add arrow showing direction
                if dx != 0 or dy != 0:
                    self.ax.arrow(path_x[i] + dx*0.3, path_y[i] + dy*0.3, 
                                dx*0.2, dy*0.2, head_width=0.1, head_length=0.1, 
                                fc=self.colors['path'], ec=self.colors['path'], alpha=0.8)
    
    def draw_nodes(self):
        """Draw all nodes with appropriate colors and labels"""
        for node in self.graph_mapper.nodes.values():
            x, y = node.position
            
            # Determine node color
            if node.isDeadEnd:
                color = self.colors['dead_end']
                marker = 's'
                size = 150
            elif node.id in self.graph_mapper.frontierQueue:
                color = self.colors['frontier']
                marker = 's'
                size = 120
            else:
                color = self.colors['visited']
                marker = 's'
                size = 100
            
            # Draw node
            self.ax.scatter(x, y, c=color, marker=marker, s=size, 
                        edgecolors='black', linewidth=1, zorder=3)
            
            # Add node label
            label = f"{node.id}"
            if node.unexploredExits:
                label += f"\n({len(node.unexploredExits)} exits)"
            
            self.ax.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points',
                        fontsize=8, ha='left', va='bottom', 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    def draw_robot(self):
        """Draw robot at current position with direction indicator"""
        x, y = self.graph_mapper.currentPosition
        
        # Draw robot
        self.ax.scatter(x, y, c=self.colors['robot'], marker='o', s=200, 
                    edgecolors='darkred', linewidth=2, zorder=5)
        
        # Draw direction arrow
        direction_vectors = {
            'north': (0, 0.3),
            'south': (0, -0.3),
            'east': (0.3, 0),
            'west': (-0.3, 0)
        }
        
        if self.graph_mapper.currentDirection in direction_vectors:
            dx, dy = direction_vectors[self.graph_mapper.currentDirection]
            self.ax.arrow(x, y, dx, dy, head_width=0.15, head_length=0.1,
                        fc='darkred', ec='darkred', linewidth=2, zorder=6)
        
        # Add robot label
        self.ax.annotate(f'ROBOT\n{self.graph_mapper.currentDirection.upper()}', 
                        (x, y), xytext=(-20, -30), textcoords='offset points',
                        fontsize=10, ha='center', va='top', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.8, edgecolor='darkred'),
                        color='white')
    
    def update_title(self):
        """Update plot title with current statistics"""
        total_nodes = len(self.graph_mapper.nodes)
        dead_ends = sum(1 for node in self.graph_mapper.nodes.values() if node.isDeadEnd)
        frontiers = len(self.graph_mapper.frontierQueue)
        movements = len(self.movement_path) - 1
        
        title = f'Robot Exploration Map - Nodes: {total_nodes} | Dead Ends: {dead_ends} | Frontiers: {frontiers} | Movements: {movements}'
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        self.ax.set_xlabel('X Position (Grid Units)', fontweight='bold')
        self.ax.set_ylabel('Y Position (Grid Units)', fontweight='bold')
    
    def save_map_image(self, filename=None):
        """Save current map as image"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"robot_map_{timestamp}.png"
        
        try:
            plt.savefig(filename, dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
            print(f"💾 Map saved as: {filename}")
        except Exception as e:
            print(f"❌ Failed to save map: {e}")
    
    def show_live_map(self):
        """Show live updating map"""
        plt.ion()  # Turn on interactive mode
        self.setup_plot()
        plt.show(block=False)
    
    def close_map(self):
        """Close map window"""
        plt.close(self.fig)
    
    def generate_exploration_animation(self, filename="robot_exploration.gif", interval=1000):
        """Generate animated GIF of the exploration process"""
        print("🎬 Generating exploration animation...")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        fig.suptitle('Robot Autonomous Exploration Animation', fontsize=16, fontweight='bold')
        
        def animate(frame):
            ax.clear()
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            # Show path up to current frame
            current_path = self.movement_path[:frame+1]
            
            if len(current_path) > 1:
                path_x = [pos[0] for pos in current_path]
                path_y = [pos[1] for pos in current_path]
                ax.plot(path_x, path_y, color=self.colors['path'], linewidth=3, alpha=0.8)
            
            # Show current position...
            if current_path:
                x, y = current_path[-1]
                ax.scatter(x, y, c=self.colors['robot'], marker='o', s=300, 
                        edgecolors='darkred', linewidth=3, zorder=5)
            
            # Set consistent bounds
            if self.movement_path:
                all_x = [pos[0] for pos in self.movement_path]
                all_y = [pos[1] for pos in self.movement_path]
                ax.set_xlim(min(all_x) - 1, max(all_x) + 1)
                ax.set_ylim(min(all_y) - 1, max(all_y) + 1)
            
            ax.set_title(f'Exploration Step: {frame + 1}/{len(self.movement_path)}', 
                        fontsize=14, fontweight='bold')
        
        try:
            anim = FuncAnimation(fig, animate, frames=len(self.movement_path), 
                            interval=interval, repeat=True)
            anim.save(filename, writer='pillow', fps=1)
            print(f"🎬 Animation saved as: {filename}")
            plt.close(fig)
        except Exception as e:
            print(f"❌ Failed to create animation: {e}")
            plt.close(fig)

# ===== Movement Controller =====
class MovementController:
    def __init__(self, chassis, visualizer=None):
        self.chassis = chassis
        self.visualizer = visualizer
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        
        # PID Parameters
        self.KP = 1.5
        self.KI = 0.3
        self.KD = 4
        self.RAMP_UP_TIME = 0.7
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
        
        # Ramp-up parameters
        min_speed = 0.1
        max_speed = 1.5
        
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
                    self.chassis.drive_speed(x=speed * direction, y=0, z=0, timeout=1)
                else:
                    self.chassis.drive_speed(x=speed * direction, y=0, z=0, timeout=1)

                # Stop condition
                if abs(relative_position - target_distance) < 0.017:
                    print(f"✅ Target reached! Final position: {current_position:.3f}")
                    self.chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
                    target_reached = True
                    break
                    
        except KeyboardInterrupt:
            print("Movement interrupted by user.")
    
    def rotate_90_degrees_right(self):
        """Rotate 90 degrees clockwise"""
        print("🔄 Rotating 90° RIGHT...")
        time.sleep(0.2)
        self.chassis.drive_speed(x=0, y=0, z=45, timeout=self.ROTATE_TIME)
        time.sleep(self.ROTATE_TIME + 0.3)
        time.sleep(0.2)
        print("✅ Right rotation completed!")

    def rotate_90_degrees_left(self):
        """Rotate 90 degrees counter-clockwise"""
        print("🔄 Rotating 90° LEFT...")
        time.sleep(0.2)
        self.chassis.drive_speed(x=0, y=0, z=-45, timeout=self.ROTATE_LEFT_TIME)
        time.sleep(self.ROTATE_LEFT_TIME + 0.3)
        time.sleep(0.2)
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

    def reverse_to_previous_node(self):
        """NEW: Reverse 0.6m to go back to previous node without rotating"""
        global ROBOT_FACE
        print("🔙 BACKTRACKING - Reversing to previous node...")
        
        # Determine current axis based on robot face
        axis_test = 'x'
        if ROBOT_FACE % 2 == 0:
            axis_test = 'y'
        elif ROBOT_FACE % 2 == 1:
            axis_test = 'x'
        
        print(f"🔙 Reversing 0.6m on {axis_test}-axis for backtrack")
        
        # Move backward using negative direction
        self.move_forward_with_pid(0.6, axis_test, direction=-1)
        
        print("✅ Reverse backtrack completed!")
    
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

        # Wall detection - NOW STORES ABSOLUTE DIRECTIONS
        self.walls = {
            'north': False,
            'south': False,
            'east': False,
            'west': False
        }

        # Legacy support (will be removed)
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

        # Additional info
        self.marker = False
        self.lastVisited = datetime.now().isoformat()
        self.sensorReadings = {}
        
        # IMPORTANT: Store the ABSOLUTE direction robot was facing when first scanned
        self.initialScanDirection = None

# ===== Graph Mapper =====
class GraphMapper:
    def __init__(self, visualizer=None):
        self.nodes = {}
        self.currentPosition = (0, 0)
        self.currentDirection = 'north'  # ABSOLUTE direction robot is facing
        self.frontierQueue = []
        self.pathStack = []
        self.visitedNodes = set()
        self.previous_node = None
        self.visualizer = visualizer
        
        # Override methods เพื่อใช้ priority-based exploration
        self.find_next_exploration_direction = self.find_next_exploration_direction_with_priority
        self.update_unexplored_exits_absolute = self.update_unexplored_exits_with_priority

    def get_node_id(self, position):
        return f"{position[0]}_{position[1]}"
    
    def create_node(self, position):
        node_id = self.get_node_id(position)
        if node_id not in self.nodes:
            node = GraphNode(node_id, position)
            # Store the absolute direction robot was facing when node was first created
            node.initialScanDirection = self.currentDirection
            self.nodes[node_id] = node
            self.visitedNodes.add(node_id)
            
            # Update visualizer if available
            if self.visualizer:
                self.visualizer.update_plot()
                
        return self.nodes[node_id]

    def get_current_node(self):
        node_id = self.get_node_id(self.currentPosition)
        return self.nodes.get(node_id)
    
    def update_current_node_walls_absolute(self, left_wall, right_wall, front_wall):
        """NEW: Update walls using ABSOLUTE directions"""
        current_node = self.get_current_node()
        if current_node:
            # Map relative sensor readings to absolute directions
            direction_map = {
                'north': {'front': 'north', 'left': 'west', 'right': 'east'},
                'south': {'front': 'south', 'left': 'east', 'right': 'west'},
                'east': {'front': 'east', 'left': 'north', 'right': 'south'},
                'west': {'front': 'west', 'left': 'south', 'right': 'north'}
            }
            
            current_mapping = direction_map[self.currentDirection]
            
            # Update absolute wall information
            current_node.walls[current_mapping['front']] = front_wall
            current_node.walls[current_mapping['left']] = left_wall
            current_node.walls[current_mapping['right']] = right_wall
            
            # Legacy support - update old format too
            current_node.wallFront = front_wall
            current_node.wallLeft = left_wall
            current_node.wallRight = right_wall
            
            current_node.lastVisited = datetime.now().isoformat()
            
            # NEW: Mark node as fully scanned
            current_node.fullyScanned = True
            current_node.scanTimestamp = datetime.now().isoformat()
            
            self.update_unexplored_exits_absolute(current_node)
            self.build_connections()
            
            # Update visualizer
            if self.visualizer:
                self.visualizer.update_plot()

    def update_unexplored_exits_with_priority(self, node):
        """Update unexplored exits with priority ordering"""
        node.unexploredExits = []
        
        x, y = node.position
        
        # กำหนดลำดับการตรวจสอบตามความสำคัญ LEFT-FIRST
        # แต่เก็บเป็นทิศทางสัมบูรณ์
        direction_map = {
            'north': {'front': 'north', 'left': 'west', 'right': 'east', 'back': 'south'},
            'south': {'front': 'south', 'left': 'east', 'right': 'west', 'back': 'north'},
            'east': {'front': 'east', 'left': 'north', 'right': 'south', 'back': 'west'},
            'west': {'front': 'west', 'left': 'south', 'right': 'north', 'back': 'east'}
        }
        
        current_mapping = direction_map[self.currentDirection]
        
        # ลำดับความสำคัญ
        priority_order = ['left', 'front', 'right', 'back']
        
        possible_directions = {
            'north': (x, y + 1),
            'south': (x, y - 1),
            'east': (x + 1, y),
            'west': (x - 1, y)
        }
        
        print(f"🧭 Updating unexplored exits for {node.id} at {node.position}")
        print(f"🔍 Wall status: {node.walls}")
        print(f"🤖 Robot facing: {self.currentDirection}")
        
        # ตรวจสอบตามลำดับความสำคัญและเพิ่มเข้า unexploredExits
        for relative_dir in priority_order:
            absolute_dir = current_mapping[relative_dir]
            target_pos = possible_directions[absolute_dir]
            target_node_id = self.get_node_id(target_pos)
            
            # ตรวจสอบเงื่อนไขเหมือนเดิม
            is_blocked = node.walls.get(absolute_dir, True)
            already_explored = absolute_dir in node.exploredDirections
            target_exists = target_node_id in self.nodes
            target_fully_explored = False
            if target_exists:
                target_node = self.nodes[target_node_id]
                target_fully_explored = target_node.fullyScanned
            
            print(f"   📍 {relative_dir} ({absolute_dir}):")
            print(f"      🚧 Blocked: {is_blocked}")
            print(f"      ✅ Already explored: {already_explored}")
            print(f"      🏗️  Target exists: {target_exists}")
            print(f"      🔍 Target fully explored: {target_fully_explored}")
            
            should_explore = (not is_blocked and 
                            not already_explored and 
                            (not target_exists or not target_fully_explored))
            
            if should_explore:
                node.unexploredExits.append(absolute_dir)
                print(f"      ✅ ADDED to unexplored exits! (Priority: {relative_dir})")
            else:
                print(f"      ❌ NOT added to unexplored exits")
        
        print(f"🎯 Final unexplored exits (ordered by priority): {node.unexploredExits}")
        
        # อัปเดต frontier queue
        has_unexplored = len(node.unexploredExits) > 0
        
        if has_unexplored and node.id not in self.frontierQueue:
            self.frontierQueue.append(node.id)
            print(f"🚀 Added {node.id} to frontier queue")
        elif not has_unexplored and node.id in self.frontierQueue:
            self.frontierQueue.remove(node.id)
            print(f"🧹 Removed {node.id} from frontier queue")
        
        # Dead end detection
        blocked_count = sum(1 for blocked in node.walls.values() if blocked)