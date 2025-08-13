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

ROBOT_FACE = 1 
CURRENT_TARGET_YAW = 0.0


class MapVisualizer:
    def __init__(self, grid_size=0.6, enable_realtime=True, save_images=True):
        """
        Map Visualizer for Robot Exploration
        
        Args:
            grid_size (float): Size of each grid cell in meters (default: 0.6m)
            enable_realtime (bool): Enable real-time visualization
            save_images (bool): Save map images during exploration
        """
        self.grid_size = grid_size
        self.enable_realtime = enable_realtime
        self.save_images = save_images
        
        # Visualization settings
        self.node_size = 0.4  # Size of node markers
        self.wall_width = 0.05  # Width of wall lines
        self.robot_size = 0.3  # Size of robot marker
        
        # Colors
        self.colors = {
            'visited_node': '#87CEEB',      # Sky blue
            'current_node': '#FF4500',      # Orange red
            'frontier_node': '#32CD32',     # Lime green
            'dead_end': '#8B0000',          # Dark red
            'wall': '#2F4F4F',              # Dark slate gray
            'path': '#FFD700',              # Gold
            'robot': '#FF1493',             # Deep pink
            'unexplored': '#D3D3D3',        # Light gray
            'background': '#F5F5F5'         # White smoke
        }
        
        # Map data storage
        self.nodes = {}
        self.walls = []
        self.robot_position = (0, 0)
        self.robot_direction = 'north'
        self.robot_path = [(0, 0)]
        self.exploration_history = []
        
        # Matplotlib setup
        if self.enable_realtime:
            plt.ion()  # Interactive mode
            self.fig, self.ax = plt.subplots(figsize=(12, 10))
            self.fig.patch.set_facecolor(self.colors['background'])
            self.ax.set_facecolor(self.colors['background'])
            
            # Setup plot parameters
            self.ax.set_aspect('equal')
            self.ax.grid(True, alpha=0.3, linestyle='--')
            self.ax.set_title('🤖 Robot Exploration Map (Real-time)', 
                            fontsize=16, fontweight='bold', pad=20)
            
            # Create legend
            self._setup_legend()
            
            # Thread-safe plotting
            self.plot_lock = threading.Lock()
        
        print("🗺️ MapVisualizer initialized successfully!")
        if enable_realtime:
            print("📊 Real-time visualization enabled")
        if save_images:
            print("💾 Image saving enabled")
    
    def _setup_legend(self):
        """Setup legend for the map visualization"""
        legend_elements = [
            plt.scatter([], [], c=self.colors['current_node'], s=200, 
                       marker='o', label='🤖 Current Position'),
            plt.scatter([], [], c=self.colors['visited_node'], s=100, 
                       marker='s', label='✅ Visited Node'),
            plt.scatter([], [], c=self.colors['frontier_node'], s=100, 
                       marker='^', label='🚀 Frontier Node'),
            plt.scatter([], [], c=self.colors['dead_end'], s=100, 
                       marker='X', label='🚫 Dead End'),
            plt.Line2D([0], [0], color=self.colors['wall'], linewidth=4, 
                      label='🧱 Wall'),
            plt.Line2D([0], [0], color=self.colors['path'], linewidth=2, 
                      label='🛤️ Robot Path')
        ]
        
        self.ax.legend(handles=legend_elements, loc='upper left', 
                      bbox_to_anchor=(0.02, 0.98), framealpha=0.9)
    
    def update_from_graph_mapper(self, graph_mapper, movement_controller=None):
        """Update visualization data from GraphMapper instance"""
        with self.plot_lock if self.enable_realtime else threading.Lock():
            # Update nodes
            self.nodes = {}
            for node_id, node in graph_mapper.nodes.items():
                self.nodes[node_id] = {
                    'position': node.position,
                    'walls': node.walls,
                    'is_dead_end': node.isDeadEnd,
                    'visited': node.visited,
                    'fully_scanned': node.fullyScanned,
                    'unexplored_exits': node.unexploredExits,
                    'visit_count': node.visitCount,
                    'last_visited': node.lastVisited
                }
            
            # Update robot position and direction
            self.robot_position = graph_mapper.currentPosition
            self.robot_direction = graph_mapper.currentDirection
            
            # Add to path if position changed
            if len(self.robot_path) == 0 or self.robot_path[-1] != self.robot_position:
                self.robot_path.append(self.robot_position)
            
            # Update frontier queue
            self.frontier_nodes = graph_mapper.frontierQueue
            
            # Add exploration event
            timestamp = datetime.now().strftime("%H:%M:%S")
            stats = {}
            if movement_controller:
                drift_status = movement_controller.get_drift_correction_status()
                movement_status = movement_controller.get_movement_status()
                stats = {
                    'nodes_visited': drift_status['nodes_visited'],
                    'corrections': drift_status['total_corrections'],
                    'consecutive_forward': movement_status['consecutive_forward'],
                    'consecutive_backward': movement_status['consecutive_backward']
                }
            
            self.exploration_history.append({
                'timestamp': timestamp,
                'position': self.robot_position,
                'direction': self.robot_direction,
                'total_nodes': len(self.nodes),
                'frontiers': len(self.frontier_nodes),
                'stats': stats
            })
    
    def _generate_walls_from_nodes(self):
        """Generate wall lines from node wall information"""
        walls = []
        
        for node_id, node_data in self.nodes.items():
            x, y = node_data['position']
            node_walls = node_data['walls']
            
            # Convert grid position to visualization coordinates
            grid_x = x * self.grid_size
            grid_y = y * self.grid_size
            half_grid = self.grid_size / 2
            
            # Generate wall lines based on absolute directions
            if node_walls.get('north', False):
                walls.append([
                    [grid_x - half_grid, grid_y + half_grid],
                    [grid_x + half_grid, grid_y + half_grid]
                ])
            
            if node_walls.get('south', False):
                walls.append([
                    [grid_x - half_grid, grid_y - half_grid],
                    [grid_x + half_grid, grid_y - half_grid]
                ])
            
            if node_walls.get('east', False):
                walls.append([
                    [grid_x + half_grid, grid_y - half_grid],
                    [grid_x + half_grid, grid_y + half_grid]
                ])
            
            if node_walls.get('west', False):
                walls.append([
                    [grid_x - half_grid, grid_y - half_grid],
                    [grid_x - half_grid, grid_y + half_grid]
                ])
        
        return walls
    
    def _draw_direction_arrow(self, x, y, direction, size=0.2):
        """Draw direction arrow for robot orientation"""
        # Direction vectors
        direction_vectors = {
            'north': (0, size),
            'south': (0, -size),
            'east': (size, 0),
            'west': (-size, 0)
        }
        
        if direction in direction_vectors:
            dx, dy = direction_vectors[direction]
            self.ax.arrow(x, y, dx, dy, 
                         head_width=size/3, head_length=size/4,
                         fc=self.colors['robot'], ec=self.colors['robot'],
                         linewidth=2, alpha=0.9)
    
    def plot_current_map(self, title_suffix="", show_stats=True):
        """Plot current exploration map"""
        if not self.enable_realtime:
            self.fig, self.ax = plt.subplots(figsize=(12, 10))
            self.fig.patch.set_facecolor(self.colors['background'])
            self.ax.set_facecolor(self.colors['background'])
        
        with self.plot_lock if self.enable_realtime else threading.Lock():
            # Clear previous plot
            self.ax.clear()
            
            # Plot nodes
            for node_id, node_data in self.nodes.items():
                x, y = node_data['position']
                grid_x, grid_y = x * self.grid_size, y * self.grid_size
                
                # Determine node color and marker
                if (x, y) == self.robot_position:
                    color = self.colors['current_node']
                    marker = 'o'
                    size = 300
                    alpha = 1.0
                elif node_data['is_dead_end']:
                    color = self.colors['dead_end']
                    marker = 'X'
                    size = 200
                    alpha = 0.8
                elif node_id in self.frontier_nodes:
                    color = self.colors['frontier_node']
                    marker = '^'
                    size = 150
                    alpha = 0.9
                else:
                    color = self.colors['visited_node']
                    marker = 's'
                    size = 100
                    alpha = 0.7
                
                # Plot node
                self.ax.scatter(grid_x, grid_y, c=color, marker=marker, 
                              s=size, alpha=alpha, edgecolors='black', linewidth=1)
                
                # Add node label
                label = f"({x},{y})"
                if node_data.get('visit_count', 1) > 1:
                    label += f"\n×{node_data['visit_count']}"
                
                self.ax.annotate(label, (grid_x, grid_y), 
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.8, fontweight='bold')
            
            # Plot walls
            walls = self._generate_walls_from_nodes()
            for wall in walls:
                wall_line = np.array(wall)
                self.ax.plot(wall_line[:, 0], wall_line[:, 1], 
                           color=self.colors['wall'], linewidth=4, alpha=0.8)
            
            # Plot robot path
            if len(self.robot_path) > 1:
                path_array = np.array(self.robot_path) * self.grid_size
                self.ax.plot(path_array[:, 0], path_array[:, 1], 
                           color=self.colors['path'], linewidth=3, alpha=0.7,
                           linestyle='-', marker='.')
            
            # Plot robot with direction arrow
            robot_x, robot_y = self.robot_position[0] * self.grid_size, self.robot_position[1] * self.grid_size
            self.ax.scatter(robot_x, robot_y, c=self.colors['robot'], 
                          marker='o', s=400, alpha=1.0, edgecolors='white', linewidth=2)
            self._draw_direction_arrow(robot_x, robot_y, self.robot_direction)
            
            # Set plot properties
            self.ax.set_aspect('equal')
            self.ax.grid(True, alpha=0.3, linestyle='--')
            
            # Calculate plot bounds
            if self.nodes:
                positions = [node['position'] for node in self.nodes.values()]
                min_x = min(pos[0] for pos in positions) * self.grid_size - self.grid_size
                max_x = max(pos[0] for pos in positions) * self.grid_size + self.grid_size
                min_y = min(pos[1] for pos in positions) * self.grid_size - self.grid_size
                max_y = max(pos[1] for pos in positions) * self.grid_size + self.grid_size
                
                self.ax.set_xlim(min_x, max_x)
                self.ax.set_ylim(min_y, max_y)
            
            # Title with current statistics
            base_title = f"🤖 Robot Exploration Map {title_suffix}"
            if show_stats and self.exploration_history:
                latest = self.exploration_history[-1]
                stats_text = f" | 📍 ({self.robot_position[0]},{self.robot_position[1]}) | 🧭 {self.robot_direction} | 🗺️ {latest['total_nodes']} nodes | 🚀 {latest['frontiers']} frontiers"
                base_title += stats_text
            
            self.ax.set_title(base_title, fontsize=14, fontweight='bold', pad=20)
            
            # Labels
            self.ax.set_xlabel('Distance (meters)', fontsize=12)
            self.ax.set_ylabel('Distance (meters)', fontsize=12)
            
            # Setup legend
            if not self.enable_realtime:
                self._setup_legend()
            else:
                # Update legend for real-time mode
                legend_elements = [
                    plt.scatter([], [], c=self.colors['current_node'], s=200, 
                               marker='o', label='🤖 Current Position'),
                    plt.scatter([], [], c=self.colors['visited_node'], s=100, 
                               marker='s', label='✅ Visited Node'),
                    plt.scatter([], [], c=self.colors['frontier_node'], s=100, 
                               marker='^', label='🚀 Frontier Node'),
                    plt.scatter([], [], c=self.colors['dead_end'], s=100, 
                               marker='X', label='🚫 Dead End'),
                    plt.Line2D([0], [0], color=self.colors['wall'], linewidth=4, 
                              label='🧱 Wall'),
                    plt.Line2D([0], [0], color=self.colors['path'], linewidth=2, 
                              label='🛤️ Robot Path')
                ]
                
                self.ax.legend(handles=legend_elements, loc='upper left', 
                              bbox_to_anchor=(0.02, 0.98), framealpha=0.9)
            
            # Refresh display
            if self.enable_realtime:
                plt.draw()
                plt.pause(0.1)
    
    def save_map_image(self, filename=None, dpi=300):
        """Save current map as image file"""
        if not self.save_images:
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"robot_map_{timestamp}.png"
        
        try:
            # Create a copy for saving (to avoid interfering with real-time display)
            save_fig, save_ax = plt.subplots(figsize=(12, 10))
            save_fig.patch.set_facecolor(self.colors['background'])
            save_ax.set_facecolor(self.colors['background'])
            
            # Plot on save figure (similar to plot_current_map but on save_ax)
            # [Same plotting logic as plot_current_map but using save_ax instead of self.ax]
            
            save_fig.savefig(filename, dpi=dpi, bbox_inches='tight', 
                           facecolor=self.colors['background'], 
                           edgecolor='none')
            plt.close(save_fig)  # Clean up
            
            print(f"💾 Map saved as: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Error saving map image: {e}")
            return None
    
    def generate_exploration_summary(self):
        """Generate text summary of exploration"""
        if not self.exploration_history:
            return "No exploration data available."
        
        summary = []
        summary.append("=" * 50)
        summary.append("📊 EXPLORATION SUMMARY")
        summary.append("=" * 50)
        
        # Basic stats
        latest = self.exploration_history[-1]
        summary.append(f"🗺️ Total nodes mapped: {latest['total_nodes']}")
        summary.append(f"🚀 Active frontiers: {latest['frontiers']}")
        summary.append(f"🛤️ Path length: {len(self.robot_path)} waypoints")
        summary.append(f"📍 Current position: {self.robot_position}")
        summary.append(f"🧭 Current direction: {self.robot_direction}")
        
        # Map boundaries
        if self.nodes:
            positions = [node['position'] for node in self.nodes.values()]
            min_x, max_x = min(pos[0] for pos in positions), max(pos[0] for pos in positions)
            min_y, max_y = min(pos[1] for pos in positions), max(pos[1] for pos in positions)
            summary.append(f"🏁 Map bounds: X({min_x} to {max_x}), Y({min_y} to {max_y})")
            summary.append(f"📏 Map size: {max_x-min_x+1} × {max_y-min_y+1} grid cells")
        
        # Node type counts
        dead_ends = sum(1 for node in self.nodes.values() if node['is_dead_end'])
        fully_scanned = sum(1 for node in self.nodes.values() if node['fully_scanned'])
        
        summary.append(f"🚫 Dead ends: {dead_ends}")
        summary.append(f"🔍 Fully scanned nodes: {fully_scanned}")
        
        # Movement stats from latest data
        if latest['stats']:
            stats = latest['stats']
            summary.append(f"\n🔧 MOVEMENT STATISTICS:")
            summary.append(f"   📊 Total node visits: {stats.get('nodes_visited', 'N/A')}")
            summary.append(f"   🔄 Drift corrections: {stats.get('corrections', 'N/A')}")
            summary.append(f"   ⏭️ Consecutive forward: {stats.get('consecutive_forward', 'N/A')}")
            summary.append(f"   ⏪ Consecutive backward: {stats.get('consecutive_backward', 'N/A')}")
        
        # Exploration timeline
        if len(self.exploration_history) > 1:
            summary.append(f"\n📅 EXPLORATION TIMELINE:")
            summary.append(f"   🚀 Started: {self.exploration_history[0]['timestamp']}")
            summary.append(f"   🏁 Latest: {self.exploration_history[-1]['timestamp']}")
            summary.append(f"   ⏱️ Updates: {len(self.exploration_history)}")
        
        summary.append("=" * 50)
        
        return "\n".join(summary)
    
    def show_map(self, block=True):
        """Show the current map visualization"""
        if self.enable_realtime:
            if block:
                plt.show(block=True)
        else:
            self.plot_current_map()
            plt.show(block=block)
    
    def close(self):
        """Clean up visualization resources"""
        if self.enable_realtime:
            plt.ioff()
            if hasattr(self, 'fig'):
                plt.close(self.fig)
        print("🗺️ MapVisualizer closed successfully!")

# ===== Integration Helper Functions =====

def integrate_visualizer_with_exploration(graph_mapper, movement_controller, 
                                        enable_realtime=True, save_images=True):
    """
    Helper function to integrate MapVisualizer with existing exploration code
    
    Usage:
        visualizer = integrate_visualizer_with_exploration(graph_mapper, movement_controller)
    """
    visualizer = MapVisualizer(enable_realtime=enable_realtime, save_images=save_images)
    
    # Initial update
    visualizer.update_from_graph_mapper(graph_mapper, movement_controller)
    visualizer.plot_current_map()
    
    return visualizer

def update_visualization_during_exploration(visualizer, graph_mapper, movement_controller,
                                          auto_save=False, save_interval=10):
    """
    Helper function to update visualization during exploration
    Call this function after each exploration step
    
    Args:
        visualizer: MapVisualizer instance
        graph_mapper: GraphMapper instance
        movement_controller: MovementController instance
        auto_save: Automatically save images
        save_interval: Save every N updates (if auto_save=True)
    """
    # Update data
    visualizer.update_from_graph_mapper(graph_mapper, movement_controller)
    
    # Plot current state
    visualizer.plot_current_map()
    
    # Auto-save if enabled
    if auto_save and len(visualizer.exploration_history) % save_interval == 0:
        step_num = len(visualizer.exploration_history)
        filename = f"exploration_step_{step_num:03d}.png"
        visualizer.save_map_image(filename)
    
    return visualizer

# เพิ่มคลาสใหม่สำหรับติดตามการเคลื่อนไหว
class MovementTracker:
    def __init__(self):
        self.consecutive_forward_moves = 0
        self.consecutive_backward_moves = 0
        self.movement_history = []  # เก็บประวัติการเคลื่อนไหว
        self.last_movement_type = None  # 'forward', 'backward', 'rotation'
        
    def record_movement(self, movement_type):
        """บันทึกการเคลื่อนไหวและอัพเดทตัวนับ"""
        self.movement_history.append(movement_type)
        
        if movement_type == 'forward':
            if self.last_movement_type == 'forward':
                self.consecutive_forward_moves += 1
            else:
                self.consecutive_forward_moves = 1
            # รีเซ็ตตัวนับถอยหลัง
            self.consecutive_backward_moves = 0
            
        elif movement_type == 'backward':
            if self.last_movement_type == 'backward':
                self.consecutive_backward_moves += 1
            else:
                self.consecutive_backward_moves = 1
            # รีเซ็ตตัวนับเดินหน้า
            self.consecutive_forward_moves = 0
            
        elif movement_type == 'rotation':
            # การหมุนจะรีเซ็ตทั้งสองตัวนับ
            self.consecutive_forward_moves = 0
            self.consecutive_backward_moves = 0
        
        self.last_movement_type = movement_type
        
        print(f"📊 Movement recorded: {movement_type}")
        print(f"   🔄 Consecutive forward: {self.consecutive_forward_moves}")
        print(f"   ↩️ Consecutive backward: {self.consecutive_backward_moves}")
    
    def has_consecutive_forward_moves(self, threshold=2):
        """เช็คว่ามีการเดินหน้าติดกันตามจำนวนที่กำหนดหรือไม่"""
        return self.consecutive_forward_moves >= threshold
    
    def has_consecutive_backward_moves(self, threshold=2):
        """เช็คว่ามีการถอยหลังติดกันตามจำนวนที่กำหนดหรือไม่"""
        return self.consecutive_backward_moves >= threshold
    
    def reset_counters(self):
        """รีเซ็ตตัวนับทั้งหมด"""
        self.consecutive_forward_moves = 0
        self.consecutive_backward_moves = 0
        self.last_movement_type = None
        print("🔄 Movement counters reset!")
    
    def get_movement_status(self):
        """ส่งคืนสถานะการเคลื่อนไหวปัจจุบัน"""
        return {
            'consecutive_forward': self.consecutive_forward_moves,
            'consecutive_backward': self.consecutive_backward_moves,
            'last_movement': self.last_movement_type,
            'history_length': len(self.movement_history)
        }

class AttitudeHandler:
    def __init__(self):
        self.current_yaw = 0.0
        self.current_pitch = 0.0
        self.current_roll = 0.0
        self.target_yaw = 0.0
        self.yaw_tolerance = 3
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
        # ✅ เริ่ม monitor ก่อนหมุน
        self.start_monitoring(chassis)

        if self.is_at_target_yaw(target_yaw):
            print(f"✅ Chassis already at correct yaw: {self.current_yaw:.1f}° (target: {target_yaw}°)")
            self.stop_monitoring(chassis)  # ✅ หยุด monitor
            return True

        gimbal_to_target = target_yaw - self.current_yaw
        gimbal_diff = self.normalize_angle(gimbal_to_target)
        robot_rotation = -gimbal_diff

        print(f"🔧 Correcting chassis yaw: from {self.current_yaw:.1f}° to {target_yaw}°")
        print(f"📐 Gimbal needs to change: {gimbal_diff:.1f}°")
        print(f"📐 Robot will rotate: {robot_rotation:.1f}°")

        try:
            if abs(robot_rotation) > self.yaw_tolerance:
                correction_speed = 60
                print(f"🔄 Rotating robot {robot_rotation:.1f}°")
                chassis.move(x=0, y=0, z=robot_rotation, z_speed=correction_speed).wait_for_completed()
                time.sleep(0.3)

            final_check = self.is_at_target_yaw(target_yaw)

            self.stop_monitoring(chassis)  # ✅ หยุด monitor หลังหมุนเสร็จ

            if final_check:
                print(f"✅ Successfully corrected chassis yaw to {self.current_yaw:.1f}°")
                return True
            else:
                print(f"⚠️ Chassis yaw correction incomplete: {self.current_yaw:.1f}° (target: {target_yaw}°)")
                return False

        except Exception as e:
            self.stop_monitoring(chassis)  # ✅ หยุด monitor แม้ error
            print(f"❌ Failed to correct chassis yaw: {e}")
            return False

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
        self.KP = 2.08
        self.KI = 0.25
        self.KD = 10
        self.RAMP_UP_TIME = 0.7
        self.ROTATE_TIME = 2.11  # Right turn
        self.ROTATE_LEFT_TIME = 1.9  # Left turn

        # เพิ่ม MovementTracker
        self.movement_tracker = MovementTracker()

        # เพิ่มระบบติดตาม attitude drift correction
        self.nodes_visited_count = 0  # นับจำนวนโหนดที่ผ่านไป
        self.DRIFT_CORRECTION_INTERVAL = 10  # ทุก 9 โหนด
        self.DRIFT_CORRECTION_ANGLE = 2  # หมุนเพิ่ม 3 องศาไปทางขวา
        self.total_drift_corrections = 0  # นับจำนวนครั้งที่แก้ไข
        
        # *** เพิ่มตัวแปรใหม่สำหรับป้องกัน drift correction ซ้ำ ***
        self.last_correction_at = 0  # เก็บ node count ครั้งสุดท้ายที่ทำ correction

        # Subscribe to position updates
        self.chassis.sub_position(freq=20, callback=self.position_handler)
        time.sleep(0.25)
    
    def position_handler(self, position_info):
        self.current_x = position_info[0]
        self.current_y = position_info[1]
        self.current_z = position_info[2]

    def increment_node_visit_for_backtrack_with_correction(self, attitude_handler):
        """เพิ่มจำนวนโหนดที่เยียมชมสำหรับการ backtrack และ trigger drift correction ถ้าถึงเวลา"""
        self.nodes_visited_count += 1
        print(f"📊 Backtrack node visit count: {self.nodes_visited_count}")
        
        # เช็คว่าต้องทำ drift correction หรือไม่ และยังไม่เคยทำที่ count นี้
        if (self.nodes_visited_count % self.DRIFT_CORRECTION_INTERVAL == 0 and 
            self.nodes_visited_count != self.last_correction_at):
            
            print(f"🔧 BACKTRACK DRIFT CORRECTION TRIGGERED!")
            print(f"   📍 After {self.nodes_visited_count} total nodes visited (including backtrack)")
            print(f"   🔄 Correction #{self.total_drift_corrections + 1}")
            
            # ทำ drift correction
            success = self.perform_attitude_drift_correction(attitude_handler)
            self.last_correction_at = self.nodes_visited_count  # บันทึกว่าทำ correction แล้ว
            
            if success:
                print(f"✅ Backtrack drift correction completed!")
                return True
            else:
                print(f"⚠️ Backtrack drift correction had issues, but continuing...")
                return False
                
        return False

    def increment_node_visit_main_exploration(self, attitude_handler):
        """เพิ่มจำนวนโหนดที่เยียมชมสำหรับ main exploration loop และ trigger drift correction ถ้าถึงเวลา"""
        self.nodes_visited_count += 1
        print(f"📊 Main exploration node visit count: {self.nodes_visited_count}")
        
        # เช็คว่าต้องทำ drift correction หรือไม่ และยังไม่เคยทำที่ count นี้
        if (self.nodes_visited_count % self.DRIFT_CORRECTION_INTERVAL == 0 and 
            self.nodes_visited_count != self.last_correction_at):
            
            print(f"🔧 MAIN EXPLORATION DRIFT CORRECTION TRIGGERED!")
            print(f"   📍 After {self.nodes_visited_count} total nodes visited")
            print(f"   🔄 Correction #{self.total_drift_corrections + 1}")
            
            # ทำ drift correction
            success = self.perform_attitude_drift_correction(attitude_handler)
            self.last_correction_at = self.nodes_visited_count  # บันทึกว่าทำ correction แล้ว
            
            if success:
                print(f"✅ Main exploration drift correction completed!")
                return True
            else:
                print(f"⚠️ Main exploration drift correction had issues, but continuing...")
                return False
                
        return False

    def perform_attitude_drift_correction(self, attitude_handler):
        global CURRENT_TARGET_YAW
        print(f"⚙️ === PERFORMING ATTITUDE DRIFT CORRECTION ===")

        # ✅ เริ่ม monitor
        attitude_handler.start_monitoring(self.chassis)

        current_yaw_before = attitude_handler.current_yaw
        target_after_correction = attitude_handler.normalize_angle(
            current_yaw_before + self.DRIFT_CORRECTION_ANGLE
        )
        CURRENT_TARGET_YAW = target_after_correction

        try:
            success = attitude_handler.correct_yaw_to_target(self.chassis, target_after_correction)

            # ✅ หยุด monitor หลังเสร็จ
            attitude_handler.stop_monitoring(self.chassis)

            if success:
                self.total_drift_corrections += 1
                return True
            else:
                return False

        except Exception as e:
            attitude_handler.stop_monitoring(self.chassis)  # ✅ หยุดแม้ error
            print(f"❌ Error during attitude drift correction: {e}")
            return False


    def move_forward_with_pid(self, target_distance, axis, direction=1, allow_yaw_correction=True, attitude_handler=None):
        """Move forward using PID control with movement tracking"""
        # บันทึกการเคลื่อนไหว
        movement_type = 'forward' if direction == 1 else 'backward'
        self.movement_tracker.record_movement(movement_type)
        
        # เช็คว่ามีการเคลื่อนไหวติดกันหรือไม่ (เฉพาะกรณีอนุญาตแก้ yaw)
        if allow_yaw_correction:
            if self.movement_tracker.has_consecutive_forward_moves(2):
                print("⚠️ DETECTED: 2 consecutive forward moves!")
                target_angle = attitude_handler.normalize_angle(CURRENT_TARGET_YAW)
                print(f"🎯 Target yaw: {target_angle}°")
                attitude_handler.correct_yaw_to_target(self.chassis, target_angle)
                
            if self.movement_tracker.has_consecutive_backward_moves(2):
                print("⚠️ DETECTED: 2 consecutive backward moves!")
                target_angle = attitude_handler.normalize_angle(CURRENT_TARGET_YAW)
                print(f"🎯 Target yaw: {target_angle}°")
                attitude_handler.correct_yaw_to_target(self.chassis, target_angle)

        # เช็คว่ามีการเคลื่อนไหวติดกันหรือไม่
        if self.movement_tracker.has_consecutive_forward_moves(2):
            print("⚠️ DETECTED: 2 consecutive forward moves!")
            # print("🔍 PATTERN DETECTED: 2+ consecutive forward moves")
            target_angle = attitude_handler.normalize_angle(CURRENT_TARGET_YAW)
            
            print(f"🎯 Target yaw: {target_angle}°")
            success = attitude_handler.correct_yaw_to_target(self.chassis, target_angle)
            
        if self.movement_tracker.has_consecutive_backward_moves(2):
            print("⚠️ DETECTED: 2 consecutive backward moves!")
            target_angle = attitude_handler.normalize_angle(CURRENT_TARGET_YAW)
            
            print(f"🎯 Target yaw: {target_angle}°")
            success = attitude_handler.correct_yaw_to_target(self.chassis, target_angle)
        
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
                if abs(relative_position - target_distance) < 0.02:
                    print(f"✅ Target reached! Final position: {current_position:.3f}")
                    self.chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
                    target_reached = True
                    break
                    
        except KeyboardInterrupt:
            print("Movement interrupted by user.")
    
    def rotate_90_degrees_right(self, attitude_handler=None):
        global CURRENT_TARGET_YAW
        print("🔄 Rotating 90° RIGHT...")
        self.movement_tracker.record_movement('rotation')
        time.sleep(0.2)

        # ✅ เริ่ม monitor
        attitude_handler.start_monitoring(self.chassis)

        CURRENT_TARGET_YAW += 90
        target_angle = attitude_handler.normalize_angle(CURRENT_TARGET_YAW)

        print(f"🎯 Target yaw: {target_angle}°")
        success = attitude_handler.correct_yaw_to_target(self.chassis, target_angle)

        # ✅ หยุด monitor
        attitude_handler.stop_monitoring(self.chassis)

        if success:
            print("✅ Right rotation completed!")
        else:
            print("⚠️ Right rotation may be incomplete")

        time.sleep(0.2)

    def rotate_90_degrees_left(self, attitude_handler=None):
        global CURRENT_TARGET_YAW
        print("🔄 Rotating 90° LEFT...")
        self.movement_tracker.record_movement('rotation')
        time.sleep(0.2)

        # ✅ เริ่ม monitor
        attitude_handler.start_monitoring(self.chassis)

        CURRENT_TARGET_YAW -= 90
        target_angle = attitude_handler.normalize_angle(CURRENT_TARGET_YAW)

        print(f"🎯 Target yaw: {target_angle}°")
        success = attitude_handler.correct_yaw_to_target(self.chassis, target_angle)

        # ✅ หยุด monitor
        attitude_handler.stop_monitoring(self.chassis)

        if success:
            print("✅ Left rotation completed!")
        else:
            print("⚠️ Left rotation may be incomplete")

        time.sleep(0.2)
    
    def reverse_from_dead_end(self):
        """Reverse robot from dead end position"""
        global ROBOT_FACE
        print("🔙 DEAD END DETECTED - Reversing...")
        # บันทึกการถอยหลัง
        self.movement_tracker.record_movement('backward')
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
        self.movement_tracker.record_movement('backward')

        global ROBOT_FACE
        print("🔙 BACKTRACKING - Reversing to previous node...")

        axis_test = 'x'
        if ROBOT_FACE % 2 == 0:
            axis_test = 'y'
        elif ROBOT_FACE % 2 == 1:
            axis_test = 'x'
        
        print(f"🔙 Reversing 0.6m on {axis_test}-axis for backtrack")
        
        # ❌ ปิด yaw correction ระหว่าง backtracking
        self.move_forward_with_pid(0.6, axis_test, direction=-1, allow_yaw_correction=False)
        
        print("✅ Reverse backtrack completed!")
    
    def cleanup(self):
        """Clean up position subscription"""
        try:
            self.chassis.unsub_position()
        except:
            pass

    def get_drift_correction_status(self):
        """ส่งคืนสถานะการแก้ไข drift ปัจจุบัน"""
        return {
            'nodes_visited': self.nodes_visited_count,
            'next_correction_at': ((self.nodes_visited_count // self.DRIFT_CORRECTION_INTERVAL) + 1) * self.DRIFT_CORRECTION_INTERVAL,
            'nodes_until_correction': self.DRIFT_CORRECTION_INTERVAL - (self.nodes_visited_count % self.DRIFT_CORRECTION_INTERVAL),
            'total_corrections': self.total_drift_corrections,
            'correction_interval': self.DRIFT_CORRECTION_INTERVAL,
            'correction_angle': self.DRIFT_CORRECTION_ANGLE,
            'last_correction_at': self.last_correction_at
        }
    
    def reset_drift_correction(self):
        """รีเซ็ตระบบแก้ไข drift (สำหรับ testing หรือเริ่มใหม่)"""
        self.nodes_visited_count = 0
        self.total_drift_corrections = 0
        print("🔄 Drift correction system reset!")

    def get_movement_status(self):
        """ส่งคืนสถานะการเคลื่อนไหวปัจจุบัน"""
        return self.movement_tracker.get_movement_status()
    
    def reset_movement_tracking(self):
        """รีเซ็ตการติดตามการเคลื่อนไหว"""
        self.movement_tracker.reset_counters()

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
    def __init__(self):
        self.nodes = {}
        self.currentPosition = (0, 0)
        self.currentDirection = 'north'  # ABSOLUTE direction robot is facing
        self.frontierQueue = []
        self.pathStack = []
        self.visitedNodes = set()
        self.previous_node = None
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

    def update_unexplored_exits_absolute(self, node):
        """FIXED: Update unexplored exits using ABSOLUTE directions"""
        node.unexploredExits = []
        
        x, y = node.position
        
        # Define all possible directions from this node (ABSOLUTE)
        possible_directions = {
            'north': (x, y + 1),
            'south': (x, y - 1),
            'east': (x + 1, y),
            'west': (x - 1, y)
        }
        
        print(f"🧭 Updating unexplored exits for {node.id} at {node.position}")
        print(f"🔍 Wall status: {node.walls}")
        
        # Check each ABSOLUTE direction for unexplored exits
        for direction, target_pos in possible_directions.items():
            target_node_id = self.get_node_id(target_pos)
            
            # Check if this direction is blocked by wall
            is_blocked = node.walls.get(direction, True)
            
            # Check if already explored
            already_explored = direction in node.exploredDirections
            
            # Check if target node exists and is fully explored
            target_exists = target_node_id in self.nodes
            target_fully_explored = False
            if target_exists:
                target_node = self.nodes[target_node_id]
                target_fully_explored = target_node.fullyScanned
            
            print(f"   📍 Direction {direction}:")
            print(f"      🚧 Blocked: {is_blocked}")
            print(f"      ✅ Already explored: {already_explored}")
            print(f"      🏗️  Target exists: {target_exists}")
            print(f"      🔍 Target fully explored: {target_fully_explored}")
            
            # Add to unexplored exits if:
            # 1. Not blocked by wall AND
            # 2. Not already explored from this node AND
            # 3. Target doesn't exist OR target exists but hasn't been fully scanned
            should_explore = (not is_blocked and 
                            not already_explored and 
                            (not target_exists or not target_fully_explored))
            
            if should_explore:
                node.unexploredExits.append(direction)
                print(f"      ✅ ADDED to unexplored exits!")
            else:
                print(f"      ❌ NOT added to unexplored exits")
        
        print(f"🎯 Final unexplored exits for {node.id}: {node.unexploredExits}")
        
        # Update frontier queue
        has_unexplored = len(node.unexploredExits) > 0
        
        if has_unexplored and node.id not in self.frontierQueue:
            self.frontierQueue.append(node.id)
            print(f"🚀 Added {node.id} to frontier queue")
        elif not has_unexplored and node.id in self.frontierQueue:
            self.frontierQueue.remove(node.id)
            print(f"🧹 Removed {node.id} from frontier queue")
        
        # Dead end detection using absolute directions
        blocked_count = sum(1 for blocked in node.walls.values() if blocked)
        is_dead_end = blocked_count >= 3  # 3 or more walls = dead end
        node.isDeadEnd = is_dead_end
        
        if is_dead_end:
            print(f"🚫 DEAD END CONFIRMED at {node.id} - {blocked_count} walls detected!")
            if node.id in self.frontierQueue:
                self.frontierQueue.remove(node.id)
                print(f"🧹 Removed dead end {node.id} from frontier queue")
    
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
    
    def get_next_position_from(self, position, direction):
        """Helper method to calculate next position from given position and direction"""
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
        """Check if current node or given node is a dead end"""
        if node is None:
            node = self.get_current_node()
        
        if not node:
            return False
        
        return node.isDeadEnd
    
    def get_next_position(self, direction):
        """Calculate next position based on current position and ABSOLUTE direction"""
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
        """Check if robot can move to target ABSOLUTE direction"""
        current_node = self.get_current_node()
        if not current_node:
            return False
        
        # Check if target direction has a wall using absolute direction
        is_blocked = current_node.walls.get(target_direction, True)
        return not is_blocked
    
    def rotate_to_absolute_direction(self, target_direction, movement_controller, attitude_handler):
        """NEW: Rotate robot to face target ABSOLUTE direction"""
        global ROBOT_FACE
        global CURRENT_TARGET_YAW
        print(f"🎯 Rotating from {self.currentDirection} to {target_direction}")
        
        if self.currentDirection == target_direction:
            print(f"✅ Already facing {target_direction}")
            return
        
        direction_order = ['north', 'east', 'south', 'west']
        current_idx = direction_order.index(self.currentDirection)
        target_idx = direction_order.index(target_direction)
        
        # Calculate shortest rotation
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
        
        # Update current direction
        self.currentDirection = target_direction
        print(f"✅ Now facing {self.currentDirection}")

    def handle_dead_end(self, movement_controller):
        """Handle dead end situation by reversing"""
        print(f"🚫 === DEAD END HANDLER ACTIVATED ===")
        current_node = self.get_current_node()

        if current_node:
            print(f"📍 Dead end at position: {current_node.position}")
            print(f"🧱 Walls: {current_node.walls}")

        # Use the reverse method
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

        return True
    
    def move_to_absolute_direction(self, target_direction, movement_controller, attitude_handler):
        """NEW: Move to target ABSOLUTE direction with proper rotation"""
        global ROBOT_FACE
        print(f"🎯 Moving to ABSOLUTE direction: {target_direction}")
        
        # Check if movement is possible
        if not self.can_move_to_direction_absolute(target_direction):
            print(f"❌ BLOCKED! Cannot move to {target_direction} - wall detected!")
            return False
        
        # First rotate to face the target direction
        self.rotate_to_absolute_direction(target_direction, movement_controller, attitude_handler)

        # เช็คสถานะการเคลื่อนไหวก่อนเคลื่อนที่
        movement_status = movement_controller.get_movement_status()
        print(f"📊 Current movement status: {movement_status}")

        # Determine axis for movement
        axis_test = 'x'
        if ROBOT_FACE % 2 == 0:
            axis_test = 'y'
        elif ROBOT_FACE % 2 == 1:
            axis_test = 'x'
        
        print(f"🚀 Moving forward on {axis_test}-axis")

        # Move forward (จะบันทึกเป็น 'forward' ใน MovementTracker)
        movement_controller.move_forward_with_pid(0.6, axis_test, direction=1)
        
        # Update position
        self.currentPosition = self.get_next_position(target_direction)
        
        # Mark this direction as explored from the previous node
        if hasattr(self, 'previous_node') and self.previous_node:
            if target_direction not in self.previous_node.exploredDirections:
                self.previous_node.exploredDirections.append(target_direction)
            
            # Remove from unexplored exits
            if target_direction in self.previous_node.unexploredExits:
                self.previous_node.unexploredExits.remove(target_direction)
                print(f"🔄 Removed {target_direction} from unexplored exits of {self.previous_node.id}")
        
        print(f"✅ Successfully moved to {self.currentPosition}")
        return True

    def reverse_to_absolute_direction(self, target_direction, movement_controller, attitude_handler):
        """NEW: Reverse to target ABSOLUTE direction for backtracking"""
        global ROBOT_FACE
        print(f"🔙 BACKTRACK: Reversing to ABSOLUTE direction: {target_direction}")
        
        # Calculate what direction we need to reverse to
        reverse_direction_map = {
            'north': 'south',
            'south': 'north',
            'east': 'west',
            'west': 'east'
        }
        
        required_facing_direction = reverse_direction_map[target_direction]
        
        # First rotate to face the direction OPPOSITE to where we want to go
        self.rotate_to_absolute_direction(required_facing_direction, movement_controller, attitude_handler)
        
        # Now reverse (which will move us in the target direction)
        movement_controller.reverse_to_previous_node()
        
        # Update position
        self.currentPosition = self.get_next_position(target_direction)
        
        print(f"✅ Successfully reversed to {self.currentPosition}, still facing {self.currentDirection}")
        return True

    def find_next_exploration_direction_with_priority(self):
        """Find next exploration direction with LEFT-first priority"""
        current_node = self.get_current_node()
        if not current_node:
            return None
        
        if self.is_dead_end(current_node):
            print(f"🚫 Current node is a dead end - no exploration directions available")
            return None
        
        print(f"🧭 Current robot facing: {self.currentDirection}")
        print(f"🔍 Available unexplored exits: {current_node.unexploredExits}")
        
        # กำหนดลำดับความสำคัญตามทิศทางสัมพันธ์ (LEFT-FIRST STRATEGY)
        # แปลงทิศทางสัมบูรณ์กลับเป็นทิศทางสัมพันธ์เพื่อจัดลำดับ
        direction_map = {
            'north': {'front': 'north', 'left': 'west', 'right': 'east', 'back': 'south'},
            'south': {'front': 'south', 'left': 'east', 'right': 'west', 'back': 'north'},
            'east': {'front': 'east', 'left': 'north', 'right': 'south', 'back': 'west'},
            'west': {'front': 'west', 'left': 'south', 'right': 'north', 'back': 'east'}
        }
        
        current_mapping = direction_map[self.currentDirection]
        
        # สร้าง reverse mapping (จากทิศทางสัมบูรณ์เป็นทิศทางสัมพันธ์)
        reverse_mapping = {v: k for k, v in current_mapping.items()}
        
        # ลำดับความสำคัญ: ซ้าย → หน้า → ขวา → หลัง
        priority_order = ['left', 'front', 'right', 'back']
        
        print(f"🎯 Checking exploration priority order: {priority_order}")
        
        # ตรวจสอบตามลำดับความสำคัญ
        for relative_direction in priority_order:
            # แปลงเป็นทิศทางสัมบูรณ์
            absolute_direction = current_mapping.get(relative_direction)
            
            if absolute_direction and absolute_direction in current_node.unexploredExits:
                if self.can_move_to_direction_absolute(absolute_direction):
                    print(f"✅ Selected direction: {relative_direction} ({absolute_direction})")
                    return absolute_direction
                else:
                    print(f"❌ {relative_direction} ({absolute_direction}) is blocked by wall!")
                    # ลบออกจาก unexplored exits เพราะมีกำแพง
                    current_node.unexploredExits.remove(absolute_direction)
        
        print(f"❌ No valid exploration direction found")
        return None

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
        is_dead_end = blocked_count >= 3
        node.isDeadEnd = is_dead_end
        
        if is_dead_end:
            print(f"🚫 DEAD END CONFIRMED at {node.id} - {blocked_count} walls detected!")
            if node.id in self.frontierQueue:
                self.frontierQueue.remove(node.id)
                print(f"🧹 Removed dead end {node.id} from frontier queue")

    def find_next_exploration_direction(self):
        """Find the next ABSOLUTE direction to explore"""
        current_node = self.get_current_node()
        if not current_node:
            return None
        
        if self.is_dead_end(current_node):
            print(f"🚫 Current node is a dead end - no exploration directions available")
            return None
        
        # Return first unexplored exit (now using absolute directions)
        if current_node.unexploredExits:
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
                        self.is_path_clear_absolute(current_pos, neighbor_pos, direction)):
                        
                        visited.add(neighbor_pos)
                        new_path = path + [direction]
                        queue.append((neighbor_pos, new_path))
        
        return None  # No path found
    
    def is_path_clear_absolute(self, from_pos, to_pos, direction):
        """Check if path between two adjacent nodes is clear using absolute directions"""
        from_node_id = self.get_node_id(from_pos)
        to_node_id = self.get_node_id(to_pos)
        
        if from_node_id not in self.nodes or to_node_id not in self.nodes:
            return False
        
        from_node = self.nodes[from_node_id]
        
        # Check if there's a wall blocking the path in absolute direction
        is_blocked = from_node.walls.get(direction, True)
        return not is_blocked
    
    def execute_path_to_frontier_with_reverse(self, path, movement_controller, attitude_handler):
        """NEW: Execute path using reverse movements for backtracking WITH NODE COUNTING"""
        print(f"🗺️ Executing REVERSE path to frontier: {path}")
        
        # เช็คสถานะการเคลื่อนไหวก่อนเริ่ม backtrack
        movement_status = movement_controller.get_movement_status()
        print(f"📊 Movement status before backtracking: {movement_status}")

        drift_corrections_during_backtrack = 0
        initial_correction_count = movement_controller.total_drift_corrections

        for i, step_direction in enumerate(path):
            print(f"🔍 Step {i+1}/{len(path)}: Current position: {self.currentPosition}, moving {step_direction}")
            
            # Use reverse movement for backtracking (more efficient)
            success = self.reverse_to_absolute_direction(step_direction, movement_controller, attitude_handler)
            
            if not success:
                print(f"❌ Failed to reverse {step_direction} during backtracking!")
                return False
            
            # *** เพิ่มการนับโหนดสำหรับการ backtrack และเช็ค drift correction ***
            needs_correction = movement_controller.increment_node_visit_for_backtrack_with_correction(attitude_handler)
            
            # ถ้าเกิด drift correction ระหว่าง backtrack
            if needs_correction:
                drift_corrections_during_backtrack = movement_controller.total_drift_corrections - initial_correction_count
                print(f"✅ Backtrack drift correction #{drift_corrections_during_backtrack} completed during step {i+1}!")
            
            time.sleep(0.2)  # Brief pause between moves
        
        if drift_corrections_during_backtrack > 0:
            print(f"🔧 Total drift corrections during this backtrack: {drift_corrections_during_backtrack}")
        
        print(f"✅ Successfully reached frontier at {self.currentPosition}")
        return True
    
    def find_nearest_frontier(self):
        """Find frontier with better validation and prioritization"""
        print("🔍 === FINDING NEAREST FRONTIER ===")
        
        if not self.frontierQueue:
            print("🔄 No frontiers in queue - rebuilding...")
            self.rebuild_frontier_queue()
            
            if not self.frontierQueue:
                print("🎉 No unexplored areas found - exploration complete!")
                return None, None, None
        
        # Validate and prioritize frontiers
        valid_frontiers = []
        
        print(f"🔍 Checking {len(self.frontierQueue)} frontier candidates...")
        
        for frontier_id in self.frontierQueue[:]:
            if frontier_id not in self.nodes:
                print(f"❌ Removing non-existent frontier {frontier_id}")
                continue
                
            frontier_node = self.nodes[frontier_id]
            
            # Re-validate unexplored exits
            print(f"\n🔍 Validating frontier {frontier_id} at {frontier_node.position}:")
            print(f"   📋 Claimed unexplored exits: {frontier_node.unexploredExits}")
            
            # Re-check each unexplored exit
            valid_exits = []
            for exit_direction in frontier_node.unexploredExits[:]:
                target_pos = self.get_next_position_from(frontier_node.position, exit_direction)
                target_node_id = self.get_node_id(target_pos)
                
                # Check if this exit is still valid
                target_exists = target_node_id in self.nodes
                if target_exists:
                    target_node = self.nodes[target_node_id]
                    target_fully_explored = target_node.fullyScanned
                    print(f"      🎯 {exit_direction} -> {target_pos}: exists={target_exists}, fully_explored={target_fully_explored}")
                    
                    # Only consider it unexplored if target doesn't exist or isn't fully explored
                    if not target_fully_explored:
                        valid_exits.append(exit_direction)
                        print(f"         ✅ Still valid for exploration")
                    else:
                        print(f"         ❌ Target already fully explored")
                else:
                    valid_exits.append(exit_direction)
                    print(f"      🎯 {exit_direction} -> {target_pos}: NEW AREA - valid for exploration")
            
            # Update the node's unexplored exits with validated list
            frontier_node.unexploredExits = valid_exits
            
            if valid_exits:
                valid_frontiers.append(frontier_id)
                print(f"   ✅ Frontier {frontier_id} is VALID with exits: {valid_exits}")
            else:
                print(f"   ❌ Frontier {frontier_id} has NO valid unexplored exits")
        
        # Update frontier queue with only valid frontiers
        self.frontierQueue = valid_frontiers
        
        if not valid_frontiers:
            print("🎉 No valid frontiers remaining - exploration complete!")
            return None, None, None
        
        print(f"\n🎯 Found {len(valid_frontiers)} valid frontiers: {valid_frontiers}")
        
        # Find the nearest valid frontier
        best_frontier = None
        best_direction = None
        shortest_path = None
        min_distance = float('inf')
        
        for frontier_id in valid_frontiers:
            frontier_node = self.nodes[frontier_id]
            
            # Find path to this frontier
            path = self.find_path_to_frontier(frontier_id)
            
            if path is not None:
                distance = len(path)
                print(f"   📍 {frontier_id}: distance={distance}, exits={frontier_node.unexploredExits}")
                
                if distance < min_distance:
                    min_distance = distance
                    best_frontier = frontier_id
                    best_direction = frontier_node.unexploredExits[0]  # Take first unexplored direction
                    shortest_path = path
            else:
                print(f"   ❌ {frontier_id}: No path found!")
        
        if best_frontier:
            print(f"\n🏆 SELECTED: {best_frontier} with direction {best_direction} (distance: {min_distance})")
            print(f"🗺️ Path: {shortest_path}")
        else:
            print(f"\n❌ No reachable frontiers found!")
        
        return best_frontier, best_direction, shortest_path
    
    def rebuild_frontier_queue(self):
        """Rebuild frontier queue with comprehensive validation"""
        print("🔄 === REBUILDING FRONTIER QUEUE ===")
        self.frontierQueue = []
        
        for node_id, node in self.nodes.items():
            print(f"\n🔍 Checking node {node_id} at {node.position}:")
            
            # Re-validate unexplored exits for this node
            valid_exits = []
            
            if hasattr(node, 'unexploredExits'):
                print(f"   📋 Current unexplored exits: {node.unexploredExits}")
                
                for exit_direction in node.unexploredExits:
                    target_pos = self.get_next_position_from(node.position, exit_direction)
                    target_node_id = self.get_node_id(target_pos)
                    
                    # Validate this exit
                    target_exists = target_node_id in self.nodes
                    if target_exists:
                        target_node = self.nodes[target_node_id]
                        if not target_node.fullyScanned:
                            valid_exits.append(exit_direction)
                            print(f"      ✅ {exit_direction} -> {target_pos}: Target not fully explored")
                        else:
                            print(f"      ❌ {exit_direction} -> {target_pos}: Target already fully explored")
                    else:
                        valid_exits.append(exit_direction)
                        print(f"      ✅ {exit_direction} -> {target_pos}: NEW AREA")
            
            # Update node's unexplored exits with validated list
            node.unexploredExits = valid_exits
            
            # Add to frontier queue if it has valid unexplored exits
            if valid_exits:
                self.frontierQueue.append(node_id)
                print(f"   🚀 ADDED to frontier queue with exits: {valid_exits}")
            else:
                print(f"   ❌ No valid unexplored exits - not added to frontier")
        
        print(f"\n✅ Frontier queue rebuilt: {len(self.frontierQueue)} frontiers found")
        print(f"🎯 Active frontiers: {self.frontierQueue}")
    
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
            print(f"   🔍 Fully Scanned: {node.fullyScanned}")
            print(f"   🧱 Walls (absolute): {node.walls}")
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

# ===== Main Exploration Functions =====
def scan_current_node_absolute(gimbal, chassis, sensor, tof_handler, graph_mapper):
    """NEW: Scan current node and update graph with ABSOLUTE directions"""
    print(f"\n🗺️ === Scanning Node at {graph_mapper.currentPosition} ===")
    
    current_node = graph_mapper.create_node(graph_mapper.currentPosition)
    
    # Check if node has been fully scanned before
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
    
    # Only scan if node hasn't been fully scanned before
    print(f"🆕 First time visiting node {current_node.id} - performing full scan")
    print(f"🧭 Robot currently facing: {graph_mapper.currentDirection}")
    
    # Lock wheels
    chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
    time.sleep(0.2)
    
    speed = 480
    scan_results = {}
    ep_chassis_fix = ep_robot.chassis
    
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

    if front_distance <= 19.0 : # ถ้าใกล้เกิน19เซน
        move_distance = -(23 - front_distance) #*-1 เพื่อให้ถอยหลัง อ่านได้18เซน move distance=-1*(25-18)=-7cm ถอยหลัง 7cm
        print(f"⚠️ FRONT too close ({front_distance:.2f}cm)! Moving back {move_distance:.2f}m")
        ep_chassis.move(x=move_distance/100, y=0, xy_speed=0.2).wait_for_completed()
        time.sleep(0.2)

    # if 45 > front_distance >= 25:
    #     move_distance=  (front_distance-28)
    #     ep_chassis.move(x=move_distance/100, y=0, xy_speed=0.2).wait_for_completed()
    #     time.sleep(0.2)

    # Scan left (physical: -90°)
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
    
    if left_distance < 15:
        move_distance = 20 - left_distance
        print(f"⚠️ LEFT too close ({left_distance:.2f}cm)! Moving right {move_distance:.2f}m")
        ep_chassis.move(x=0.01, y=move_distance/100, xy_speed=0.5).wait_for_completed()
        time.sleep(0.3)

    # Scan right (physical: 90°)
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

    if right_distance < 15:
        move_distance = -(21 - right_distance)
        print(f"⚠️ RIGHT too close ({right_distance:.2f}cm)! Moving left {move_distance:.2f}m")
        ep_chassis.move(x=0.01, y=move_distance/100, xy_speed=0.5).wait_for_completed()
        time.sleep(0.3)



    print(f"📏 RIGHT scan result: {right_distance:.2f}cm - {'WALL' if right_wall else 'OPEN'}")
    
    # Return to center
    gimbal.moveto(pitch=0, yaw=0, pitch_speed=speed, yaw_speed=speed).wait_for_completed()
    time.sleep(0.2)
    
    # Unlock wheels
    chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0, timeout=0.1)
    time.sleep(0.2)
    
    # NEW: Update graph with wall information using ABSOLUTE directions
    graph_mapper.update_current_node_walls_absolute(left_wall, right_wall, front_wall)
    current_node.sensorReadings = scan_results
    
    print(f"✅ Node {current_node.id} scan complete:")
    print(f"   🧱 Walls detected (relative): Left={left_wall}, Right={right_wall}, Front={front_wall}")
    print(f"   🧱 Walls stored (absolute): {current_node.walls}")
    print(f"   📏 Distances: Left={left_distance:.1f}cm, Right={right_distance:.1f}cm, Front={front_distance:.1f}cm")
    
    return scan_results

def explore_autonomously_with_absolute_directions(gimbal, chassis, sensor, tof_handler, graph_mapper, movement_controller, attitude_handler, max_nodes=20):
    """Main autonomous exploration with attitude drift correction INCLUDING BACKTRACKING"""
    print("\n🚀 === STARTING AUTONOMOUS EXPLORATION WITH COMPREHENSIVE DRIFT CORRECTION ===")
    print(f"🎯 Wall Detection Threshold: {tof_handler.WALL_THRESHOLD}cm")
    print(f"🔧 Attitude Drift Correction: Every {movement_controller.DRIFT_CORRECTION_INTERVAL} nodes (+{movement_controller.DRIFT_CORRECTION_ANGLE}° right)")
    
    nodes_explored = 0
    scanning_iterations = 0
    dead_end_reversals = 0
    backtrack_attempts = 0
    reverse_backtracks = 0
    
    while nodes_explored < max_nodes:
        print(f"\n{'='*50}")
        print(f"--- EXPLORATION STEP {nodes_explored + 1} ---")
        print(f"🤖 Current position: {graph_mapper.currentPosition}")
        print(f"🧭 Current direction (absolute): {graph_mapper.currentDirection}")
        
        # แสดงสถานะ drift correction
        drift_status = movement_controller.get_drift_correction_status()
        print(f"🔧 Comprehensive Drift Correction Status:")
        print(f"   📊 Total nodes visited (including backtrack): {drift_status['nodes_visited']}")
        print(f"   ⏳ Next correction at node: {drift_status['next_correction_at']}")
        print(f"   ⏰ Nodes until correction: {drift_status['nodes_until_correction']}")
        print(f"   🔄 Total corrections done: {drift_status['total_corrections']}")
        print(f"   📍 Last correction at node: {drift_status['last_correction_at']}")
        print(f"{'='*50}")
        
        if nodes_explored % 5 == 0:  # แสดง map ทุกๆ 5 nodes
            print(f"\n📊 === INTERMEDIATE MAP (Node {nodes_explored}) ===")
            plot_exploration_map(graph_mapper, show_plot=True)

        print(f"\n🎉 === FINAL EXPLORATION MAP ===")
        plot_exploration_map(graph_mapper, show_plot=True)
        plot_exploration_progress(graph_mapper)
            
        # *** เพิ่มจำนวนโหนดสำหรับ main exploration และเช็ค drift correction ***
        needs_drift_correction = movement_controller.increment_node_visit_main_exploration(attitude_handler)
        
        if needs_drift_correction:
            print(f"✅ Main exploration drift correction completed!")
        
        # Check if current node needs scanning
        current_node = graph_mapper.create_node(graph_mapper.currentPosition)
        
        if not current_node.fullyScanned:
            print("🔍 NEW NODE - Performing full scan...")
            scan_results = scan_current_node_absolute(gimbal, chassis, sensor, tof_handler, graph_mapper)
            scanning_iterations += 1
            
            # Check if this scan revealed a dead end
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
        else:
            print("⚡ REVISITED NODE - Using cached scan data (no physical scanning)")
            graph_mapper.update_unexplored_exits_absolute(current_node)
            graph_mapper.build_connections()
        
        nodes_explored += 1
        
        # Print current graph state
        graph_mapper.print_graph_summary()
        
        # Find next direction to explore
        graph_mapper.previous_node = current_node
        
        # STEP 1: Try to find unexplored direction from current node
        next_direction = graph_mapper.find_next_exploration_direction()
        
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
        
        # STEP 2: Backtracking logic
        backtrack_attempts += 1
        
        frontier_id, frontier_direction, path = graph_mapper.find_nearest_frontier()
        
        if frontier_id and path is not None and frontier_direction:
            print(f"🎯 Found frontier node {frontier_id} with unexplored direction: {frontier_direction}")
            print(f"🗺️ Path to frontier: {path} (distance: {len(path)} steps)")
            print("🔙 REVERSE BACKTRACK: Using reverse movements WITH drift correction!")
            
            try:
                # *** เปลี่ยนการเรียกใช้ให้ส่ง attitude_handler ไปด้วย ***
                success = graph_mapper.execute_path_to_frontier_with_reverse(path, movement_controller, attitude_handler)
                
                if success:
                    reverse_backtracks += 1
                    print(f"✅ Successfully REVERSE backtracked to frontier at {graph_mapper.currentPosition}")
                    print(f"   📊 Total reverse backtracks: {reverse_backtracks}")
                    
                    # แสดงสถานะ drift correction หลัง backtrack
                    updated_drift_status = movement_controller.get_drift_correction_status()
                    print(f"   🔧 Total nodes after backtrack: {updated_drift_status['nodes_visited']}")
                    print(f"   🔄 Total corrections: {updated_drift_status['total_corrections']}")
                    
                    time.sleep(0.2)
                    continue
                    
                else:
                    print(f"❌ Failed to execute reverse backtracking path!")
                    break
                    
            except Exception as e:
                print(f"❌ Error during reverse backtracking: {e}")
                break
        else:
            # STEP 3: Final check
            print("🎉 No more frontiers found!")
            print("🔄 Performing final frontier scan...")
            graph_mapper.rebuild_frontier_queue()
            
            if graph_mapper.frontierQueue:
                print(f"🚀 Found {len(graph_mapper.frontierQueue)} missed frontiers - continuing...")
                continue
            else:
                print("🎉 EXPLORATION DEFINITELY COMPLETE!")
                break
        
        if nodes_explored >= max_nodes:
            print(f"⚠️ Reached maximum nodes limit ({max_nodes})")
            break
    
    # Final statistics
    final_drift_status = movement_controller.get_drift_correction_status()
    
    print(f"\n🎉 === EXPLORATION COMPLETED ===")
    print(f"📊 PERFORMANCE SUMMARY:")
    print(f"   🗺️ Total exploration steps: {nodes_explored}")
    print(f"   📊 Total nodes visited (including backtrack): {final_drift_status['nodes_visited']}")
    print(f"   🔍 Physical scans performed: {scanning_iterations}")
    print(f"   🔙 Dead end reversals: {dead_end_reversals}")
    print(f"   🔄 Backtrack attempts: {backtrack_attempts}")
    print(f"   🔙 Reverse backtracks: {reverse_backtracks}")
    print(f"   ⚡ Scans saved by caching: {nodes_explored - scanning_iterations}")
    
    print(f"\n🔧 COMPREHENSIVE ATTITUDE DRIFT CORRECTION SUMMARY:")
    print(f"   📊 Total nodes counted: {final_drift_status['nodes_visited']} (exploration + backtracking)")
    print(f"   🔄 Total corrections performed: {final_drift_status['total_corrections']}")
    print(f"   🎯 Total angle corrected: {final_drift_status['total_corrections'] * final_drift_status['correction_angle']}°")
    print(f"   📍 Correction interval: Every {final_drift_status['correction_interval']} nodes")
    print(f"   🔧 Correction angle: +{final_drift_status['correction_angle']}° per correction")
    print(f"   📈 Last correction at node: {final_drift_status['last_correction_at']}")
    
    backtrack_node_count = final_drift_status['nodes_visited'] - nodes_explored
    if final_drift_status['nodes_visited'] > 0:
        drift_frequency = final_drift_status['total_corrections'] / final_drift_status['nodes_visited']
        backtrack_percentage = (backtrack_node_count / final_drift_status['nodes_visited']) * 100
        print(f"   📊 Drift correction frequency: {drift_frequency:.2f} corrections per total node")
        print(f"   🔙 Backtrack nodes: {backtrack_node_count} ({backtrack_percentage:.1f}% of total)")
    
    graph_mapper.print_graph_summary()
    generate_exploration_report_absolute(graph_mapper, nodes_explored, dead_end_reversals, reverse_backtracks, final_drift_status)


def generate_exploration_report_absolute(graph_mapper, nodes_explored, dead_end_reversals=0, reverse_backtracks=0, final_drift_status=None):
    """Generate comprehensive exploration report with absolute direction info"""
    print(f"\n{'='*60}")
    print("📋 FINAL EXPLORATION REPORT (ABSOLUTE DIRECTIONS)")
    print(f"{'='*60}")
    
    # Basic statistics
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
    
    # Efficiency metrics
    revisited_nodes = nodes_explored - total_nodes
    if revisited_nodes > 0:
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
    
    # Wall statistics using absolute directions
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
    
    # Movement efficiency summary
    print(f"\n🔙 MOVEMENT EFFICIENCY:")
    print(f"   Dead end reversals: {dead_end_reversals}")
    print(f"   Reverse backtracks: {reverse_backtracks}")
    print(f"   Total reverse movements: {dead_end_reversals + reverse_backtracks}")
    print(f"   Time saved vs 180° turns: ~{(dead_end_reversals + reverse_backtracks) * 2:.1f} seconds")
    
    # Unexplored areas
    if graph_mapper.frontierQueue:
        print(f"\n🔍 UNEXPLORED AREAS:")
        for frontier_id in graph_mapper.frontierQueue:
            node = graph_mapper.nodes[frontier_id]
            print(f"   📍 {node.position}: {len(node.unexploredExits)} unexplored exits {node.unexploredExits}")
    
    print(f"\n⭐ ABSOLUTE DIRECTION BENEFITS:")
    
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
        time.sleep(0.3)
        
        print(f"🎯 Wall Detection Threshold: {tof_handler.WALL_THRESHOLD}cm")
        
        # Start autonomous exploration with absolute directions
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

print(f"\n📊 Generating visual reports...")
create_final_map_report(graph_mapper)