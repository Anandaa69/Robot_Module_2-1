#!/usr/bin/env python3
"""
Maze Visualizer - โปรแกรมสำหรับแสดงผลข้อมูลเขาวงกตจากไฟล์ JSON
วาดเป็น grid พร้อมกำแพง เส้นทางหุ่นยนต์ และพื้นที่ที่สำรวจแล้ว
รวมถึงการแสดง markers ที่พบ
"""

import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import numpy as np
import argparse
import os
from datetime import datetime
from matplotlib import animation

class MazeVisualizer:
    def __init__(self, json_file):
        """
        Initialize maze visualizer with data from JSON file
        """
        self.json_file = json_file
        self.maze_data = None
        self.fig = None
        self.ax = None
        
        # Colors for visualization
        self.colors = {
            'explored': '#E8F4FD',      # Light blue for explored areas
            'unexplored': '#F5F5F5',    # Light gray for unexplored areas
            'wall': '#2C3E50',          # Dark blue-gray for walls
            'robot_path': '#E74C3C',    # Red for robot path
            'start_position': '#27AE60', # Green for start position
            'dead_end': '#F39C12',      # Orange for dead ends
            'grid_lines': '#BDC3C7',    # Gray for grid lines
            'red_marker': '#FF0000',    # Red for red markers
            'marker_detected': '#FF69B4' # Pink for marker detection zones
        }
        
        self.load_data()

    def create_animation(self, save_file="maze_animation.mp4"):
        if 'robot_path' not in self.maze_data or len(self.maze_data['robot_path']) < 2:
            print("❌ ไม่มีข้อมูลเส้นทางหุ่นยนต์เพียงพอสำหรับทำอนิเมชัน")
            return

        bounds = self.maze_data['metadata']['boundaries']
        min_x, max_x = bounds['min_x'], bounds['max_x']
        min_y, max_y = bounds['min_y'], bounds['max_y']

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(min_x - 0.5, max_x + 0.5)
        ax.set_ylim(min_y - 0.5, max_y + 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()

        # วาดพื้นและกำแพงคงที่
        self.ax = ax
        self._draw_grid_background(min_x, max_x, min_y, max_y)
        self._draw_walls()
        self._draw_start_position()
        self._draw_dead_ends()
        self._draw_markers()  # เพิ่มการวาด markers
        self._add_grid_lines(min_x, max_x, min_y, max_y)

        path_line, = ax.plot([], [], color=self.colors['robot_path'], linewidth=3)
        marker, = ax.plot([], [], 'o', color=self.colors['robot_path'])

        path = self.maze_data['robot_path']

        def init():
            path_line.set_data([], [])
            marker.set_data([], [])
            return path_line, marker

        def update(frame):
            x_data = [p[0] for p in path[:frame+1]]
            y_data = [p[1] for p in path[:frame+1]]
            path_line.set_data(x_data, y_data)
            marker.set_data(x_data[-1], y_data[-1])
            return path_line, marker

        ani = animation.FuncAnimation(fig, update, frames=len(path), init_func=init,
                                    blit=True, interval=300, repeat=False)

        ani.save("maze_animation.gif", writer="pillow", fps=3)
        print(f"🎥 Animation saved to: {save_file}")
        plt.close(fig)
    
    def load_data(self):
        """Load maze data from JSON file"""
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                self.maze_data = json.load(f)
            print(f"✅ Successfully loaded maze data from: {self.json_file}")
            print(f"   📊 Nodes: {len(self.maze_data['nodes'])}")
            print(f"   🗺️ Grid size: {self.maze_data['metadata']['boundaries']['width']}x{self.maze_data['metadata']['boundaries']['height']}")
            print(f"   🧱 Walls: {len(self.maze_data['walls'])}")
            
            # นับ markers ที่พบ
            marker_count = 0
            for node_data in self.maze_data['nodes'].values():
                if 'marker_scan_results' in node_data:
                    for direction_data in node_data['marker_scan_results'].values():
                        if direction_data.get('found_red', False):
                            marker_count += 1
            print(f"   🔴 Red markers detected: {marker_count}")
            
        except FileNotFoundError:
            print(f"❌ File not found: {self.json_file}")
            raise
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON format: {e}")
            raise
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def create_visualization(self, save_file=None, show_path=True, show_visit_counts=True, 
                           show_dead_ends=True, show_markers=True):
        """
        Create maze visualization
        """
        if not self.maze_data:
            print("❌ No maze data loaded")
            return
        
        # Get boundaries
        bounds = self.maze_data['metadata']['boundaries']
        min_x, max_x = bounds['min_x'], bounds['max_x']
        min_y, max_y = bounds['min_y'], bounds['max_y']
        width, height = bounds['width'], bounds['height']
        
        # Create figure with proper size
        fig_width = max(10, width * 2)
        fig_height = max(8, height * 2)
        self.fig, self.ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Set up the grid
        self.ax.set_xlim(min_x - 0.5, max_x + 0.5)
        self.ax.set_ylim(min_y - 0.5, max_y + 0.5)
        self.ax.set_aspect('equal')
        
        # Draw grid background
        self._draw_grid_background(min_x, max_x, min_y, max_y)
        
        # Draw walls
        self._draw_walls()
        
        # Draw robot path
        if show_path:
            self._draw_robot_path()
        
        # Draw special markers
        self._draw_start_position()
        
        if show_dead_ends:
            self._draw_dead_ends()
        
        if show_visit_counts:
            self._draw_visit_counts()
        
        # Draw markers ที่สำคัญที่สุด!
        if show_markers:
            self._draw_markers()
        
        # Add grid lines
        self._add_grid_lines(min_x, max_x, min_y, max_y)
        
        # Customize plot
        self._customize_plot()
        
        # Add title and info
        self._add_title_and_info()

        self.ax.invert_xaxis()
        
        # Save or show
        if save_file:
            plt.savefig(save_file, dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
            print(f"📁 Visualization saved to: {save_file}")
            plt.close()  # Close the figure to prevent display
        else:
            # Auto-save with timestamp even if no save file specified
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            auto_save_file = f"maze_visualization_{timestamp}.png"
            plt.savefig(auto_save_file, dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
            print(f"📁 Visualization auto-saved to: {auto_save_file}")
            plt.close()  # Close the figure to prevent display
    
    def _draw_grid_background(self, min_x, max_x, min_y, max_y):
        """Draw background grid cells"""
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                cell_key = f"{x},{y}"
                if cell_key in self.maze_data['grid_representation']:
                    cell_data = self.maze_data['grid_representation'][cell_key]
                    color = self.colors['explored'] if cell_data['explored'] else self.colors['unexplored']
                else:
                    color = self.colors['unexplored']
                
                # Draw cell background
                rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, 
                                    linewidth=0, facecolor=color, alpha=0.7)
                self.ax.add_patch(rect)
    
    def _draw_walls(self):
        """Draw walls as thick lines"""
        wall_thickness = 0.1
        
        for wall_key, wall_data in self.maze_data['walls'].items():
            x, y = wall_data['position']
            direction = wall_data['direction']
            
            if direction == 'north':
                wall_rect = patches.Rectangle((x - 0.5, y + 0.5 - wall_thickness/2), 
                                            1, wall_thickness, 
                                            facecolor=self.colors['wall'], 
                                            edgecolor=self.colors['wall'])
            elif direction == 'south':
                wall_rect = patches.Rectangle((x - 0.5, y - 0.5 - wall_thickness/2), 
                                            1, wall_thickness, 
                                            facecolor=self.colors['wall'], 
                                            edgecolor=self.colors['wall'])
            elif direction == 'east':
                wall_rect = patches.Rectangle((x + 0.5 - wall_thickness/2, y - 0.5), 
                                            wall_thickness, 1, 
                                            facecolor=self.colors['wall'], 
                                            edgecolor=self.colors['wall'])
            elif direction == 'west':
                wall_rect = patches.Rectangle((x - 0.5 - wall_thickness/2, y - 0.5), 
                                            wall_thickness, 1, 
                                            facecolor=self.colors['wall'], 
                                            edgecolor=self.colors['wall'])
            
            self.ax.add_patch(wall_rect)
    
    def _draw_robot_path(self):
        """Draw robot's path as a line"""
        if 'robot_path' not in self.maze_data or len(self.maze_data['robot_path']) < 2:
            return
        
        path = self.maze_data['robot_path']
        x_coords = [pos[0] for pos in path]
        y_coords = [pos[1] for pos in path]
        
        # Draw path line
        self.ax.plot(x_coords, y_coords, color=self.colors['robot_path'], 
                    linewidth=3, alpha=0.8, linestyle='-', marker='o', 
                    markersize=4, markerfacecolor=self.colors['robot_path'])
        
        # Add direction arrows
        for i in range(len(path) - 1):
            start_x, start_y = path[i]
            end_x, end_y = path[i + 1]
            dx, dy = end_x - start_x, end_y - start_y
            
            # Add small arrow
            if dx != 0 or dy != 0:
                self.ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                            arrowprops=dict(arrowstyle='->', color=self.colors['robot_path'], 
                                            lw=2, alpha=0.6))
    
    def _draw_start_position(self):
        """Mark the starting position"""
        start_pos = self.maze_data['metadata']['robot_start_position']
        circle = patches.Circle(start_pos, 0.15, color=self.colors['start_position'], 
                            zorder=10, alpha=0.9)
        self.ax.add_patch(circle)
        
        # Add text label
        self.ax.text(start_pos[0], start_pos[1] - 0.7, 'START', 
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    color=self.colors['start_position'])
    
    def _draw_dead_ends(self):
        """Mark dead end positions"""
        for node_data in self.maze_data['nodes'].values():
            if node_data['is_dead_end']:
                x, y = node_data['position']
                triangle = patches.RegularPolygon((x, y), 3, radius=0.2, 
                                                color=self.colors['dead_end'], 
                                                alpha=0.8, zorder=8)
                self.ax.add_patch(triangle)
                
                # Add text label
                self.ax.text(x, y + 0.6, 'DEAD END', ha='center', va='center', 
                        fontsize=8, color=self.colors['dead_end'], fontweight='bold')
    
    def _draw_markers(self):
        """Draw markers detected by the robot"""
        marker_positions = []
        
        for node_data in self.maze_data['nodes'].values():
            if 'marker_scan_results' not in node_data:
                continue
                
            node_x, node_y = node_data['position']
            
            for direction, scan_data in node_data['marker_scan_results'].items():
                if scan_data.get('found_red', False):
                    # คำนวณตำแหน่งของ marker ตามทิศทางและระยะทาง
                    distance = scan_data.get('distance', 0) / 100.0  # แปลง cm เป็น grid units
                    angle = scan_data.get('angle', 0)
                    
                    # คำนวณตำแหน่ง marker
                    if direction == 'right' or angle == 90:
                        marker_x = node_x + 0.8  # ด้านขวา
                        marker_y = node_y
                    elif direction == 'left' or angle == -90:
                        marker_x = node_x - 0.8  # ด้านซ้าย
                        marker_y = node_y
                    elif direction == 'front' or angle == 0:
                        marker_x = node_x  # ด้านหน้า
                        marker_y = node_y + 0.8
                    elif direction == 'back' or angle == 180:
                        marker_x = node_x  # ด้านหลัง
                        marker_y = node_y - 0.8
                    else:
                        # ใช้ทิศทางจาก compass_direction ถ้ามี
                        compass = scan_data.get('compass_direction', '')
                        if 'ตะวันออก' in compass or 'E' in compass:
                            marker_x = node_x + 0.8
                            marker_y = node_y
                        elif 'ตะวันตก' in compass or 'W' in compass:
                            marker_x = node_x - 0.8
                            marker_y = node_y
                        elif 'เหนือ' in compass or 'N' in compass:
                            marker_x = node_x
                            marker_y = node_y + 0.8
                        elif 'ใต้' in compass or 'S' in compass:
                            marker_x = node_x
                            marker_y = node_y - 0.8
                        else:
                            marker_x = node_x + 0.8  # default ด้านขวา
                            marker_y = node_y
                    
                    marker_pos = (marker_x, marker_y)
                    if marker_pos not in marker_positions:
                        marker_positions.append(marker_pos)
                        
                        # วาด marker เป็นสี่เหลี่ยมสีแดง
                        rect = patches.Rectangle((marker_x - 0.1, marker_y - 0.1), 0.2, 0.2,
                                               facecolor=self.colors['red_marker'],
                                               edgecolor='darkred', linewidth=2, 
                                               alpha=0.9, zorder=15)
                        self.ax.add_patch(rect)
                        
                        # เพิ่มป้ายกำกับ
                        self.ax.text(marker_x, marker_y - 0.35, 'RED', ha='center', va='center',
                                   fontsize=8, fontweight='bold', color=self.colors['red_marker'])
                        
                        # วาดเส้นเชื่อมจากโหนดไปยัง marker
                        self.ax.plot([node_x, marker_x], [node_y, marker_y], 
                                   color=self.colors['marker_detected'], 
                                   linestyle='--', linewidth=2, alpha=0.6, zorder=12)
                        
                        # แสดงระยะทาง
                        mid_x = (node_x + marker_x) / 2
                        mid_y = (node_y + marker_y) / 2
                        distance_cm = scan_data.get('distance', 0)
                        self.ax.text(mid_x, mid_y + 0.15, f'{distance_cm:.1f}cm', 
                                   ha='center', va='center', fontsize=7, 
                                   color=self.colors['marker_detected'], 
                                   bbox=dict(boxstyle='round,pad=0.2', 
                                           facecolor='white', alpha=0.8))
        
        print(f"🔴 Drew {len(marker_positions)} red markers")
    
    def _draw_visit_counts(self):
        """Draw visit counts on each explored cell"""
        for node_data in self.maze_data['nodes'].values():
            x, y = node_data['position']
            visit_count = node_data['visit_count']
            
            if visit_count > 1:  # Only show if visited more than once
                # Background circle for better readability
                circle = patches.Circle((x, y), 0.12, color='white', 
                                    alpha=0.8, zorder=9)
                self.ax.add_patch(circle)
                
                # Visit count text
                self.ax.text(x, y, str(visit_count), ha='center', va='center', 
                        fontsize=10, fontweight='bold', color='black', zorder=10)
    
    def _add_grid_lines(self, min_x, max_x, min_y, max_y):
        """Add grid lines for better visualization"""
        # Vertical lines
        for x in range(min_x, max_x + 2):
            self.ax.axvline(x - 0.5, color=self.colors['grid_lines'], 
                        linewidth=0.5, alpha=0.5, zorder=1)
        
        # Horizontal lines
        for y in range(min_y, max_y + 2):
            self.ax.axhline(y - 0.5, color=self.colors['grid_lines'], 
                        linewidth=0.5, alpha=0.5, zorder=1)
    
    def _customize_plot(self):
        """Customize plot appearance"""
        # Remove ticks and labels
        self.ax.set_xticks(range(self.maze_data['metadata']['boundaries']['min_x'], 
                                self.maze_data['metadata']['boundaries']['max_x'] + 1))
        self.ax.set_yticks(range(self.maze_data['metadata']['boundaries']['min_y'], 
                                self.maze_data['metadata']['boundaries']['max_y'] + 1))
        
        # Add coordinate labels
        self.ax.set_xlabel('X Coordinate', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('Y Coordinate', fontsize=12, fontweight='bold')
        
        # Invert y-axis to match traditional maze view (origin at top-left)
        self.ax.invert_yaxis()
        
        # Set background color
        self.ax.set_facecolor('white')
    
    def _add_title_and_info(self):
        """Add title and information"""
        metadata = self.maze_data['metadata']
        
        # นับ markers
        marker_count = 0
        for node_data in self.maze_data['nodes'].values():
            if 'marker_scan_results' in node_data:
                for direction_data in node_data['marker_scan_results'].values():
                    if direction_data.get('found_red', False):
                        marker_count += 1
        
        # Main title
        title = f"🗺️ Maze Exploration Visualization"
        self.ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Information text
        info_text = (f"📊 Nodes Explored: {metadata['total_nodes_explored']} | "
                    f"🗺️ Grid Size: {metadata['boundaries']['width']}×{metadata['boundaries']['height']} | "
                    f"🧱 Walls: {len(self.maze_data['walls'])} | "
                    f"🛤️ Path Length: {len(self.maze_data['robot_path'])} | "
                    f"🔴 Red Markers: {marker_count}")
        
        # Add info text below the plot
        self.fig.text(0.5, 0.02, info_text, ha='center', va='bottom', 
                    fontsize=10, style='italic')
        
        # Add legend
        self._add_legend()
    
    def _add_legend(self):
        """Add legend to explain the visualization"""
        legend_elements = [
            patches.Patch(color=self.colors['explored'], alpha=0.7, label='Explored Area'),
            patches.Patch(color=self.colors['unexplored'], alpha=0.7, label='Unexplored Area'),
            patches.Patch(color=self.colors['wall'], label='Wall'),
            plt.Line2D([0], [0], color=self.colors['robot_path'], linewidth=3, label='Robot Path'),
            patches.Patch(color=self.colors['start_position'], label='Start Position'),
            patches.Patch(color=self.colors['dead_end'], label='Dead End'),
            patches.Patch(color=self.colors['red_marker'], label='Red Marker'),
            plt.Line2D([0], [0], color=self.colors['marker_detected'], linestyle='--', label='Marker Detection')
        ]
        
        self.ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), 
                    frameon=True, fancybox=True, shadow=True)


def quick_visualize(json_file=None, auto_save=True, output_name=None):
    """
    Quick visualization function - automatically creates and saves maze plot
    """
    # Auto-detect JSON file if not provided
    if json_file is None:
        possible_files = ['maze_data.json', 'test_maze_data.json']
        json_file = None
        for file in possible_files:
            if os.path.exists(file):
                json_file = file
                break
        
        if json_file is None:
            print("❌ No maze data file found. Looking for:")
            for file in possible_files:
                print(f"   - {file}")
            return None
    
    if not os.path.exists(json_file):
        print(f"❌ File not found: {json_file}")
        return None
    
    try:
        print(f"🎨 Creating maze visualization from: {json_file}")
        
        # Create visualizer
        visualizer = MazeVisualizer(json_file)
        
        # Generate output filename
        if output_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(json_file))[0]
            output_name = f"{base_name}_plot_{timestamp}.png"
        
        # Create visualization - always save to file
        visualizer.create_visualization(
            save_file=output_name,
            show_path=True,
            show_visit_counts=True,
            show_dead_ends=True,
            show_markers=True  # เพิ่มการแสดง markers
        )
        
        print("✅ Visualization complete!")
        return output_name
        
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Visualize maze exploration data from JSON file')
    parser.add_argument('json_file', nargs='?', default=None, 
                    help='Path to JSON file containing maze data (auto-detects if not specified)')
    parser.add_argument('--save', '-s', help='Save visualization to specific file')
    parser.add_argument('--no-path', action='store_true', help='Hide robot path')
    parser.add_argument('--no-visits', action='store_true', help='Hide visit counts')
    parser.add_argument('--no-dead-ends', action='store_true', help='Hide dead end markers')
    parser.add_argument('--no-markers', action='store_true', help='Hide detected markers')
    parser.add_argument('--quick', '-q', action='store_true', help='Quick mode - auto-detect file and save')
    
    args = parser.parse_args()
    
    # Quick mode
    if args.quick:
        output_file = quick_visualize()
        if output_file:
            print(f"🎯 Quick visualization saved to: {output_file}")
        return
    
    # Auto-detect file if not provided
    json_file = args.json_file
    if json_file is None:
        possible_files = ['maze_data.json', 'test_maze_data.json']
        for file in possible_files:
            if os.path.exists(file):
                json_file = file
                print(f"📂 Auto-detected: {json_file}")
                break
        
        if json_file is None:
            print("❌ No maze data file found. Looking for:")
            for file in possible_files:
                print(f"   - {file}")
            return
    
    # Check if file exists
    if not os.path.exists(json_file):
        print(f"❌ File not found: {json_file}")
        print("Please make sure you have run the maze exploration program first.")
        return
    
    try:
        # Create visualizer
        visualizer = MazeVisualizer(json_file)
        
        # Generate output filename if save is requested but no filename provided
        save_file = args.save
        if save_file is True or save_file == '':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(json_file))[0]
            save_file = f"{base_name}_visualization_{timestamp}.png"
        
        # Create visualization - will auto-save if no save file specified
        print("🎨 Creating maze visualization...")
        visualizer.create_visualization(
            save_file=save_file,
            show_path=not args.no_path,
            show_visit_counts=not args.no_visits,
            show_dead_ends=not args.no_dead_ends,
            show_markers=not args.no_markers  # เพิ่มตัวเลือก markers
        )
        
        print("✅ Visualization complete!")
        
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # หาที่อยู่ของไฟล์ .py ปัจจุบัน
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ตั้งค่า path ของ JSON ให้อยู่ในโฟลเดอร์เดียวกับ .py
    json_path = os.path.join(script_dir, "maze_data.json")

    if not os.path.exists(json_path):
        print(f"❌ ไม่พบไฟล์ maze_data.json ในโฟลเดอร์: {script_dir}")
    else:
        visualizer = MazeVisualizer(json_path)
        print("🎨 Creating maze visualization...")
        visualizer.create_visualization(
            save_file=os.path.join(script_dir, "maze_visualization_with_markers.png"),
            show_path=True,
            show_visit_counts=True,
            show_dead_ends=True,
            show_markers=True
        )
        print(f"✅ Visualization saved to: {os.path.join(script_dir, 'maze_visualization_with_markers.png')}")
        visualizer.create_animation(os.path.join(script_dir, "maze_animation.gif"))