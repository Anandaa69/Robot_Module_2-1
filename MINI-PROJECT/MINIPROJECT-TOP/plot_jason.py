#!/usr/bin/env python3
"""
Maze Visualizer - โปรแกรมสำหรับแสดงผลข้อมูลเขาวงกตจากไฟล์ JSON
วาดเป็น grid พร้อมกำแพง เส้นทางหุ่นยนต์ และพื้นที่ที่สำรวจแล้ว
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

class MazeVisualizer:
    def __init__(self, json_file):
        """
        Initialize maze visualizer with data from JSON file
        """
        self.json_file =r"D:\downsyndrome\year2_1\Robot_Module_2-1\maze_data.json"
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
            'grid_lines': '#BDC3C7'     # Gray for grid lines
        }
        
        self.load_data()
    
    def load_data(self):
        """Load maze data from JSON file"""
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                self.maze_data = json.load(f)
            print(f"✅ Successfully loaded maze data from: {self.json_file}")
            print(f"   📊 Nodes: {len(self.maze_data['nodes'])}")
            print(f"   🗺️ Grid size: {self.maze_data['metadata']['boundaries']['width']}x{self.maze_data['metadata']['boundaries']['height']}")
            print(f"   🧱 Walls: {len(self.maze_data['walls'])}")
        except FileNotFoundError:
            print(f"❌ File not found: {self.json_file}")
            raise
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON format: {e}")
            raise
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def create_visualization(self, save_file=None, show_path=True, show_visit_counts=True, show_dead_ends=True):
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
        
        # Add grid lines
        self._add_grid_lines(min_x, max_x, min_y, max_y)
        
        # Customize plot
        self._customize_plot()
        
        # Add title and info
        self._add_title_and_info()
        
        # Save or show
        if save_file:
            plt.savefig(save_file, dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
            print(f"📁 Visualization saved to: {save_file}")
            plt.close()  # Close the figure to prevent display
        else:
            plt.show()
    
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
        
        # Main title
        title = f"🗺️ Maze Exploration Visualization"
        self.ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Information text
        info_text = (f"📊 Nodes Explored: {metadata['total_nodes_explored']} | "
                    f"🗺️ Grid Size: {metadata['boundaries']['width']}×{metadata['boundaries']['height']} | "
                    f"🧱 Walls: {len(self.maze_data['walls'])} | "
                    f"🛤️ Path Length: {len(self.maze_data['robot_path'])}")
        
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
            patches.Patch(color=self.colors['dead_end'], label='Dead End')
        ]
        
        self.ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), 
                    frameon=True, fancybox=True, shadow=True)


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Visualize maze exploration data from JSON file')
    parser.add_argument('json_file', nargs='?', default='maze_data.json', 
                    help='Path to JSON file containing maze data (default: maze_data.json)')
    parser.add_argument('--save', '-s', help='Save visualization to file')
    parser.add_argument('--no-path', action='store_true', help='Hide robot path')
    parser.add_argument('--no-visits', action='store_true', help='Hide visit counts')
    parser.add_argument('--no-dead-ends', action='store_true', help='Hide dead end markers')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.json_file):
        print(f"❌ File not found: {args.json_file}")
        print("Please make sure you have run the maze exploration program first.")
        return
    
    try:
        # Create visualizer
        visualizer = MazeVisualizer(args.json_file)
        
        # Generate output filename if save is requested but no filename provided
        save_file = args.save
        if save_file is True or save_file == '':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_file = f"maze_visualization_{timestamp}.png"
        
        # Create visualization
        print("🎨 Creating maze visualization...")
        visualizer.create_visualization(
            save_file=save_file,
            show_path=not args.no_path,
            show_visit_counts=not args.no_visits,
            show_dead_ends=not args.no_dead_ends
        )
        
        print("✅ Visualization complete!")
        
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()