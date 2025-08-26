import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Arrow
import numpy as np
from datetime import datetime
import json
from collections import defaultdict

class RobotMapVisualizer:
    """
    Class สำหรับการวาดแผนที่และ plot การเคลื่อนไหวของหุ่นยนต์
    
    Features:
    - Real-time map visualization
    - Robot path tracking
    - Wall detection display
    - Dead end marking
    - Frontier visualization
    - Movement statistics
    - Export capabilities
    """
    
    def __init__(self, cell_size=1.0, figsize=(12, 10)):
        """
        Initialize the map visualizer
        
        Args:
            cell_size (float): Size of each grid cell in meters
            figsize (tuple): Figure size for matplotlib
        """
        self.cell_size = cell_size
        self.figsize = figsize
        
        # Data storage
        self.nodes = {}
        self.robot_path = []
        self.current_position = (0, 0)
        self.current_direction = 'north'
        self.movement_history = []
        
        # Visual settings
        self.colors = {
            'wall': '#2C3E50',           # Dark blue-gray
            'open_space': '#ECF0F1',     # Light gray
            'robot': '#E74C3C',          # Red
            'path': '#3498DB',           # Blue
            'dead_end': '#8E44AD',       # Purple
            'frontier': '#F39C12',       # Orange
            'unexplored': '#95A5A6',     # Gray
            'grid': '#BDC3C7',           # Light gray
            'text': '#2C3E50'            # Dark gray
        }
        
        # Statistics
        self.stats = {
            'nodes_explored': 0,
            'walls_detected': 0,
            'dead_ends_found': 0,
            'backtrack_count': 0,
            'total_distance': 0.0
        }
        
        # Initialize matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.setup_plot()
    
    def setup_plot(self):
        """Setup the initial plot configuration"""
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3, color=self.colors['grid'])
        self.ax.set_title('Robot Autonomous Exploration Map', fontsize=16, fontweight='bold')
        
        # Set initial limits
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        
        # Labels
        self.ax.set_xlabel('X Position (m)', fontsize=12)
        self.ax.set_ylabel('Y Position (m)', fontsize=12)
        
        # Enable interactive mode
        plt.ion()
    
    def update_from_graph_mapper(self, graph_mapper):
        """
        Update visualizer data from GraphMapper instance
        
        Args:
            graph_mapper: GraphMapper instance from your code
        """
        self.nodes = graph_mapper.nodes
        self.current_position = graph_mapper.currentPosition
        self.current_direction = graph_mapper.currentDirection
        
        # Update statistics
        self.stats['nodes_explored'] = len(self.nodes)
        self.stats['walls_detected'] = self._count_walls()
        self.stats['dead_ends_found'] = sum(1 for node in self.nodes.values() if node.isDeadEnd)
    
    def add_robot_position(self, position, direction=None, live_update=False):
        """
        Add robot position to path history
        Args:
            position (tuple): (x, y) coordinates
            direction (str): Robot facing direction
            live_update (bool): ถ้า True จะวาดตำแหน่งนี้ทันที
        """
        self.robot_path.append({
            'position': position,
            'direction': direction or self.current_direction,
            'timestamp': datetime.now(),
            'step': len(self.robot_path)
        })
        
        self.current_position = position
        if direction:
            self.current_direction = direction
        
        # Calculate total distance
        if len(self.robot_path) > 1:
            prev_pos = self.robot_path[-2]['position']
            curr_pos = position
            distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
            self.stats['total_distance'] += distance

        if live_update:
            self._plot_last_step()

    def _plot_last_step(self):
        if len(self.robot_path) < 2:
            return
        
        prev_pos = self.robot_path[-2]['position']
        curr_pos = self.robot_path[-1]['position']
        
        # plot path segment
        self.ax.plot(
            [prev_pos[0], curr_pos[0]],
            [prev_pos[1], curr_pos[1]],
            color=self.colors['path'], linewidth=2, alpha=0.7
        )
        
        # update robot marker
        self._plot_robot()
        
        plt.draw()
        plt.pause(0.01)


    
    def plot_map(self, show_path=True, show_walls=True, show_frontiers=True, 
                show_dead_ends=True, show_robot=True, show_grid_labels=True):
        """
        Plot the complete map with all features
        
        Args:
            show_path (bool): Show robot movement path
            show_walls (bool): Show detected walls
            show_frontiers (bool): Show frontier nodes
            show_dead_ends (bool): Show dead end nodes
            show_robot (bool): Show current robot position
            show_grid_labels (bool): Show grid coordinate labels
        """
        # Clear the plot
        self.ax.clear()
        self.setup_plot()
        
        # Calculate map bounds
        if self.nodes:
            positions = [node.position for node in self.nodes.values()]
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            
            min_x, max_x = min(x_coords) - 1, max(x_coords) + 1
            min_y, max_y = min(y_coords) - 1, max(y_coords) + 1
            
            self.ax.set_xlim(min_x, max_x)
            self.ax.set_ylim(min_y, max_y)
        
        # Plot explored nodes (background)
        self._plot_explored_nodes()
        
        # Plot walls
        if show_walls:
            self._plot_walls()
        
        # Plot robot path
        if show_path and len(self.robot_path) > 1:
            self._plot_robot_path()
        
        # Plot special nodes
        if show_dead_ends:
            self._plot_dead_ends()
        
        if show_frontiers:
            self._plot_frontiers()
        
        # Plot current robot position
        if show_robot:
            self._plot_robot()
        
        # Plot grid labels
        if show_grid_labels:
            self._plot_grid_labels()
        
        # Add legend
        self._add_legend()
        
        # Add statistics
        self._add_statistics()
        
        # Refresh the plot
        plt.draw()
        plt.pause(0.01)
    
    def _plot_explored_nodes(self):
        """Plot explored nodes as grid cells"""
        for node in self.nodes.values():
            x, y = node.position
            # Draw explored cell
            rect = patches.Rectangle((x - 0.4, y - 0.4), 0.8, 0.8,
                                linewidth=1, edgecolor='black', 
                                facecolor=self.colors['open_space'], alpha=0.7)
            self.ax.add_patch(rect)
            
            # Add node ID text
            self.ax.text(x, y, f"{node.id}", ha='center', va='center', 
                        fontsize=8, color=self.colors['text'], alpha=0.6)
    
    def _plot_walls(self):
        """Plot walls detected by sensors"""
        wall_thickness = 0.05
        
        for node in self.nodes.values():
            x, y = node.position
            
            if hasattr(node, 'walls') and node.walls:
                # North wall
                if node.walls.get('north', False):
                    rect = patches.Rectangle((x - 0.5, y + 0.4), 1.0, wall_thickness,
                                        facecolor=self.colors['wall'], alpha=0.8)
                    self.ax.add_patch(rect)
                
                # South wall
                if node.walls.get('south', False):
                    rect = patches.Rectangle((x - 0.5, y - 0.45), 1.0, wall_thickness,
                                        facecolor=self.colors['wall'], alpha=0.8)
                    self.ax.add_patch(rect)
                
                # East wall
                if node.walls.get('east', False):
                    rect = patches.Rectangle((x + 0.4, y - 0.5), wall_thickness, 1.0,
                                        facecolor=self.colors['wall'], alpha=0.8)
                    self.ax.add_patch(rect)
                
                # West wall
                if node.walls.get('west', False):
                    rect = patches.Rectangle((x - 0.45, y - 0.5), wall_thickness, 1.0,
                                        facecolor=self.colors['wall'], alpha=0.8)
                    self.ax.add_patch(rect)
    
    def _plot_robot_path(self):
        """Plot robot movement path"""
        if len(self.robot_path) < 2:
            return
        
        # Extract path coordinates
        x_coords = [step['position'][0] for step in self.robot_path]
        y_coords = [step['position'][1] for step in self.robot_path]
        
        # Plot path line
        self.ax.plot(x_coords, y_coords, color=self.colors['path'], 
                    linewidth=2, alpha=0.7, marker='o', markersize=3, 
                    label=f'Robot Path ({len(self.robot_path)} steps)')
        
        # Add step numbers
        for i, step in enumerate(self.robot_path[::5]):  # Show every 5th step
            x, y = step['position']
            self.ax.text(x + 0.1, y + 0.1, str(i * 5), fontsize=8, 
                        color=self.colors['path'], alpha=0.8)
    
    def _plot_dead_ends(self):
        """Plot dead end nodes"""
        for node in self.nodes.values():
            if node.isDeadEnd:
                x, y = node.position
                circle = Circle((x, y), 0.15, color=self.colors['dead_end'], 
                            alpha=0.8, zorder=5)
                self.ax.add_patch(circle)
                self.ax.text(x, y - 0.3, 'DEAD\nEND', ha='center', va='top',
                            fontsize=7, color=self.colors['dead_end'], 
                            fontweight='bold')
    
    def _plot_frontiers(self):
        """Plot frontier nodes (unexplored areas)"""
        for node in self.nodes.values():
            if hasattr(node, 'unexploredExits') and node.unexploredExits:
                x, y = node.position
                # Draw frontier marker
                triangle = patches.RegularPolygon((x, y), 3, 0.12, 
                                                color=self.colors['frontier'],
                                                alpha=0.8, zorder=4)
                self.ax.add_patch(triangle)
                
                # Show unexplored directions
                directions_text = ','.join(node.unexploredExits)
                self.ax.text(x, y + 0.3, f'→{directions_text}', ha='center', va='bottom',
                            fontsize=6, color=self.colors['frontier'], fontweight='bold')
    
    def _plot_robot(self):
        """Plot current robot position and orientation"""
        x, y = self.current_position
        
        # Robot body (circle)
        robot_circle = Circle((x, y), 0.2, color=self.colors['robot'], 
                            alpha=0.9, zorder=10)
        self.ax.add_patch(robot_circle)
        
        # Robot direction indicator (arrow)
        direction_vectors = {
            'north': (0, 0.3),
            'south': (0, -0.3),
            'east': (0.3, 0),
            'west': (-0.3, 0)
        }
        
        if self.current_direction in direction_vectors:
            dx, dy = direction_vectors[self.current_direction]
            arrow = Arrow(x, y, dx, dy, width=0.15, 
                        color='white', alpha=0.9, zorder=11)
            self.ax.add_patch(arrow)
        
        # Robot label
        self.ax.text(x, y - 0.5, 'ROBOT', ha='center', va='top',
                    fontsize=8, color=self.colors['robot'], fontweight='bold')
    
    def _plot_grid_labels(self):
        """Add coordinate labels to grid intersections"""
        if not self.nodes:
            return
        
        positions = [node.position for node in self.nodes.values()]
        x_coords = set(pos[0] for pos in positions)
        y_coords = set(pos[1] for pos in positions)
        
        for x in x_coords:
            for y in y_coords:
                if (x, y) not in [node.position for node in self.nodes.values()]:
                    # Unexplored grid point
                    self.ax.plot(x, y, 'x', color=self.colors['unexplored'], 
                            markersize=4, alpha=0.5)
    
    def _add_legend(self):
        """Add legend to the plot"""
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors['robot'],
                    markersize=10, label='Current Robot Position'),
            plt.Line2D([0], [0], color=self.colors['path'], linewidth=2, label='Robot Path'),
            plt.Rectangle((0, 0), 1, 1, facecolor=self.colors['wall'], label='Walls'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors['dead_end'],
                    markersize=8, label='Dead Ends'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=self.colors['frontier'],
                    markersize=8, label='Frontiers'),
            plt.Rectangle((0, 0), 1, 1, facecolor=self.colors['open_space'], 
                        alpha=0.7, label='Explored Area')
        ]
        
        self.ax.legend(handles=legend_elements, loc='upper right', 
                    bbox_to_anchor=(1.15, 1), fontsize=9)
    
    def _add_statistics(self):
        """Add exploration statistics to the plot"""
        stats_text = f"""Exploration Statistics:
Nodes Explored: {self.stats['nodes_explored']}
Walls Detected: {self.stats['walls_detected']}
Dead Ends: {self.stats['dead_ends_found']}
Path Steps: {len(self.robot_path)}
Total Distance: {self.stats['total_distance']:.1f}m
Current Position: {self.current_position}
Facing: {self.current_direction.upper()}"""
        
        self.ax.text(0.02, 0.98, stats_text, transform=self.ax.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.8), fontsize=9)
    
    def _count_walls(self):
        """Count total walls detected"""
        wall_count = 0
        for node in self.nodes.values():
            if hasattr(node, 'walls') and node.walls:
                wall_count += sum(1 for is_wall in node.walls.values() if is_wall)
        return wall_count
    
    def animate_exploration(self, update_interval=0.5):
        """
        Create animated visualization of exploration process
        
        Args:
            update_interval (float): Time between updates in seconds
        """
        plt.ioff()  # Turn off interactive mode
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        def animate(frame):
            ax.clear()
            # Plot up to current frame
            temp_path = self.robot_path[:frame+1]
            # ... implement frame-by-frame animation
            
        # Use matplotlib animation
        from matplotlib.animation import FuncAnimation
        anim = FuncAnimation(fig, animate, frames=len(self.robot_path),
                           interval=int(update_interval * 1000), repeat=False)
        
        plt.show()
        return anim
    
    def save_map(self, filename, format='png', dpi=300):
        """
        Save current map to file
        
        Args:
            filename (str): Output filename
            format (str): File format ('png', 'pdf', 'svg', etc.)
            dpi (int): Resolution for raster formats
        """
        plt.savefig(filename, format=format, dpi=dpi, bbox_inches='tight')
        print(f"✅ Map saved to {filename}")
    
    def export_data(self, filename):
        """
        Export exploration data to JSON file
        
        Args:
            filename (str): Output JSON filename
        """
        export_data = {
            'exploration_summary': self.stats,
            'robot_path': [
                {
                    'position': step['position'],
                    'direction': step['direction'],
                    'step': step['step'],
                    'timestamp': step['timestamp'].isoformat()
                }
                for step in self.robot_path
            ],
            'nodes': {
                node_id: {
                    'position': node.position,
                    'walls': getattr(node, 'walls', {}),
                    'is_dead_end': node.isDeadEnd,
                    'unexplored_exits': getattr(node, 'unexploredExits', []),
                    'fully_scanned': getattr(node, 'fullyScanned', False)
                }
                for node_id, node in self.nodes.items()
            },
            'current_position': self.current_position,
            'current_direction': self.current_direction
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Exploration data exported to {filename}")
    
    def create_summary_report(self):
        """Create a comprehensive exploration summary"""
        if not self.nodes:
            return "No exploration data available."
        
        # Calculate advanced statistics
        positions = [node.position for node in self.nodes.values()]
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        map_width = max(x_coords) - min(x_coords) + 1
        map_height = max(y_coords) - min(y_coords) + 1
        map_area = map_width * map_height
        
        coverage_efficiency = len(self.nodes) / map_area * 100 if map_area > 0 else 0
        
        # Direction analysis
        direction_counts = defaultdict(int)
        for step in self.robot_path:
            direction_counts[step['direction']] += 1
        
        report = f"""
🤖 ROBOT EXPLORATION SUMMARY REPORT
{'='*50}

📊 BASIC STATISTICS:
   • Nodes Explored: {self.stats['nodes_explored']}
   • Total Path Steps: {len(self.robot_path)}
   • Total Distance Traveled: {self.stats['total_distance']:.2f}m
   • Dead Ends Found: {self.stats['dead_ends_found']}
   • Walls Detected: {self.stats['walls_detected']}

🗺️ MAP DIMENSIONS:
   • Width: {map_width} units
   • Height: {map_height} units  
   • Total Area: {map_area} grid cells
   • Coverage Efficiency: {coverage_efficiency:.1f}%

🧭 MOVEMENT ANALYSIS:
"""
        
        for direction, count in sorted(direction_counts.items()):
            percentage = count / len(self.robot_path) * 100 if self.robot_path else 0
            report += f"   • {direction.upper()}: {count} steps ({percentage:.1f}%)\n"
        
        report += f"""
🎯 EXPLORATION EFFICIENCY:
   • Average Steps per Node: {len(self.robot_path) / len(self.nodes):.1f}
   • Dead End Rate: {self.stats['dead_ends_found'] / len(self.nodes) * 100:.1f}%
   • Wall Density: {self.stats['walls_detected'] / (len(self.nodes) * 4) * 100:.1f}%

📍 CURRENT STATUS:
   • Robot Position: {self.current_position}
   • Robot Facing: {self.current_direction.upper()}
   • Exploration Complete: {'Yes' if not any(getattr(node, 'unexploredExits', []) for node in self.nodes.values()) else 'No'}

{'='*50}
Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return report

# ===== INTEGRATION FUNCTIONS =====

def integrate_with_exploration_loop(graph_mapper, movement_controller, visualizer):
    """
    Integration function to use with your main exploration loop
    
    Example usage in your explore_autonomously_with_absolute_directions function:
    
    # Add at the beginning of your exploration loop:
    visualizer = RobotMapVisualizer()
    
    # Add after each movement:
    visualizer.update_from_graph_mapper(graph_mapper)
    visualizer.add_robot_position(graph_mapper.currentPosition, graph_mapper.currentDirection)
    visualizer.plot_map()
    
    # Add at the end:
    print(visualizer.create_summary_report())
    visualizer.save_map("final_exploration_map.png")
    visualizer.export_data("exploration_data.json")
    """
    
    # Update visualizer with current state
    visualizer.update_from_graph_mapper(graph_mapper)
    visualizer.add_robot_position(graph_mapper.currentPosition, graph_mapper.currentDirection)
    
    # Plot current state
    visualizer.plot_map()
    
    return visualizer

# ===== EXAMPLE USAGE =====

def example_usage():
    """
    Example of how to use the RobotMapVisualizer
    """
    
    # Create visualizer instance
    visualizer = RobotMapVisualizer(cell_size=0.6, figsize=(14, 10))
    
    # Simulate some exploration data
    example_nodes = {
        '0_0': type('Node', (), {
            'id': '0_0',
            'position': (0, 0),
            'walls': {'north': False, 'south': True, 'east': False, 'west': True},
            'isDeadEnd': False,
            'unexploredExits': ['north', 'east'],
            'fullyScanned': True
        })(),
        '1_0': type('Node', (), {
            'id': '1_0', 
            'position': (1, 0),
            'walls': {'north': False, 'south': False, 'east': True, 'west': False},
            'isDeadEnd': False,
            'unexploredExits': ['north'],
            'fullyScanned': True
        })(),
        '0_1': type('Node', (), {
            'id': '0_1',
            'position': (0, 1), 
            'walls': {'north': True, 'south': False, 'east': False, 'west': True},
            'isDeadEnd': True,
            'unexploredExits': [],
            'fullyScanned': True
        })()
    }
    
    # Add example data
    visualizer.nodes = example_nodes
    visualizer.current_position = (1, 0)
    visualizer.current_direction = 'north'
    
    # Add path history
    path_steps = [(0, 0), (1, 0), (0, 0), (0, 1)]
    for i, pos in enumerate(path_steps):
        visualizer.add_robot_position(pos, ['north', 'east', 'west', 'north'][i])
    
    # Plot the map
    visualizer.plot_map()
    
    # Show summary
    print(visualizer.create_summary_report())
    
    # Save results
    visualizer.save_map("example_map.png")
    visualizer.export_data("example_data.json")
    
    plt.show()

if __name__ == "__main__":
    example_usage()