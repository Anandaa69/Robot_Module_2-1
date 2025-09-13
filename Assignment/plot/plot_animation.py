import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np
from pathlib import Path
import argparse

def load_maze_data(filename="Assignment/data/maze_data_enhanced.json"):
    """Load maze data from JSON file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {filename}.")
        return None

def create_static_maze_plot(maze_data, save_path="Assignment/plots/"):
    """Create static maze visualization"""
    if not maze_data:
        return None
    
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    metadata = maze_data["metadata"]
    bounds = metadata["boundaries"]
    nodes = maze_data["nodes"]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = {
        'wall': '#2C3E50', 'open_space': '#ECF0F1', 'robot_path': '#E74C3C',
        'dead_end': '#C0392B', 'normal_node': '#3498DB', 'start_node': '#27AE60',
        'backtrack': '#9B59B6'
    }
    
    # Grid background
    for x in range(bounds["min_x"] - 1, bounds["max_x"] + 2):
        for y in range(bounds["min_y"] - 1, bounds["max_y"] + 2):
            rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, linewidth=0.5, 
                                   edgecolor='lightgray', facecolor=colors['open_space'], alpha=0.3)
            ax.add_patch(rect)
    
    # Draw walls
    wall_thickness = 0.1
    for node_id, node_data in nodes.items():
        x, y = node_data["position"]
        walls = node_data["walls"]
        
        wall_coords = [
            (walls["north"], (x - 0.5, y + 0.5 - wall_thickness/2), (1, wall_thickness)),
            (walls["south"], (x - 0.5, y - 0.5 - wall_thickness/2), (1, wall_thickness)),
            (walls["east"], (x + 0.5 - wall_thickness/2, y - 0.5), (wall_thickness, 1)),
            (walls["west"], (x - 0.5 - wall_thickness/2, y - 0.5), (wall_thickness, 1))
        ]
        
        for has_wall, (wx, wy), (width, height) in wall_coords:
            if has_wall:
                wall = patches.Rectangle((wx, wy), width, height, 
                                       facecolor=colors['wall'], edgecolor='none')
                ax.add_patch(wall)
    
    # Draw exploration path
    if "exploration_path" in maze_data:
        for path_segment in maze_data["exploration_path"]:
            from_pos = path_segment["from_position"]
            to_pos = path_segment["to_position"]
            action_type = path_segment["action_type"]
            
            color_map = {
                "backtrack": (colors['backtrack'], 0.6, 2),
                "dead_end_reverse": (colors['dead_end'], 0.8, 3),
                "default": (colors['robot_path'], 0.7, 2)
            }
            
            color, alpha, linewidth = color_map.get(action_type, color_map["default"])
            ax.annotate('', xy=to_pos, xytext=from_pos,
                       arrowprops=dict(arrowstyle='->', color=color, alpha=alpha, lw=linewidth))
    
    # Draw nodes
    for node_id, node_data in nodes.items():
        x, y = node_data["position"]
        
        if (x, y) == (0, 0):
            color, size, marker = colors['start_node'], 200, 's'
        elif node_data["is_dead_end"]:
            color, size, marker = colors['dead_end'], 100, 'X'
        else:
            color, size, marker = colors['normal_node'], 80, 'o'
        
        ax.scatter(x, y, c=color, s=size, marker=marker, edgecolors='black', linewidth=1, zorder=5)
        ax.text(x, y-0.15, f"{x},{y}", ha='center', va='top', fontsize=8, weight='bold', zorder=6)
        
        if node_data.get("visit_count", 0) > 1:
            ax.text(x+0.3, y+0.3, f"×{node_data['visit_count']}", ha='left', va='bottom', 
                   fontsize=6, color='red', weight='bold', zorder=6)
    
    # Plot settings
    ax.set_xlim(bounds["min_x"] - 1, bounds["max_x"] + 1)
    ax.set_ylim(bounds["min_y"] - 1, bounds["max_y"] + 1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    
    total_nodes = metadata["total_nodes"]
    total_steps = metadata.get("total_steps", "N/A")
    drift_corrections = metadata.get("drift_corrections", 0)
    
    title = (f"Robot Maze Exploration Map\nNodes: {total_nodes} | Steps: {total_steps} | "
             f"Drift Corrections: {drift_corrections}")
    ax.set_title(title, fontsize=14, weight='bold', pad=20)
    
    # Legend
    legend_elements = [
        plt.scatter([], [], c=colors['start_node'], s=100, marker='s', label='Start Position'),
        plt.scatter([], [], c=colors['normal_node'], s=80, marker='o', label='Normal Node'),
        plt.scatter([], [], c=colors['dead_end'], s=100, marker='X', label='Dead End'),
        plt.Line2D([0], [0], color=colors['robot_path'], lw=2, label='Exploration Path'),
        plt.Line2D([0], [0], color=colors['backtrack'], lw=2, label='Backtrack Path'),
        patches.Rectangle((0, 0), 1, 1, facecolor=colors['wall'], label='Wall')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    
    filename = f"{save_path}maze_static_map.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Static map saved: {filename}")
    
    return fig

def create_animated_visualization(maze_data, save_path="Assignment/plots/"):
    """Create animated maze exploration visualization"""
    if not maze_data:
        return None
    
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    metadata = maze_data["metadata"]
    bounds = metadata["boundaries"]
    nodes = maze_data["nodes"]
    exploration_history = maze_data.get("exploration_history", [])
    decision_points = maze_data.get("decision_points", [])
    
    if not exploration_history:
        print("No exploration history found for animation")
        return None
    
    # Setup plot
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(bounds["min_x"] - 1.5, bounds["max_x"] + 1.5)
    ax.set_ylim(bounds["min_y"] - 1.5, bounds["max_y"] + 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    
    colors = {
        'wall': '#2C3E50', 'robot_current': '#E74C3C', 'robot_trail': '#3498DB',
        'scanned_node': '#27AE60', 'dead_end': '#C0392B', 'backtrack': '#9B59B6',
        'decision': '#F39C12'
    }
    
    # Initialize animated elements
    robot_scatter = ax.scatter([], [], s=300, c=colors['robot_current'], 
                              marker='o', edgecolors='black', linewidth=2, zorder=10)
    status_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=10, 
                         verticalalignment='top',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    title = f"Robot Maze Exploration Animation\nTotal Steps: {metadata.get('total_steps', 'N/A')}"
    ax.set_title(title, fontsize=14, weight='bold', pad=20)
    
    # Storage for dynamic elements
    trail_lines = []
    wall_patches = []
    node_scatters = []
    
    def draw_walls_for_node(node_pos, walls, alpha=1.0):
        """Draw walls for a specific node"""
        x, y = node_pos
        wall_thickness = 0.1
        
        wall_coords = [
            (walls.get("north", False), (x - 0.5, y + 0.5 - wall_thickness/2), (1, wall_thickness)),
            (walls.get("south", False), (x - 0.5, y - 0.5 - wall_thickness/2), (1, wall_thickness)),
            (walls.get("east", False), (x + 0.5 - wall_thickness/2, y - 0.5), (wall_thickness, 1)),
            (walls.get("west", False), (x - 0.5 - wall_thickness/2, y - 0.5), (wall_thickness, 1))
        ]
        
        patches_added = []
        for has_wall, (wx, wy), (width, height) in wall_coords:
            if has_wall:
                wall = patches.Rectangle((wx, wy), width, height, 
                                       facecolor=colors['wall'], edgecolor='none', alpha=alpha)
                ax.add_patch(wall)
                patches_added.append(wall)
                wall_patches.append(wall)
        
        return patches_added
    
    def animate_frame(frame_num):
        """Animate a single frame"""
        if frame_num >= len(exploration_history):
            return [robot_scatter, status_text]
        
        step_data = exploration_history[frame_num]
        
        # Update robot position
        robot_pos = step_data["position"]
        robot_scatter.set_offsets([robot_pos])
        
        # Update status text
        status = f"Step: {step_data['step']}\n"
        status += f"Position: {robot_pos}\n"
        status += f"Direction: {step_data['direction']}\n"
        status += f"Action: {step_data['action_type']}\n"
        
        if step_data.get('sensor_readings'):
            readings = step_data['sensor_readings']
            status += "Sensors: "
            for direction, value in readings.items():
                if isinstance(value, (int, float)):
                    status += f"{direction[0].upper()}:{value:.1f} "
            status += "cm\n"
        
        if step_data.get('frontier_queue'):
            status += f"Frontiers: {len(step_data['frontier_queue'])}"
        
        status_text.set_text(status)
        
        # Draw walls if this is a scan step
        if step_data['action_type'] in ['scan', 'cached_scan'] and step_data.get('walls'):
            draw_walls_for_node(robot_pos, step_data['walls'])
        
        # Add node marker
        if step_data['action_type'] in ['scan', 'cached_scan']:
            color, size = colors['scanned_node'], 150
        elif step_data['action_type'] == 'dead_end':
            color, size = colors['dead_end'], 200
        else:
            color, size = colors['robot_trail'], 80
        
        node_scatter = ax.scatter(robot_pos[0], robot_pos[1], 
                                 c=color, s=size, alpha=0.7, zorder=5)
        node_scatters.append(node_scatter)
        
        # Draw trail line from previous position
        if frame_num > 0:
            prev_step = exploration_history[frame_num - 1]
            prev_pos = prev_step["position"]
            
            line_props = {
                'backtrack': (colors['backtrack'], 3, 0.8),
                'dead_end': (colors['dead_end'], 4, 0.9)
            }
            
            line_color, line_width, alpha = line_props.get(
                step_data['action_type'], 
                (colors['robot_trail'], 2, 0.6)
            )
            
            line = ax.plot([prev_pos[0], robot_pos[0]], [prev_pos[1], robot_pos[1]], 
                          color=line_color, linewidth=line_width, alpha=alpha, zorder=3)[0]
            trail_lines.append(line)
        
        # Mark decision points
        for decision in decision_points:
            if decision.get('step', -1) == step_data['step']:
                decision_pos = decision['position']
                decision_type = decision['decision_type']
                
                marker_props = {
                    'dead_end': ('X', colors['dead_end']),
                    'backtrack_to_frontier': ('^', colors['backtrack']),
                    'default': ('D', colors['decision'])
                }
                
                marker, color = marker_props.get(decision_type, marker_props['default'])
                
                decision_scatter = ax.scatter(decision_pos[0], decision_pos[1], 
                                             c=color, s=100, marker=marker, 
                                             alpha=0.9, zorder=8, edgecolors='black')
                node_scatters.append(decision_scatter)
        
        return [robot_scatter, status_text] + node_scatters + trail_lines
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate_frame, 
        frames=len(exploration_history),
        interval=800,  # 800ms between frames
        blit=False, repeat=True
    )
    
    return fig, anim

def save_animation(fig, anim, save_path="Assignment/plots/", filename="maze_exploration_animation.gif"):
    """Save animation as GIF and optionally MP4"""
    full_path = f"{save_path}{filename}"
    print(f"Saving animation to: {full_path}")
    print("This may take several minutes...")
    
    try:
        # Save as GIF
        writer = animation.PillowWriter(fps=1.2, metadata=dict(artist='Robot Maze Explorer'))
        anim.save(full_path, writer=writer)
        print(f"Animation saved successfully as GIF!")
        
        # Try to save as MP4 if ffmpeg is available
        try:
            mp4_path = full_path.replace('.gif', '.mp4')
            writer_mp4 = animation.FFMpegWriter(fps=1.2, metadata=dict(artist='Robot Maze Explorer'))
            anim.save(mp4_path, writer=writer_mp4)
            print(f"MP4 version also saved: {mp4_path}")
        except Exception as e:
            print(f"Could not save MP4 version (ffmpeg not available): {e}")
            
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Make sure you have Pillow installed: pip install Pillow")

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='Visualize robot maze exploration')
    parser.add_argument('--input', '-i', default='Assignment/data/maze_data_enhanced.json',
                       help='Input JSON file path')
    parser.add_argument('--output', '-o', default='Assignment/plots/',
                       help='Output directory for plots')
    parser.add_argument('--static', '-s', action='store_true',
                       help='Create static plot only')
    parser.add_argument('--animation', '-a', action='store_true',
                       help='Create animation only')
    parser.add_argument('--show', action='store_true',
                       help='Show plots in window (in addition to saving)')
    
    args = parser.parse_args()
    
    # Load maze data
    print(f"Loading maze data from: {args.input}")
    maze_data = load_maze_data(args.input)
    
    if not maze_data:
        print("Failed to load maze data. Exiting.")
        return
    
    print(f"Loaded maze data with {maze_data['metadata']['total_nodes']} nodes")
    
    # Create visualizations based on arguments
    if not args.animation:  # Create static by default, or if explicitly requested
        print("Creating static maze visualization...")
        fig_static = create_static_maze_plot(maze_data, args.output)
        if fig_static and args.show:
            plt.show()
    
    if not args.static:  # Create animation by default, or if explicitly requested
        print("Creating animated maze visualization...")
        fig_anim, anim = create_animated_visualization(maze_data, args.output)
        
        if fig_anim and anim:
            # Save animation
            save_animation(fig_anim, anim, args.output)
            
            # Show animation if requested
            if args.show:
                plt.show()
        else:
            print("Failed to create animation")

if __name__ == "__main__":
    # If no command line args, create both static and animated versions
    import sys
    if len(sys.argv) == 1:
        print("Creating both static and animated visualizations...")
        
        maze_data = load_maze_data()
        if maze_data:
            print("Creating static maze map...")
            create_static_maze_plot(maze_data)
            
            print("Creating animated visualization...")
            fig, anim = create_animated_visualization(maze_data)
            if fig and anim:
                save_animation(fig, anim)
                print("All visualizations created successfully!")
            else:
                print("Animation creation failed")
        else:
            print("Failed to load maze data")
    else:
        main()