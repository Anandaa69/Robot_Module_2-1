#!/usr/bin/env python3
"""
Exploration Data Analyzer
Advanced analysis tools for robot exploration data
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import argparse
import os
from datetime import datetime
import pandas as pd

class ExplorationAnalyzer:
    def __init__(self, data_file):
        self.data_file = data_file
        self.exploration_data = None
        self.nodes = {}
        self.metadata = {}
        self.statistics = {}
        self.load_data()
    
    def load_data(self):
        """Load exploration data from JSON file"""
        with open(self.data_file, 'r', encoding='utf-8') as f:
            self.exploration_data = json.load(f)
        
        self.nodes = self.exploration_data.get('nodes', {})
        self.metadata = self.exploration_data.get('metadata', {})
        self.statistics = self.exploration_data.get('statistics', {})
        
        print(f"📊 Loaded {len(self.nodes)} nodes from {self.data_file}")
    
    def analyze_exploration_efficiency(self):
        """Analyze exploration efficiency metrics"""
        print("\n🔍 === EXPLORATION EFFICIENCY ANALYSIS ===")
        
        total_nodes = len(self.nodes)
        dead_ends = sum(1 for node in self.nodes.values() if node.get('isDeadEnd', False))
        frontiers = len([node for node in self.nodes.values() 
                        if node.get('unexploredExits', [])])
        fully_scanned = sum(1 for node in self.nodes.values() 
                           if node.get('fullyScanned', False))
        
        print(f"📊 Basic Metrics:")
        print(f"   Total nodes explored: {total_nodes}")
        print(f"   Fully scanned nodes: {fully_scanned}")
        print(f"   Dead ends found: {dead_ends}")
        print(f"   Active frontiers: {frontiers}")
        print(f"   Dead end ratio: {dead_ends/total_nodes*100:.1f}%")
        
        # Calculate connectivity
        connections = 0
        for node in self.nodes.values():
            neighbors = node.get('neighbors', {})
            connections += sum(1 for neighbor_id in neighbors.values() 
                             if neighbor_id is not None)
        
        avg_connectivity = connections / (2 * total_nodes) if total_nodes > 0 else 0
        print(f"   Average connectivity: {avg_connectivity:.2f}")
        
        # Wall density analysis
        if 'wall_statistics' in self.statistics:
            wall_stats = self.statistics['wall_statistics']
            total_walls = wall_stats.get('total_walls', 0)
            total_openings = wall_stats.get('total_openings', 0)
            if total_walls + total_openings > 0:
                wall_density = total_walls / (total_walls + total_openings) * 100
                print(f"   Wall density: {wall_density:.1f}%")
        
        return {
            'total_nodes': total_nodes,
            'dead_ends': dead_ends,
            'frontiers': frontiers,
            'fully_scanned': fully_scanned,
            'avg_connectivity': avg_connectivity,
            'dead_end_ratio': dead_ends/total_nodes*100 if total_nodes > 0 else 0
        }
    
    def analyze_spatial_distribution(self):
        """Analyze spatial distribution of nodes and features"""
        print("\n🗺️ === SPATIAL DISTRIBUTION ANALYSIS ===")
        
        if not self.nodes:
            print("No nodes to analyze")
            return
        
        positions = [node['position'] for node in self.nodes.values()]
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        # Basic spatial stats
        print(f"📐 Spatial Metrics:")
        print(f"   X range: {min(x_coords)} to {max(x_coords)} (span: {max(x_coords) - min(x_coords)})")
        print(f"   Y range: {min(y_coords)} to {max(y_coords)} (span: {max(y_coords) - min(y_coords)})")
        print(f"   Center of mass: ({np.mean(x_coords):.1f}, {np.mean(y_coords):.1f})")
        
        # Analyze node density
        area = (max(x_coords) - min(x_coords) + 1) * (max(y_coords) - min(y_coords) + 1)
        density = len(self.nodes) / area if area > 0 else 0
        print(f"   Node density: {density:.3f} nodes per unit area")
        
        # Analyze dead end distribution
        dead_end_positions = [node['position'] for node in self.nodes.values() 
                             if node.get('isDeadEnd', False)]
        if dead_end_positions:
            dead_x = [pos[0] for pos in dead_end_positions]
            dead_y = [pos[1] for pos in dead_end_positions]
            print(f"   Dead end center: ({np.mean(dead_x):.1f}, {np.mean(dead_y):.1f})")
        
        return {
            'x_range': (min(x_coords), max(x_coords)),
            'y_range': (min(y_coords), max(y_coords)),
            'center_of_mass': (np.mean(x_coords), np.mean(y_coords)),
            'node_density': density,
            'total_area': area
        }
    
    def analyze_wall_patterns(self):
        """Analyze wall patterns and maze structure"""
        print("\n🧱 === WALL PATTERN ANALYSIS ===")
        
        # Count walls by direction
        wall_counts = {'north': 0, 'south': 0, 'east': 0, 'west': 0}
        total_scanned = 0
        
        for node in self.nodes.values():
            walls = node.get('walls', {})
            if walls:
                total_scanned += 1
                for direction, has_wall in walls.items():
                    if has_wall:
                        wall_counts[direction] += 1
        
        print(f"🔍 Wall Statistics:")
        print(f"   Nodes with wall data: {total_scanned}")
        for direction, count in wall_counts.items():
            percentage = count / total_scanned * 100 if total_scanned > 0 else 0
            print(f"   {direction.capitalize()} walls: {count} ({percentage:.1f}%)")
        
        # Analyze wall symmetry
        if total_scanned > 0:
            north_south_ratio = wall_counts['north'] / max(wall_counts['south'], 1)
            east_west_ratio = wall_counts['east'] / max(wall_counts['west'], 1)
            print(f"   North/South ratio: {north_south_ratio:.2f}")
            print(f"   East/West ratio: {east_west_ratio:.2f}")
        
        # Identify potential corridors (nodes with exactly 2 openings)
        corridors = []
        junctions = []
        
        for node_id, node in self.nodes.items():
            walls = node.get('walls', {})
            if walls:
                wall_count = sum(1 for has_wall in walls.values() if has_wall)
                opening_count = 4 - wall_count
                
                if opening_count == 2:
                    corridors.append(node_id)
                elif opening_count >= 3:
                    junctions.append(node_id)
        
        print(f"   Corridor nodes (2 openings): {len(corridors)}")
        print(f"   Junction nodes (3+ openings): {len(junctions)}")
        
        return {
            'wall_counts': wall_counts,
            'total_scanned': total_scanned,
            'corridors': len(corridors),
            'junctions': len(junctions)
        }
    
    def create_path_analysis(self):
        """Analyze potential paths and connectivity"""
        print("\n🛤️ === PATH ANALYSIS ===")
        
        # Build adjacency list
        adjacency = {}
        for node_id, node in self.nodes.items():
            adjacency[node_id] = []
            neighbors = node.get('neighbors', {})
            for direction, neighbor_id in neighbors.items():
                if neighbor_id and neighbor_id in self.nodes:
                    # Check if path is not blocked by wall
                    walls = node.get('walls', {})
                    if not walls.get(direction, True):
                        adjacency[node_id].append(neighbor_id)
        
        # Calculate connectivity metrics
        connected_components = self.find_connected_components(adjacency)
        largest_component = max(connected_components, key=len) if connected_components else []
        
        print(f"🔗 Connectivity Analysis:")
        print(f"   Connected components: {len(connected_components)}")
        print(f"   Largest component size: {len(largest_component)}")
        print(f"   Connectivity ratio: {len(largest_component)/len(self.nodes)*100:.1f}%")
        
        # Find longest path
        if largest_component:
            longest_path = self.find_longest_path(adjacency, largest_component)
            print(f"   Longest path length: {len(longest_path)-1} steps")
        
        return {
            'connected_components': len(connected_components),
            'largest_component_size': len(largest_component),
            'connectivity_ratio': len(largest_component)/len(self.nodes)*100 if self.nodes else 0
        }
    
    def find_connected_components(self, adjacency):
        """Find connected components using DFS"""
        visited = set()
        components = []
        
        def dfs(node, component):
            visited.add(node)
            component.append(node)
            for neighbor in adjacency.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, component)
        
        for node_id in adjacency:
            if node_id not in visited:
                component = []
                dfs(node_id, component)
                components.append(component)
        
        return components
    
    def find_longest_path(self, adjacency, nodes):
        """Find longest path in the connected component"""
        max_length = 0
        longest_path = []
        
        def dfs_path(start, current, path, visited):
            nonlocal max_length, longest_path
            
            if len(path) > max_length:
                max_length = len(path)
                longest_path = path.copy()
            
            for neighbor in adjacency.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs_path(start, neighbor, path, visited)
                    path.pop()
                    visited.remove(neighbor)
        
        # Try starting from each node (for disconnected or complex graphs)
        for start_node in nodes[:min(10, len(nodes))]:  # Limit for performance
            visited = {start_node}
            dfs_path(start_node, start_node, [start_node], visited)
        
        return longest_path
    
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report"""
        print("\n" + "="*60)
        print("📋 COMPREHENSIVE EXPLORATION ANALYSIS REPORT")
        print("="*60)
        
        # Basic info
        print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Data Source: {self.data_file}")
        if 'timestamp' in self.metadata:
            print(f"🕐 Exploration Date: {self.metadata['timestamp']}")
        
        # Run all analyses
        efficiency = self.analyze_exploration_efficiency()
        spatial = self.analyze_spatial_distribution()
        walls = self.analyze_wall_patterns()
        paths = self.create_path_analysis()
        
        # Summary scores
        print(f"\n⭐ EXPLORATION QUALITY SCORES:")
        
        # Coverage score (based on area explored vs dead ends)
        coverage_score = min(100, (efficiency['total_nodes'] / max(spatial['total_area'], 1)) * 100)
        print(f"   Coverage Score: {coverage_score:.1f}/100")
        
        # Efficiency score (fewer dead ends = better)
        efficiency_score = max(0, 100 - efficiency['dead_end_ratio'])
        print(f"   Efficiency Score: {efficiency_score:.1f}/100")
        
        # Connectivity score
        connectivity_score = paths['connectivity_ratio']
        print(f"   Connectivity Score: {connectivity_score:.1f}/100")
        
        # Overall score
        overall_score = (coverage_score + efficiency_score + connectivity_score) / 3
        print(f"   Overall Score: {overall_score:.1f}/100")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if efficiency['dead_end_ratio'] > 30:
            print("   🚫 High dead end ratio - consider different exploration strategy")
        if paths['connectivity_ratio'] < 80:
            print("   🔗 Low connectivity - some areas may be unreachable")
        if walls['junctions'] < efficiency['total_nodes'] * 0.1:
            print("   🛤️ Few junctions found - environment may be mostly corridors")
        
        print("="*60)
        
        return {
            'efficiency': efficiency,
            'spatial': spatial,
            'walls': walls,
            'paths': paths,
            'scores': {
                'coverage': coverage_score,
                'efficiency': efficiency_score,
                'connectivity': connectivity_score,
                'overall': overall_score
            }
        }
    
    def create_heatmap_visualization(self, save_path=None):
        """Create heatmap showing various metrics"""
        if not self.nodes:
            print("No data to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Exploration Analysis Heatmaps', fontsize=16, fontweight='bold')
        
        # Get map bounds
        positions = [node['position'] for node in self.nodes.values()]
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Create grids
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        
        # 1. Wall density heatmap
        wall_grid = np.zeros((height, width))
        for node in self.nodes.values():
            x, y = node['position']
            walls = node.get('walls', {})
            wall_count = sum(1 for wall in walls.values() if wall)
            wall_grid[max_y - y, x - min_x] = wall_count
        
        im1 = axes[0,0].imshow(wall_grid, cmap='Reds', aspect='equal')
        axes[0,0].set_title('Wall Density')
        plt.colorbar(im1, ax=axes[0,0])
        
        # 2. Visit frequency (if available)
        visit_grid = np.zeros((height, width))
        for node in self.nodes.values():
            x, y = node['position']
            visit_count = node.get('visitCount', 1)
            visit_grid[max_y - y, x - min_x] = visit_count
        
        im2 = axes[0,1].imshow(visit_grid, cmap='Blues', aspect='equal')
        axes[0,1].set_title('Visit Frequency')
        plt.colorbar(im2, ax=axes[0,1])
        
        # 3. Node types
        type_grid = np.zeros((height, width))
        for node in self.nodes.values():
            x, y = node['position']
            if node.get('isDeadEnd', False):
                type_grid[max_y - y, x - min_x] = 3  # Dead end
            elif node.get('unexploredExits', []):
                type_grid[max_y - y, x - min_x] = 2  # Frontier
            elif node['position'] == [0, 0]:
                type_grid[max_y - y, x - min_x] = 1  # Start
            else:
                type_grid[max_y - y, x - min_x] = 0  # Normal
        
        im3 = axes[1,0].imshow(type_grid, cmap='viridis', aspect='equal')
        axes[1,0].set_title('Node Types')
        plt.colorbar(im3, ax=axes[1,0])
        
        # 4. Sensor readings average
        sensor_grid = np.zeros((height, width))
        for node in self.nodes.values():
            x, y = node['position']
            readings = node.get('sensorReadings', {})
            if readings:
                avg_reading = np.mean(list(readings.values()))
                sensor_grid[max_y - y, x - min_x] = avg_reading
        
        im4 = axes[1,1].imshow(sensor_grid, cmap='plasma', aspect='equal')
        axes[1,1].set_title('Average Sensor Readings')
        plt.colorbar(im4, ax=axes[1,1])
        
        # Set ticks for all subplots
        for ax in axes.flat:
            ax.set_xticks(range(0, width, max(1, width//10)))
            ax.set_yticks(range(0, height, max(1, height//10)))
            ax.set_xticklabels([f"{min_x + i}" for i in range(0, width, max(1, width//10))])
            ax.set_yticklabels([f"{max_y - i}" for i in range(0, height, max(1, height//10))])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Heatmap saved to: {save_path}")
        
        plt.show()
        return fig

def main():
    parser = argparse.ArgumentParser(description='Analyze robot exploration data')
    parser.add_argument('data_file', nargs='?', help='JSON data file to analyze')
    parser.add_argument('--heatmap', action='store_true', help='Generate heatmap visualization')
    parser.add_argument('--output', '-o', help='Output file path for visualizations')
    
    args = parser.parse_args()
    
    # Find data file if not specified
    if not args.data_file:
        data_dir = 'exploration_data'
        if os.path.exists(data_dir):
            json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
            if json_files:
                json_files.sort(key=lambda x: os.path.getmtime(os.path.join(data_dir, x)), reverse=True)
                args.data_file = os.path.join(data_dir, json_files[0])
                print(f"📁 Using most recent data file: {args.data_file}")
            else:
                print("❌ No JSON files found in exploration_data directory")
                return 1
        else:
            print("❌ No data file specified and exploration_data directory not found")
            return 1
    
    try:
        # Create analyzer
        analyzer = ExplorationAnalyzer(args.data_file)
        
        # Generate comprehensive report
        report = analyzer.generate_comprehensive_report()
        
        # Create heatmap if requested
        if args.heatmap:
            output_path = args.output or f"exploration_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            analyzer.create_heatmap_visualization(output_path)
        
        print("\n✅ Analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())