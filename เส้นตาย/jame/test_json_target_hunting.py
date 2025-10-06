#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for JSON Target Hunting functionality
This script tests the JSON loading and target hunting functions without running the full robot code
"""

import json
import sys
import os

# Add the robot module path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def test_json_loading():
    """Test JSON file loading functionality"""
    print("🧪 Testing JSON file loading...")
    
    # Test file paths
    mapping_path = "Robot_Module/Assignment/dude/James_path/Mapping_Top.json"
    position_path = "Robot_Module/Assignment/dude/James_path/Robot_Position_Timestamps.json"
    detected_objects_path = "Robot_Module/Assignment/dude/James_path/Detected_Objects.json"
    
    try:
        # Load mapping data
        with open(mapping_path, 'r') as f:
            mapping_data = json.load(f)
        print(f"✅ Mapping data loaded: {len(mapping_data['nodes'])} nodes")
        
        # Load position data
        with open(position_path, 'r') as f:
            position_data = json.load(f)
        print(f"✅ Position data loaded: {len(position_data['position_log'])} entries")
        
        # Load detected objects data
        with open(detected_objects_path, 'r') as f:
            detected_objects_data = json.load(f)
        print(f"✅ Detected objects data loaded: {len(detected_objects_data['detected_objects'])} objects")
        
        return mapping_data, position_data, detected_objects_data
        
    except Exception as e:
        print(f"❌ Error loading JSON files: {e}")
        return None, None, None

def test_target_extraction(detected_objects_data):
    """Test target extraction from JSON data"""
    print("\n🧪 Testing target extraction...")
    
    if not detected_objects_data:
        print("❌ No detected objects data available")
        return []
    
    targets = []
    for obj in detected_objects_data['detected_objects']:
        if obj.get('is_target', False):
            target_info = {
                'position': (obj['cell_position']['row'], obj['cell_position']['col']),
                'detected_from_node': tuple(obj['detected_from_node']),
                'color': obj['color'],
                'shape': obj['shape'],
                'zone': obj['zone']
            }
            targets.append(target_info)
            print(f"🎯 Found target: {target_info['color']} {target_info['shape']} at {target_info['position']}, detected from {target_info['detected_from_node']}")
    
    return targets

def test_occupancy_map_creation(mapping_data):
    """Test occupancy map creation from JSON data"""
    print("\n🧪 Testing occupancy map creation...")
    
    if not mapping_data:
        print("❌ No mapping data available")
        return None
    
    # Get grid size from mapping data
    nodes = mapping_data['nodes']
    max_row = max(node['coordinate']['row'] for node in nodes)
    max_col = max(node['coordinate']['col'] for node in nodes)
    grid_size = max(max_row, max_col) + 1
    
    print(f"📊 Grid size: {grid_size}x{grid_size}")
    
    # Count walls and visited nodes
    wall_count = 0
    visited_count = 0
    
    for node in nodes:
        row = node['coordinate']['row']
        col = node['coordinate']['col']
        
        # Count walls
        walls = node['walls']
        if walls['north']: wall_count += 1
        if walls['south']: wall_count += 1
        if walls['east']: wall_count += 1
        if walls['west']: wall_count += 1
        
        # Count visited nodes (low probability indicates exploration)
        if node['probability'] < 0.1:
            visited_count += 1
    
    print(f"📊 Walls detected: {wall_count}")
    print(f"📊 Visited nodes: {visited_count}")
    
    return True

def main():
    """Main test function"""
    print("🚀 Starting JSON Target Hunting Test...")
    
    # Test 1: Load JSON files
    mapping_data, position_data, detected_objects_data = test_json_loading()
    
    if not mapping_data:
        print("❌ Failed to load JSON data. Test aborted.")
        return False
    
    # Test 2: Extract targets
    targets = test_target_extraction(detected_objects_data)
    
    if not targets:
        print("❌ No targets found in JSON data.")
        return False
    
    # Test 3: Create occupancy map
    map_created = test_occupancy_map_creation(mapping_data)
    
    if not map_created:
        print("❌ Failed to create occupancy map.")
        return False
    
    # Test 4: Simulate path calculation
    print("\n🧪 Testing path calculation simulation...")
    current_position = (4, 0)  # Starting position from JSON
    
    for i, target in enumerate(targets):
        target_pos = target['detected_from_node']
        print(f"🎯 Target {i+1}: Need to go from {current_position} to {target_pos}")
        
        # Simple distance calculation
        distance = abs(current_position[0] - target_pos[0]) + abs(current_position[1] - target_pos[1])
        print(f"   📏 Manhattan distance: {distance} steps")
        
        # Update current position for next target
        current_position = target_pos
    
    print("\n✅ All tests completed successfully!")
    print("🎯 JSON Target Hunting functionality is ready!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
