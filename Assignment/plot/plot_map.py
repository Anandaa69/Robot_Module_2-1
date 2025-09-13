import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np
from collections import defaultdict
import json

# ข้อมูลแผนที่จาก JSON
maze_data = {
    "metadata": {
        "timestamp": "2025-09-13T11:22:29.687115",
        "total_nodes": 8,
        "boundaries": {
            "min_x": -2,
            "max_x": 1,
            "min_y": 0,
            "max_y": 1
        },
        "drift_corrections": 1
    },
    "nodes": {
        "0_0": {
            "position": [0, 0],
            "walls": {"north": False, "south": True, "east": True, "west": True},
            "is_dead_end": True,
            "explored_directions": ["north"],
            "unexplored_exits": []
        },
        "0_1": {
            "position": [0, 1],
            "walls": {"north": True, "south": False, "east": False, "west": False},
            "is_dead_end": False,
            "explored_directions": ["west", "east"],
            "unexplored_exits": []
        },
        "-1_1": {
            "position": [-1, 1],
            "walls": {"north": True, "south": False, "east": False, "west": False},
            "is_dead_end": False,
            "explored_directions": ["south", "west"],
            "unexplored_exits": []
        },
        "-1_0": {
            "position": [-1, 0],
            "walls": {"north": False, "south": True, "east": True, "west": True},
            "is_dead_end": True,
            "explored_directions": [],
            "unexplored_exits": []
        },
        "-2_1": {
            "position": [-2, 1],
            "walls": {"north": True, "south": False, "east": False, "west": True},
            "is_dead_end": False,
            "explored_directions": ["south"],
            "unexplored_exits": []
        },
        "-2_0": {
            "position": [-2, 0],
            "walls": {"north": False, "south": True, "east": True, "west": True},
            "is_dead_end": True,
            "explored_directions": [],
            "unexplored_exits": []
        },
        "1_1": {
            "position": [1, 1],
            "walls": {"north": True, "south": False, "east": True, "west": False},
            "is_dead_end": False,
            "explored_directions": ["south"],
            "unexplored_exits": []
        },
        "1_0": {
            "position": [1, 0],
            "walls": {"north": False, "south": True, "east": True, "west": True},
            "is_dead_end": True,
            "explored_directions": [],
            "unexplored_exits": []
        }
    }
}

def create_static_maze_plot():
    """สร้างกราฟแผนที่แบบภาพนิ่ง"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # สี่เหลี่ยมแต่ละช่อง
    cell_size = 1.0
    
    # วาดห้องทั้งหมด
    for node_id, node_data in maze_data['nodes'].items():
        x, y = node_data['position']
        
        # วาดพื้นห้อง
        if node_data['is_dead_end']:
            color = '#ffcccc'  # สีแดงอ่อนสำหรับทางตัน
        else:
            color = '#ccffcc'  # สีเขียวอ่อนสำหรับทางผ่าน
            
        rect = patches.Rectangle((x-0.4, y-0.4), 0.8, 0.8, 
                                linewidth=1, edgecolor='black', 
                                facecolor=color, alpha=0.7)
        ax.add_patch(rect)
        
        # วาดกำแพง
        wall_thickness = 0.1
        if node_data['walls']['north']:
            wall = patches.Rectangle((x-0.5, y+0.4), 1.0, wall_thickness, 
                                   facecolor='black')
            ax.add_patch(wall)
        if node_data['walls']['south']:
            wall = patches.Rectangle((x-0.5, y-0.5), 1.0, wall_thickness, 
                                   facecolor='black')
            ax.add_patch(wall)
        if node_data['walls']['east']:
            wall = patches.Rectangle((x+0.4, y-0.5), wall_thickness, 1.0, 
                                   facecolor='black')
            ax.add_patch(wall)
        if node_data['walls']['west']:
            wall = patches.Rectangle((x-0.5, y-0.5), wall_thickness, 1.0, 
                                   facecolor='black')
            ax.add_patch(wall)
        
        # แสดงพิกัด
        ax.text(x, y, f'({x},{y})', ha='center', va='center', 
                fontsize=10, fontweight='bold')
        
        # แสดงทิศทางที่สำรวจแล้ว
        if node_data['explored_directions']:
            directions_text = ', '.join(node_data['explored_directions'])
            ax.text(x, y-0.25, f"สำรวจ: {directions_text}", 
                   ha='center', va='center', fontsize=8, 
                   style='italic', color='blue')
    
    # วาดเส้นเชื่อมระหว่างห้อง (ทางเดิน)
    for node_id, node_data in maze_data['nodes'].items():
        x, y = node_data['position']
        
        # เช็คการเชื่อมต่อและวาดเส้น
        if not node_data['walls']['north']:
            next_pos = (x, y+1)
            next_key = f"{next_pos[0]}_{next_pos[1]}"
            if next_key in maze_data['nodes']:
                ax.plot([x, x], [y+0.4, y+0.6], 'g-', linewidth=3, alpha=0.8)
        
        if not node_data['walls']['east']:
            next_pos = (x+1, y)
            next_key = f"{next_pos[0]}_{next_pos[1]}"
            if next_key in maze_data['nodes']:
                ax.plot([x+0.4, x+0.6], [y, y], 'g-', linewidth=3, alpha=0.8)
    
    # ตั้งค่ากราฟ
    ax.set_xlim(maze_data['metadata']['boundaries']['min_x']-1, 
                maze_data['metadata']['boundaries']['max_x']+1)
    ax.set_ylim(maze_data['metadata']['boundaries']['min_y']-1, 
                maze_data['metadata']['boundaries']['max_y']+1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('แผนที่เขาวงกต - ผลการสำรวจ\n' + 
                f"จำนวนห้องทั้งหมด: {maze_data['metadata']['total_nodes']} ห้อง", 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('พิกัด X', fontsize=12)
    ax.set_ylabel('พิกัด Y', fontsize=12)
    
    # Legend
    legend_elements = [
        patches.Patch(color='#ccffcc', alpha=0.7, label='ทางผ่าน'),
        patches.Patch(color='#ffcccc', alpha=0.7, label='ทางตัน'),
        patches.Patch(color='black', label='กำแพง'),
        plt.Line2D([0], [0], color='green', lw=3, label='ทางเดิน')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    return fig, ax

def create_animated_maze_plot():
    """สร้างกราฟแผนที่แบบอนิเมชัน"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # เรียงลำดับ nodes ตาม timestamp (สมมุติตามลำดับใน dict)
    nodes_list = list(maze_data['nodes'].items())
    
    def animate(frame):
        ax.clear()
        
        # แสดง nodes ที่ค้นพบแล้วจนถึงเฟรมปัจจุบัน
        for i in range(min(frame + 1, len(nodes_list))):
            node_id, node_data = nodes_list[i]
            x, y = node_data['position']
            
            # วาดพื้นห้อง
            if node_data['is_dead_end']:
                color = '#ffcccc'  # สีแดงอ่อนสำหรับทางตัน
            else:
                color = '#ccffcc'  # สีเขียวอ่อนสำหรับทางผ่าน
                
            # เอฟเฟกต์การปรากฏ
            alpha = 0.3 if i == frame else 0.7
            rect = patches.Rectangle((x-0.4, y-0.4), 0.8, 0.8, 
                                    linewidth=2 if i == frame else 1, 
                                    edgecolor='red' if i == frame else 'black', 
                                    facecolor=color, alpha=alpha)
            ax.add_patch(rect)
            
            # วาดกำแพง
            wall_thickness = 0.1
            wall_color = 'red' if i == frame else 'black'
            if node_data['walls']['north']:
                wall = patches.Rectangle((x-0.5, y+0.4), 1.0, wall_thickness, 
                                       facecolor=wall_color)
                ax.add_patch(wall)
            if node_data['walls']['south']:
                wall = patches.Rectangle((x-0.5, y-0.5), 1.0, wall_thickness, 
                                       facecolor=wall_color)
                ax.add_patch(wall)
            if node_data['walls']['east']:
                wall = patches.Rectangle((x+0.4, y-0.5), wall_thickness, 1.0, 
                                       facecolor=wall_color)
                ax.add_patch(wall)
            if node_data['walls']['west']:
                wall = patches.Rectangle((x-0.5, y-0.5), wall_thickness, 1.0, 
                                       facecolor=wall_color)
                ax.add_patch(wall)
            
            # แสดงพิกัด
            ax.text(x, y, f'({x},{y})', ha='center', va='center', 
                    fontsize=12 if i == frame else 10, 
                    fontweight='bold', 
                    color='red' if i == frame else 'black')
            
            # แสดงทิศทางที่สำรวจแล้ว
            if node_data['explored_directions'] and i <= frame:
                directions_text = ', '.join(node_data['explored_directions'])
                ax.text(x, y-0.25, f"สำรวจ: {directions_text}", 
                       ha='center', va='center', fontsize=8, 
                       style='italic', color='blue')
        
        # วาดเส้นเชื่อมระหว่างห้องที่ค้นพบแล้ว
        for i in range(min(frame + 1, len(nodes_list))):
            node_id, node_data = nodes_list[i]
            x, y = node_data['position']
            
            if not node_data['walls']['north']:
                next_pos = (x, y+1)
                next_key = f"{next_pos[0]}_{next_pos[1]}"
                if next_key in maze_data['nodes']:
                    # เช็คว่า node ปลายทางถูกค้นพบแล้วหรือยัง
                    next_index = next((j for j, (nid, _) in enumerate(nodes_list) 
                                     if nid == next_key), None)
                    if next_index is not None and next_index <= frame:
                        ax.plot([x, x], [y+0.4, y+0.6], 'g-', linewidth=3, alpha=0.8)
            
            if not node_data['walls']['east']:
                next_pos = (x+1, y)
                next_key = f"{next_pos[0]}_{next_pos[1]}"
                if next_key in maze_data['nodes']:
                    next_index = next((j for j, (nid, _) in enumerate(nodes_list) 
                                     if nid == next_key), None)
                    if next_index is not None and next_index <= frame:
                        ax.plot([x+0.4, x+0.6], [y, y], 'g-', linewidth=3, alpha=0.8)
        
        # ตั้งค่ากราฟ
        ax.set_xlim(maze_data['metadata']['boundaries']['min_x']-1, 
                    maze_data['metadata']['boundaries']['max_x']+1)
        ax.set_ylim(maze_data['metadata']['boundaries']['min_y']-1, 
                    maze_data['metadata']['boundaries']['max_y']+1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'แผนที่เขาวงกต - การสำรวจแบบเรียลไทม์\n' + 
                    f"ห้องที่ค้นพบ: {min(frame + 1, len(nodes_list))}/{len(nodes_list)} ห้อง", 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('พิกัด X', fontsize=12)
        ax.set_ylabel('พิกัด Y', fontsize=12)
        
        # Legend
        legend_elements = [
            patches.Patch(color='#ccffcc', alpha=0.7, label='ทางผ่าน'),
            patches.Patch(color='#ffcccc', alpha=0.7, label='ทางตัน'),
            patches.Patch(color='black', label='กำแพง'),
            plt.Line2D([0], [0], color='green', lw=3, label='ทางเดิน'),
            patches.Patch(color='red', alpha=0.3, label='ห้องที่กำลังสำรวจ')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # สร้างอนิเมชัน
    anim = animation.FuncAnimation(fig, animate, frames=len(nodes_list)+2, 
                                  interval=1500, repeat=True, blit=False)
    
    plt.tight_layout()
    return fig, anim

# โหลดข้อมูลแผนที่จากไฟล์
maze_data = load_maze_data()

# สร้างและแสดงกราฟภาพนิ่ง
print("\nสร้างกราฟแผนที่แบบภาพนิ่ง...")
static_fig, static_ax = create_static_maze_plot(maze_data)
plt.show()

# สร้างและแสดงกราฟแบบอนิเมชัน
print("\nสร้างกราฟแผนที่แบบอนิเมชัน...")
anim_fig, animation_obj = create_animated_maze_plot(maze_data)

# บันทึกอนิเมชันเป็นไฟล์ GIF (ถ้าต้องการ)
save_animation = input("\nต้องการบันทึกอนิเมชันเป็นไฟล์ GIF หรือไม่? (y/n): ").lower().strip()
if save_animation == 'y' or save_animation == 'yes':
    filename = input("ใส่ชื่อไฟล์ (ไม่ต้องใส่นามสกุล): ").strip()
    if not filename:
        filename = "maze_exploration"
    try:
        animation_obj.save(f'{filename}.gif', writer='pillow', fps=0.67)
        print(f"บันทึกอนิเมชันเป็นไฟล์ {filename}.gif แล้ว")
    except Exception as e:
        print(f"ไม่สามารถบันทึกไฟล์ได้: {e}")

plt.show()

print("\n=== ข้อมูลสรุปแผนที่ ===")
print(f"จำนวนห้องทั้งหมด: {maze_data['metadata']['total_nodes']} ห้อง")
print(f"ขอบเขตพิกัด: X({maze_data['metadata']['boundaries']['min_x']} ถึง {maze_data['metadata']['boundaries']['max_x']}), Y({maze_data['metadata']['boundaries']['min_y']} ถึง {maze_data['metadata']['boundaries']['max_y']})")

dead_ends = [node for node in maze_data['nodes'].values() if node['is_dead_end']]
print(f"จำนวนทางตัน: {len(dead_ends)} ห้อง")
print(f"จำนวนทางผ่าน: {maze_data['metadata']['total_nodes'] - len(dead_ends)} ห้อง")