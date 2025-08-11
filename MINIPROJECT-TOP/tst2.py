import time
import robomaster
from robomaster import robot, camera
from robomaster import vision
import numpy as np
from scipy.ndimage import median_filter
from datetime import datetime
import cv2

class GraphNode:
    def __init__(self, node_id, position):
        self.id = node_id
        self.position = position  # (x, y)
        
        # Wall detection
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
        self.visited = True  # Current node is visited when created
        self.visitCount = 1
        self.exploredDirections = []
        self.unexploredExits = []
        self.isDeadEnd = False
        
        # Additional info
        self.marker = False
        self.lastVisited = datetime.now().isoformat()
        self.sensorReadings = {}
        self.detected_marker_ids = []  # เก็บ id marker ที่เจอใน node นี้
        
    def to_dict(self):
        """Convert node to dictionary for display"""
        return {
            'id': self.id,
            'position': self.position,
            'walls': {
                'left': self.wallLeft,
                'right': self.wallRight, 
                'front': self.wallFront,
                'back': self.wallBack
            },
            'neighbors': self.neighbors,
            'exploredDirections': self.exploredDirections,
            'unexploredExits': self.unexploredExits,
            'isDeadEnd': self.isDeadEnd,
            'visitCount': self.visitCount,
            'marker': self.marker,
            'sensorReadings': self.sensorReadings,
            'detected_marker_ids': self.detected_marker_ids
        }

class GraphMapper:
    def __init__(self):
        self.nodes = {}  # node_id -> GraphNode
        self.currentPosition = (0, 0)  # Starting position
        self.currentDirection = 'north'  # Robot facing direction
        self.frontierQueue = []  # Nodes with unexplored exits
        self.pathStack = []  # For backtracking
        
    def get_node_id(self, position):
        """Generate unique node ID from position"""
        return f"{position[0]}_{position[1]}"
    
    def create_node(self, position):
        """Create new node at given position"""
        node_id = self.get_node_id(position)
        if node_id not in self.nodes:
            self.nodes[node_id] = GraphNode(node_id, position)
        return self.nodes[node_id]
    
    def get_current_node(self):
        """Get current node"""
        node_id = self.get_node_id(self.currentPosition)
        return self.nodes.get(node_id)
    
    def update_current_node_walls(self, left_wall, right_wall, front_wall):
        """Update wall information for current node"""
        current_node = self.get_current_node()
        if current_node:
            current_node.wallLeft = left_wall
            current_node.wallRight = right_wall
            current_node.wallFront = front_wall
            current_node.lastVisited = datetime.now().isoformat()
            current_node.visitCount += 1
            
            # Update unexplored exits based on walls
            self.update_unexplored_exits(current_node)
    
    def update_unexplored_exits(self, node):
        """Update unexplored exits based on wall detection"""
        node.unexploredExits = []
        
        # Check each direction relative to current facing direction
        direction_map = {
            'north': {'front': 'north', 'left': 'west', 'right': 'east', 'back': 'south'},
            'south': {'front': 'south', 'left': 'east', 'right': 'west', 'back': 'north'},
            'east': {'front': 'east', 'left': 'north', 'right': 'south', 'back': 'west'},
            'west': {'front': 'west', 'left': 'south', 'right': 'north', 'back': 'east'}
        }
        
        directions = direction_map[self.currentDirection]
        
        # Check for unexplored exits (no walls)
        if not node.wallFront:
            exit_direction = directions['front']
            if exit_direction not in node.exploredDirections:
                node.unexploredExits.append(exit_direction)
                
        if not node.wallLeft:
            exit_direction = directions['left']
            if exit_direction not in node.exploredDirections:
                node.unexploredExits.append(exit_direction)
                
        if not node.wallRight:
            exit_direction = directions['right']
            if exit_direction not in node.exploredDirections:
                node.unexploredExits.append(exit_direction)
        
        # Update frontier queue
        if node.unexploredExits and node.id not in self.frontierQueue:
            self.frontierQueue.append(node.id)
        elif not node.unexploredExits and node.id in self.frontierQueue:
            self.frontierQueue.remove(node.id)
            
        # Check if it's a dead end
        node.isDeadEnd = (node.wallFront and node.wallLeft and node.wallRight)
    
    def print_graph_summary(self):
        """Print current graph state"""
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
            print(f"   🧱 Walls: L:{node.wallLeft} R:{node.wallRight} F:{node.wallFront} B:{node.wallBack}")
            print(f"   🔍 Unexplored exits: {node.unexploredExits}")
            print(f"   ✅ Explored directions: {node.exploredDirections}")
            print(f"   🎯 Is dead end: {node.isDeadEnd}")
            print(f"   📊 Visit count: {node.visitCount}")
            print(f"   🔖 Marker detected: {node.marker}")
            
            if node.detected_marker_ids:
                print(f"   🆔 Marker IDs: {node.detected_marker_ids}")
            
            if node.sensorReadings:
                print(f"   📡 Sensor readings:")
                for direction, reading in node.sensorReadings.items():
                    print(f"      {direction}: {reading:.2f}cm")
        
        print("-"*60)
        if self.frontierQueue:
            print(f"🚀 Next exploration targets: {self.frontierQueue}")
        else:
            print("🎉 No more frontiers - exploration complete!")
        print("="*60)

class ToFSensorHandler:
    def __init__(self):
        # ค่า Calibration จาก Linear Regression
        self.CALIBRATION_SLOPE = 0.0894 
        self.CALIBRATION_Y_INTERCEPT = 3.8409
        
        # Median Filter settings
        self.WINDOW_SIZE = 5
        self.tof_buffer = []
        
        # Wall detection threshold (cm)
        self.WALL_THRESHOLD = 25.0  # 25cm
        
        # เก็บค่าเซ็นเซอร์สำหรับแต่ละตำแหน่ง
        self.readings = {
            'front': [],
            'left': [],
            'right': []
        }
        
        self.current_scan_direction = None
        self.collecting_data = False
        
    def calibrate_tof_value(self, raw_tof_mm):
        """แปลงค่า ToF ดิบ (mm) เป็นระยะทางที่สอบเทียบแล้ว (cm)"""
        calibrated_cm = (self.CALIBRATION_SLOPE * raw_tof_mm) + self.CALIBRATION_Y_INTERCEPT
        return calibrated_cm
    
    def apply_median_filter(self, data, window_size):
        """ใช้ Median Filter กับข้อมูล"""
        if len(data) == 0:
            return 0.0 
        if len(data) < window_size:
            return data[-1] 
        else:
            filtered = median_filter(data[-window_size:], size=window_size)
            return filtered[-1]
    
    def tof_data_handler(self, sub_info):
        """Callback สำหรับรับข้อมูลจากเซ็นเซอร์ ToF"""
        if not self.collecting_data or not self.current_scan_direction:
            return
            
        raw_tof_mm = sub_info[0]  # ค่าดิบจากเซ็นเซอร์ (mm)
        
        # Calibrate ค่า
        calibrated_tof_cm = self.calibrate_tof_value(raw_tof_mm)
        
        # เพิ่มข้อมูลลง buffer
        self.tof_buffer.append(calibrated_tof_cm)
        
        # ใช้ Median Filter
        filtered_tof_cm = self.apply_median_filter(self.tof_buffer, self.WINDOW_SIZE)
        
        # เก็บข้อมูลตามทิศทางที่สแกน
        if len(self.tof_buffer) <= 15:  # เก็บ 15 ค่าต่อทิศทาง
            self.readings[self.current_scan_direction].append({
                'raw_mm': raw_tof_mm,
                'calibrated_cm': calibrated_tof_cm,
                'filtered_cm': filtered_tof_cm,
                'timestamp': datetime.now().isoformat()
            })
        
        # แสดงผลแบบ real-time
        wall_status = "🧱 WALL" if filtered_tof_cm <= self.WALL_THRESHOLD else "🚪 OPEN"
        print(f"[{self.current_scan_direction.upper()}] {filtered_tof_cm:.2f}cm {wall_status}")
    
    def start_scanning(self, direction):
        """เริ่มสแกนทิศทางที่กำหนด"""
        self.current_scan_direction = direction
        self.tof_buffer.clear()
        if direction not in self.readings:
            self.readings[direction] = []
        else:
            self.readings[direction].clear()
        self.collecting_data = True
        
    def stop_scanning(self, unsub_distance_func):
        """หยุดสแกน"""
        self.collecting_data = False
        try:
            unsub_distance_func()
        except:
            pass
    
    def get_average_distance(self, direction):
        """คำนวณระยะทางเฉลี่ยสำหรับทิศทางที่กำหนด"""
        if direction not in self.readings or len(self.readings[direction]) == 0:
            return 0.0
            
        filtered_values = [reading['filtered_cm'] for reading in self.readings[direction]]
        return np.mean(filtered_values)
    
    def is_wall_detected(self, direction):
        """ตรวจสอบว่ามีกำแพงในทิศทางนั้นหรือไม่"""
        avg_distance = self.get_average_distance(direction)
        return avg_distance <= self.WALL_THRESHOLD and avg_distance > 0

class MarkerInfo:
    """ข้อมูล Marker ที่ตรวจพบ"""
    def __init__(self, x, y, w, h, marker_id):
        self._x = x
        self._y = y
        self._w = w
        self._h = h
        self._id = marker_id

    @property
    def id(self):
        return self._id

class MarkerVisionHandler:
    def __init__(self, graph_mapper):
        self.graph_mapper = graph_mapper
        self.markers = []  # เก็บ marker ที่ detect ล่าสุด
        self.marker_detected = False  # ตัวแปรสถานะการเจอ marker
        self.detection_count = 0  # นับจำนวนครั้งที่เจอ marker
    
    def on_detect_marker(self, marker_info):
        """Callback function สำหรับ marker detection (ปรับปรุงแล้ว)"""
        number = len(marker_info)
        self.markers.clear()
        
        if number > 0:
            self.marker_detected = True
            self.detection_count += 1
            
            # เก็บข้อมูล markers ทั้งหมด
            for i in range(number):
                x, y, w, h, marker_id = marker_info[i]
                self.markers.append(MarkerInfo(x, y, w, h, marker_id))
                print(f"🔖 Marker detected: ID={marker_id}, pos=({x:.3f},{y:.3f}), size=({w:.3f},{h:.3f})")
            
            # อัปเดต node ปัจจุบัน
            current_node = self.graph_mapper.get_current_node()
            if current_node:
                current_node.marker = True
                current_node.lastVisited = datetime.now().isoformat()
                current_node.detected_marker_ids = [m.id for m in self.markers]
                print(f"✅ Updated node {current_node.id} with {len(self.markers)} markers")
        else:
            # ไม่เจอ marker - ไม่ต้องเปลี่ยนสถานะ
            pass
    
    def draw_markers_on_image(self, img):
        """วาดกรอบและข้อมูล marker บนภาพ (ไม่ใช้แล้ว)"""
        return img
    
    def get_detection_summary(self):
        """ข้อมูลสรุปการตรวจจับ marker"""
        return {
            'detected': self.marker_detected,
            'count': len(self.markers),
            'total_detections': self.detection_count,
            'marker_ids': [m.id for m in self.markers] if self.markers else []
        }

def simple_marker_detection_sequence(vision, marker_handler, duration=10):
    """การตรวจจับ marker แบบง่าย - ไม่มี video stream"""
    print(f"\n🔖 === เริ่มการตรวจจับ Marker (ระยะเวลา {duration} วินาที) ===")
    
    # Subscribe marker detection
    print("🔍 เริ่ม marker detection...")
    result = vision.sub_detect_info(name="marker", callback=marker_handler.on_detect_marker)
    
    if not result:
        print("❌ ไม่สามารถเริ่ม marker detection ได้")
        return {'detected': False, 'count': 0, 'total_detections': 0, 'marker_ids': []}
    
    start_time = time.time()
    last_status_time = time.time()
    
    print("📡 กำลังตรวจจับ marker (ไม่แสดงภาพ)...")
    
    try:
        while time.time() - start_time < duration:
            # แสดงสถานะทุกๆ 2 วินาที
            if time.time() - last_status_time >= 2:
                summary = marker_handler.get_detection_summary()
                remaining_time = duration - (time.time() - start_time)
                print(f"📊 เวลาเหลือ: {remaining_time:.1f}s | "
                    f"Detected: {summary['detected']} | "
                    f"Markers: {summary['count']} | "
                    f"Total: {summary['total_detections']}")
                
                # หากเจอ marker แล้วให้แสดงรายละเอียด
                if summary['detected']:
                    print(f"   🆔 Marker IDs: {summary['marker_ids']}")
                
                last_status_time = time.time()
            
            # หยุดชั่วคราว
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("⚡ ผู้ใช้หยุดการตรวจจับ")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการตรวจจับ marker: {e}")
    
    finally:
        # ปิด marker detection
        print("🛑 หยุด marker detection...")
        try:
            vision.unsub_detect_info(name="marker")
        except:
            pass
        
        # แสดงผลสรุป
        summary = marker_handler.get_detection_summary()
        print(f"\n📊 สรุปผลการตรวจจับ:")
        print(f"   🔖 พบ marker: {summary['detected']}")
        print(f"   📊 จำนวน marker ปัจจุบัน: {summary['count']}")
        print(f"   🎯 จำนวนครั้งที่ตรวจพบ: {summary['total_detections']}")
        print(f"   🆔 Marker IDs: {summary['marker_ids']}")
        
        return summary

def graph_mapping_scan_sequence(gimbal, chassis, sensor, tof_handler, graph_mapper):
    """ลำดับการสแกนและสร้าง Graph Mapping"""
    print("\n🗺️  === เริ่มการสแกนและสร้าง Graph Mapping ===")
    
    # สร้าง node ปัจจุบัน
    current_node = graph_mapper.create_node(graph_mapper.currentPosition)
    
    # ล็อคล้อ
    print("🔒 ล็อคล้อทั้ง 4 ล้อ...")
    chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
    time.sleep(0.5)
    
    speed = 540
    scan_results = {}
    
    # 1. สแกนด้านหน้า (0°)
    print("\n1. 🔍 สแกนด้านหน้า (0°)...")
    gimbal.moveto(pitch=0, yaw=0, pitch_speed=speed, yaw_speed=speed).wait_for_completed()
    time.sleep(0.3)
    
    tof_handler.start_scanning('front')
    sensor.sub_distance(freq=25, callback=tof_handler.tof_data_handler)
    time.sleep(0.6)
    tof_handler.stop_scanning(sensor.unsub_distance)
    
    front_distance = tof_handler.get_average_distance('front')
    front_wall = tof_handler.is_wall_detected('front')
    scan_results['front'] = front_distance
    print(f"   📏 ระยะทางเฉลี่ย: {front_distance:.2f}cm {'🧱' if front_wall else '🚪'}")
    
    # 2. สแกนด้านซ้าย (90°)
    print("\n2. 🔍 สแกนด้านซ้าย (90°)...")
    gimbal.moveto(pitch=0, yaw=90, pitch_speed=speed, yaw_speed=speed).wait_for_completed()
    time.sleep(0.3)
    
    tof_handler.start_scanning('left')
    sensor.sub_distance(freq=25, callback=tof_handler.tof_data_handler)
    time.sleep(0.6)
    tof_handler.stop_scanning(sensor.unsub_distance)
    
    left_distance = tof_handler.get_average_distance('left')
    left_wall = tof_handler.is_wall_detected('left')
    scan_results['left'] = left_distance
    print(f"   📏 ระยะทางเฉลี่ย: {left_distance:.2f}cm {'🧱' if left_wall else '🚪'}")
    
    # 3. สแกนด้านขวา (-90°)
    print("\n3. 🔍 สแกนด้านขวา (-90°)...")
    gimbal.moveto(pitch=0, yaw=-90, pitch_speed=speed, yaw_speed=speed).wait_for_completed()
    time.sleep(0.3)
    
    tof_handler.start_scanning('right')
    sensor.sub_distance(freq=25, callback=tof_handler.tof_data_handler)
    time.sleep(0.6)
    tof_handler.stop_scanning(sensor.unsub_distance)
    
    right_distance = tof_handler.get_average_distance('right')
    right_wall = tof_handler.is_wall_detected('right')
    scan_results['right'] = right_distance
    print(f"   📏 ระยะทางเฉลี่ย: {right_distance:.2f}cm {'🧱' if right_wall else '🚪'}")
    
    # 4. กลับสู่ตำแหน่งเริ่มต้น
    print("\n4. 🔄 กลับสู่ตำแหน่งเริ่มต้น...")
    gimbal.moveto(pitch=0, yaw=0, pitch_speed=speed, yaw_speed=speed).wait_for_completed()
    time.sleep(0.3)
    
    # ปลดล็อคล้อ
    print("🔓 ปลดล็อคล้อ...")
    chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0, timeout=0.1)
    time.sleep(0.2)
    
    # อัปเดต Graph Node
    graph_mapper.update_current_node_walls(left_wall, right_wall, front_wall)
    current_node.sensorReadings = scan_results
    
    print(f"\n✅ เสร็จสิ้นการสแกน Node {current_node.id}")
    print(f"🧱 กำแพง: ซ้าย={left_wall}, ขวา={right_wall}, หน้า={front_wall}")
    
    return scan_results

if __name__ == '__main__':
    print("🤖 กำลังเชื่อมต่อหุ่นยนต์...")
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    
    ep_gimbal = ep_robot.gimbal
    ep_chassis = ep_robot.chassis
    ep_sensor = ep_robot.sensor
    ep_camera = ep_robot.camera
    ep_vision = ep_robot.vision
    
    tof_handler = ToFSensorHandler()
    graph_mapper = GraphMapper()
    marker_handler = MarkerVisionHandler(graph_mapper)
    
    try:
        print("✅ Recalibrating gimbal...")
        ep_gimbal.recenter(pitch_speed=100, yaw_speed=100).wait_for_completed()
        print("✅ Gimbal recalibrated.")
        
        print("🔄 รีเซ็ตตำแหน่งเริ่มต้น...")
        ep_gimbal.moveto(pitch=0, yaw=0, pitch_speed=50, yaw_speed=50).wait_for_completed()
        time.sleep(1)
        
        print(f"🎯 Wall Detection Threshold: {tof_handler.WALL_THRESHOLD}cm")
        print(f"🎯 ใช้ Calibration: slope={tof_handler.CALIBRATION_SLOPE}, intercept={tof_handler.CALIBRATION_Y_INTERCEPT}")
        
        # เริ่มสแกน Map Node ปัจจุบัน
        scan_results = graph_mapping_scan_sequence(ep_gimbal, ep_chassis, ep_sensor, tof_handler, graph_mapper)
        
        # เริ่นระบบตรวจจับ marker แบบง่าย
        marker_summary = simple_marker_detection_sequence(ep_vision, marker_handler, duration=10)
        
        # แสดงผลสรุป Graph
        graph_mapper.print_graph_summary()
        
        # แสดงสรุป marker detection
        print(f"\n🔖 ===== สรุปผลการตรวจจับ MARKER =====")
        print(f"🔍 พบ marker: {marker_summary['detected']}")
        print(f"📊 จำนวน marker: {marker_summary['count']}")
        print(f"🎯 ตรวจพบทั้งหมด: {marker_summary['total_detections']} ครั้ง")
        print(f"🆔 Marker IDs: {marker_summary['marker_ids']}")
    
    finally:
        print("\n🛑 ปิดการทำงานและเชื่อมต่อหุ่นยนต์...")
        try:
            ep_vision.unsub_detect_info(name="marker")
        except:
            pass
        ep_robot.close()