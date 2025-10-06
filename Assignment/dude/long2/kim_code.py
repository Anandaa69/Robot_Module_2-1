# ระบบตรวจจับและยิงวัตถุสำหรับ RoboMaster
# แยกออกมาจาก walifwpare.py เพื่อให้ใช้งานได้อย่างอิสระ

import cv2
import numpy as np
import time
import threading
from datetime import datetime
from robomaster import blaster, gimbal, camera
import csv

#############################
# Object Detection Constants
#############################

# --- ค่าคงที่สำหรับการคำนวณระยะทาง ---
WIDTH_SQUARE = 7.0
DIAMETER_CIRCLE = 7.0
WIDTH_RECT_H = 9.0
HEIGHT_RECT_H = 6.0
WIDTH_RECT_V = 6.0
HEIGHT_RECT_V = 9.0

# ระยะโฟกัสของกล้องที่ได้จากการคาลิเบรต (หน่วย: pixels)
FOCAL_LENGTH_PIXELS = 600

# กำหนดค่า Tolerance สำหรับการ "เจอตรงกลาง" (หน่วย: pixels)
CENTER_TOLERANCE = 20

# --- ค่าคงที่สำหรับควบคุม Gimbal ---
YAW_KP = 0.12
PITCH_KP = 0.15
MAX_YAW_SPEED = 60
MAX_PITCH_SPEED = 60

# ตั้งค่าสำหรับการยิงน้ำ
FIRE_COOLDOWN = 2.0  # ระยะเวลาหน่วงระหว่างการยิง (วินาที)

# ตัวแปรสำหรับการกรองเพิ่มเติม
MIN_AREA = 1500
MIN_WIDTH = 40
MIN_HEIGHT = 40
MAX_ASPECT_RATIO = 4.0
MIN_ASPECT_RATIO = 0.25

# --- กำหนดช่วงสีและสีของกรอบ (HSV + BGR สำหรับ bounding box) ---
color_ranges = {
    "red":            ([0, 120, 70],   [10, 255, 255],   (0, 0, 255)),
    "dark_green":     ([70, 150, 20],   [90, 255, 70],    (0, 255, 0)),
    "yellow_brown":   ([15, 150, 80],   [30, 255, 180],   (0, 255, 255)),
    "dark_blue":      ([110, 150, 30],  [130, 255, 100],  (255, 0, 0)),
}

# ช่วงสีแดงเพิ่มเติม (สำหรับ HSV wraparound)
red_ranges_extended = [
    ([0, 120, 70], [10, 255, 255]),      # แดงส้ม
    ([170, 120, 70], [180, 255, 255]),   # แดงม่วง
    ([0, 80, 100], [15, 255, 255]),      # แดงอ่อน
    ([165, 80, 100], [180, 255, 255])    # แดงเข้ม
]

# ตัวแปรสำหรับติดตามวัตถุที่ถูกยิง
shot_objects = {}  # เก็บข้อมูลวัตถุที่ถูกยิง {(x, y): [{'color': str, 'shape': str, 'timestamp': datetime, 'distance': float}]}
shot_objects_lock = threading.Lock()  # ป้องกันการเข้าถึงข้อมูลพร้อมกัน

# ตัวแปรสำหรับ object detection
object_detection_enabled = True
current_frame = None
frame_lock = threading.Lock()

# ตัวแปรสำหรับ Color Range Adjustment Window
current_color_index = 0  # 0: red, 1: dark_green, 2: yellow_brown, 3: dark_blue
color_names = ["red", "dark_green", "yellow_brown", "dark_blue"]

# ตัวแปรสำหรับ Mask Display Window
show_mask = True  # เปิด/ปิดการแสดง mask
enable_equalization = True  # เปิด/ปิดการปรับความสว่าง
show_detection_window = True  # เปิด/ปิดหน้าต่างแสดงผลการตรวจจับ

# ตัวแปรสำหรับ Detection Display Thread
detection_thread_running = False

#############################
# Object Detection Functions
#############################

def create_red_mask(hsv_image):
    """สร้าง mask สำหรับสีแดงที่ปรับปรุงแล้ว รองรับ HSV wraparound และปรับตามสภาพแสง"""
    # วิเคราะห์ความสว่างเฉลี่ยของภาพ
    avg_brightness = np.mean(hsv_image[:,:,2])
    
    # เลือกช่วงสีแดงที่เหมาะสมตามสภาพแสง
    red_ranges = adaptive_red_ranges(avg_brightness)
    
    # สร้าง mask รวมจากหลายช่วงสีแดง
    combined_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
    
    for lower, upper in red_ranges:
        mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # ปรับปรุง mask ด้วย morphological operations ตามสภาพแสง
    if avg_brightness < 100:  # แสงน้อย - ใช้ kernel ใหญ่ขึ้น
        kernel = np.ones((4, 4), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)
    elif avg_brightness > 180:  # แสงมาก - ใช้ kernel เล็กลง
        kernel = np.ones((2, 2), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
    else:  # แสงปกติ
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
    
    return combined_mask

def enhance_red_detection(roi_frame):
    """ปรับปรุงภาพเพื่อการตรวจจับสีแดงที่ดีขึ้น"""
    # แปลงเป็น HSV
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    
    # วิเคราะห์ความสว่างเฉลี่ยของภาพ
    avg_brightness = np.mean(hsv[:,:,2])
    
    # ปรับปรุงความอิ่มตัวของสี (Saturation) เพื่อให้สีแดงชัดขึ้น
    if avg_brightness < 100:  # สภาพแสงน้อย
        hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.4)  # เพิ่ม saturation มากขึ้น
        hsv[:,:,2] = cv2.add(hsv[:,:,2], 20)  # เพิ่มความสว่างมากขึ้น
    elif avg_brightness > 200:  # สภาพแสงมาก
        hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.1)  # เพิ่ม saturation น้อย
        hsv[:,:,2] = cv2.subtract(hsv[:,:,2], 5)  # ลดความสว่างเล็กน้อย
    else:  # สภาพแสงปกติ
        hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.2)  # เพิ่ม saturation ปกติ
        hsv[:,:,2] = cv2.add(hsv[:,:,2], 10)  # เพิ่มความสว่างปกติ
    
    # จำกัดค่าให้อยู่ในช่วงที่ถูกต้อง
    hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
    hsv[:,:,2] = np.clip(hsv[:,:,2], 0, 255)
    
    return hsv

def adaptive_red_ranges(avg_brightness):
    """ปรับช่วงสีแดงตามสภาพแสง"""
    if avg_brightness < 80:  # แสงน้อยมาก
        return [
            ([0, 100, 30], [15, 255, 255]),      # ขยายช่วง H และลด S minimum
            ([165, 100, 30], [180, 255, 255]),
            ([0, 60, 50], [20, 255, 255]),       # ช่วงกว้างสำหรับแสงน้อย
            ([160, 60, 50], [180, 255, 255])
        ]
    elif avg_brightness > 180:  # แสงมากมาก
        return [
            ([0, 140, 100], [8, 255, 255]),      # จำกัดช่วง H และเพิ่ม S minimum
            ([172, 140, 100], [180, 255, 255]),
            ([0, 120, 120], [12, 255, 255]),     # ช่วงแคบสำหรับแสงมาก
            ([168, 120, 120], [180, 255, 255])
        ]
    else:  # แสงปกติ
        return red_ranges_extended

def detect_shape(contour):
    """ปรับปรุงการตรวจสอบรูปร่างให้เข้มงวดขึ้น"""
    shape = "Unknown"
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    
    # ตรวจสอบขนาดขั้นต่ำ
    if area < MIN_AREA:
        return "Unknown"
    
    # คำนวณ Solidity อีกครั้งเพื่อความแน่นอน
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0:
        solidity = float(area) / hull_area
        if solidity < 0.85:  # เพิ่มความเข้มงวด
            return "Unknown"
    
    # คำนวณ Extent (อัตราส่วนของพื้นที่ต่อพื้นที่ bounding box)
    x, y, w, h = cv2.boundingRect(contour)
    bounding_area = w * h
    if bounding_area > 0:
        extent = float(area) / bounding_area
        if extent < 0.7:  # วัตถุต้องเติมพื้นที่อย่างน้อย 70% ของ bounding box
            return "Unknown"
    
    # ตรวจสอบขนาดขั้นต่ำของ bounding box
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        return "Unknown"
    
    # ตรวจสอบสัดส่วน (ไม่ให้ยาวเกินไป)
    aspect_ratio = float(w) / h
    if aspect_ratio > MAX_ASPECT_RATIO or aspect_ratio < MIN_ASPECT_RATIO:
        return "Unknown"
    
    # คำนวณ Circularity สำหรับวงกลม
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    # ตรวจสอบรูปร่าง
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    num_vertices = len(approx)

    if num_vertices == 4:
        # ตรวจสอบว่ามุมใกล้เคียง 90 องศา (เป็นสี่เหลี่ยม)
        angles = []
        for i in range(4):
            p1 = approx[i][0]
            p2 = approx[(i+1)%4][0]
            p3 = approx[(i+2)%4][0]
            
            # คำนวณมุม
            v1 = p1 - p2
            v2 = p3 - p2
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle) * 180 / np.pi
            angles.append(angle)
        
        # ตรวจสอบว่ามุมใกล้เคียง 90 องศา (80-100 องศา)
        valid_angles = sum(1 for a in angles if 80 <= a <= 100)
        if valid_angles < 3:  # ต้องมีมุมที่ถูกต้องอย่างน้อย 3 มุม
            return "Unknown"
        
        # ตรวจสอบ aspect ratio
        if 0.9 <= aspect_ratio <= 1.1:
            shape = "Square"
        elif aspect_ratio > 1.2:  # เพิ่มความเข้มงวด
            shape = "Rectangle-H"
        elif aspect_ratio < 0.8:  # เพิ่มความเข้มงวด
            shape = "Rectangle-V"
        else:
            return "Unknown"
            
    elif num_vertices > 6 and circularity > 0.8:  # ตรวจสอบ circularity
        shape = "Circle"
        
    return shape

def process_frame_for_objects(frame):
    """ประมวลผลภาพเพื่อตรวจจับวัตถุและส่งคืนข้อมูลเป้าหมาย"""
    if frame is None:
        return None, False
    
    frame_height, frame_width, _ = frame.shape

    roi_w = 800
    roi_h = 450
    roi_x = (frame_width - roi_w) // 2
    roi_y = (frame_height - roi_h) // 2
    
    roi_center_x = roi_w // 2
    roi_center_y = roi_h // 2

    roi_frame = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    
    # ปรับความสว่างของภาพเพื่อความเสถียรในการ detect สี
    lab = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(lab[:,:,0])
    roi_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # ใช้ฟังก์ชันปรับปรุงสำหรับการตรวจจับสีแดง
    hsv = enhance_red_detection(roi_frame)
    
    # ตัวแปรสำหรับเก็บข้อมูลเป้าหมายที่ใหญ่ที่สุดในเฟรมนี้
    biggest_contour_area = 0
    target_x_offset = 0
    target_y_offset = 0
    found_target_in_frame = False
    target_info = None

    for color_name, (lower, upper, box_color) in color_ranges.items():
        # ใช้ฟังก์ชันพิเศษสำหรับสีแดง
        if color_name == "red":
            mask = create_red_mask(hsv)
        else:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            # เพิ่ม Morphology Operations เพื่อปรับปรุง mask สำหรับสีอื่น
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=1)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > MIN_AREA:
                # ตรวจสอบขนาดและสัดส่วนของ bounding box
                x, y, w, h = cv2.boundingRect(cnt)
                
                # ตรวจสอบขนาดขั้นต่ำ
                if w < MIN_WIDTH or h < MIN_HEIGHT:
                    continue
                
                # ตรวจสอบสัดส่วน
                aspect_ratio = float(w) / h
                if aspect_ratio > MAX_ASPECT_RATIO or aspect_ratio < MIN_ASPECT_RATIO:
                    continue
                
                # คำนวณ Solidity เพื่อกรอง noise หรือ contour ที่ผิดปกติ
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = float(area) / hull_area
                    if solidity < 0.85:
                        continue

                shape_name = detect_shape(cnt)
                
                if shape_name != "Unknown":
                    distance_cm = 0
                    new_x, new_y = 0, 0
                    
                    if shape_name == "Circle":
                        (x_c, y_c), radius = cv2.minEnclosingCircle(cnt)
                        new_x = int(x_c - roi_center_x)
                        new_y = int(roi_center_y - y_c)
                        
                        if radius > 0:
                            pixel_diameter = 2 * radius
                            distance_cm = (DIAMETER_CIRCLE * FOCAL_LENGTH_PIXELS) / pixel_diameter
                    
                    else: # Square or Rectangle
                        x, y, w, h = cv2.boundingRect(cnt)
                        obj_center_x = x + w // 2
                        obj_center_y = y + h // 2
                        new_x = int(obj_center_x - roi_center_x)
                        new_y = int(roi_center_y - obj_center_y)
                        
                        if shape_name == "Square":
                            distance_cm = (WIDTH_SQUARE * FOCAL_LENGTH_PIXELS) / h
                        elif shape_name == "Rectangle-V":
                            distance_cm = (HEIGHT_RECT_V * FOCAL_LENGTH_PIXELS) / h
                        elif shape_name == "Rectangle-H":
                            distance_cm = (WIDTH_RECT_H * FOCAL_LENGTH_PIXELS) / w

                    # ตรวจสอบว่าเป็น contour ที่ใหญ่ที่สุดที่เจอในเฟรมนี้หรือไม่
                    if area > biggest_contour_area:
                        biggest_contour_area = area
                        target_x_offset = new_x
                        target_y_offset = new_y
                        found_target_in_frame = True
                        target_info = {
                            'color': color_name,
                            'shape': shape_name,
                            'distance': distance_cm,
                            'x_offset': new_x,
                            'y_offset': new_y
                        }

                    # ส่วนตรวจสอบและคืนค่าเมื่อเจอตรงกลาง
                    if abs(new_x) <= CENTER_TOLERANCE and abs(new_y) <= CENTER_TOLERANCE:
                        print(f"เจอตรงกลางของ {color_name} {shape_name}! | พิกัด: ({new_x}, {new_y}) | ระยะทาง: {distance_cm:.1f} cm")
                        return target_info, True

    if found_target_in_frame:
        return target_info, False
    
    return None, False

#############################
# Shooting System Class
#############################

class ShootingSystem:
    """คลาสสำหรับจัดการระบบการยิงและตรวจจับวัตถุ"""
    
    def __init__(self, ep_robot):
        """
        เริ่มต้นระบบการยิง
        
        Args:
            ep_robot: RoboMaster robot instance
        """
        self.ep_robot = ep_robot
        self.ep_gimbal = ep_robot.gimbal
        self.ep_blaster = ep_robot.blaster
        self.ep_camera = ep_robot.camera
        self.last_fire_time = 0
        
        # เริ่มต้น blaster
        self.ep_blaster.set_fire_count(300)
        
    def fire_at_target(self, target_info, robot_position=None):
        """
        ยิงเป้าหมายที่ระบุ
        
        Args:
            target_info: ข้อมูลเป้าหมาย
            robot_position: ตำแหน่งของหุ่นยนต์ (สำหรับบันทึก)
        
        Returns:
            bool: True ถ้ายิงสำเร็จ, False ถ้าไม่สำเร็จ
        """
        current_time = time.time()
        if current_time - self.last_fire_time >= FIRE_COOLDOWN:
            print("🔥 ยิงเป้าหมาย!")
            
            # ปรับ gimbal ขึ้นเล็กน้อยก่อนยิง
            self.ep_gimbal.move(pitch=8, pitch_speed=40).wait_for_completed()
            time.sleep(0.2)
            
            # ยิง
            self.ep_blaster.fire(fire_type=blaster.WATER_FIRE, times=70)
            self.last_fire_time = current_time
            
            # บันทึกวัตถุที่ถูกยิง
            if robot_position is not None:
                record_shot_object(target_info, robot_position)
            
            # กลับลงมาหลังยิง
            self.ep_gimbal.move(pitch=-8, pitch_speed=30).wait_for_completed()
            
            return True
        else:
            print(f"⏳ รอ cooldown อีก {FIRE_COOLDOWN - (current_time - self.last_fire_time):.1f} วินาที")
            return False
    
    def track_and_shoot_target(self, target_info, max_tracking_time=10.0):
        """
        ติดตามและยิงเป้าหมาย
        
        Args:
            target_info: ข้อมูลเป้าหมายเริ่มต้น
            max_tracking_time: เวลาสูงสุดในการติดตาม (วินาที)
        
        Returns:
            bool: True ถ้ายิงสำเร็จ, False ถ้าไม่สำเร็จ
        """
        print(f"🎯 เริ่มติดตามเป้าหมาย: {target_info['color']} {target_info['shape']}")
        
        tracking_start_time = time.time()
        
        while time.time() - tracking_start_time < max_tracking_time:
            try:
                img = self.ep_camera.read_cv2_image(strategy="newest", timeout=0.1)
                if img is not None:
                    current_target, is_centered = process_frame_for_objects(img)
                    
                    if current_target:
                        if is_centered:
                            # วัตถุอยู่ตรงกลาง - ยิง!
                            if self.fire_at_target(current_target):
                                print("✅ การยิงเสร็จสิ้น")
                                return True
                        else:
                            # ปรับ gimbal ไปที่เป้าหมาย
                            yaw_speed = YAW_KP * current_target['x_offset']
                            pitch_speed = PITCH_KP * current_target['y_offset']
                            
                            # จำกัดความเร็วสูงสุด
                            yaw_speed = np.clip(yaw_speed, -MAX_YAW_SPEED, MAX_YAW_SPEED)
                            pitch_speed = np.clip(pitch_speed, -MAX_PITCH_SPEED, MAX_PITCH_SPEED)
                            
                            self.ep_gimbal.drive_speed(yaw_speed=yaw_speed, pitch_speed=pitch_speed)
                            time.sleep(0.05)
                    else:
                        # หาเป้าหมายไม่เจอ ให้หยุดติดตาม
                        print("⚠️ สูญเสียเป้าหมาย หยุดการติดตาม")
                        self.ep_gimbal.drive_speed(yaw_speed=0, pitch_speed=0)
                        break
                        
            except Exception as e:
                print(f"⚠️ ข้อผิดพลาดในการติดตามเป้าหมาย: {e}")
                break
        
        # หยุด gimbal
        self.ep_gimbal.drive_speed(yaw_speed=0, pitch_speed=0)
        print("⏰ หมดเวลาติดตาม")
        return False
    
    def scan_and_shoot_in_direction(self, yaw_angle, scan_time=3.0, robot_position=None):
        """
        สแกนและยิงวัตถุในทิศทางที่กำหนด
        
        Args:
            yaw_angle: มุม yaw ที่จะสแกน
            scan_time: เวลาในการสแกน (วินาที)
            robot_position: ตำแหน่งของหุ่นยนต์
        
        Returns:
            bool: True ถ้าพบและยิงวัตถุ, False ถ้าไม่พบ
        """
        print(f"🔍 สแกนหาวัตถุในทิศทาง {yaw_angle} องศา")
        
        # เลื่อน gimbal ไปยังทิศทางที่กำหนด
        self.ep_gimbal.moveto(pitch=0, yaw=yaw_angle, pitch_speed=200, yaw_speed=200).wait_for_completed()
        time.sleep(0.3)
        
        scan_start_time = time.time()
        object_detected = False
        target_acquired = False
        
        while time.time() - scan_start_time < scan_time:
            try:
                img = self.ep_camera.read_cv2_image(strategy="newest", timeout=0.1)
                if img is not None:
                    target_info, is_centered = process_frame_for_objects(img)
                    
                    if target_info:
                        print(f"🎯 พบวัตถุ: {target_info['color']} {target_info['shape']} ที่ระยะ {target_info['distance']:.1f}cm")
                        target_acquired = True
                        
                        if is_centered:
                            # วัตถุอยู่ตรงกลาง - ยิง!
                            if self.fire_at_target(target_info, robot_position):
                                object_detected = True
                                break
                        else:
                            # ปรับ gimbal ไปที่เป้าหมาย
                            yaw_speed = YAW_KP * target_info['x_offset']
                            pitch_speed = PITCH_KP * target_info['y_offset']
                            
                            # จำกัดความเร็วสูงสุด
                            yaw_speed = np.clip(yaw_speed, -MAX_YAW_SPEED, MAX_YAW_SPEED)
                            pitch_speed = np.clip(pitch_speed, -MAX_PITCH_SPEED, MAX_PITCH_SPEED)
                            
                            self.ep_gimbal.drive_speed(yaw_speed=yaw_speed, pitch_speed=pitch_speed)
                            time.sleep(0.05)
                    else:
                        # ไม่เจอวัตถุ - หยุด gimbal
                        self.ep_gimbal.drive_speed(yaw_speed=0, pitch_speed=0)
                        time.sleep(0.1)
                        
            except Exception as e:
                print(f"⚠️ ข้อผิดพลาดในการสแกน: {e}")
                break
        
        # ถ้าพบเป้าหมายแต่ยังไม่ได้ยิง ให้ต่อเนื่องจนกว่าจะยิงเสร็จ
        if target_acquired and not object_detected:
            print("🎯 ต่อเนื่องการติดตามเป้าหมาย...")
            object_detected = self.track_and_shoot_target(target_info, max_tracking_time=8.0)
        
        # หยุด gimbal และกลับไปตำแหน่งสแกน
        self.ep_gimbal.drive_speed(yaw_speed=0, pitch_speed=0)
        if object_detected:
            # ถ้าเพิ่งยิงเสร็จ ให้กลับไปตำแหน่งสแกน
            self.ep_gimbal.moveto(pitch=0, yaw=yaw_angle, pitch_speed=200, yaw_speed=200).wait_for_completed()
        
        return object_detected
    
    def scan_low_objects(self, robot_position=None):
        """
        สแกนหาวัตถุที่อยู่ต่ำกว่าพื้นที่ ROI ปกติ
        
        Args:
            robot_position: ตำแหน่งของหุ่นยนต์
        
        Returns:
            bool: True ถ้าพบและยิงวัตถุ, False ถ้าไม่พบ
        """
        print(f"\n🔍 Scanning for low objects...")
        
        # ปรับ gimbal ลงเล็กน้อยเพื่อดูพื้นที่ต่ำกว่า
        original_pitch = 0
        scan_pitch = -15  # ลงมา 15 องศา
        
        try:
            self.ep_gimbal.moveto(pitch=scan_pitch, yaw=0, pitch_speed=100, yaw_speed=100).wait_for_completed()
            time.sleep(0.3)
            
            # สแกนหาวัตถุในมุมต่ำ
            scan_start_time = time.time()
            object_detected = False
            target_acquired = False
            
            while time.time() - scan_start_time < 2.0:  # สแกนเป็นเวลา 2 วินาที
                try:
                    img = self.ep_camera.read_cv2_image(strategy="newest", timeout=0.1)
                    if img is not None:
                        target_info, is_centered = process_frame_for_objects(img)
                        
                        if target_info:
                            print(f"🎯 พบวัตถุในมุมต่ำ: {target_info['color']} {target_info['shape']} ที่ระยะ {target_info['distance']:.1f}cm")
                            target_acquired = True
                            
                            if is_centered:
                                # วัตถุอยู่ตรงกลาง - ยิง!
                                if self.fire_at_target(target_info, robot_position):
                                    object_detected = True
                                    break
                            else:
                                # ปรับ gimbal ไปที่เป้าหมาย
                                yaw_speed = YAW_KP * target_info['x_offset']
                                pitch_speed = PITCH_KP * target_info['y_offset']
                                
                                # จำกัดความเร็วสูงสุด
                                yaw_speed = np.clip(yaw_speed, -MAX_YAW_SPEED, MAX_YAW_SPEED)
                                pitch_speed = np.clip(pitch_speed, -MAX_PITCH_SPEED, MAX_PITCH_SPEED)
                                
                                self.ep_gimbal.drive_speed(yaw_speed=yaw_speed, pitch_speed=pitch_speed)
                                time.sleep(0.05)
                        else:
                            # ไม่เจอวัตถุ - หยุด gimbal
                            self.ep_gimbal.drive_speed(yaw_speed=0, pitch_speed=0)
                            time.sleep(0.1)
                            
                except Exception as e:
                    print(f"⚠️ ข้อผิดพลาดในการสแกนวัตถุในมุมต่ำ: {e}")
                    break
            
            # ถ้าพบเป้าหมายแต่ยังไม่ได้ยิง ให้ต่อเนื่องจนกว่าจะยิงเสร็จ
            if target_acquired and not object_detected:
                print("🎯 ต่อเนื่องการติดตามเป้าหมายในมุมต่ำ...")
                object_detected = self.track_and_shoot_target(target_info, max_tracking_time=8.0)
            
            # หยุด gimbal และกลับไปตำแหน่งปกติ
            self.ep_gimbal.drive_speed(yaw_speed=0, pitch_speed=0)
            self.ep_gimbal.moveto(pitch=original_pitch, yaw=0, pitch_speed=100, yaw_speed=100).wait_for_completed()
            
            if object_detected:
                print("✅ พบและยิงวัตถุในมุมต่ำเสร็จสิ้น")
            else:
                print("ℹ️ ไม่พบวัตถุในมุมต่ำ")
                
            return object_detected
                
        except Exception as e:
            print(f"⚠️ ข้อผิดพลาดในการสแกนมุมต่ำ: {e}")
            # กลับไปตำแหน่งปกติในกรณีเกิดข้อผิดพลาด
            try:
                self.ep_gimbal.moveto(pitch=original_pitch, yaw=0, pitch_speed=100, yaw_speed=100).wait_for_completed()
            except:
                pass
            return False

#############################
# Shot Object Tracking Functions
#############################

def record_shot_object(target_info, robot_position):
    """บันทึกข้อมูลวัตถุที่ถูกยิง"""
    global shot_objects
    
    if target_info is None:
        return
    
    cell_pos = tuple(robot_position)
    shot_data = {
        'color': target_info['color'],
        'shape': target_info['shape'],
        'distance': target_info['distance'],
        'timestamp': datetime.now(),
        'x_offset': target_info.get('x_offset', 0),
        'y_offset': target_info.get('y_offset', 0)
    }
    
    with shot_objects_lock:
        if cell_pos not in shot_objects:
            shot_objects[cell_pos] = []
        shot_objects[cell_pos].append(shot_data)
    
    print(f"📝 บันทึกวัตถุที่ถูกยิง: {target_info['color']} {target_info['shape']} ที่เซลล์ {cell_pos}")

def get_shot_objects_in_cell(cell_pos):
    """ดึงข้อมูลวัตถุที่ถูกยิงในเซลล์ที่กำหนด"""
    with shot_objects_lock:
        return shot_objects.get(tuple(cell_pos), [])

def get_all_shot_objects():
    """ดึงข้อมูลวัตถุที่ถูกยิงทั้งหมด"""
    with shot_objects_lock:
        return dict(shot_objects)

def save_shot_objects(filename="shot_objects.csv"):
    """บันทึกข้อมูลวัตถุที่ถูกยิงลง CSV"""
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['cell_x', 'cell_y', 'color', 'shape', 'distance', 'timestamp', 'x_offset', 'y_offset'])
            
            with shot_objects_lock:
                for (cell_x, cell_y), objects in shot_objects.items():
                    for obj in objects:
                        writer.writerow([
                            cell_x, cell_y,
                            obj['color'], obj['shape'],
                            obj['distance'], obj['timestamp'],
                            obj['x_offset'], obj['y_offset']
                        ])
        
        print(f"💾 บันทึกข้อมูลวัตถุที่ถูกยิงลงไฟล์ {filename}")
        
    except Exception as e:
        print(f"⚠️ ข้อผิดพลาดในการบันทึกไฟล์: {e}")

#############################
# Utility Functions
#############################

def create_trackbar_window():
    """สร้างหน้าต่าง trackbar สำหรับปรับค่าสี"""
    cv2.namedWindow('Color Range Adjustment', cv2.WINDOW_NORMAL)
    
    # สร้าง trackbars สำหรับ H, S, V (lower และ upper)
    cv2.createTrackbar('H Lower', 'Color Range Adjustment', 0, 179, lambda x: None)
    cv2.createTrackbar('S Lower', 'Color Range Adjustment', 0, 255, lambda x: None)
    cv2.createTrackbar('V Lower', 'Color Range Adjustment', 0, 255, lambda x: None)
    cv2.createTrackbar('H Upper', 'Color Range Adjustment', 179, 179, lambda x: None)
    cv2.createTrackbar('S Upper', 'Color Range Adjustment', 255, 255, lambda x: None)
    cv2.createTrackbar('V Upper', 'Color Range Adjustment', 255, 255, lambda x: None)
    
    # ตั้งค่าเริ่มต้นจากสีปัจจุบัน
    update_trackbars_from_current_color()

def update_trackbars_from_current_color():
    """อัปเดต trackbar ให้ตรงกับสีปัจจุบัน"""
    global current_color_index
    current_color = color_names[current_color_index]
    lower, upper, _ = color_ranges[current_color]
    
    cv2.setTrackbarPos('H Lower', 'Color Range Adjustment', lower[0])
    cv2.setTrackbarPos('S Lower', 'Color Range Adjustment', lower[1])
    cv2.setTrackbarPos('V Lower', 'Color Range Adjustment', lower[2])
    cv2.setTrackbarPos('H Upper', 'Color Range Adjustment', upper[0])
    cv2.setTrackbarPos('S Upper', 'Color Range Adjustment', upper[1])
    cv2.setTrackbarPos('V Upper', 'Color Range Adjustment', upper[2])

def update_color_range():
    """อัปเดตช่วงสีจาก trackbar"""
    global current_color_index, color_ranges
    
    try:
        h_lower = cv2.getTrackbarPos('H Lower', 'Color Range Adjustment')
        s_lower = cv2.getTrackbarPos('S Lower', 'Color Range Adjustment')
        v_lower = cv2.getTrackbarPos('V Lower', 'Color Range Adjustment')
        h_upper = cv2.getTrackbarPos('H Upper', 'Color Range Adjustment')
        s_upper = cv2.getTrackbarPos('S Upper', 'Color Range Adjustment')
        v_upper = cv2.getTrackbarPos('V Upper', 'Color Range Adjustment')
        
        current_color = color_names[current_color_index]
        lower = [h_lower, s_lower, v_lower]
        upper = [h_upper, s_upper, v_upper]
        box_color = color_ranges[current_color][2]  # เก็บสีของกรอบเดิม
        
        color_ranges[current_color] = (lower, upper, box_color)
        
    except:
        pass  # ถ้า trackbar ยังไม่ได้สร้าง ให้ข้ามไป

def print_shot_objects_summary():
    """แสดงสรุปวัตถุที่ถูกยิงทั้งหมด"""
    print("\n" + "="*60)
    print("📊 สรุปวัตถุที่ถูกยิงทั้งหมด")
    print("="*60)
    
    with shot_objects_lock:
        if not shot_objects:
            print("   ไม่มีวัตถุที่ถูกยิง")
            return
        
        total_objects = 0
        for cell_pos, objects in shot_objects.items():
            total_objects += len(objects)
            print(f"\n📍 เซลล์ {cell_pos}:")
            print(f"   • จำนวนวัตถุที่ถูกยิง: {len(objects)}")
            print(f"   • รายละเอียดวัตถุในแต่ละเซลล์:")
            
            for i, obj in enumerate(objects, 1):
                print(f"     {i}. {obj['color']} {obj['shape']} - ระยะ {obj['distance']:.1f}cm")
                print(f"        เวลา: {obj['timestamp'].strftime('%H:%M:%S')}")
                print(f"        ตำแหน่ง: ({obj['x_offset']}, {obj['y_offset']})")
        
        print(f"\n🎯 รวมวัตถุที่ถูกยิงทั้งหมด: {total_objects} วัตถุ")
        print("="*60)

if __name__ == "__main__":
    print("🎯 Shooting System Module")
    print("ใช้โมดูลนี้โดยการ import ในไฟล์หลัก")
    print("ตัวอย่าง:")
    print("from shooting_system import ShootingSystem")
    print("shooting_system = ShootingSystem(ep_robot)")