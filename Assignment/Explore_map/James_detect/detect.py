import cv2
import numpy as np
import sys
import robomaster
from robomaster import robot
import time
from scipy.spatial import distance as dist # ใช้คำนวณระยะห่างระหว่างจุด
from collections import OrderedDict

# ======================================================================
# คลาส ObjectDetector ที่อัปเกรดเป็น ObjectTracker
# ======================================================================
class ObjectTracker:
    def __init__(self, template_paths, max_disappeared=30):
        print("🖼️  กำลังโหลดและประมวลผลภาพ Templates...")
        self.templates = self._load_templates(template_paths)
        if not self.templates:
            sys.exit("❌ ไม่สามารถโหลดไฟล์ Template ได้")
        print("✅ โหลด Templates สำเร็จ")

        # --- ส่วนของ Object Tracking ---
        self.next_object_id = 0
        # self.objects จะเก็บข้อมูลทั้งหมดของวัตถุที่กำลังติดตาม
        # { objectID: {'centroid': (x,y), 'locked_shape': 'Square', 'locked_color': 'Red', ...} }
        self.objects = OrderedDict()
        self.disappeared = OrderedDict() # นับจำนวนเฟรมที่วัตถุหายไป
        self.max_disappeared = max_disappeared # จำนวนเฟรมสูงสุดที่ยอมให้วัตถุหายไปก่อนจะลืม

    def _load_templates(self, template_paths):
        processed_templates = {}
        for shape_name, path in template_paths.items():
            template_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if template_img is None: continue
            blurred = cv2.GaussianBlur(template_img, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                processed_templates[shape_name] = max(contours, key=cv2.contourArea)
        return processed_templates

    def _get_raw_detections(self, frame):
        # ฟังก์ชันนี้จะทำการตรวจจับดิบๆ ในแต่ละเฟรม (เหมือนโค้ดเดิม)
        # แต่จะคืนค่าเป็น list ของ (contour, shape, color)
        # ... (โค้ดการประมวลผลภาพและตรวจจับสี/รูปทรง) ...
        blurred_frame = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v_eq = clahe.apply(v)
        normalized_hsv = cv2.merge([h, s, v_eq])

        color_ranges = {
            'Red': [(np.array([0, 120, 70]), np.array([10, 255, 255])), (np.array([170, 120, 70]), np.array([180, 255, 255]))],
            'Yellow': [(np.array([20, 100, 100]), np.array([30, 255, 255]))],
            'Green': [(np.array([35, 60, 30]), np.array([85, 255, 120]))],
            'Blue': [(np.array([90, 60, 30]), np.array([130, 255, 255]))]
        }
        
        masks = {}
        for color_name, ranges in color_ranges.items():
            mask_parts = [cv2.inRange(normalized_hsv, lower, upper) for lower, upper in ranges]
            masks[color_name] = cv2.bitwise_or(mask_parts[0], mask_parts[1]) if len(mask_parts) > 1 else mask_parts[0]

        combined_mask = cv2.bitwise_or(masks['Red'], cv2.bitwise_or(masks['Yellow'], cv2.bitwise_or(masks['Green'], masks['Blue'])))
        kernel = np.ones((5, 5), np.uint8)
        cleaned_mask = cv2.morphologyEx(cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1), cv2.MORPH_CLOSE, kernel, iterations=2)
        
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        raw_detections = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 1500: continue

            # ตรวจจับรูปทรง (เหมือนเดิม)
            shape = "Unknown"
            best_match_score, initial_shape = float('inf'), "Unknown"
            for shape_name, template_cnt in self.templates.items():
                score = cv2.matchShapes(cnt, template_cnt, cv2.CONTOURS_MATCH_I1, 0.0)
                if score < best_match_score: best_match_score, initial_shape = score, shape_name
            
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
            if len(approx) == 4:
                _, (w, h), _ = cv2.minAreaRect(cnt)
                aspect_ratio = max(w, h) / min(w, h) if min(w,h) > 0 else 0
                shape = "Square" if 0.95 <= aspect_ratio <= 1.15 else ("Rectangle_H" if w < h else "Rectangle_V")
            elif best_match_score < 0.56: shape = initial_shape

            # ตรวจจับสี (เหมือนเดิม)
            if shape != "Unknown":
                contour_mask = np.zeros(frame.shape[:2], dtype="uint8")
                cv2.drawContours(contour_mask, [cnt], -1, 255, -1)
                max_mean, found_color = 0, "Unknown"
                for color_name, mask in masks.items():
                    mean_val = cv2.mean(mask, mask=contour_mask)[0]
                    if mean_val > max_mean: max_mean, found_color = mean_val, color_name
                
                if max_mean > 25:
                    raw_detections.append({'contour': cnt, 'shape': shape, 'color': found_color})
        return raw_detections

    def update(self, frame):
        raw_detections = self._get_raw_detections(frame)
        
        # คำนวณจุดศูนย์กลางของวัตถุที่เจอในเฟรมปัจจุบัน
        input_centroids = np.zeros((len(raw_detections), 2), dtype="int")
        for i, detection in enumerate(raw_detections):
            M = cv2.moments(detection['contour'])
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            input_centroids[i] = (cX, cY)

        # ถ้าไม่มีวัตถุที่กำลังติดตามอยู่เลย ให้ลงทะเบียนวัตถุใหม่ทั้งหมด
        if len(self.objects) == 0:
            for i in range(len(raw_detections)):
                objectID = self.next_object_id
                self.objects[objectID] = {
                    'centroid': input_centroids[i],
                    'locked_shape': raw_detections[i]['shape'],
                    'locked_color': raw_detections[i]['color'],
                    'contour': raw_detections[i]['contour'] # เก็บ contour ปัจจุบันไว้วาด
                }
                print(f"✨ ลงทะเบียนวัตถุใหม่! ID {objectID}: {self.objects[objectID]['locked_color']} {self.objects[objectID]['locked_shape']}")
                self.disappeared[objectID] = 0
                self.next_object_id += 1
        else:
            # จับคู่วัตถุเก่ากับวัตถุใหม่ที่ใกล้ที่สุด
            object_ids = list(self.objects.keys())
            previous_centroids = np.array([d['centroid'] for d in self.objects.values()])
            D = dist.cdist(previous_centroids, input_centroids)
            
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols: continue
                
                # จับคู่สำเร็จ อัปเดตข้อมูล
                objectID = object_ids[row]
                self.objects[objectID]['centroid'] = input_centroids[col]
                self.objects[objectID]['contour'] = raw_detections[col]['contour']
                self.disappeared[objectID] = 0
                used_rows.add(row)
                used_cols.add(col)

            # ตรวจสอบวัตถุที่ไม่ได้ถูกจับคู่
            unused_rows = set(range(previous_centroids.shape[0])).difference(used_rows)
            unused_cols = set(range(input_centroids.shape[0])).difference(used_cols)

            # ถ้าวัตถุเก่าหายไป
            for row in unused_rows:
                objectID = object_ids[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    print(f"❌ ลบวัตถุ ID {objectID} เนื่องจากหายไปนาน")
                    del self.objects[objectID]
                    del self.disappeared[objectID]

            # ถ้ามีวัตถุใหม่เกิดขึ้น
            for col in unused_cols:
                objectID = self.next_object_id
                self.objects[objectID] = {
                    'centroid': input_centroids[col],
                    'locked_shape': raw_detections[col]['shape'],
                    'locked_color': raw_detections[col]['color'],
                    'contour': raw_detections[col]['contour']
                }
                print(f"✨ ลงทะเบียนวัตถุใหม่! ID {objectID}: {self.objects[objectID]['locked_color']} {self.objects[objectID]['locked_shape']}")
                self.disappeared[objectID] = 0
                self.next_object_id += 1

        # คืนค่าวัตถุที่ติดตามได้ทั้งหมด
        return list(self.objects.values())

# ======================================================================
# ฟังก์ชันสำหรับรับ Input จากผู้ใช้ (เหมือนเดิม)
# ======================================================================
def get_target_choice():
    VALID_SHAPES = ["Circle", "Square", "Rectangle_H", "Rectangle_V"]
    VALID_COLORS = ["Red", "Yellow", "Green", "Blue"]
    print("\n--- 🎯 กำหนดลักษณะของเป้าหมาย ---")
    while True:
        shape = input(f"เลือกรูปทรง ({'/'.join(VALID_SHAPES)}): ").strip().title()
        if shape in VALID_SHAPES: break
        print("⚠️ รูปทรงไม่ถูกต้อง, กรุณาลองใหม่")
    while True:
        color = input(f"เลือกสี ({'/'.join(VALID_COLORS)}): ").strip().title()
        if color in VALID_COLORS: break
        print("⚠️ สีไม่ถูกต้อง, กรุณาลองใหม่")
    print(f"✅ เป้าหมายคือ: {shape} สี {color}. เริ่มการค้นหา!")
    return shape, color

# ======================================================================
# Main (ปรับปรุงเล็กน้อยเพื่อเรียกใช้ Tracker)
# ======================================================================
if __name__ == '__main__':
    # เพิ่มการติดตั้ง scipy หากยังไม่มี
    try:
        import scipy
    except ImportError:
        print("กำลังติดตั้งไลบรารีที่จำเป็น: scipy")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
        print("ติดตั้งสำเร็จ!")

    target_shape, target_color = get_target_choice()
    
    template_files = {
        "Circle": "./Assignment/image_processing/template/circle1.png",
        "Square": "./Assignment/image_processing/template/square.png",
        "Rectangle_H": "./Assignment/image_processing/template/rec_h.png",
        "Rectangle_V": "./Assignment/image_processing/template/rec_v.png",
    }
    # เปลี่ยนมาใช้ ObjectTracker
    tracker = ObjectTracker(template_paths=template_files)

    ep_robot = robot.Robot()
    
    try:
        print("\n🤖 กำลังเชื่อมต่อกับหุ่นยนต์ Robomaster...")
        ep_robot.initialize(conn_type="ap") 
        print("✅ เชื่อมต่อสำเร็จ!")

        print("\n📷 กำลังเปิดกล้องของหุ่นยนต์...")
        ep_robot.camera.start_video_stream(display=False, resolution="720p")
        print("✅ เปิดกล้องสำเร็จ. กด 'q' บนหน้าต่างวิดีโอเพื่อออกจากโปรแกรม")
        
        while True:
            frame = ep_robot.camera.read_cv2_image(timeout=2)
            if frame is None: continue
            
            output_frame = frame.copy()
            # เรียกใช้ tracker.update() แทน detector.detect()
            tracked_objects = tracker.update(frame)

            for obj in tracked_objects:
                # ใช้ค่าที่ล็อกไว้จาก tracker
                is_target = (obj["locked_shape"] == target_shape and obj["locked_color"] == target_color)
                
                x, y, w, h = cv2.boundingRect(obj["contour"]) # ใช้ contour ปัจจุบันในการวาด
                box_color = (0, 0, 255) if is_target else (0, 255, 0)
                thickness = 4 if is_target else 2
                
                cv2.rectangle(output_frame, (x, y), (x+w, y+h), box_color, thickness)
                
                label = f"{obj['locked_shape']}, {obj['locked_color']}"
                if is_target:
                    label = "!!! TARGET FOUND !!!"
                    cv2.putText(output_frame, label, (x, y - 15), cv2.FONT_HERSHEY_TRIPLEX, 0.7, box_color, 2)
                else:
                    cv2.putText(output_frame, label, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('Robomaster Camera Feed - Press "q" to exit', output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
                
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดร้ายแรง: {e}")
        
    finally:
        print("\n🔌 กำลังปิดการเชื่อมต่อ...")
        try:
             ep_robot.camera.stop_video_stream()
             ep_robot.close()
        except Exception: pass
        cv2.destroyAllWindows()
        print("✅ ปิดการเชื่อมต่อเรียบร้อย")