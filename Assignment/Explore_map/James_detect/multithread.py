import cv2
import numpy as np
import sys
import robomaster
from robomaster import robot
import time
from scipy.spatial import distance as dist
from collections import OrderedDict
import threading # เพิ่ม Library สำหรับ Multi-thread

# ======================================================================
# คลาส ObjectTracker (เหมือนเดิม ไม่มีการเปลี่ยนแปลง)
# ======================================================================
class ObjectTracker:
    def __init__(self, template_paths, max_disappeared=30):
        print("🖼️  Loading and processing template images...")
        self.templates = self._load_templates(template_paths)
        if not self.templates:
            sys.exit("❌ Could not load template files.")
        print("✅ Templates loaded successfully.")
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

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
    
    # ฟังก์ชัน update สำหรับการ tracking (อาจจะไม่ได้ใช้แต่คงโครงสร้างไว้)
    def update(self, frame):
        # ... โค้ด update เดิม ...
        # ในตัวอย่างนี้จะเรียก _get_raw_detections ตรงๆ เพื่อความง่าย
        return self._get_raw_detections(frame)


# ======================================================================
# คลาสใหม่: VideoStream สำหรับ Thread-1 (ดึงภาพจากกล้อง)
# ======================================================================
class VideoStream:
    def __init__(self, camera):
        self.camera = camera
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        # เริ่ม Thread ที่จะรันฟังก์ชัน self.update
        print("▶️ Starting video stream thread...")
        t = threading.Thread(target=self.update, args=())
        t.daemon = True # ทำให้ Thread ปิดตัวเองเมื่อโปรแกรมหลักปิด
        t.start()
        return self

    def update(self):
        # ลูปนี้จะทำงานตลอดเวลาใน background
        while not self.stopped:
            try:
                # อ่านเฟรมจากกล้อง
                frame = self.camera.read_cv2_image(timeout=2)
                with self.lock:
                    self.frame = frame
            except Exception as e:
                print(f"Error reading frame from camera: {e}")
                time.sleep(0.5)

    def read(self):
        # ฟังก์ชันสำหรับให้ Thread อื่นมาดึงเฟรมล่าสุดไปใช้
        with self.lock:
            frame = self.frame
        return frame

    def stop(self):
        # สั่งให้ Thread หยุดทำงาน
        self.stopped = True

# ======================================================================
# คลาสใหม่: FrameProcessor สำหรับ Thread-2 (ประมวลผลภาพ)
# ======================================================================
class FrameProcessor:
    def __init__(self, video_stream, tracker, target_shape, target_color):
        self.video_stream = video_stream
        self.tracker = tracker
        self.target_shape = target_shape
        self.target_color = target_color
        self.processed_frame = None
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        print("▶️ Starting frame processor thread...")
        t = threading.Thread(target=self.process_loop, args=())
        t.daemon = True
        t.start()
        return self

    def process_loop(self):
        while not self.stopped:
            # ดึงเฟรมล่าสุดจาก VideoStream
            frame = self.video_stream.read()
            if frame is None:
                time.sleep(0.01) # รอสักครู่ถ้ายังไม่มีเฟรม
                continue
            
            output_frame = frame.copy()
            
            # --- ส่วนการประมวลผล (โค้ดเดิม) ---
            detected_objects = self.tracker.update(frame)

            for obj in detected_objects:
                is_target = (obj["shape"] == self.target_shape and obj["color"] == self.target_color)
                
                x, y, w, h = cv2.boundingRect(obj["contour"])
                box_color = (0, 0, 255) if is_target else (0, 255, 0)
                thickness = 4 if is_target else 2
                
                cv2.rectangle(output_frame, (x, y), (x + w, y + h), box_color, thickness)
                
                label = f"{obj['shape']}, {obj['color']}"
                if is_target:
                    label = "!!! TARGET FOUND !!!"
                    cv2.putText(output_frame, label, (x, y - 15), cv2.FONT_HERSHEY_TRIPLEX, 0.7, box_color, 2)
                else:
                    cv2.putText(output_frame, label, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # --- จบส่วนการประมวลผล ---

            # เก็บเฟรมที่ประมวลผลเสร็จแล้ว
            with self.lock:
                self.processed_frame = output_frame

    def read_processed(self):
        # ให้ Main Thread มาดึงภาพที่ประมวลผลเสร็จแล้วไปแสดง
        with self.lock:
            frame = self.processed_frame
        return frame

    def stop(self):
        self.stopped = True

# ======================================================================
# ฟังก์ชันสำหรับรับ Input จากผู้ใช้ (เหมือนเดิม)
# ======================================================================
def get_target_choice():
    VALID_SHAPES = ["Circle", "Square", "Rectangle_H", "Rectangle_V"]
    VALID_COLORS = ["Red", "Yellow", "Green", "Blue"]
    print("\n--- 🎯 Define Target Characteristics ---")
    while True:
        shape = input(f"Select Shape ({'/'.join(VALID_SHAPES)}): ").strip().title()
        if shape in VALID_SHAPES: break
        print("⚠️ Invalid shape, please try again.")
    while True:
        color = input(f"Select Color ({'/'.join(VALID_COLORS)}): ").strip().title()
        if color in VALID_COLORS: break
        print("⚠️ Invalid color, please try again.")
    print(f"✅ Target defined: {color} {shape}. Starting real-time detection!")
    return shape, color

# ======================================================================
# Main (ปรับปรุงใหม่เพื่อใช้ Multi-thread)
# ======================================================================
if __name__ == '__main__':
    try:
        import scipy
    except ImportError:
        print("Installing required library: scipy")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
        print("Installation successful!")

    target_shape, target_color = get_target_choice()
    
    template_files = {
        "Circle": "./Assignment/image_processing/template/circle1.png",
        "Square": "./Assignment/image_processing/template/square.png",
        "Rectangle_H": "./Assignment/image_processing/template/rec_h.png",
        "Rectangle_V": "./Assignment/image_processing/template/rec_v.png",
    }
    tracker = ObjectTracker(template_paths=template_files)

    ep_robot = robot.Robot()
    vs = None
    fp = None
    
    try:
        print("\n🤖 Connecting to Robomaster robot...")
        ep_robot.initialize(conn_type="ap") 
        print("✅ Connection successful!")

        print("\n📷 Starting camera...")
        ep_robot.camera.start_video_stream(display=False, resolution="720p")
        time.sleep(1) # รอให้กล้องเริ่มทำงานนิ่งๆ

        # 1. เริ่ม Thread-1: VideoStream
        vs = VideoStream(ep_robot.camera).start()
        
        # 2. เริ่ม Thread-2: FrameProcessor
        fp = FrameProcessor(vs, tracker, target_shape, target_color).start()
        
        print("✅ All threads are running. Displaying processed feed.")
        print("   Press 'q' on the video window to exit.")

        # 3. Main Thread ทำหน้าที่แค่แสดงผล
        while True:
            # ดึงเฟรมที่ประมวลผลเสร็จแล้วมาแสดง
            processed_frame = fp.read_processed()

            if processed_frame is not None:
                cv2.imshow('Robomaster Camera Feed - Press "q" to exit', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # หน่วงเวลาเล็กน้อยเพื่อลดการใช้ CPU ของ Main Thread
            time.sleep(0.01)

    except Exception as e:
        print(f"❌ A critical error occurred: {e}")
        
    finally:
        print("\n🔌 Shutting down all threads and connections...")
        if fp:
            fp.stop() # สั่งให้ Thread-2 หยุด
        if vs:
            vs.stop() # สั่งให้ Thread-1 หยุด
        
        time.sleep(1) # รอให้ Thread ปิดตัวเรียบร้อย
        
        try:
            ep_robot.camera.stop_video_stream()
            ep_robot.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("✅ Shutdown complete.")