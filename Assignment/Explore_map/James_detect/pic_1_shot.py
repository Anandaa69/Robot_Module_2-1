import cv2
import numpy as np
import sys
import robomaster
from robomaster import robot
import time
import math # เพิ่ม math เข้ามาเพื่อใช้คำนวณ
import matplotlib.pyplot as plt

class ObjectTracker:
    def __init__(self, template_paths):
        print("🖼️  Loading and processing template images...")
        self.templates = self._load_templates(template_paths)
        if not self.templates:
            sys.exit("❌ Could not load template files.")
        print("✅ Templates loaded successfully.")

    def _load_templates(self, template_paths):
        # ... (โค้ดส่วนนี้เหมือนเดิม) ...
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

    # ==================================================================
    # VVVV ฟังก์ชันใหม่สำหรับคำนวณมุม VVVV
    # ==================================================================
    def _get_angle(self, pt1, pt2, pt0):
        """คำนวณองศาของมุมที่จุด pt0"""
        dx1 = pt1[0] - pt0[0]
        dy1 = pt1[1] - pt0[1]
        dx2 = pt2[0] - pt0[0]
        dy2 = pt2[1] - pt0[1]
        dot_product = dx1 * dx2 + dy1 * dy2
        magnitude1 = math.sqrt(dx1 * dx1 + dy1 * dy1)
        magnitude2 = math.sqrt(dx2 * dx2 + dy2 * dy2)
        
        # ป้องกันการหารด้วยศูนย์
        if magnitude1 * magnitude2 == 0:
            return 0
            
        angle_rad = math.acos(dot_product / (magnitude1 * magnitude2))
        return math.degrees(angle_rad)

    # ==================================================================
    # VVVV ฟังก์ชันที่ปรับปรุงด้วย "การตรวจสอบมุม" VVVV
    # ==================================================================
    def _get_raw_detections(self, frame):
        # ... (ส่วนประมวลผลภาพเบื้องต้นเหมือนเดิม) ...
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
            area = cv2.contourArea(cnt)
            if area < 1500: continue

            shape = "Uncertain"

            # 1. ตรวจสี (เหมือนเดิม)
            contour_mask = np.zeros(frame.shape[:2], dtype="uint8")
            cv2.drawContours(contour_mask, [cnt], -1, 255, -1)
            max_mean, found_color = 0, "Unknown"
            for color_name, mask in masks.items():
                mean_val = cv2.mean(mask, mask=contour_mask)[0]
                if mean_val > max_mean: max_mean, found_color = mean_val, color_name
            if max_mean <= 25: continue

            # 2. คำนวณค่าทางเรขาคณิต
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

            # 3. ตรวจสอบเงื่อนไขรูปทรงแบบใหม่
            # 3.1) เช็คความเป็นวงกลมก่อน (ยังคงมีประโยชน์)
            circularity = (4 * np.pi * area) / (peri * peri) if peri > 0 else 0
            if circularity > 0.84:
                shape = "Circle"
            
            # 3.2) ถ้าไม่กลม และมี 4 จุดยอด -> **ทำการตรวจสอบมุม**
            elif len(approx) == 4:
                # แปลง approx ให้เป็น list ของ tuple เพื่อให้ใช้ง่าย
                points = [tuple(p[0]) for p in approx]
                
                # คำนวณมุมทั้ง 4
                angles = []
                for i in range(4):
                    p0 = points[i]
                    p_prev = points[(i - 1) % 4]
                    p_next = points[(i + 1) % 4]
                    angles.append(self._get_angle(p_prev, p_next, p0))

                # ตรวจสอบว่าทุกมุมอยู่ในช่วง 90 +/- 15 องศา หรือไม่
                is_rectangle = all(75 <= angle <= 105 for angle in angles)
                
                if is_rectangle:
                    # ถ้ามุมถูกต้อง ค่อยมาเช็คว่าเป็น Square หรือ Rectangle
                    _, (w, h), _ = cv2.minAreaRect(cnt)
                    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
                    if 0.90 <= aspect_ratio <= 1.10:
                        shape = "Square"
                    else:
                        shape = "Rectangle_H" if w < h else "Rectangle_V"
                # ถ้ามุมไม่ถูกต้อง -> shape จะยังคงเป็น "Uncertain" ซึ่งคือเป้าหมายของเรา
            
            raw_detections.append({'contour': cnt, 'shape': shape, 'color': found_color})
            
        return raw_detections

# (ส่วนที่เหลือของโค้ด: get_target_choice, scan_and_process_maze_wall, __main__ เหมือนเดิมทั้งหมด)
# ... (คัดลอกส่วนที่เหลือจากโค้ดก่อนหน้ามาวางได้เลย) ...
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
    print(f"✅ Target defined: {color} {shape}. Starting scan.")
    return shape, color

def scan_and_process_maze_wall(frame, tracker, target_shape, target_color):
    if frame is None:
        print("❌ Cannot process a null frame.")
        return None, "", ""
    print("🧠 Processing captured frame...")
    output_image = frame.copy()
    height, width, _ = frame.shape
    mid_x = width // 2
    cv2.line(output_image, (mid_x, 0), (mid_x, height), (255, 255, 0), 2)
    cv2.putText(output_image, "LEFT", (mid_x - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(output_image, "RIGHT", (mid_x + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    detections = tracker._get_raw_detections(frame)
    left_detections, right_detections = [], []
    for det in detections:
        M = cv2.moments(det['contour'])
        if M["m00"] == 0: continue
        cX = int(M["m10"] / M["m00"])
        if cX < mid_x: left_detections.append(det)
        else: right_detections.append(det)
    left_summary, right_summary = [], []
    def process_side(detections, side_summary):
        for det in detections:
            shape, color, contour = det['shape'], det['color'], det['contour']
            is_target = (shape == target_shape and color == target_color)
            is_uncertain = (shape == "Uncertain")
            x, y, w, h = cv2.boundingRect(contour)
            if is_target:
                box_color, thickness, label = (0, 0, 255), 4, f"!!! TARGET: {color} {shape} !!!"
                summary_text = f"  - TARGET FOUND: {color} {shape}"
            elif is_uncertain:
                box_color, thickness, label = (0, 255, 255), 2, f"Uncertain Shape ({color})"
                summary_text = f"  - Uncertain object detected (Color: {color}). Needs closer inspection."
            else:
                box_color, thickness, label = (0, 255, 0), 2, f"Object: {color} {shape}"
                summary_text = f"  - Found object: {color} {shape}"
            cv2.rectangle(output_image, (x, y), (x+w, y+h), box_color, thickness)
            cv2.putText(output_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
            side_summary.append(summary_text)
    process_side(left_detections, left_summary)
    process_side(right_detections, right_summary)
    final_left_summary = f"--- Left Side Detections ---\n" + ('\n'.join(left_summary) if left_summary else "  - No objects found.")
    final_right_summary = f"--- Right Side Detections ---\n" + ('\n'.join(right_summary) if right_summary else "  - No objects found.")
    return output_image, final_left_summary, final_right_summary

if __name__ == '__main__':
    # ... (ส่วน main เหมือนเดิม) ...
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
    captured_frame = None

    try:
        print("\n🤖 Connecting to Robomaster robot...")
        ep_robot.initialize(conn_type="ap")
        print("✅ Connection successful!")
        print("📷 Starting camera for a snapshot...")
        ep_robot.camera.start_video_stream(display=False, resolution="720p")
        print("   Waiting 2 seconds for the camera to stabilize...")
        time.sleep(2)
        print("📸 Capturing a single frame...")
        # captured_frame = cv2.imread("test_image.png") # <<<< โหลดภาพทดสอบของคุณที่นี่
        captured_frame = ep_robot.camera.read_cv2_image(timeout=5) # <<<< ใช้บรรทัดนี้เมื่อต่อกับหุ่น
        if captured_frame is None:
            raise RuntimeError("Failed to capture or read frame.")
        else:
            print("✅ Frame captured successfully.")
    except Exception as e:
        print(f"❌ An error occurred during connection or capture: {e}")
    finally:
        print("\n🔌 Closing robot connection...")
        try:
            ep_robot.camera.stop_video_stream()
            ep_robot.close()
            print("✅ Robot connection closed.")
        except Exception as e:
            print(f"   (Note) Error during cleanup, but continuing: {e}")

    if captured_frame is not None:
        result_image, left_summary, right_summary = scan_and_process_maze_wall(
            captured_frame, tracker, target_shape, target_color
        )
        print("\n--- SCAN COMPLETE ---")
        print(left_summary)
        print(right_summary)
        print("\nDisplaying result image with Matplotlib...")
        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 8))
        plt.imshow(result_image_rgb)
        plt.title("Maze Wall Scan Analysis Result")
        plt.axis('off')
        plt.show()
    else:
        print("\n❌ No frame was captured. Skipping processing and display.")