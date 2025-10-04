# pic_1_shot.py  (night-robust hue detection)
import cv2
import numpy as np
import sys
import robomaster
from robomaster import robot
import time
import math
import matplotlib.pyplot as plt

# ===============================
# Night-robust color pre-processing (No changes)
# ===============================
def apply_awb(bgr):
    if hasattr(cv2, "xphoto") and hasattr(cv2.xphoto, "createLearningBasedWB"):
        wb = cv2.xphoto.createLearningBasedWB()
        try:
            wb.setSaturationThreshold(0.99)
        except Exception: pass
        return wb.balanceWhite(bgr)
    elif hasattr(cv2, "xphoto"):
        wb = cv2.xphoto.createSimpleWB()
        try:
            wb.setP(0.5)
        except Exception: pass
        return wb.balanceWhite(bgr)
    else:
        return bgr

def retinex_msrcp(bgr, sigmas=(15, 80, 250), eps=1e-6):
    img = bgr.astype(np.float32) + 1.0
    intensity = img.mean(axis=2)
    log_I = np.log(intensity + eps)
    msr = np.zeros_like(intensity, dtype=np.float32)
    for s in sigmas:
        blur = cv2.GaussianBlur(intensity, (0, 0), s)
        msr += (log_I - np.log(blur + eps))
    msr /= float(len(sigmas))
    msr -= msr.min()
    msr /= (msr.max() + eps)
    msr = (msr * 255.0).astype(np.float32)
    scale = (msr + 1.0) / (intensity + eps)
    out = np.clip(img * scale[..., None], 0, 255).astype(np.uint8)
    return out

def night_enhance_pipeline(bgr):
    den = cv2.fastNlMeansDenoisingColored(bgr, None, 7, 7, 7, 21)
    awb = apply_awb(den)
    ret = retinex_msrcp(awb)
    return ret

# =====================================================
# ObjectTracker (No changes)
# =====================================================
class ObjectTracker:
    def __init__(self, template_paths):
        print("🖼️  Loading and processing template images...")
        self.templates = self._load_templates(template_paths)
        if not self.templates:
            sys.exit("❌ Could not load template files.")
        print("✅ Templates loaded successfully.")

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

    def _get_angle(self, pt1, pt2, pt0):
        dx1 = pt1[0] - pt0[0]; dy1 = pt1[1] - pt0[1]
        dx2 = pt2[0] - pt0[0]; dy2 = pt2[1] - pt0[1]
        dot_product = dx1 * dx2 + dy1 * dy2
        magnitude1 = math.sqrt(dx1 * dx1 + dy1 * dy1)
        magnitude2 = math.sqrt(dx2 * dx2 + dy2 * dy2)
        if magnitude1 * magnitude2 == 0: return 0
        angle_rad = math.acos(dot_product / (magnitude1 * magnitude2))
        return math.degrees(angle_rad)

    def _get_raw_detections(self, frame):
        enhanced = night_enhance_pipeline(frame)
        enhanced = cv2.GaussianBlur(enhanced, (5, 5), 0)
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        v_eq = clahe.apply(v)
        normalized_hsv = cv2.merge([h, s, v_eq])
        color_ranges = {
            'Red': [np.array([0, 80, 40]), np.array([10, 255, 255]), np.array([170, 80, 40]), np.array([180, 255, 255])],
            'Yellow': [np.array([20, 60, 40]), np.array([35, 255, 255])],
            'Green': [np.array([35, 40, 30]), np.array([85, 255, 255])],
            'Blue': [np.array([90, 40, 30]), np.array([130, 255, 255])]
        }
        masks = {}
        lower1, upper1, lower2, upper2 = color_ranges['Red']
        masks['Red'] = cv2.inRange(normalized_hsv, lower1, upper1) | cv2.inRange(normalized_hsv, lower2, upper2)
        for name in ['Yellow', 'Green', 'Blue']:
            lower, upper = color_ranges[name]
            masks[name] = cv2.inRange(normalized_hsv, lower, upper)
        combined_mask = masks['Red'] | masks['Yellow'] | masks['Green'] | masks['Blue']
        kernel = np.ones((5, 5), np.uint8)
        cleaned_mask = cv2.morphologyEx(cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1), cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1500: continue
            contour_mask = np.zeros(frame.shape[:2], dtype="uint8")
            cv2.drawContours(contour_mask, [cnt], -1, 255, -1)
            max_mean, found_color = 0, "Unknown"
            for color_name, m in masks.items():
                mean_val = cv2.mean(m, mask=contour_mask)[0]
                if mean_val > max_mean:
                    max_mean, found_color = mean_val, color_name
            if max_mean <= 20: continue
            shape = "Uncertain"
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            circularity = (4 * np.pi * area) / (peri * peri) if peri > 0 else 0
            if circularity > 0.84:
                shape = "Circle"
            elif len(approx) == 4:
                points = [tuple(p[0]) for p in approx]
                angles = [self._get_angle(points[(i - 1) % 4], points[(i + 1) % 4], p) for i, p in enumerate(points)]
                if all(75 <= ang <= 105 for ang in angles):
                    _, (w, h), _ = cv2.minAreaRect(cnt)
                    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
                    shape = "Square" if 0.90 <= aspect_ratio <= 1.10 else "Rectangle_H" if w < h else "Rectangle_V"
            raw_detections.append({'contour': cnt, 'shape': shape, 'color': found_color})
        return raw_detections

# =====================================================
# ส่วนโต้ตอบและวาดผล (ปรับปรุงใหม่)
# =====================================================
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
        return None, "", "", "", []

    frame = frame[352:352+360, 14:14+1215]
    print("🧠 Processing captured frame...")
    output_image = frame.copy()
    height, width, _ = frame.shape

    DIVIDER_X1, DIVIDER_X2 = 400, 800
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.line(output_image, (DIVIDER_X1, 0), (DIVIDER_X1, height), (255, 255, 0), 2)
    cv2.line(output_image, (DIVIDER_X2, 0), (DIVIDER_X2, height), (255, 255, 0), 2)
    cv2.putText(output_image, "LEFT", (DIVIDER_X1 // 2 - 50, 40), font, 1, (255, 255, 0), 2)
    cv2.putText(output_image, "CENTER", ((DIVIDER_X1 + DIVIDER_X2) // 2 - 70, 40), font, 1, (255, 255, 0), 2)
    cv2.putText(output_image, "RIGHT", ((DIVIDER_X2 + width) // 2 - 60, 40), font, 1, (255, 255, 0), 2)

    detections = tracker._get_raw_detections(frame)
    
    summary_lines = {'Left': [], 'Center': [], 'Right': []}
    detailed_results = []  # <--- NEW: List to store detailed object info
    object_id_counter = 1   # <--- NEW: Counter for object IDs

    for det in detections:
        shape, color, contour = det['shape'], det['color'], det['contour']
        x, y, w, h = cv2.boundingRect(contour)
        obj_start_x, obj_end_x = x, x + w

        zone_label = "Unknown"
        if obj_end_x < DIVIDER_X1: zone_label = "Left"
        elif obj_start_x >= DIVIDER_X2: zone_label = "Right"
        elif obj_start_x >= DIVIDER_X1 and obj_end_x < DIVIDER_X2: zone_label = "Center"
        elif obj_start_x < DIVIDER_X1 and obj_end_x >= DIVIDER_X1: zone_label = "Corner (L/C)"
        elif obj_start_x < DIVIDER_X2 and obj_end_x >= DIVIDER_X2: zone_label = "Corner (C/R)"

        is_target = (shape == target_shape and color == target_color)
        is_uncertain = (shape == "Uncertain")
        
        if is_target:
            box_color, thickness, summary_prefix = (0, 0, 255), 4, "TARGET FOUND"
        elif is_uncertain:
            box_color, thickness, summary_prefix = (0, 255, 255), 2, "Uncertain object"
        else:
            box_color, thickness, summary_prefix = (0, 255, 0), 2, "Found object"
        
        summary_text = f"  - {summary_prefix}: {color} {shape}"

        # Add to summary lists
        if "Left" in zone_label or "(L/C)" in zone_label:
            summary_lines['Left'].append(f"{summary_text} [ID: {object_id_counter}]")
        if "Center" in zone_label or "(L/C)" in zone_label or "(C/R)" in zone_label:
            summary_lines['Center'].append(f"{summary_text} [ID: {object_id_counter}]")
        if "Right" in zone_label or "(C/R)" in zone_label:
            summary_lines['Right'].append(f"{summary_text} [ID: {object_id_counter}]")
            
        # --- NEW: Store detailed info for the legend ---
        detailed_results.append({
            "id": object_id_counter,
            "color": color,
            "shape": shape,
            "zone": zone_label,
            "is_target": is_target
        })

        # --- MODIFIED: Draw only the ID on the image ---
        label_on_box = str(object_id_counter)
        cv2.rectangle(output_image, (x, y), (x + w, y + h), box_color, thickness)
        cv2.putText(output_image, label_on_box, (x + 5, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
        
        object_id_counter += 1 # Increment for the next object

    final_left_summary = f"--- Left Zone Detections ---\n" + ('\n'.join(summary_lines['Left']) if summary_lines['Left'] else "  - No objects found.")
    final_center_summary = f"--- Center Zone Detections ---\n" + ('\n'.join(summary_lines['Center']) if summary_lines['Center'] else "  - No objects found.")
    final_right_summary = f"--- Right Zone Detections ---\n" + ('\n'.join(summary_lines['Right']) if summary_lines['Right'] else "  - No objects found.")

    return output_image, final_left_summary, final_center_summary, final_right_summary, detailed_results

# =====================================================
# __main__ (ปรับปรุงเพื่อแสดง Legend)
# =====================================================
if __name__ == '__main__':
    # ... (ส่วนการติดตั้ง library และรับ input เหมือนเดิม) ...
    try: import scipy
    except ImportError:
        print("Installing required library: scipy")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
        print("Installation successful!")
    try: _ = cv2.xphoto
    except Exception: print("🔎 Tip: For best AWB, install opencv-contrib-python")

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

    # ... (ส่วนการเชื่อมต่อหุ่นยนต์และถ่ายภาพเหมือนเดิม) ...
    try:
        print("\n🤖 Connecting to Robomaster robot...")
        ep_robot.initialize(conn_type="ap")
        print("✅ Connection successful!")
        print("📷 Starting camera for a snapshot...")
        ep_robot.camera.start_video_stream(display=False, resolution="720p")
        print("   Waiting 2 seconds for the camera to stabilize...")
        time.sleep(2)
        print("📸 Capturing a single frame...")
        captured_frame = ep_robot.camera.read_cv2_image(timeout=5)
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
        # --- MODIFIED: รับ 'detailed_results' เพิ่ม ---
        result_image, left_summary, center_summary, right_summary, detailed_results = scan_and_process_maze_wall(
            captured_frame, tracker, target_shape, target_color
        )
        print("\n--- SCAN COMPLETE ---")
        print(left_summary)
        print(center_summary)
        print(right_summary)
        
        print("\nDisplaying result image with Matplotlib...")
        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        
        # --- NEW: สร้างและแสดงผล Legend ---
        fig = plt.figure(figsize=(12, 10)) # เพิ่มความสูงของ Figure
        plt.imshow(result_image_rgb)
        plt.title("Maze Wall Scan Analysis Result (3-Zone)")
        plt.axis('off')

        # สร้างข้อความสำหรับ Legend
        if detailed_results:
            legend_lines = ["--- Object Details Legend ---"]
            for obj in detailed_results:
                target_str = " (TARGET!)" if obj['is_target'] else ""
                line = f"ID {obj['id']}: {obj['color']} {obj['shape']} in Zone [{obj['zone']}] {target_str}"
                legend_lines.append(line)
            legend_text = "\n".join(legend_lines)
        else:
            legend_text = "--- No objects detected ---"

        # ปรับ layout เพื่อเพิ่มที่ว่างด้านล่าง
        plt.subplots_adjust(bottom=0.25)
        
        # แสดง Legend ในพื้นที่ว่าง
        plt.figtext(0.5, 0.01, legend_text, 
                    ha="center", va="bottom", 
                    fontsize=10, 
                    bbox={"facecolor":"white", "alpha":0.8, "pad":5})

        plt.show()
    else:
        print("\n❌ No frame was captured. Skipping processing and display.")