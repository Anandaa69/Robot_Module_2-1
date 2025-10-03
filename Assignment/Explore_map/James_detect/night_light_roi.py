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
# Night-robust color pre-processing
# ===============================
def apply_awb(bgr):
    """
    Auto White Balance:
    - ใช้ LearningBasedWB ถ้ามี (opencv-contrib-python)
    - ไม่มีก็ fallback เป็น SimpleWB
    """
    if hasattr(cv2, "xphoto") and hasattr(cv2.xphoto, "createLearningBasedWB"):
        wb = cv2.xphoto.createLearningBasedWB()
        try:
            # ลดการล้นของ highlight ตอนกลางคืน/ไฟหลอด
            wb.setSaturationThreshold(0.99)
        except Exception:
            pass
        return wb.balanceWhite(bgr)
    elif hasattr(cv2, "xphoto"):
        wb = cv2.xphoto.createSimpleWB()
        # median-based ปานกลาง
        try:
            wb.setP(0.5)
        except Exception:
            pass
        return wb.balanceWhite(bgr)
    else:
        # ไม่มี xphoto ก็คืนภาพเดิม (ยังได้ประโยชน์จาก Retinex/CLAHE ด้านล่าง)
        return bgr

def retinex_msrcp(bgr, sigmas=(15, 80, 250), eps=1e-6):
    """
    MSRCP Retinex (ฉบับย่อ): ช่วยให้คอนทราสต์สม่ำเสมอขึ้นในภาพมืด/มีเงา
    """
    img = bgr.astype(np.float32) + 1.0
    intensity = img.mean(axis=2)
    log_I = np.log(intensity + eps)
    msr = np.zeros_like(intensity, dtype=np.float32)
    for s in sigmas:
        blur = cv2.GaussianBlur(intensity, (0, 0), s)
        msr += (log_I - np.log(blur + eps))
    msr /= float(len(sigmas))

    # นอร์มัลไลซ์ 0..255
    msr -= msr.min()
    msr /= (msr.max() + eps)
    msr = (msr * 255.0).astype(np.float32)

    # color preservation
    scale = (msr + 1.0) / (intensity + eps)
    out = np.clip(img * scale[..., None], 0, 255).astype(np.uint8)
    return out

def night_enhance_pipeline(bgr):
    """
    Pipeline กลางคืน: denoise -> AWB -> Retinex
    """
    den = cv2.fastNlMeansDenoisingColored(bgr, None, 7, 7, 7, 21)
    awb = apply_awb(den)
    ret = retinex_msrcp(awb)
    return ret

# =====================================================
# ObjectTracker (เหมือนเดิม + ปรับ _get_raw_detections ให้ทนแสง)
# =====================================================
class ObjectTracker:
    def __init__(self, template_paths):
        print("🖼️  Loading and processing template images...")
        self.templates = self._load_templates(template_paths)
        if not self.templates:
            sys.exit("❌ Could not load template files.")
        print("✅ Templates loaded successfully.")

    def _load_templates(self, template_paths):
        # โหลดคอนทัวร์ของ template (เหมือนเดิม)
        processed_templates = {}
        for shape_name, path in template_paths.items():
            template_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if template_img is None:
                continue
            blurred = cv2.GaussianBlur(template_img, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                processed_templates[shape_name] = max(contours, key=cv2.contourArea)
        return processed_templates

    # ---------- geometry helper ----------
    def _get_angle(self, pt1, pt2, pt0):
        """คำนวณองศาของมุมที่จุด pt0"""
        dx1 = pt1[0] - pt0[0]
        dy1 = pt1[1] - pt0[1]
        dx2 = pt2[0] - pt0[0]
        dy2 = pt2[1] - pt0[1]
        dot_product = dx1 * dx2 + dy1 * dy2
        magnitude1 = math.sqrt(dx1 * dx1 + dy1 * dy1)
        magnitude2 = math.sqrt(dx2 * dx2 + dy2 * dy2)
        if magnitude1 * magnitude2 == 0:
            return 0
        angle_rad = math.acos(dot_product / (magnitude1 * magnitude2))
        return math.degrees(angle_rad)

    # ---------- RAW detections (ปรับให้ robust ต่อกลางคืน) ----------
    def _get_raw_detections(self, frame):
        # 1) Night-robust preproc
        enhanced = night_enhance_pipeline(frame)

        # 2) เพิ่ม blur เล็กน้อย
        enhanced = cv2.GaussianBlur(enhanced, (5, 5), 0)

        # 3) ไป HSV และทำ CLAHE ที่ช่อง V
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        # กลางคืนมืด -> เพิ่ม clipLimit ขึ้นเล็กน้อย
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        v_eq = clahe.apply(v)
        normalized_hsv = cv2.merge([h, s, v_eq])

        # 4) กำหนดช่วงสีแบบ "รับกลางคืน" (S/V ต่ำลง)
        color_ranges = {
            'Red': [
                np.array([0,   80,  40]), np.array([10, 255, 255]),
                np.array([170, 80,  40]), np.array([180, 255, 255])
            ],
            'Yellow': [np.array([20, 60, 40]), np.array([35, 255, 255])],
            'Green':  [np.array([35, 40, 30]), np.array([85, 255, 255])],
            'Blue':   [np.array([90, 40, 30]), np.array([130,255, 255])]
        }
        # สร้าง mask ต่อสี
        masks = {}
        # Red มี 2 ช่วง H
        lower1, upper1, lower2, upper2 = color_ranges['Red']
        mask_red = cv2.inRange(normalized_hsv, lower1, upper1) | cv2.inRange(normalized_hsv, lower2, upper2)
        masks['Red'] = mask_red
        for name in ['Yellow', 'Green', 'Blue']:
            lower, upper = color_ranges[name]
            masks[name] = cv2.inRange(normalized_hsv, lower, upper)

        # 5) รวม mask + ทำความสะอาด
        combined_mask = masks['Red'] | masks['Yellow'] | masks['Green'] | masks['Blue']
        kernel = np.ones((5, 5), np.uint8)
        cleaned_mask = cv2.morphologyEx(
            cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1),
            cv2.MORPH_CLOSE, kernel, iterations=2
        )

        # 6) หา contours
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1500:
                continue

            # -------- ตรวจสี (เลือกสีที่มีสัดส่วน mask มากสุดในคอนทัวร์) --------
            contour_mask = np.zeros(frame.shape[:2], dtype="uint8")
            cv2.drawContours(contour_mask, [cnt], -1, 255, -1)
            max_mean, found_color = 0, "Unknown"
            for color_name, m in masks.items():
                mean_val = cv2.mean(m, mask=contour_mask)[0]
                if mean_val > max_mean:
                    max_mean, found_color = mean_val, color_name
            # กลางคืนบางที mean ต่ำ ให้เกณฑ์อ่อนกว่าเดิม
            if max_mean <= 20:
                continue

            # -------- ตรวจรูปทรง --------
            shape = "Uncertain"
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

            # 3.1) วงกลม (circularity)
            circularity = (4 * np.pi * area) / (peri * peri) if peri > 0 else 0
            if circularity > 0.84:
                shape = "Circle"

            # 3.2) สี่เหลี่ยม (เช็คมุม ~ 90°)
            elif len(approx) == 4:
                points = [tuple(p[0]) for p in approx]
                angles = []
                for i in range(4):
                    p0 = points[i]
                    p_prev = points[(i - 1) % 4]
                    p_next = points[(i + 1) % 4]
                    angles.append(self._get_angle(p_prev, p_next, p0))

                is_rectangle = all(75 <= ang <= 105 for ang in angles)
                if is_rectangle:
                    _, (w, h), _ = cv2.minAreaRect(cnt)
                    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
                    if 0.90 <= aspect_ratio <= 1.10:
                        shape = "Square"
                    else:
                        shape = "Rectangle_H" if w < h else "Rectangle_V"

            # เก็บผล
            raw_detections.append({'contour': cnt, 'shape': shape, 'color': found_color})

        return raw_detections

# =====================================================
# ส่วนโต้ตอบและวาดผล (ตามไฟล์เดิม)
# =====================================================
def get_target_choice():
    VALID_SHAPES = ["Circle", "Square", "Rectangle_H", "Rectangle_V"]
    VALID_COLORS = ["Red", "Yellow", "Green", "Blue"]
    print("\n--- 🎯 Define Target Characteristics ---")
    while True:
        shape = input(f"Select Shape ({'/'.join(VALID_SHAPES)}): ").strip().title()
        if shape in VALID_SHAPES:
            break
        print("⚠️ Invalid shape, please try again.")
    while True:
        color = input(f"Select Color ({'/'.join(VALID_COLORS)}): ").strip().title()
        if color in VALID_COLORS:
            break
        print("⚠️ Invalid color, please try again.")
    print(f"✅ Target defined: {color} {shape}. Starting scan.")
    return shape, color

def scan_and_process_maze_wall(frame, tracker, target_shape, target_color):
    if frame is None:
        print("❌ Cannot process a null frame.")
        return None, "", ""
    
    # --- CROP FRAME TO ROI ---
    # พิกัด (x=14, y=352, w=1215, h=360)
    frame = frame[352:352+360, 14:14+1215]
    # -------------------------

    print("🧠 Processing captured frame...")
    output_image = frame.copy()
    height, width, _ = frame.shape
    mid_x = width // 2
    cv2.line(output_image, (mid_x, 0), (mid_x, height), (255, 255, 0), 2)
    cv2.putText(output_image, "LEFT", (mid_x - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(output_image, "RIGHT", (mid_x + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    detections = tracker._get_raw_detections(frame)

    # แยกซ้าย/ขวา
    left_detections, right_detections = [], []
    for det in detections:
        M = cv2.moments(det['contour'])
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        if cX < mid_x:
            left_detections.append(det)
        else:
            right_detections.append(det)

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
            cv2.rectangle(output_image, (x, y), (x + w, y + h), box_color, thickness)
            cv2.putText(output_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
            side_summary.append(summary_text)

    process_side(left_detections, left_summary)
    process_side(right_detections, right_summary)

    final_left_summary = f"--- Left Side Detections ---\n" + ('\n'.join(left_summary) if left_summary else "  - No objects found.")
    final_right_summary = f"--- Right Side Detections ---\n" + ('\n'.join(right_summary) if right_summary else "  - No objects found.")

    return output_image, final_left_summary, final_right_summary

# =====================================================
# __main__ (เหมือนเดิม)
# =====================================================
if __name__ == '__main__':
    try:
        import scipy
    except ImportError:
        print("Installing required library: scipy")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
        print("Installation successful!")

    # แนะนำ: opencv-contrib-python สำหรับ xphoto (AWB)
    try:
        _ = cv2.xphoto  # type: ignore
    except Exception:
        print("🔎 Tip: For best AWB, install opencv-contrib-python")

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
        # ทดสอบออฟไลน์: captured_frame = cv2.imread("test_image.png")
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