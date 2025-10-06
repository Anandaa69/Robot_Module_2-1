import cv2
import numpy as np
import sys
import robomaster
from robomaster import robot
import time
import math

# ======================================================================
# ส่วนจัดการแสง (เหมือนเดิม)
# ======================================================================
def apply_awb(bgr):
    if hasattr(cv2, "xphoto") and hasattr(cv2.xphoto, "createLearningBasedWB"):
        wb = cv2.xphoto.createLearningBasedWB()
        wb.setSaturationThreshold(0.99)
        return wb.balanceWhite(bgr)
    elif hasattr(cv2, "xphoto"):
        wb = cv2.xphoto.createSimpleWB()
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
    denoised = cv2.fastNlMeansDenoisingColored(bgr, None, 7, 7, 7, 21)
    awb_image = apply_awb(denoised)
    retinex_image = retinex_msrcp(awb_image)
    return retinex_image

# ======================================================================
# คลาส ObjectDetector (แก้ไขแล้ว)
# ======================================================================
class ObjectDetector:
    def __init__(self):
        print("✅ ObjectDetector พร้อมใช้งาน (โหมดวิเคราะห์รูปทรง + จัดการแสงขั้นสูง)")

    def _get_angle(self, pt1, pt2, pt0):
        dx1 = pt1[0][0] - pt0[0][0]
        dy1 = pt1[0][1] - pt0[0][1]
        dx2 = pt2[0][0] - pt0[0][0]
        dy2 = pt2[0][1] - pt0[0][1]
        dot_product = float(dx1 * dx2 + dy1 * dy2)
        magnitude1 = math.sqrt(dx1*dx1 + dy1*dy1)
        magnitude2 = math.sqrt(dx2*dx2 + dy2*dy2)
        if magnitude1 * magnitude2 == 0: return 0
        cos_angle = max(-1.0, min(1.0, dot_product / (magnitude1 * magnitude2)))
        angle_rad = math.acos(cos_angle)
        return math.degrees(angle_rad)

    def detect(self, frame):
        enhanced_frame = night_enhance_pipeline(frame)
        hsv = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        v_eq = clahe.apply(v)
        normalized_hsv = cv2.merge([h, s, v_eq])
        detected_objects = []

        # --- FIX: ขยายช่วงค่าสีเขียวให้กว้างขึ้น ---
        color_ranges = {
            'Red':    [ (np.array([0, 80, 40]), np.array([10, 255, 255])), (np.array([170, 80, 40]), np.array([180, 255, 255])) ],
            'Yellow': [ (np.array([20, 60, 40]), np.array([35, 255, 255])) ],
            'Green':  [ (np.array([35, 25, 25]), np.array([90, 255, 255])) ], # <--- ค่าที่แก้ไขแล้ว
            'Blue':   [ (np.array([90, 40, 30]), np.array([130, 255, 255])) ]
        }
        
        masks = {}
        for color_name, list_of_ranges in color_ranges.items():
            mask_parts = [cv2.inRange(normalized_hsv, lower, upper) for lower, upper in list_of_ranges]
            final_mask = mask_parts[0]
            for i in range(1, len(mask_parts)):
                final_mask = cv2.bitwise_or(final_mask, mask_parts[i])
            masks[color_name] = final_mask
        
        # (สำหรับดีบัก) หากต้องการดูว่า mask สีเขียวเห็นอะไรบ้าง ให้ uncomment บรรทัดถัดไป
        # cv2.imshow("Green Mask Debug", masks['Green'])
        
        combined_mask = cv2.bitwise_or(masks['Red'], cv2.bitwise_or(masks['Yellow'], cv2.bitwise_or(masks['Green'], masks['Blue'])))
        kernel = np.ones((5, 5), np.uint8)
        cleaned_mask = cv2.morphologyEx(cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 2000: continue
            shape = "Unknown"
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            
            circularity = (4 * np.pi * area) / (peri * peri) if peri > 0 else 0
            if circularity > 0.84:
                shape = "Circle"
            elif len(approx) == 4:
                angles = [self._get_angle(approx[i-1], approx[(i+1)%4], approx[i]) for i in range(4)]
                if all(75 <= angle <= 105 for angle in angles):
                    _, (w, h), _ = cv2.minAreaRect(cnt)
                    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
                    shape = "Square" if aspect_ratio < 1.15 else ("Rectangle_H" if w > h else "Rectangle_V")

            if shape != "Unknown":
                contour_mask = np.zeros(frame.shape[:2], dtype="uint8")
                cv2.drawContours(contour_mask, [cnt], -1, 255, -1)
                max_mean, found_color = 0, "Unknown"
                for name, mask in masks.items():
                    mean_val = cv2.mean(mask, mask=contour_mask)[0]
                    if mean_val > max_mean:
                        max_mean, found_color = mean_val, name
                if max_mean > 20:
                    detected_objects.append({"shape": shape, "color": found_color, "contour": cnt})

        return detected_objects

# ======================================================================
# ส่วนที่เหลือ (เหมือนเดิม)
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

if __name__ == '__main__':
    try:
        _ = cv2.xphoto
    except AttributeError:
        print("\n[คำแนะนำ] เพื่อการจัดการแสงที่ดีที่สุด (Auto White Balance),")
        print("โปรดติดตั้ง: pip install opencv-contrib-python\n")
    target_shape, target_color = get_target_choice()
    detector = ObjectDetector()
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
            if frame is None:
                time.sleep(0.1)
                continue
            output_frame = frame.copy()
            detected_objects = detector.detect(frame)
            for obj in detected_objects:
                is_target = (obj["shape"] == target_shape and obj["color"] == target_color)
                x, y, _, _ = cv2.boundingRect(obj["contour"])
                box_color = (0, 0, 255) if is_target else (0, 255, 0)
                thickness = 4 if is_target else 2
                cv2.drawContours(output_frame, [obj["contour"]], -1, box_color, thickness)
                label_text = f"{obj['shape']}, {obj['color']}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                text_thickness = 2
                if is_target:
                    label_text = "!!! TARGET FOUND !!!"
                    font = cv2.FONT_HERSHEY_TRIPLEX
                    font_scale = 0.7
                (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, text_thickness)
                rect_y_start = y - text_height - 10
                if rect_y_start < 0: rect_y_start = y + 10
                rect_top_left = (x, rect_y_start)
                rect_bottom_right = (x + text_width, rect_y_start + text_height + baseline)
                cv2.rectangle(output_frame, rect_top_left, rect_bottom_right, box_color, cv2.FILLED)
                text_origin = (x, rect_y_start + text_height)
                cv2.putText(output_frame, label_text, text_origin, font, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)
            cv2.imshow('Robomaster Camera Feed - Press "q" to exit', output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดร้ายแรง: {e}")
    finally:
        print("\n🔌 กำลังปิดการเชื่อมต่อ...")
        try:
            ep_robot.camera.stop_video_stream()
            ep_robot.close()
        except Exception as close_error:
            print(f"เกิดข้อผิดพลาดเล็กน้อยขณะปิดการเชื่อมต่อ: {close_error}")
        cv2.destroyAllWindows()
        print("✅ ปิดการเชื่อมต่อเรียบร้อย")
        