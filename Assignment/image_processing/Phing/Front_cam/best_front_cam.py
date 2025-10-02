import cv2
import numpy as np
import sys

# ======================================================================
# คลาส ObjectDetector ที่เพิ่มการกำจัด Noise
# ======================================================================
class ObjectDetector:
    def __init__(self, template_paths):
        print("🖼️  กำลังโหลดและประมวลผลภาพ Templates...")
        self.templates = self._load_templates(template_paths)
        if not self.templates:
            print("❌ ไม่สามารถโหลดไฟล์ Template ได้, กรุณาตรวจสอบว่าไฟล์ภาพอยู่ในโฟลเดอร์ที่ถูกต้อง")
            sys.exit(1)
        print(f"✅ โหลด Templates สำเร็จ: {list(self.templates.keys())}")

    def _load_templates(self, template_paths):
        processed_templates = {}
        for shape_name, path in template_paths.items():
            template_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if template_img is None:
                print(f"⚠️ คำเตือน: ไม่พบไฟล์ template ที่: {path}")
                continue
            
            blurred = cv2.GaussianBlur(template_img, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                template_contour = max(contours, key=cv2.contourArea)
                processed_templates[shape_name] = template_contour
        
        return processed_templates

    def detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detected_objects = []

        # ช่วงสี
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([90, 255, 255])
        lower_blue = np.array([95, 80, 80])
        upper_blue = np.array([130, 255, 255])

        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        combined_mask = cv2.bitwise_or(mask_green, mask_blue)

        # กำจัด Noise
        kernel = np.ones((7, 7), np.uint8)
        opened_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 2500:
                continue

            best_match_score = float('inf')
            best_match_shape = "Unknown"

            for shape_name, template_cnt in self.templates.items():
                score = cv2.matchShapes(cnt, template_cnt, cv2.CONTOURS_MATCH_I1, 0.0)
                if score < best_match_score:
                    best_match_score = score
                    best_match_shape = shape_name

            shape = "Unknown"
            match_threshold = 0.4
            if best_match_score < match_threshold:
                shape = best_match_shape

            peri = cv2.arcLength(cnt, True)
            if peri > 0:
                circularity = (4 * np.pi * area) / (peri ** 2)
                if circularity > 0.82 and shape != "Circle":
                    shape = "Circle" 

            if shape != "Unknown":
                mask = np.zeros(frame.shape[:2], dtype="uint8")
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                
                color = "Green" if cv2.mean(mask_green, mask=mask)[0] > 1 else "Blue"

                detected_objects.append({
                    "shape": shape,
                    "color": color,
                    "contour": cnt,
                })

        return detected_objects

# ======================================================================
# ฟังก์ชันสำหรับรับ Input จากผู้ใช้
# *** ย้ายมาไว้ก่อน __main__ ***
# ======================================================================
def get_target_choice():
    VALID_SHAPES = ["Circle", "Square", "Triangle"]
    VALID_COLORS = ["Green", "Blue"]

    print("\n--- 🎯 กำหนดลักษณะของเป้าหมาย ---")
    
    while True:
        prompt = f"เลือกรูปทรงของเป้าหมาย ({'/'.join(VALID_SHAPES)}): "
        shape = input(prompt).strip().title()
        if shape in VALID_SHAPES:
            break
        print("⚠️ รูปทรงไม่ถูกต้อง, กรุณาลองใหม่")

    while True:
        prompt = f"เลือกสีของเป้าหมาย ({'/'.join(VALID_COLORS)}): "
        color = input(prompt).strip().title()
        if color in VALID_COLORS:
            break
        print("⚠️ สีไม่ถูกต้อง, กรุณาลองใหม่")

    print(f"✅ เป้าหมายคือ: {shape} สี {color}. เริ่มการค้นหา!")
    return shape, color

# ======================================================================
# Main (ส่วนนี้จะทำงานหลังจากโปรแกรมรู้จักฟังก์ชัน get_target_choice แล้ว)
# ======================================================================
if __name__ == '__main__':
    target_shape, target_color = get_target_choice()

    template_files = {
        "Circle": "./Assignment/image_processing/template/circle1.png",
        "Square": "./Assignment/image_processing/template/rectangle1.png",
        "Triangle": "./Assignment/image_processing/template/triangle1.png"
    }
    detector = ObjectDetector(template_paths=template_files)

    print("\n💻 กำลังเปิดกล้องเว็บแคม...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ ไม่สามารถเปิดกล้องเว็บแคมได้")
        exit()
    print("✅ เปิดกล้องเว็บแคมสำเร็จ. กด 'q' เพื่อออกจากโปรแกรม")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            
            frame = cv2.flip(frame, 1)
            output_frame = frame.copy()

            detected_objects = detector.detect(frame)

            for obj in detected_objects:
                is_target = (obj["shape"] == target_shape and obj["color"] == target_color)
                x, y, _, _ = cv2.boundingRect(obj["contour"])

                if is_target:
                    cv2.drawContours(output_frame, [obj["contour"]], -1, (0, 0, 255), 4)
                    cv2.putText(output_frame, "!!! TARGET FOUND !!!", (x, y - 15), 
                                cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.drawContours(output_frame, [obj["contour"]], -1, (0, 255, 0), 3)
                    label = f"{obj['shape']}, {obj['color']}"
                    cv2.putText(output_frame, label, (x, y - 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('Live Webcam Feed - Press "q" to exit', output_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("🔌 ปิดการเชื่อมต่อกล้องเรียบร้อย")