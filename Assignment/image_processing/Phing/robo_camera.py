import cv2
import numpy as np
import sys
import robomaster
from robomaster import robot
import time

# ======================================================================
# คลาส ObjectDetector (อัปเดตให้รองรับ 4 สี)
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

        # --- UPDATED: เพิ่มช่วงสี แดง และ เหลือง ---
        color_ranges = {
            'Red': [(np.array([0, 120, 70]), np.array([10, 255, 255])), (np.array([170, 120, 70]), np.array([180, 255, 255]))],
            'Yellow': [(np.array([22, 93, 0]), np.array([45, 255, 255]))],
            'Green': [(np.array([40, 50, 50]), np.array([90, 255, 255]))],
            'Blue': [(np.array([95, 80, 80]), np.array([130, 255, 255]))]
        }

        # สร้าง Masks สำหรับแต่ละสี
        masks = {}
        for color_name, ranges in color_ranges.items():
            mask_parts = [cv2.inRange(hsv, lower, upper) for lower, upper in ranges]
            masks[color_name] = cv2.bitwise_or(mask_parts[0], mask_parts[1]) if len(mask_parts) > 1 else mask_parts[0]

        # รวม Mask ทั้งหมดเข้าด้วยกัน
        combined_mask = masks['Red']
        for color_name in ['Yellow', 'Green', 'Blue']:
            combined_mask = cv2.bitwise_or(combined_mask, masks[color_name])

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
            if best_match_score < 0.4:
                shape = best_match_shape

            peri = cv2.arcLength(cnt, True)
            if peri > 0:
                circularity = (4 * np.pi * area) / (peri ** 2)
                if circularity > 0.82:
                    shape = "Circle" 

            if shape != "Unknown":
                # --- UPDATED: ตรรกะการระบุสีสำหรับ 4 สี ---
                contour_mask = np.zeros(frame.shape[:2], dtype="uint8")
                cv2.drawContours(contour_mask, [cnt], -1, 255, -1)
                
                found_color = "Unknown"
                for color_name, mask in masks.items():
                    # ตรวจสอบว่า contour ที่เจอ มีส่วนที่ตรงกับ mask ของสีใดมากที่สุด
                    if cv2.mean(mask, mask=contour_mask)[0] > 20: # ค่า threshold อาจปรับได้
                        found_color = color_name
                        break
                
                if found_color != "Unknown":
                    detected_objects.append({
                        "shape": shape,
                        "color": found_color,
                        "contour": cnt,
                    })
        return detected_objects

# ======================================================================
# ฟังก์ชันสำหรับรับ Input จากผู้ใช้ (อัปเดตตัวเลือก)
# ======================================================================
def get_target_choice():
    # --- UPDATED: เพิ่มตัวเลือกรูปทรงและสี ---
    VALID_SHAPES = ["Circle", "Square", "Rectangle_H", "Rectangle_V"]
    VALID_COLORS = ["Red", "Yellow", "Green", "Blue"]

    print("\n--- 🎯 กำหนดลักษณะของเป้าหมาย ---")
    
    while True:
        prompt = f"เลือกรูปทรง ({'/'.join(VALID_SHAPES)}): "
        shape = input(prompt).strip().title()
        if shape in VALID_SHAPES:
            break
        print("⚠️ รูปทรงไม่ถูกต้อง, กรุณาลองใหม่")

    while True:
        prompt = f"เลือกสี ({'/'.join(VALID_COLORS)}): "
        color = input(prompt).strip().title()
        if color in VALID_COLORS:
            break
        print("⚠️ สีไม่ถูกต้อง, กรุณาลองใหม่")

    print(f"✅ เป้าหมายคือ: {shape} สี {color}. เริ่มการค้นหา!")
    return shape, color

# ======================================================================
# Main (อัปเดต Path ของ Template)
# ======================================================================
if __name__ == '__main__':
    target_shape, target_color = get_target_choice()

    # --- UPDATED: เพิ่ม Path สำหรับเทมเพลตใหม่ ---
    template_files = {
        "Circle": "./Assignment/image_processing/template/circle1.png",
        "Square": "./Assignment/image_processing/template/square.png",
        "Rectangle_H": "./Assignment/image_processing/template/rec_h.png",
        "Rectangle_V": "./Assignment/image_processing/template/rec_v.png",
    }
    detector = ObjectDetector(template_paths=template_files)

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
                print("...กำลังรอรับภาพจากหุ่นยนต์...")
                time.sleep(0.1)
                continue
            
            output_frame = frame.copy()
            detected_objects = detector.detect(frame)

            for obj in detected_objects:
                is_target = (obj["shape"] == target_shape and obj["color"] == target_color)
                x, y, _, _ = cv2.boundingRect(obj["contour"])

                # กำหนดสีของกรอบและป้ายชื่อ
                box_color = (0, 0, 255) if is_target else (0, 255, 0)
                thickness = 4 if is_target else 2
                
                cv2.drawContours(output_frame, [obj["contour"]], -1, box_color, thickness)
                
                if is_target:
                    label = "!!! TARGET FOUND !!!"
                    cv2.putText(output_frame, label, (x, y - 15), 
                                cv2.FONT_HERSHEY_TRIPLEX, 0.7, box_color, 2)
                else:
                    label = f"{obj['shape']}, {obj['color']}"
                    cv2.putText(output_frame, label, (x, y - 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('Robomaster Camera Feed - Press "q" to exit', output_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("ได้รับคำสั่งให้ออก...")
                break
                
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดร้ายแรง: {e}")
        
    finally:
        print("\n🔌 กำลังปิดการเชื่อมต่อ...")
        if ep_robot._sdk_connection is not None:
             ep_robot.camera.stop_video_stream()
             ep_robot.close()
        cv2.destroyAllWindows()
        print("✅ ปิดการเชื่อมต่อเรียบร้อย")