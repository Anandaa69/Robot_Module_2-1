import cv2
import numpy as np
import sys
import robomaster
from robomaster import robot, camera

# ======================================================================
# คลาส ObjectDetector (ใช้ Template Matching)
# *** ส่วนนี้ไม่ต้องแก้ไขใดๆ ทำงานได้ทั้งกับเว็บแคมและ Robomaster ***
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
            contours, _ = cv2.findContours(template_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                template_contour = max(contours, key=cv2.contourArea)
                processed_templates[shape_name] = template_contour
        return processed_templates

    def detect(self, frame):
        # ไม่มีการ flip ภาพที่นี่ เพราะเราจะรับภาพจากหุ่นยนต์มาตรงๆ
        output = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # ช่วงสี (เหมือนเดิม)
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([90, 255, 255])
        lower_blue = np.array([95, 80, 80])
        upper_blue = np.array([130, 255, 255])

        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        combined_mask = cv2.bitwise_or(mask_green, mask_blue)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1000:
                continue

            best_match_score = float('inf')
            best_match_shape = "Unknown"

            for shape_name, template_cnt in self.templates.items():
                score = cv2.matchShapes(cnt, template_cnt, cv2.CONTOURS_MATCH_I1, 0.0)
                if score < best_match_score:
                    best_match_score = score
                    best_match_shape = shape_name

            match_threshold = 0.4
            if best_match_score < match_threshold:
                shape = best_match_shape
            else:
                shape = "Unknown"

            if shape != "Unknown":
                mask = np.zeros(frame.shape[:2], dtype="uint8")
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                
                if cv2.mean(mask_green, mask=mask)[0] > 1:
                    color = "Green"
                else:
                    color = "Blue"

                x, y, w, h = cv2.boundingRect(cnt)
                cv2.drawContours(output, [cnt], -1, (0, 255, 0), 3)
                cv2.putText(output, f"{shape}, {color} (Score: {best_match_score:.2f})", 
                            (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return output

# ======================================================================
# Main (ปรับแก้สำหรับใช้กับกล้อง Robomaster)
# ======================================================================
if __name__ == '__main__':
    # กำหนด path ของไฟล์ template (เหมือนเดิม)
    template_files = {
        "Circle": "./Assignment/image_processing/templ/circle.jpg",
        "Square": "./Assignment/image_processing/templ/rectangle.jpg",
        "Triangle": "./Assignment/image_processing/templ/triangle.jpg"
    }

    # สร้าง instance ของ detector และส่ง path ของ templates เข้าไป
    detector = ObjectDetector(template_paths=template_files)

    ep_robot = None
    try:
        # --- ส่วนที่ 1: การเชื่อมต่อกับหุ่นยนต์ ---
        print("\n🤖 กำลังเชื่อมต่อกับหุ่นยนต์ Robomaster...")
        ep_robot = robot.Robot()
        ep_robot.initialize(conn_type="ap") # หรือ "sta" หากเชื่อมต่อผ่าน router

        ep_camera = ep_robot.camera
        ep_camera.start_video_stream(display=False, resolution=camera.STREAM_720P)
        print("✅ เปิดสตรีมวิดีโอจากกล้องหุ่นยนต์แล้ว กด 'q' เพื่อออกจากโปรแกรม")

        # --- ส่วนที่ 2: ลูปการทำงานหลัก ---
        while True:
            # อ่านเฟรมภาพจากกล้องหุ่นยนต์
            frame = ep_camera.read_cv2_image()
            if frame is None:
                # ถ้ายังไม่ได้รับเฟรม ให้รอสักครู่แล้ววนลูปใหม่
                continue

            # เรียกใช้ detector เพื่อประมวลผล (ส่วนนี้เหมือนเดิม)
            processed_frame = detector.detect(frame)

            # แสดงผลลัพธ์
            cv2.imshow('Live Robomaster Feed - Press "q" to exit', processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("...กำลังปิดโปรแกรม")
                break
                
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดร้ายแรง: {e}")
        
    finally:
        # --- ส่วนที่ 3: การปิดการเชื่อมต่อ ---
        cv2.destroyAllWindows()
        if ep_robot is not None:
            ep_camera.stop_video_stream()
            ep_robot.close()
            print("🔌 ปิดการเชื่อมต่อกับหุ่นยนต์เรียบร้อย")