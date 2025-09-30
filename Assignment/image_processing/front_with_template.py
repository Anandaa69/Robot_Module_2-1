import cv2
import numpy as np
import sys

# ======================================================================
# คลาส ObjectDetector ที่ปรับปรุงให้ใช้ Template Matching
# ======================================================================
class ObjectDetector:
    def __init__(self, template_paths):
        """
        Constructor ของคลาส, ทำการโหลดและประมวลผลภาพ Template ตอนเริ่มต้น
        """
        print("🖼️  กำลังโหลดและประมวลผลภาพ Templates...")
        self.templates = self._load_templates(template_paths)
        if not self.templates:
            # ถ้าโหลด template ไม่สำเร็จ ให้หยุดการทำงาน
            print("❌ ไม่สามารถโหลดไฟล์ Template ได้, กรุณาตรวจสอบว่าไฟล์ภาพอยู่ในโฟลเดอร์ที่ถูกต้อง")
            sys.exit(1) # ออกจากโปรแกรม
        print(f"✅ โหลด Templates สำเร็จ: {list(self.templates.keys())}")


    def _load_templates(self, template_paths):
        """
        ฟังก์ชันสำหรับโหลดภาพ template, แปลงเป็น grayscale, และดึง contour หลัก
        """
        processed_templates = {}
        for shape_name, path in template_paths.items():
            # อ่านภาพ template แบบ grayscale
            template_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if template_img is None:
                print(f"⚠️ คำเตือน: ไม่พบไฟล์ template ที่: {path}")
                continue
            
            # หา contours ของรูปทรงใน template
            # เราคาดว่าในภาพ template จะมีรูปทรงสีขาวอันเดียวบนพื้นดำ
            contours, _ = cv2.findContours(template_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # เลือก contour ที่ใหญ่ที่สุดเป็นตัวแทนของ template
                template_contour = max(contours, key=cv2.contourArea)
                processed_templates[shape_name] = template_contour
        
        return processed_templates

    def detect(self, frame):
        frame = cv2.flip(frame, 1)
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

            # --- วิเคราะห์รูปทรงด้วย Template Matching ---
            best_match_score = float('inf') # ตั้งค่าเริ่มต้นให้สูงมากๆ
            best_match_shape = "Unknown"

            # เปรียบเทียบ contour ที่เจอในวิดีโอ (cnt) กับ contour ของ template ทุกอัน
            for shape_name, template_cnt in self.templates.items():
                # cv2.matchShapes คำนวณ "ความแตกต่าง" ระหว่าง 2 รูปทรง
                # ค่ายิ่งน้อยแปลว่ายิ่งเหมือนกัน
                score = cv2.matchShapes(cnt, template_cnt, cv2.CONTOURS_MATCH_I1, 0.0)
                
                if score < best_match_score:
                    best_match_score = score
                    best_match_shape = shape_name

            # ตั้งค่าเกณฑ์ (threshold) ถ้าค่าความแตกต่างน้อยกว่านี้ ถึงจะยอมรับว่าเป็นรูปทรงนั้น
            # คุณสามารถปรับค่านี้ได้ ถ้า 0.4 เข้มงวดไปให้เพิ่ม, ถ้าหละหลวมไปให้ลด
            match_threshold = 0.4 
            if best_match_score < match_threshold:
                shape = best_match_shape
            else:
                shape = "Unknown"

            if shape != "Unknown":
                # --- วิเคราะห์สี (เหมือนเดิม) ---
                mask = np.zeros(frame.shape[:2], dtype="uint8")
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                
                if cv2.mean(mask_green, mask=mask)[0] > 1:
                    color = "Green"
                else:
                    color = "Blue"

                # --- วาดผลลัพธ์ ---
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.drawContours(output, [cnt], -1, (0, 255, 0), 3)
                cv2.putText(output, f"{shape}, {color} (Score: {best_match_score:.2f})", 
                            (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return output

# ======================================================================
# Main (สำหรับใช้กับเว็บแคม)
# ======================================================================
if __name__ == '__main__':
    # กำหนด path ของไฟล์ template
    template_files = {
        "Circle": "./Assignment/image_processing/templ/circle.jpg",
        "Square": "./Assignment/image_processing/templ/rectangle.jpg",
        "Triangle": "./Assignment/image_processing/templ/triangle.jpg"
    }

    # สร้าง instance ของ detector และส่ง path ของ templates เข้าไป
    detector = ObjectDetector(template_paths=template_files)

    print("\n💻 กำลังเปิดกล้องเว็บแคม...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ ไม่สามารถเปิดกล้องเว็บแคมได้")
        exit()

    print("✅ เปิดกล้องเว็บแคมสำเร็จ. กด 'q' เพื่อออกจากโปรแกรม")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            processed_frame = detector.detect(frame)
            cv2.imshow('Live Webcam Feed - Press "q" to exit', processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("🔌 ปิดการเชื่อมต่อกล้องเรียบร้อย")