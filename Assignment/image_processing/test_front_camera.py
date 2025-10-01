import cv2
import numpy as np

# ======================================================================
# สร้าง Template Shapes (ใช้ contour)
# ======================================================================
def create_template_shapes():
    templates = {}

    # สามเหลี่ยม
    triangle = np.array([[[100, 200]], [[200, 200]], [[150, 100]]])
    templates["Triangle"] = triangle

    # สี่เหลี่ยม
    square = np.array([[[100, 100]], [[200, 100]], [[200, 200]], [[100, 200]]])
    templates["Square"] = square

    # วงกลม
    circle_img = np.zeros((300, 300), dtype="uint8")
    cv2.circle(circle_img, (150, 150), 80, 255, -1)
    cnts, _ = cv2.findContours(circle_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    templates["Circle"] = cnts[0]

    return templates

# ======================================================================
# คลาส ObjectDetector
# ======================================================================
class ObjectDetector:
    def __init__(self, templates):
        self.templates = templates

    def detect(self, frame):
        frame = cv2.flip(frame, 1)
        output = frame.copy()

        # 1. แปลงภาพเป็น HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 2. กำหนดช่วงสี
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([90, 255, 255])
        lower_blue = np.array([95, 80, 80])
        upper_blue = np.array([130, 255, 255])

        # 3. สร้าง mask
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        combined_mask = cv2.bitwise_or(mask_green, mask_blue)

        # 4. หา contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 5. วิเคราะห์แต่ละ contour
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1000:
                continue

            # --- Template Matching ---
            best_shape = "Unknown"
            best_score = 1.0
            for shape_name, tmpl in self.templates.items():
                score = cv2.matchShapes(cnt, tmpl, 1, 0.0)
                if score < best_score:
                    best_score = score
                    best_shape = shape_name

            shape = best_shape

            # --- วิเคราะห์สี ---
            mask = np.zeros(frame.shape[:2], dtype="uint8")
            cv2.drawContours(mask, [cnt], -1, 255, -1)

            if cv2.mean(mask_green, mask=mask)[0] > 1:
                color = "Green"
            elif cv2.mean(mask_blue, mask=mask)[0] > 1:
                color = "Blue"
            else:
                color = "Unknown"

            # --- วาดผลลัพธ์ ---
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.drawContours(output, [cnt], -1, (0, 255, 0), 3)
            cv2.putText(output, f"{shape}, {color}", (x, y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return output

# ======================================================================
# Main (สำหรับใช้กับเว็บแคม)
# ======================================================================
if __name__ == '__main__':
    print("💻 กำลังเปิดกล้องเว็บแคม...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ ไม่สามารถเปิดกล้องเว็บแคมได้")
        exit()

    print("✅ เปิดกล้องเว็บแคมสำเร็จ. กด 'q' เพื่อออกจากโปรแกรม")
    templates = create_template_shapes()
    detector = ObjectDetector(templates)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("⚠️ ไม่สามารถอ่านเฟรมจากกล้องได้")
                break

            processed_frame = detector.detect(frame)
            cv2.imshow('Live Webcam Feed - Press "q" to exit', processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("...กำลังปิดโปรแกรม")
                break

    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดร้ายแรง: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("📷 Resolution:", cap.get(cv2.CAP_PROP_FRAME_WIDTH), "x", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("🔌 ปิดการเชื่อมต่อกล้องเรียบร้อย")
