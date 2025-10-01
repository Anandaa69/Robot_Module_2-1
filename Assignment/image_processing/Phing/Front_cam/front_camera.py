import cv2
import numpy as np

# ======================================================================
# คลาส ObjectDetector ที่ปรับปรุงใหม่
# ======================================================================
class ObjectDetector:
    def detect(self, frame):
        # พลิกภาพซ้าย-ขวา (เหมือนส่องกระจก) เพื่อให้ใช้งานง่ายขึ้น
        frame = cv2.flip(frame, 1)
        output = frame.copy()

        # 1. แปลงภาพ BGR เป็น HSV color space
        # HSV เหมาะกับการแยกสีมากกว่า BGR เพราะไม่ไวต่อความสว่างของแสงเท่า
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 2. กำหนดช่วงของ "สีเขียว" และ "สีน้ำเงิน" ในระบบ HSV
        # ค่าเหล่านี้อาจจะต้องปรับจูนเล็กน้อยเพื่อให้เข้ากับสภาพแสงในห้องของคุณ
        # รูปแบบคือ [Hue (สี), Saturation (ความสด), Value (ความสว่าง)]
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([90, 255, 255])

        lower_blue = np.array([95, 80, 80])
        upper_blue = np.array([130, 255, 255])

        # 3. สร้าง Mask: หาพิกเซลที่อยู่ในช่วงสีที่กำหนด
        # ผลลัพธ์คือภาพขาว-ดำ ที่พิกเซลสีขาวคือบริเวณที่ตรงกับสีที่เราสนใจ
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        # รวม mask ทั้งสองสีเข้าด้วยกัน
        combined_mask = cv2.bitwise_or(mask_green, mask_blue)

        # 4. ค้นหา Contours (เส้นรอบรูป) จาก Mask ที่ได้
        # ตอนนี้เราจะหาเส้นรอบรูปจาก "หย่อมสี" เท่านั้น ไม่ใช่จากขอบทั้งหมดในภาพ
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 5. วนลูปเพื่อวิเคราะห์แต่ละ Contour ที่เจอ
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # กรองวัตถุที่เล็กเกินไปออก
            if area < 1000:
                continue

            # --- วิเคราะห์รูปทรง ---
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
            num_vertices = len(approx)

            shape = "Unknown"
            if num_vertices == 3:
                shape = "Triangle"
            elif num_vertices == 4:
                # สามารถคำนวณ aspect ratio เพื่อแยกสี่เหลี่ยมจัตุรัส/ผืนผ้าได้ แต่โจทย์คือสี่เหลี่ยม
                shape = "Square"
            else:
                # ใช้วิธีวัด "ความเป็นวงกลม" (Circularity) ซึ่งแม่นยำกว่าการนับจุด
                # วงกลมที่สมบูรณ์จะมีค่านี้เท่ากับ 1
                circularity = (4 * np.pi * area) / (peri ** 2)
                if circularity > 0.80:
                    shape = "Circle"

            if shape != "Unknown":
                # --- วิเคราะห์สี ---
                # เราสามารถรู้สีได้จากว่า contour นี้มาจาก mask สีอะไร
                # โดยการตรวจสอบค่าเฉลี่ยของสีในพื้นที่ contour นั้นๆ
                mask = np.zeros(frame.shape[:2], dtype="uint8")
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                
                # ตรวจสอบว่าอยู่ใน mask สีเขียวหรือสีน้ำเงินมากกว่ากัน
                if cv2.mean(mask_green, mask=mask)[0] > 1:
                    color = "Green"
                elif cv2.mean(mask_blue, mask=mask)[0] > 1:
                    color = "Blue"
                else:
                    color = "Unknown"

                # --- วาดผลลัพธ์ ---
                x, y, w, h = cv2.boundingRect(approx)
                cv2.drawContours(output, [approx], -1, (0, 255, 0), 3)
                cv2.putText(output, f"{shape}, {color}", (x, y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return output

# ======================================================================
# Main (สำหรับใช้กับเว็บแคม)
# ======================================================================
if __name__ == '__main__':
    print("💻 กำลังเปิดกล้องเว็บแคม...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ ไม่สามารถเปิดกล้องเว็บแคมได้")
        exit()

    print("✅ เปิดกล้องเว็บแคมสำเร็จ. กด 'q' เพื่อออกจากโปรแกรม")
    detector = ObjectDetector()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("⚠️ ไม่สามารถอ่านเฟรมจากกล้องได้")
                break

            # เรียกใช้ฟังก์ชัน detect ที่ปรับปรุงใหม่
            processed_frame = detector.detect(frame)

            # แสดงผลลัพธ์
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
        print("🔌 ปิดการเชื่อมต่อกล้องเรียบร้อย")