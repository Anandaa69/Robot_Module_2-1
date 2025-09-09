# calibrate_k.py
import cv2
import numpy as np
from robomaster import robot
from robomaster import camera
import time

# --- นำฟังก์ชันที่จำเป็นมาจากโค้ดเดิม ---
def create_pink_mask(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_pink = np.array([120, 10, 100])
    upper_pink = np.array([170, 100, 200])
    mask = cv2.inRange(hsv, lower_pink, upper_pink)
    mask = cv2.medianBlur(mask, 5)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

# --- ค่าคงที่สำหรับการปรับเทียบ ---
# 1. วัดระยะทางจริงจากหน้าเลนส์กล้องไปยังวัตถุ แล้วใส่ค่าที่นี่ (หน่วยเป็น cm)
Z_KNOWN_DISTANCE_CM = 50.0 

# 2. ใส่ขนาดจริงของวัตถุ (หน่วยเป็น cm)
REAL_WIDTH_CM = 24.2
REAL_HEIGHT_CM = 13.9

# --- ส่วนการทำงานหลัก ---
def main():
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_720P)
    
    print(f"!!! Calibration Mode !!!")
    print(f"Place the object at a known distance of {Z_KNOWN_DISTANCE_CM} cm from the camera lens.")
    print("Press 'c' to capture pixel dimensions and calculate K.")
    print("Press 'q' to quit.")

    try:
        while True:
            frame = ep_camera.read_cv2_image()
            if frame is None:
                continue

            # ค้นหาวัตถุด้วย Mask และ Contour (เหมือนโค้ดหลัก)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pink_mask = create_pink_mask(frame_rgb)
            contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                main_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(main_contour)

                # วาดกรอบและแสดงค่า w, h ที่วัดได้
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(frame, f"Width (px): {w}", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, f"Height (px): {h}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # เมื่อผู้ใช้กด 'c' ให้คำนวณและแสดงผลค่า K
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    if w > 0 and h > 0:
                        # ใช้สูตร K = (Z * P) / F
                        calculated_kx = (Z_KNOWN_DISTANCE_CM * w) / REAL_WIDTH_CM
                        calculated_ky = (Z_KNOWN_DISTANCE_CM * h) / REAL_HEIGHT_CM
                        
                        print("\n" + "="*30)
                        print("!!! CALIBRATION RESULTS !!!")
                        print(f"  - Known Distance (Z): {Z_KNOWN_DISTANCE_CM} cm")
                        print(f"  - Real Width (F_x): {REAL_WIDTH_CM} cm")
                        print(f"  - Real Height (F_y): {REAL_HEIGHT_CM} cm")
                        print(f"  - Measured Pixel Width (P_x): {w} px")
                        print(f"  - Measured Pixel Height (P_y): {h} px")
                        print("-" * 30)
                        print(f"==> Calculated K_X = {calculated_kx:.4f}")
                        print(f"==> Calculated K_Y = {calculated_ky:.4f}")
                        print("="*30)
                        print("Copy these K_X and K_Y values into your main detect_pid.py script.\n")

            else:
                key = cv2.waitKey(1) & 0xFF

            cv2.imshow("Calibration", frame)
            if key == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()
        ep_camera.stop_video_stream()
        ep_robot.close()

if __name__ == '__main__':
    main()