import cv2
import numpy as np
import time
from robomaster import robot, camera

# =========================
# ฟังก์ชันสร้าง Trackbar สำหรับปรับ HSV
# =========================
def setup_hsv_trackbars():
    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("H_min", "Trackbars", 140, 179, lambda x: None)
    cv2.createTrackbar("H_max", "Trackbars", 170, 179, lambda x: None)
    cv2.createTrackbar("S_min", "Trackbars", 80, 255, lambda x: None)
    cv2.createTrackbar("S_max", "Trackbars", 255, 255, lambda x: None)
    cv2.createTrackbar("V_min", "Trackbars", 120, 255, lambda x: None)
    cv2.createTrackbar("V_max", "Trackbars", 255, 255, lambda x: None)

def get_trackbar_values():
    h_min = cv2.getTrackbarPos("H_min", "Trackbars")
    h_max = cv2.getTrackbarPos("H_max", "Trackbars")
    s_min = cv2.getTrackbarPos("S_min", "Trackbars")
    s_max = cv2.getTrackbarPos("S_max", "Trackbars")
    v_min = cv2.getTrackbarPos("V_min", "Trackbars")
    v_max = cv2.getTrackbarPos("V_max", "Trackbars")
    return np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max])

# =========================
# ฟังก์ชันสร้าง mask สีชมพู (ปรับแล้ว)
# =========================
def create_pink_mask(img_rgb, lower_pink, upper_pink):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_pink, upper_pink)

    # ถ้า mask ว่างหรือเต็มเกินไป → return ไปเลยกันค้าง
    nonzero = cv2.countNonZero(mask)
    if nonzero < 50 or nonzero > mask.size * 0.9:
        return np.zeros_like(mask)

    # ลบ noise และเติมรูว่าง
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ทำให้ขอบคมขึ้น
    mask = cv2.medianBlur(mask, 3)
    edges = cv2.Canny(mask, 50, 150)
    mask = cv2.bitwise_or(mask, edges)

    return mask

# =========================
# โปรแกรมหลัก
# =========================
def main():
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_720P)

    setup_hsv_trackbars()

    try:
        while True:
            frame = ep_camera.read_cv2_image()
            if frame is None:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # อ่านค่า HSV สดๆ จาก Trackbar
            lower_pink, upper_pink = get_trackbar_values()
            mask = create_pink_mask(frame_rgb, lower_pink, upper_pink)

            # แสดงผล
            cv2.imshow("Camera", frame)
            cv2.imshow("Pink Mask", mask)

            # กด q เพื่อออก
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    finally:
        ep_camera.stop_video_stream()
        ep_robot.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
