import time
import cv2
from robomaster import robot
from robomaster import camera


if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")


    ep_camera = ep_robot.camera
    ep_gimbal = ep_robot.gimbal
    ep_gimbal.recenter().wait_for_completed()
    ep_gimbal.move(pitch=-5,yaw=0).wait_for_completed()
    # เริ่มต้น video stream (ไม่ต้อง display=True เพราะเราจะใช้ OpenCV แสดงผล)
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_720P)

    # ใช้ cv2 window เปิดภาพ
    cv2.namedWindow("RoboMaster Camera", cv2.WINDOW_NORMAL)

    while True:
        img = ep_camera.read_cv2_image(strategy="newest", timeout=1)  # ดึง frame ล่าสุด
        if img is not None:
            cv2.imshow("RoboMaster Camera", img)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):  # ถ้ากด c → แคปภาพ
            filename = f"capture_{int(time.time())}.jpg"
            cv2.imwrite(filename, img)
            print(f"📸 Saved {filename}")

        elif key == ord('q'):  # กด q เพื่อออก
            break

    # ปิดการทำงาน
    ep_camera.stop_video_stream()
    cv2.destroyAllWindows()
    ep_robot.close()
