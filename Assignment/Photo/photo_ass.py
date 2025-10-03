import time
import cv2
import os
from robomaster import robot, camera

# ==== ตั้งค่า path และชื่อไฟล์ ====
save_path = r"D:\downsyndrome\year2_1\Robot_Module_2-1\Assignment\Photo"
filename_prefix = "fontss_changeangle2_2_node"   # ตัวแปรสำหรับเปลี่ยนชื่อไฟล์

if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_camera = ep_robot.camera
    ep_gimbal = ep_robot.gimbal
    ep_gimbal.recenter().wait_for_completed()
    ep_gimbal.move(pitch=0, yaw=0).wait_for_completed()

    # เริ่ม stream
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_720P)
    cv2.namedWindow("RoboMaster Camera", cv2.WINDOW_NORMAL)

    while True:
        img = ep_camera.read_cv2_image(strategy="newest", timeout=1)
        if img is not None:
            cv2.imshow("RoboMaster Camera", img)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):  # ถ่ายภาพ
            timestamp = int(time.time())
            filename = f"{filename_prefix}.jpg"
            filepath = os.path.join(save_path, filename)

            cv2.imwrite(filepath, img)
            print(f"📸 Saved {filepath}")

            # แสดงรูปที่ถ่ายอีกหน้าต่างหนึ่ง
            cv2.imshow("Captured Photo", img)

        elif key == ord('q'):  # ออก
            break

    # ปิดการทำงาน
    ep_camera.stop_video_stream()
    cv2.destroyAllWindows()
    ep_robot.close()
