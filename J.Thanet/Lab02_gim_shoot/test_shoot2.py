import cv2
import time
import robomaster
from robomaster import robot
from robomaster import blaster
from robomaster import camera
from robomaster import vision
from robomaster import gimbal


markers = []
window_size = 720
height_window = window_size
weight_window = 1280


# ตัวแปรสำหรับ PID
Kp_yaw, Ki_yaw, Kd_yaw = 130.5, 0.1, 5.00
Kp_pitch, Ki_pitch, Kd_pitch = 130.5, 0.1, 5.00

# ตัวแปรสำหรับเก็บค่าสถานะของ PID
integral_x = 0
error_x_pre = 0
integral_y = 0
error_y_pre = 0


class MarkerInfo:
    def __init__(self, x, y, w, h, info):
        self._x = x
        self._y = y
        self._w = w
        self._h = h
        self._info = info

    @property  # พิกัดมุมบนซ้าย
    def pt1(self):
        return int((self._x - self._w / 2) * weight_window), int((self._y - self._h / 2) * height_window)

    @property  # พิกัดมุมขวาล่าง
    def pt2(self):
        return int((self._x + self._w / 2) * weight_window), int((self._y + self._h / 2) * height_window)

    @property  # ตรงกลางของ marker
    def center(self):
        return int(self._x * weight_window), int(self._y * height_window)

    @property
    def text(self):
        return self._info

    @property
    def center_x(self):  # สำหรับใช้ในการเรียงลำดับ
        return self._x

# Global variable to store the sorted markers
sorted_markers_by_x = []


def on_detect_marker(marker_info):
    global markers, sorted_markers_by_x
    markers.clear()
    for i in range(len(marker_info)):
        x, y, w, h, info = marker_info[i]
        markers.append(MarkerInfo(x, y, w, h, info))
    
    # เรียงลำดับมาร์คเกอร์ตามแกน X
    sorted_markers_by_x = sorted(markers, key=lambda m: m.center_x)


def track_and_fire_with_pid(ep_gimbal, ep_blaster, target_marker):
    """
    ใช้ PID เพื่อติดตามและยิง marker ที่กำหนด พร้อมอัปเดตตำแหน่ง marker ซ้ำตลอดเวลา
    """
    global integral_x, error_x_pre, integral_y, error_y_pre

    integral_x, error_x_pre = 0, 0
    integral_y, error_y_pre = 0, 0

    print(f"-> กำลังติดตามและเล็งยิง Marker: {target_marker.text}")

    timeout_sec = 10  # จำกัดเวลาติดตาม 10 วินาที
    start_time = time.time()

    while time.time() - start_time < timeout_sec:
        try:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=5)

            # หา marker ปัจจุบันใหม่ (ไม่ใช้ตำแหน่งเดิม)
            current_target = next((m for m in markers if m.text == target_marker.text), None)

            if current_target:
                target_x, target_y = current_target.center

                # PID
                error_x = target_x - (weight_window / 2)
                error_y = (height_window / 2) - target_y

                integral_x += error_x
                derivative_x = error_x - error_x_pre
                output_yaw = Kp_yaw * error_x + Ki_yaw * integral_x + Kd_yaw * derivative_x
                error_x_pre = error_x

                integral_y += error_y
                derivative_y = error_y - error_y_pre
                output_pitch = Kp_pitch * error_y + Ki_pitch * integral_y + Kd_pitch * derivative_y
                error_y_pre = error_y

                # Limit speed
                max_speed = 50
                output_yaw = max(-max_speed, min(max_speed, output_yaw))
                output_pitch = max(-max_speed, min(max_speed, output_pitch))

                ep_gimbal.drive_speed(yaw_speed=output_yaw, pitch_speed=output_pitch)

                # แสดงผลในหน้าจอ (optional)
                cv2.rectangle(img, current_target.pt1, current_target.pt2, (0, 255, 0), 2)
                cv2.putText(img, current_target.text, current_target.center, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.imshow("Tracking", img)
                cv2.waitKey(1)

                # ยิงเมื่ออยู่กลางเฟรม
                if abs(error_x) < 20 and abs(error_y) < 20:
                    print("--> Marker อยู่ตรงกลางแล้ว! ทำการยิง")
                    ep_blaster.fire(fire_type=blaster.INFRARED_FIRE, times=3)
                    time.sleep(2)
                    ep_gimbal.drive_speed(yaw_speed=0, pitch_speed=0)
                    cv2.destroyWindow("Tracking")
                    return True
            else:
                print("--> Marker หายจากเฟรม รอให้กลับมา...")
                ep_gimbal.drive_speed(yaw_speed=0, pitch_speed=0)
                time.sleep(0.2)
        except Exception as e:
            print(f"เกิดข้อผิดพลาด: {e}")
            continue

    print("--> ไม่สามารถเล็ง marker ได้ในเวลาที่กำหนด")
    ep_gimbal.drive_speed(yaw_speed=0, pitch_speed=0)
    cv2.destroyWindow("Tracking")
    return False


if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_blaster = ep_robot.blaster
    ep_gimbal = ep_robot.gimbal
    ep_vision = ep_robot.vision
    ep_camera = ep_robot.camera

    ep_gimbal.moveto(pitch=0, yaw=0).wait_for_completed()
    ep_camera.start_video_stream(display=False)
    time.sleep(10)

    result = ep_vision.sub_detect_info(name="marker", callback=on_detect_marker)
    print("เริ่มการตรวจจับ marker และเตรียมพร้อม...")

    # วนลูปเพื่อรอจนกว่าจะตรวจพบ marker ครบ 3 ตัว
    for i in range(200):
        img = ep_camera.read_cv2_image(strategy="newest", timeout=5.0)
        if len(markers) >= 3:
            print("ตรวจพบ marker ครบ 3 อันแล้ว!")
            break
        for j in range(len(markers)):
            cv2.rectangle(img, markers[j].pt1, markers[j].pt2, (255, 255, 255))
            cv2.putText(img, markers[j].text, markers[j].center, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.imshow("Markers", img)
        cv2.waitKey(1)
        if i == 199:
            print("ไม่พบ marker ครบ 3 อันในเวลาที่กำหนด")
            ep_robot.close()
            exit()
    
    cv2.destroyAllWindows()
    ep_gimbal.moveto(pitch=0, yaw=0).wait_for_completed() # กลับไปที่ตำแหน่งเริ่มต้น

    # กำหนดลำดับการยิง: ซ้าย(0) -> กลาง(1) -> ขวา(2) -> กลาง(1) -> ซ้าย(0)
    if len(sorted_markers_by_x) >= 3:
        fire_sequence = [
            sorted_markers_by_x[0],
            sorted_markers_by_x[1],
            sorted_markers_by_x[2],
            sorted_markers_by_x[1],
            sorted_markers_by_x[0]
        ]
        
        for target in fire_sequence:
            track_and_fire_with_pid(ep_gimbal, ep_blaster, target)

    # ทำความสะอาดและปิดการทำงาน
    result = ep_vision.unsub_detect_info(name="marker")
    ep_camera.stop_video_stream()
    ep_robot.close()