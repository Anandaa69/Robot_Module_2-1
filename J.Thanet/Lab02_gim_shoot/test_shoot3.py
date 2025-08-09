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
Kp_yaw, Ki_yaw, Kd_yaw = 0.5, 0.0001, 0.05
Kp_pitch, Ki_pitch, Kd_pitch = 0.5, 0.0001, 0.05

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

    @property # พิกัดมุมบนซ้าย
    def pt1(self):
        return int((self._x - self._w / 2) * weight_window), int((self._y - self._h / 2) * height_window)

    @property # พิกัดมุมขวาล่าง
    def pt2(self):
        return int((self._x + self._w / 2) * weight_window), int((self._y + self._h / 2) * height_window)

    @property # ตรงกลางของ marker
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
    number = len(marker_info)
    markers.clear()

    for i in range(0, number):
        x, y, w, h, info = marker_info[i]
        markers.append(MarkerInfo(x, y, w, h, info))
        # print("marker:{0} x:{1}, y:{2}, w:{3}, h:{4}".format(info, x * height_window, y * weight_window,
        #                                                     w * height_window, h * weight_window))

    global sorted_markers_by_x # เรียงลำดับมาร์คเกอร์ตามแกน X
    sorted_markers_by_x = sorted(markers, key=lambda m: m.center_x)

    # print("\n== เรียงมาร์คเกอร์ตามแนวแกน X จากซ้ายไปขวา ==")
    # for idx, m in enumerate(sorted_markers_by_x):
    #     cx, cy = m.center
    #     print(f"อันดับ {idx + 1}: Marker {m.text} - CenterX: {m.center_x:.3f} -> พิกัด Pixel: ({cx}, {cy})")


def move_to_markers_with_pid_sequence():
    global integral_x, error_x_pre, integral_y, error_y_pre

    if len(sorted_markers_by_x) < 3:
        print("ตรวจพบ marker น้อยกว่า 3 อัน ไม่สามารถทำการเคลื่อนที่ได้")
        return

    # ลำดับการเคลื่อนที่: 1 -> 2 -> 3 -> 2 -> 1
    sequence_indices = [0, 1, 2, 1, 0]

    for index in sequence_indices:
        target_marker = sorted_markers_by_x[index]
        print(f"กำลังเคลื่อนที่ไปยัง Marker {target_marker.text}...")

        # ติดตาม marker ที่เลือกจนกว่าจะอยู่ตรงกลาง
        for i in range(200): # ลูปเพื่อติดตาม marker เป็นเวลาสั้นๆ
            if len(markers) == 0:
                print("Marker หายไป! หยุดการเคลื่อนที่")
                ep_gimbal.drive_speed(yaw_speed=0, pitch_speed=0)
                break
            
            # หาก marker ที่ติดตามมีการขยับในเฟรมภาพ ให้ใช้ PID คำนวณความเร็วใหม่
            
            target_x, target_y = target_marker.center
            
            error_x = target_x - (weight_window / 2)
            error_y = (height_window / 2) - target_y

            integral_x += error_x
            derivative_x = error_x - error_x_pre
            output_yaw = Kp_yaw * error_x + Ki_yaw * integral_x + Kd_yaw * derivative_x
            error_x_pre = error_x

            integral_y += error_y
            derivative_y = error_y - error_y_pre
            output_pitch = Kp_pitch * error_y + Ki_pitch * integral_y + Kd_yaw * derivative_y
            error_y_pre = error_y

            # จำกัดความเร็ว
            max_speed = 30
            output_yaw = max(-max_speed, min(max_speed, output_yaw))
            output_pitch = max(-max_speed, min(max_speed, output_pitch))

            ep_gimbal.drive_speed(yaw_speed=output_yaw, pitch_speed=output_pitch)
            time.sleep(0.01) # หน่วงเวลาเล็กน้อยเพื่อให้ gimbal มีเวลาตอบสนอง

        print(f"ถึงตำแหน่ง Marker {target_marker.text} แล้ว.")
        ep_gimbal.drive_speed(yaw_speed=0, pitch_speed=0) # หยุด gimbal
        time.sleep(1) # รอ 1 วินาที ก่อนไป marker ถัดไป

    print("เสร็จสิ้นการเคลื่อนที่ตามลำดับ")


if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_blaster = ep_robot.blaster
    ep_gimbal = ep_robot.gimbal
    ep_vision = ep_robot.vision
    ep_camera = ep_robot.camera

    ep_gimbal.moveto(pitch=0, yaw=0).wait_for_completed()
    ep_camera.start_video_stream(display=False)
    
    # ใช้ callback เดิมเพื่ออัปเดตตำแหน่ง marker ในตัวแปร markers
    result = ep_vision.sub_detect_info(name="marker", callback=on_detect_marker)

    print("เริ่มการตรวจจับ marker...")
    for i in range(0, 500):
        img = ep_camera.read_cv2_image(strategy="newest", timeout=3)
        for j in range(0, len(markers)):
            cv2.rectangle(img, markers[j].pt1, markers[j].pt2, (255, 255, 255))
            cv2.putText(img, markers[j].text, markers[j].center, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.imshow("Markers", img)
        cv2.waitKey(1)
        if len(markers) >= 3:
            print("ตรวจพบ marker ครบ 3 อันแล้ว")
            break

    cv2.destroyAllWindows()
    
    # เรียกใช้ฟังก์ชันเคลื่อนที่ gimbal
    move_to_markers_with_pid_sequence()
    
    # ทำความสะอาดและปิดการทำงาน
    result = ep_vision.unsub_detect_info(name="marker")
    ep_camera.stop_video_stream()
    ep_robot.close()