from robomaster import robot
import cv2
import numpy as np
import csv
import time

# -*-coding:utf-8-*-
# ... (โค้ดส่วนต้นเหมือนเดิม)

import time

markers = []
window_size = 720
height_window = window_size
weight_window = 1280
# 640x360 960x540 1280x720

class MarkerInfo:

    def __init__(self, x, y, w, h, info):
        self._x = x
        self._y = y
        self._w = w
        self._h = h
        self._info = info

    @property
    def pt1(self):
        return int((self._x - self._w / 2) * weight_window), int((self._y - self._h / 2) * height_window)

    @property
    def pt2(self):
        return int((self._x + self._w / 2) * weight_window), int((self._y + self._h / 2) * height_window)

    @property
    def center(self):
        return int(self._x * weight_window), int(self._y * height_window)

    @property
    def text(self):
        return self._info


def on_detect_marker(marker_info):
    number = len(marker_info)
    markers.clear()
    for i in range(0, number):
        x, y, w, h, info = marker_info[i]
        markers.append(MarkerInfo(x, y, w, h, info))
        print("marker:{0} x:{1}, y:{2}, w:{3}, h:{4}".format(info, x * height_window, y * weight_window, w * height_window, h * weight_window))


def gimbal_recenter_and_rotate(ep_gimbal, angle=180, speed=20):
    # รีเซ็ตกิมบอลกลับศูนย์
    print("Recenter gimbal...")
    ep_gimbal.recenter().wait_for_completed()
    time.sleep(1)

    # หมุนกิมบอล 180 องศา แบบช้าๆ
    print(f"Rotate gimbal by {angle} degrees at speed {speed}...")
    ep_gimbal.rotate(yaw=angle, speed=speed).wait_for_completed()
    print("Rotation completed.")


if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_vision = ep_robot.vision
    ep_camera = ep_robot.camera
    ep_gimbal = ep_robot.gimbal

    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_720P)
    result = ep_vision.sub_detect_info(name="marker", callback=on_detect_marker)

    # รีเซ็ตและหมุนกิมบอล
    gimbal_recenter_and_rotate(ep_gimbal, angle=180, speed=20)

    for i in range(0, 5000):
        img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        for j in range(0, len(markers)):
            cv2.rectangle(img, markers[j].pt1, markers[j].pt2, (255, 255, 255))
            cv2.putText(img, markers[j].text, markers[j].center, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.imshow("Markers", img)
        cv2.waitKey(1)
    cv2.destroyAllWindows()

    result = ep_vision.unsub_detect_info(name="marker")
    cv2.destroyAllWindows()
    ep_camera.stop_video_stream()
    ep_robot.close()
