import cv2
import time
import robomaster
from robomaster import robot
from robomaster import blaster
from robomaster import camera
from robomaster import vision


markers = []
window_size = 720
height_window = window_size
weight_window = 1280


class MarkerInfo:

    def __init__(self, x, y, w, h, info):
        self._x = x
        self._y = y
        self._w = w
        self._h = h
        self._info = info

    @property #พิกัดมุมบนซ้าย
    def pt1(self):
        return int((self._x - self._w / 2) * weight_window), int((self._y - self._h / 2) * height_window)

    @property #พิกัดมุมขวาล่าง
    def pt2(self):
        return int((self._x + self._w / 2) * weight_window), int((self._y + self._h / 2) * height_window)

    @property#ตรงกลางของ marker
    def center(self):
        return int(self._x * weight_window), int(self._y * height_window)

    @property
    def text(self):
        return self._info

    @property
    def center_x(self):  # สำหรับใช้ในการเรียงลำดับ
        return self._x


def on_detect_marker(marker_info):
    number = len(marker_info)
    markers.clear()

    for i in range(0, number):
        x, y, w, h, info = marker_info[i]
        markers.append(MarkerInfo(x, y, w, h, info))
        print("marker:{0} x:{1}, y:{2}, w:{3}, h:{4}".format(info, x * height_window, y * weight_window,
                                                             w * height_window, h * weight_window))

    global sorted_markers_by_x # เรียงลำดับมาร์คเกอร์ตามแกน X
    sorted_markers_by_x = sorted(markers, key=lambda m: m.center_x)

    print("\n== เรียงมาร์คเกอร์ตามแนวแกน X จากซ้ายไปขวา ==")
    for idx, m in enumerate(sorted_markers_by_x):
        cx, cy = m.center
        print(f"อันดับ {idx + 1}: Marker {m.text} - CenterX: {m.center_x:.3f} -> พิกัด Pixel: ({cx}, {cy})")

if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_blaster = ep_robot.blaster
    ep_gimbal = ep_robot.gimbal
    ep_vision = ep_robot.vision
    ep_camera = ep_robot.camera


    ep_gimbal.moveto(pitch=0, yaw=0).wait_for_completed()
    ep_camera.start_video_stream(display=False)
    result = ep_vision.sub_detect_info(name="marker", callback=on_detect_marker)

    for i in range(0, 500):
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