import cv2
import time
import robomaster
from robomaster import robot
from robomaster import blaster
from robomaster import camera
from robomaster import vision


markers = []
detected_objects = {}  # เก็บประวัติตำแหน่งของ object แต่ละตัว
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

    @property #ตรงกลางของ marker
    def center(self):
        return int(self._x * weight_window), int(self._y * height_window)

    @property
    def center_pixel_x(self):
        return int(self._x * weight_window)
    
    @property
    def center_pixel_y(self):
        return int(self._y * height_window)

    @property
    def text(self):
        return self._info

    @property
    def center_x(self):  # สำหรับใช้ในการเรียงลำดับ (normalized coordinate)
        return self._x
    
    @property
    def center_y(self):  # normalized coordinate
        return self._y


def on_detect_marker(marker_info):
    number = len(marker_info)
    markers.clear()

    print(f"\n=== ตรวจพบ {number} markers ===")
    
    for i in range(0, number):
        x, y, w, h, info = marker_info[i]
        markers.append(MarkerInfo(x, y, w, h, info))
        
        # เก็บตำแหน่งใน dictionary
        center_x_pixel = int(x * weight_window)
        center_y_pixel = int(y * height_window)
        
        detected_objects[info] = {
            'center_normalized': (x, y),  # ค่าปกติ 0-1
            'center_pixel': (center_x_pixel, center_y_pixel),  # ค่าพิกเซล
            'size_normalized': (w, h),
            'size_pixel': (int(w * weight_window), int(h * height_window)),
            'timestamp': time.time()
        }
        
        print(f"Marker {info}:")
        print(f"  - Center (Normalized): ({x:.3f}, {y:.3f})")
        print(f"  - Center (Pixel): ({center_x_pixel}, {center_y_pixel})")
        print(f"  - Size (Normalized): ({w:.3f}, {h:.3f})")
        print(f"  - Size (Pixel): ({int(w * weight_window)}, {int(h * height_window)})")

    # เรียงลำดับมาร์คเกอร์ตามแกน X
    global sorted_markers_by_x
    sorted_markers_by_x = sorted(markers, key=lambda m: m.center_x)

    print("\n== เรียงมาร์คเกอร์ตามแนวแกน X จากซ้ายไปขวา ==")
    for idx, m in enumerate(sorted_markers_by_x):
        cx, cy = m.center
        print(f"อันดับ {idx + 1}: Marker {m.text}")
        print(f"  - Center X (Normalized): {m.center_x:.3f}")
        print(f"  - Center (Pixel): ({cx}, {cy})")


def print_all_detected_objects():
    """พิมพ์ข้อมูลทั้งหมดของ objects ที่เคยตรวจพบ"""
    print("\n" + "="*50)
    print("สรุปตำแหน่งทั้งหมดที่เคยตรวจพบ:")
    print("="*50)
    
    if not detected_objects:
        print("ยังไม่พบ object ใดๆ")
        return
    
    for obj_id, obj_data in detected_objects.items():
        print(f"\nMarker {obj_id}:")
        print(f"  - ตำแหน่งกลาง (Pixel): {obj_data['center_pixel']}")
        print(f"  - ตำแหน่งกลาง (Normalized): ({obj_data['center_normalized'][0]:.3f}, {obj_data['center_normalized'][1]:.3f})")
        print(f"  - ขนาด (Pixel): {obj_data['size_pixel']}")
        print(f"  - เวลาที่ตรวจพบล่าสุด: {time.ctime(obj_data['timestamp'])}")


def get_object_center(marker_id):
    """ฟังก์ชันสำหรับดึงตำแหน่งกลางของ marker ตาม ID"""
    if marker_id in detected_objects:
        return detected_objects[marker_id]['center_pixel']
    else:
        print(f"ไม่พบ Marker {marker_id}")
        return None


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

    frame_count = 0
    
    for i in range(0, 500):
        img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        
        # วาดกรอบและข้อความบน markers
        for j in range(0, len(markers)):
            cv2.rectangle(img, markers[j].pt1, markers[j].pt2, (0, 255, 0), 2)
            cv2.putText(img, markers[j].text, markers[j].center, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            # แสดงพิกัด center
            center_text = f"({markers[j].center_pixel_x}, {markers[j].center_pixel_y})"
            cv2.putText(img, center_text, (markers[j].center_pixel_x, markers[j].center_pixel_y + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow("Markers", img)
        
        # แสดงข้อมูลทุก 50 frames
        frame_count += 1
        if frame_count % 50 == 0:
            print_all_detected_objects()
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # พิมพ์สรุปสุดท้าย
    print_all_detected_objects()
    
    # ตัวอย่างการใช้งานฟังก์ชัน get_object_center
    print("\n=== ตัวอย่างการดึงตำแหน่ง ===")
    for marker_id in detected_objects.keys():
        center = get_object_center(marker_id)
        if center:
            print(f"Center ของ Marker {marker_id}: {center}")

    cv2.destroyAllWindows()

    result = ep_vision.unsub_detect_info(name="marker")
    cv2.destroyAllWindows()
    ep_camera.stop_video_stream()

    ep_robot.close()