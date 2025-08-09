import cv2
import time
import robomaster
from robomaster import robot
from robomaster import blaster
from robomaster import camera
from robomaster import vision


markers = []
detected_objects = {}  # เก็บประวัติตำแหน่งของ object แต่ละตัว
sorted_markers_by_x = []  # เก็บ markers ที่เรียงลำดับแล้ว
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
    global sorted_markers_by_x
    
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

    # เรียงลำดับมาร์คเกอร์ตามแกน X จากซ้ายไปขวา
    sorted_markers_by_x = sorted(markers, key=lambda m: m.center_x)

    print("== เรียงมาร์คเกอร์ตามแนวแกน X จากซ้ายไปขวา ==")
    for idx, m in enumerate(sorted_markers_by_x):
        cx, cy = m.center
        order_number = idx + 1
        
        # เพิ่มหมายเลขลำดับเข้าไปใน detected_objects
        detected_objects[m.text]['order'] = order_number
        
        print(f"ลำดับที่ {order_number}: Marker {m.text}")
        print(f"  - Center X (Normalized): {m.center_x:.3f}")
        print(f"  - Center (Pixel): ({cx}, {cy})")
        print(f"  - Size (Normalized): ({m._w:.3f}, {m._h:.3f})")
        print(f"  - Size (Pixel): ({int(m._w * weight_window)}, {int(m._h * height_window)})")


def print_all_detected_objects():
    """พิมพ์ข้อมูลทั้งหมดของ objects ที่เคยตรวจพบ (เรียงตามลำดับจากซ้ายไปขวา)"""
    print("\n" + "="*50)
    print("สรุปตำแหน่งทั้งหมดที่เคยตรวจพบ (เรียงตามลำดับ):")
    print("="*50)
    
    if not detected_objects:
        print("ยังไม่พบ object ใดๆ")
        return
    
    # เรียงตามลำดับ order (ถ้ามี) หรือตาม center_x
    sorted_objects = sorted(detected_objects.items(), 
                          key=lambda x: x[1].get('order', float(x[1]['center_normalized'][0])))
    
    for obj_id, obj_data in sorted_objects:
        order_text = f"ลำดับที่ {obj_data['order']}" if 'order' in obj_data else "ไม่มีลำดับ"
        print(f"\n{order_text}: Marker {obj_id}")
        print(f"  - ตำแหน่งกลาง (Pixel): {obj_data['center_pixel']}")
        print(f"  - ตำแหน่งกลาง (Normalized): ({obj_data['center_normalized'][0]:.3f}, {obj_data['center_normalized'][1]:.3f})")
        print(f"  - ขนาด (Pixel): {obj_data['size_pixel']}")
        print(f"  - เวลาที่ตรวจพบล่าสุด: {time.ctime(obj_data['timestamp'])}")


def get_object_by_order(order_number):
    """ฟังก์ชันสำหรับดึงข้อมูล marker ตามลำดับ (1, 2, 3, ...)"""
    for marker_id, obj_data in detected_objects.items():
        if obj_data.get('order') == order_number:
            return marker_id, obj_data
    print(f"ไม่พบ Object ลำดับที่ {order_number}")
    return None, None


def get_current_sorted_markers():
    """ฟังก์ชันสำหรับดึงรายการ markers ปัจจุบันที่เรียงลำดับแล้ว"""
    return [(idx + 1, m.text, m.center) for idx, m in enumerate(sorted_markers_by_x)]


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
        
        # วาดกรอบและข้อความบน markers (ตามลำดับที่เรียงแล้ว)
        for idx, marker in enumerate(sorted_markers_by_x):
            order_number = idx + 1
            
            # วาดกรอบสีต่างกันตามลำดับ
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
            color = colors[idx % len(colors)]
            
            cv2.rectangle(img, marker.pt1, marker.pt2, color, 3)
            
            # แสดงหมายเลขลำดับและ marker ID
            label = f"{order_number}. {marker.text}"
            cv2.putText(img, label, marker.center, cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            
            # แสดงพิกัด center
            center_text = f"({marker.center_pixel_x}, {marker.center_pixel_y})"
            cv2.putText(img, center_text, (marker.center_pixel_x, marker.center_pixel_y + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # แสดงหมายเลขลำดับใหญ่ๆ ที่มุมบนซ้าย
            cv2.putText(img, str(order_number), 
                       (marker.pt1[0], marker.pt1[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
        
        cv2.imshow("Markers", img)
        
        # แสดงข้อมูลทุก 50 frames
        frame_count += 1
        if frame_count % 50 == 0:
            print_all_detected_objects()
            
            # แสดงลำดับปัจจุบัน
            current_order = get_current_sorted_markers()
            if current_order:
                print("\n=== ลำดับปัจจุบัน (ซ้าย -> ขวา) ===")
                for order, marker_id, center in current_order:
                    print(f"{order}. Marker {marker_id} - Center: {center}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # พิมพ์สรุปสุดท้าย
    print_all_detected_objects()
    
    # ตัวอย่างการใช้งานฟังก์ชันต่างๆ
    print("\n=== ตัวอย่างการดึงข้อมูลตามลำดับ ===")
    for i in range(1, len(detected_objects) + 1):
        marker_id, obj_data = get_object_by_order(i)
        if marker_id:
            center = obj_data['center_pixel']
            print(f"ลำดับที่ {i}: Marker {marker_id} - Center: {center}")
    
    print("\n=== ลำดับสุดท้ายที่ตรวจพบ ===")
    current_order = get_current_sorted_markers()
    for order, marker_id, center in current_order:
        print(f"{order}. Marker {marker_id} - Center: {center}")

    cv2.destroyAllWindows()

    result = ep_vision.unsub_detect_info(name="marker")
    cv2.destroyAllWindows()
    ep_camera.stop_video_stream()

    ep_robot.close()