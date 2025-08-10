import cv2
import time
import robomaster
from robomaster import robot
from robomaster import gimbal
from robomaster import camera
from robomaster import vision

# ใช้ global list เพื่อเก็บข้อมูล marker ที่ตรวจจับได้
markers = []

# ขนาดหน้าต่างแสดงผล
window_width = 1280
window_height = 720

# ความเร็วในการหมุน gimbal เมื่อค้นหา marker
gimbal_search_speed = 10 # หน่วย: องศา/วินาที

# ตัวแปรสำหรับควบคุมการหมุน Gimbal
yaw_current = 0
yaw_direction = 1 # 1 = หมุนไปทางขวา, -1 = หมุนไปทางซ้าย
yaw_max_range = 60 # องศาที่ gimbal จะหมุนไปได้จากจุดศูนย์กลาง (-60 ถึง 60)

class MarkerInfo:
    """คลาสสำหรับจัดเก็บข้อมูลของ Marker ที่ตรวจจับได้"""
    def __init__(self, x, y, w, h, info):
        self._x = x
        self._y = y
        self._w = w
        self._h = h
        self._info = info

    @property # พิกัดมุมบนซ้ายของ marker
    def pt1(self):
        return int((self._x - self._w / 2) * window_width), int((self._y - self._h / 2) * window_height)

    @property # พิกัดมุมขวาล่างของ marker
    def pt2(self):
        return int((self._x + self._w / 2) * window_width), int((self._y + self._h / 2) * window_height)

    @property # พิกัดตรงกลางของ marker บนหน้าจอ
    def center(self):
        return int(self._x * window_width), int(self._y * window_height)

    @property
    def text(self):
        return self._info

    @property # พิกัด x ตรงกลางของ marker สำหรับใช้ในการจัดเรียงและควบคุม
    def center_x(self):
        return self._x

def on_detect_marker(marker_info):
    """Callback function ที่จะถูกเรียกเมื่อตรวจพบ marker"""
    global markers
    number = len(marker_info)
    markers.clear()

    for i in range(0, number):
        x, y, w, h, info = marker_info[i]
        markers.append(MarkerInfo(x, y, w, h, info))
        # พิมพ์ข้อมูล marker ที่ตรวจจับได้ (ตามโค้ดเดิม)
        print("marker:{0} x:{1}, y:{2}, w:{3}, h:{4}".format(info, x * window_width, y * window_height,
                                                             w * window_width, h * window_height))


if __name__ == '__main__':
    ep_robot = robot.Robot()
    # กำหนดให้ robomaster เชื่อมต่อผ่านโหมด Wi-Fi AP
    ep_robot.initialize(conn_type="ap")
    
    # ดึง service ต่างๆ ของหุ่นยนต์มาใช้งาน
    ep_gimbal = ep_robot.gimbal
    ep_vision = ep_robot.vision
    ep_camera = ep_robot.camera

    # ตั้งค่า gimbal ให้กลับมาที่ตำแหน่งเริ่มต้น
    ep_gimbal.moveto(pitch=0, yaw=0).wait_for_completed()
    ep_gimbal.set_led(comp="all", r=0, g=255, b=0, effect="on") # เปิดไฟ LED สีเขียว
    
    # ตั้งค่ากล้องและเริ่ม Video Stream
    # 'display=False' เพื่อประหยัดทรัพยากร เพราะเราจะนำภาพไปแสดงผลเองด้วย OpenCV
    ep_camera.start_video_stream(display=False)
    
    # ลงทะเบียน callback function สำหรับตรวจจับ marker
    result = ep_vision.sub_detect_info(name="marker", callback=on_detect_marker)
    
    # Loop หลักของการทำงาน
    while True:
        # อ่านภาพจากกล้อง
        img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        if img is None:
            continue

        # หากตรวจพบ marker
        if len(markers) > 0:
            # ใช้ marker ตัวแรกที่ตรวจเจอเป็นหลัก
            marker = markers[0]
            
            # คำนวณค่า error เพื่อให้ marker อยู่ตรงกลางหน้าจอ
            # ถ้า marker อยู่ทางซ้าย center_x จะน้อยกว่า 0.5
            # ถ้า marker อยู่ทางขวา center_x จะมากกว่า 0.5
            error = marker.center_x - 0.5
            
            # ปรับมุม yaw ของ gimbal ตามค่า error
            # ค่า 20 คือค่าคงที่ที่ใช้ในการปรับความเร็ว (p-gain)
            yaw_speed = -error * 20 
            ep_gimbal.drive_speed(yaw_speed, 0) # สั่งให้ gimbal หมุนด้วยความเร็วที่คำนวณได้
        else:
            # หากไม่พบ marker ให้ gimbal หมุนสแกน
            yaw_current += yaw_direction * gimbal_search_speed * 0.1 # 0.1 คือ interval time
            
            # หากหมุนไปจนสุดขอบ (เกิน -60 หรือ +60) ให้เปลี่ยนทิศทางการหมุน
            if yaw_current >= yaw_max_range:
                yaw_current = yaw_max_range
                yaw_direction = -1
            elif yaw_current <= -yaw_max_range:
                yaw_current = -yaw_max_range
                yaw_direction = 1
                
            # สั่งให้ gimbal หมุนไปยังตำแหน่งใหม่
            ep_gimbal.moveto(yaw=yaw_current, pitch=0).wait_for_completed(False)
            
        # วาดกรอบสี่เหลี่ยมและข้อความบนภาพที่ตรวจจับได้
        for marker in markers:
            cv2.rectangle(img, marker.pt1, marker.pt2, (255, 255, 255), 2)
            cv2.putText(img, marker.text, marker.center, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        # แสดงภาพผลลัพธ์
        cv2.imshow("Robomaster Gimbal Marker Search", img)
        
        # กด 'q' เพื่อออกจากโปรแกรม
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup: ปิดการทำงานต่างๆ ของหุ่นยนต์
    ep_gimbal.drive_speed(0, 0)
    ep_vision.unsub_detect_info(name="marker")
    ep_camera.stop_video_stream()
    cv2.destroyAllWindows()
    ep_robot.close()

