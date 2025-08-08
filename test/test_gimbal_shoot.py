import time
import cv2 # เพิ่มไลบรารี OpenCV
import robomaster
from robomaster import robot, vision, gimbal, blaster, camera

class PIDController:
    """A simple PID controller."""
    def __init__(self, kp, ki, kd, setpoint):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()

    def update(self, current_value):
        """Calculates the PID control output."""
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0:
            return 0

        error = self.setpoint - current_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt

        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        self.prev_error = error
        self.last_time = current_time
        return output

# Global list to store processed marker objects
g_processed_markers = []

class MarkerObject:
    """A class to hold processed marker information for drawing."""
    def __init__(self, x, y, w, h, info):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.info = info
        # คำนวณพิกัดสำหรับวาดบนภาพขนาด 1280x720
        self.pt1 = (int((x - w / 2) * 1280), int((y - h / 2) * 720))
        self.pt2 = (int((x + w / 2) * 1280), int((y + h / 2) * 720))
        self.center = (int(x * 1280), int(y * 720))
        self.text = info

def on_detect_marker(marker_info):
    """Callback function that updates the global list of detected markers."""
    global g_processed_markers
    g_processed_markers.clear()
    for m in marker_info:
        g_processed_markers.append(MarkerObject(*m))


if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    # ตั้งค่าความละเอียดกล้องเป็น 720p เพื่อให้ภาพชัดขึ้น
    ep_robot.camera.start_video_stream(display=False, resolution=camera.STREAM_720P)
    ep_robot.vision.sub_detect_info(name="marker", callback=on_detect_marker)
    ep_robot.gimbal.recenter().wait_for_completed()

    print("กำลังค้นหา Marker หมายเลข '1' จำนวน 3 เป้า...")
    time.sleep(2)

    # กรองหาเฉพาะ Marker หมายเลข '1' จากข้อมูลที่ประมวลผลแล้ว
    markers_1 = [m for m in g_processed_markers if m.info == '1']

    if len(markers_1) == 3:
        markers_1.sort(key=lambda m: m.x)
        
        targets = {
            "left":   markers_1[0],
            "center": markers_1[1],
            "right":  markers_1[2]
        }
        print("ระบุเป้าหมายครบทั้ง 3 จุดแล้ว: ซ้าย, กลาง, ขวา")
        
        sequence = ["left", "center", "right", "center", "left"]
        
        for target_name in sequence:
            print(f"\n--- เริ่มเล็งเป้าหมาย: {target_name.upper()} ---")
            
            # --- ปรับจูนค่า PID ให้เร็วขึ้น ---
            # เพิ่ม Kp เพื่อการตอบสนองที่เร็วขึ้น และเพิ่ม Kd เพื่อลดการสั่น
            pid_yaw = PIDController(kp=150, ki=0.5, kd=15, setpoint=0.5)
            pid_pitch = PIDController(kp=150, ki=0.5, kd=15, setpoint=0.5)
            
            locked = False
            for i in range(150):
                # --- ส่วนของการแสดงผลภาพ ---
                img = ep_robot.camera.read_cv2_image(strategy="newest", timeout=0.5)
                if img is not None:
                    # วาดกรอบและข้อความลงบน Marker ทั้งหมดที่เห็น
                    for marker in g_processed_markers:
                        cv2.rectangle(img, marker.pt1, marker.pt2, (0, 255, 0), 2)
                        cv2.putText(img, marker.text, marker.center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    cv2.imshow("Robomaster Camera Feed", img)
                    cv2.waitKey(1) # สำคัญมากสำหรับการแสดงผลหน้าต่าง

                # ค้นหาเป้าหมายปัจจุบันจากข้อมูลล่าสุด
                current_target_info = None
                temp_markers = [m for m in g_processed_markers if m.info == '1']
                temp_markers.sort(key=lambda m: m.x)

                target_index = list(targets.keys()).index(target_name)
                if len(temp_markers) > target_index:
                    current_target_info = temp_markers[target_index]

                if not current_target_info:
                    ep_robot.gimbal.drive_speed(0, 0)
                    continue

                yaw_speed = -pid_yaw.update(current_target_info.x)
                pitch_speed = pid_pitch.update(current_target_info.y)
                
                ep_robot.gimbal.drive_speed(yaw_speed=yaw_speed, pitch_speed=pitch_speed)
                
                error_x = current_target_info.x - 0.5
                error_y = current_target_info.y - 0.5
                
                if abs(error_x) < 0.015 and abs(error_y) < 0.015:
                    print(f"เป้าหมาย {target_name.upper()} ล็อคแล้ว! ทำการยิงเลเซอร์")
                    ep_robot.gimbal.drive_speed(0, 0)
                    ep_robot.blaster.fire(fire_type=blaster.INFRARED_FIRE, times=1)
                    time.sleep(1.5)
                    locked = True
                    break
                
                time.sleep(0.01) # ลด delay เพื่อการตอบสนองที่เร็วขึ้น
            
            if not locked:
                print(f"ไม่สามารถล็อคเป้าหมาย {target_name.upper()} ได้ทันเวลา")
                ep_robot.gimbal.drive_speed(0, 0)

    else:
        print(f"ข้อผิดพลาด: ตรวจพบ Marker หมายเลข '1' จำนวน {len(markers_1)} จุด (ต้องการ 3 จุด)")

    print("\n--- ลำดับการทำงานเสร็จสิ้น ---")
    cv2.destroyAllWindows() # ปิดหน้าต่างกล้องทั้งหมด
    ep_robot.vision.unsub_detect_info(name="marker")
    ep_robot.camera.stop_video_stream()
    ep_robot.close()
