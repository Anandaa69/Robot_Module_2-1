
import time
import robomaster
from robomaster import robot, vision, gimbal, blaster

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

# Global list to store raw marker data from the vision callback
g_detected_markers_list = []

def on_detect_marker(marker_info):
    """Callback function that updates the global list of detected markers."""
    global g_detected_markers_list
    g_detected_markers_list = marker_info


if __name__ == '_main_':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_robot.vision.sub_detect_info(name="marker", callback=on_detect_marker)
    ep_robot.camera.start_video_stream(display=False)
    ep_robot.gimbal.recenter().wait_for_completed()

    print("กำลังค้นหา Marker หมายเลข '1' จำนวน 3 เป้า...")
    time.sleep(3)  # รอให้ระบบ Vision ตรวจจับ Marker

    # กรองหาเฉพาะ Marker หมายเลข '1'
    markers_1 = [m for m in g_detected_markers_list if m[4] == '1']

    if len(markers_1) == 3:
        # เรียงลำดับ Marker ตามแกน X (จากซ้ายไปขวา)
        markers_1.sort(key=lambda m: m[0])
        
        targets = {
            "left":   markers_1[0],
            "center": markers_1[1],
            "right":  markers_1[2]
        }
        print("ระบุเป้าหมายครบทั้ง 3 จุดแล้ว: ซ้าย, กลาง, ขวา")
        
        # กำหนดลำดับการยิง
        sequence = ["left", "center", "right", "center", "left"]
        
        for target_name in sequence:
            print(f"\n--- เริ่มเล็งเป้าหมาย: {target_name.upper()} ---")
            
            # --- ปรับจูนค่า PID ที่นี่ ---
            # Kp: อัตราขยาย - ตอบสนองต่อค่า error ปัจจุบัน (ยิ่งมากยิ่งเร็ว แต่อาจแกว่ง)
            # Ki: อินทิกรัล - ลดค่า error ที่ค้างนิ่งๆ ให้เข้าเป้าสนิท
            # Kd: อนุพันธ์ - ลดการแกว่ง (overshoot) ทำให้เข้าเป้าได้นิ่งขึ้น
            pid_yaw = PIDController(kp=85, ki=0.5, kd=7, setpoint=0.5)
            pid_pitch = PIDController(kp=85, ki=0.5, kd=7, setpoint=0.5)
            
            locked = False
            # พยายามเล็งเป้าหมายเป็นเวลาสูงสุด 3 วินาที (150 * 0.02 วินาที)
            for i in range(150):
                # ค้นหา Marker เป้าหมายจากข้อมูลล่าสุด
                current_target_info = None
                temp_markers = [m for m in g_detected_markers_list if m[4] == '1']
                temp_markers.sort(key=lambda m: m[0])

                target_index = list(targets.keys()).index(target_name)
                if len(temp_markers) > target_index:
                    # อัปเดตตำแหน่งเป้าหมายแบบ Real-time
                    current_target_info = temp_markers[target_index]

                if not current_target_info:
                    print("ไม่พบเป้าหมายในมุมกล้อง")
                    ep_robot.gimbal.drive_speed(0, 0)
                    time.sleep(0.1)
                    continue

                current_x, current_y = current_target_info[0], current_target_info[1]
                
                # คำนวณความเร็วที่ต้องหมุนจาก PID
                yaw_speed = -pid_yaw.update(current_x)
                pitch_speed = pid_pitch.update(current_y)
                
                ep_robot.gimbal.drive_speed(yaw_speed=yaw_speed, pitch_speed=pitch_speed)
                
                # ตรวจสอบว่าเข้าเป้าแล้วหรือยัง
                error_x = current_x - 0.5
                error_y = current_y - 0.5
                
                if abs(error_x) < 0.015 and abs(error_y) < 0.015:
                    print(f"เป้าหมาย {target_name.upper()} ล็อคแล้ว!")
                    ep_robot.gimbal.drive_speed(0, 0) # หยุดหมุน
                    ep_robot.blaster.fire(fire_type=blaster.INFRARED_FIRE, times=1)

                    time.sleep(1.5) # รอหลังยิงเพื่อให้หุ่นยนต์นิ่ง
                    locked = True
                    break
                
                time.sleep(0.02)
            
            if not locked:
                print(f"ไม่สามารถล็อคเป้าหมาย {target_name.upper()} ได้ทันเวลา")
                ep_robot.gimbal.drive_speed(0, 0)

    else:
        print(f"ข้อผิดพลาด: ตรวจพบ Marker หมายเลข '1' จำนวน {len(markers_1)} จุด (ต้องการ 3 จุด)")

    print("\n--- ลำดับการทำงานเสร็จสิ้น ---")
    ep_robot.vision.unsub_detect_info(name="marker")
    ep_robot.camera.stop_video_stream()
    ep_robot.close()