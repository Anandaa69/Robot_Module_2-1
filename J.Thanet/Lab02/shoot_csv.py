import time
import cv2
import robomaster
from robomaster import robot, vision, gimbal, blaster, camera
import math
import csv
from datetime import datetime

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
        
        # Clamp integral to prevent windup
        self.integral = max(-10, min(10, self.integral))
        
        derivative = (error - self.prev_error) / dt

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        # Clamp output to reasonable range
        output = max(-100, min(100, output))

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

def on_detect_marker(marker_info):
    """Callback function that updates the global list of detected markers."""
    global g_processed_markers
    g_processed_markers.clear()
    for m in marker_info:
        g_processed_markers.append(MarkerObject(*m))

def calculate_target_angles(target_distance, target_spacing):
    """คำนวณมุม yaw ที่ต้องการสำหรับแต่ละเป้าหมาย"""
    half_spacing = target_spacing / 2
    left_angle = math.degrees(math.atan(half_spacing / target_distance))
    right_angle = math.degrees(math.atan(half_spacing / target_distance))
    center_angle = 0.0
    
    return {
        'left': -left_angle,
        'center': center_angle,
        'right': right_angle
    }

if __name__ == '__main__':
    KP = 140
    KI = 2
    KD = 3

    # KP = 140
    # KI = 2
    # KD = 3

    KP_str = str(KP).replace('.', '-')
    KI_str = str(KI).replace('.', '-')
    KD_str = str(KD).replace('.', '-')
    
    # Initialize CSV file with headers
    csv_filename = f"F:/Coder/Year2-1/Robot_Module/J.Thanet/Lab02/data/rec_track_{datetime.now().strftime('%H%M')}_P{KP_str}_I{KI_str}_D{KD_str}.csv"
    csv_headers = [
        'timestamp', 'target_name', 'iteration', 'markers_found', 
        'marker_id', 'marker_x', 'marker_y', 'marker_w', 'marker_h',
        'error_x', 'error_y', 'yaw_speed', 'pitch_speed', 
        'locked', 'fired', 'elapsed_time'
    ]
    
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_headers)
    
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_robot.camera.start_video_stream(display=False, resolution=camera.STREAM_720P)
    ep_robot.vision.sub_detect_info(name="marker", callback=on_detect_marker)
    ep_robot.gimbal.recenter().wait_for_completed()

    print("กำลังค้นหา Marker ทุกหมายเลขจำนวน 3 เป้า...")
    time.sleep(2)

    pid_config = {
        "yaw": {"kp": KP, "ki": KI, "kd": KD},
        "pitch": {"kp": KP, "ki": KI, "kd": KD}
    }

    target_distance = 1.0  # meters
    target_spacing = 0.6   # meters
    target_angles = calculate_target_angles(target_distance, target_spacing)

    pid_yaw = PIDController(kp=pid_config["yaw"]["kp"], ki=pid_config["yaw"]["ki"], kd=pid_config["yaw"]["kd"], setpoint=0.5)
    pid_pitch = PIDController(kp=pid_config["pitch"]["kp"], ki=pid_config["pitch"]["ki"], kd=pid_config["pitch"]["kd"], setpoint=0.5)

    sequence = ["left", "center", "right", "center", "left"]
    
    for target_name in sequence:
        print(f"\n--- เริ่มเล็งเป้าหมาย: {target_name.upper()} ---")
        
        locked = False
        aim_start_time = time.time()
        
        current_yaw_target_angle = target_angles[target_name] * 1.0

        ep_robot.gimbal.moveto(pitch=-10, yaw=current_yaw_target_angle).wait_for_completed()
        time.sleep(0.5) # Wait for gimbal to settle properly

        for i in range(300): # Limit tracking time
            current_target_info = None
            temp_markers = g_processed_markers
            temp_markers.sort(key=lambda m: m.x)
            print(f"พบ Markers จำนวน: {len(temp_markers)}")
            
            if temp_markers:
                if target_name == "left":
                    current_target_info = temp_markers[0]
                elif target_name == "center":
                    if len(temp_markers) >= 2:
                        current_target_info = temp_markers[1]
                    else:
                        current_target_info = temp_markers[0]
                elif target_name == "right":
                    if len(temp_markers) >= 3:
                        current_target_info = temp_markers[2]
                    elif len(temp_markers) >= 2:
                        current_target_info = temp_markers[1]
                    else:
                        current_target_info = temp_markers[0]

            if current_target_info:
                error_x = current_target_info.x - 0.5
                error_y = current_target_info.y - 0.5
                
                yaw_speed = -pid_yaw.update(current_target_info.x)
                pitch_speed = pid_pitch.update(current_target_info.y)
                
                ep_robot.gimbal.drive_speed(yaw_speed=yaw_speed, pitch_speed=pitch_speed)
                
                # บันทึกข้อมูลลง CSV (ยังไม่ยิง)
                with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([
                        datetime.now().strftime('%H:%M:%S.%f'),
                        target_name,
                        i,
                        len(temp_markers),
                        current_target_info.info,
                        current_target_info.x,
                        current_target_info.y,
                        current_target_info.w,
                        current_target_info.h,
                        error_x,
                        error_y,
                        yaw_speed,
                        pitch_speed,
                        int(locked),
                        0,  # fired = 0
                        round(time.time() - aim_start_time, 3)
                    ])

                print(f"Tracking Marker ID {current_target_info.info} | Error X: {error_x:.3f}, Y: {error_y:.3f}")

                if abs(error_x) <= 0.015 and abs(error_y) <= 0.015:
                    ep_robot.gimbal.drive_speed(0, 0)
                    time.sleep(0.15)  # Let gimbal stabilize
                    
                    print(f"เป้าหมาย {target_name.upper()} ล็อคแล้ว! ทำการยิงเลเซอร์")
                    ep_robot.blaster.fire(fire_type=blaster.INFRARED_FIRE, times=1)
                    
                    # บันทึก fired = 1
                    with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([
                            datetime.now().strftime('%H:%M:%S.%f'),
                            target_name,
                            i,
                            len(temp_markers),
                            current_target_info.info,
                            current_target_info.x,
                            current_target_info.y,
                            current_target_info.w,
                            current_target_info.h,
                            error_x,
                            error_y,
                            0,
                            0,
                            1,
                            1,
                            round(time.time() - aim_start_time, 3)
                        ])
                    
                    time.sleep(0.15)
                    locked = True
                    break
            else:
                print("Lost marker. Scanning...")
                
            time.sleep(0.2)
        
        if not locked:
            print(f"ไม่สามารถล็อคเป้าหมาย {target_name.upper()} ได้ทันเวลา")
            ep_robot.gimbal.drive_speed(0, 0)

    print(f"\n--- ลำดับการทำงานเสร็จสิ้น ---")
    print(f"ข้อมูลถูกบันทึกลงไฟล์: {csv_filename}")
    ep_robot.vision.unsub_detect_info(name="marker")
    ep_robot.camera.stop_video_stream()
    ep_robot.close()
