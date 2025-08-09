import time
import cv2
import robomaster
from robomaster import robot, vision, gimbal, blaster, camera
import json
import math

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
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_robot.camera.start_video_stream(display=False, resolution=camera.STREAM_720P)
    ep_robot.vision.sub_detect_info(name="marker", callback=on_detect_marker)
    ep_robot.gimbal.recenter().wait_for_completed()

    print("กำลังค้นหา Marker หมายเลข '1' จำนวน 3 เป้า...")
    time.sleep(2)

    # Load PID values from file
    try:
        with open("pid_values.json", "r") as f:
            pid_config = json.load(f)
    except FileNotFoundError:
        print("pid_values.json not found. Using default PID values.")
        pid_config = {
            "yaw": {"kp": 110, "ki": 3.5, "kd": 2},
            "pitch": {"kp": 110, "ki": 3.5, "kd": 2}
        }

    # Calculate target angles dynamically
    target_distance = 1.0  # meters
    target_spacing = 0.6   # meters
    target_angles = calculate_target_angles(target_distance, target_spacing)

    # Initialize PID controllers with loaded values
    pid_yaw = PIDController(kp=pid_config["yaw"]["kp"], ki=pid_config["yaw"]["ki"], kd=pid_config["yaw"]["kd"], setpoint=0.5)
    pid_pitch = PIDController(kp=pid_config["pitch"]["kp"], ki=pid_config["pitch"]["ki"], kd=pid_config["pitch"]["kd"], setpoint=0.5)

    sequence = ["left", "center", "right", "center", "left"]
    
    for target_name in sequence:
        print(f"\n--- เริ่มเล็งเป้าหมาย: {target_name.upper()} ---")
        
        locked = False
        aim_start_time = time.time()
        
        # Get the target angle for the current target name
        current_yaw_target_angle = target_angles[target_name]*1.2
        #current_yaw_target_angle = target_angles[target_name]


        # Move gimbal to the target yaw angle
        ep_robot.gimbal.moveto(pitch=-10, yaw=current_yaw_target_angle).wait_for_completed()
        time.sleep(0.5) # Wait for gimbal to settle

        # Start tracking and firing
        for i in range(300): # Limit tracking time to prevent infinite loops
            current_target_info = None
            temp_markers = [m for m in g_processed_markers if m.info == '1']
            temp_markers.sort(key=lambda m: m.x)
            print(f"พบ Markers จำนวน: {len(temp_markers)}")
            # Find the marker closest to the center of the screen
            # if temp_markers:
            #     if target_name == "left" and len(temp_markers) >= 1:
            #         current_target_info = temp_markers[0]  # ซ้ายสุด
            #     elif target_name == "center" and len(temp_markers) >= 2:
            #         current_target_info = temp_markers[1]  # กลาง  
            #     elif target_name == "right" and len(temp_markers) >= 3:
            #         current_target_info = temp_markers[2]  # ขวาสุด
            #     elif target_name == "center" and len(temp_markers) == 2:
            #         current_target_info = temp_markers[1]  # กรณีเหลือ 2 เป้า
            #     else:
            #         current_target_info = None  # ไม่พบเป้าที่ต้องการ
            
            if temp_markers:
                if target_name == "left":
                    current_target_info = temp_markers[0]
                elif target_name == "center":
                    if len(temp_markers) >= 2:
                        current_target_info = temp_markers[1]  # กลาง
                    else:
                        current_target_info = temp_markers[0]  # ถ้าเหลือ 1 ตัว
                elif target_name == "right":
                    if len(temp_markers) >= 3:
                        current_target_info = temp_markers[2]  # ขวาสุด
                    elif len(temp_markers) >= 2:
                        current_target_info = temp_markers[1]  # ถ้าเหลือ 2 ตัว
                    else:
                        current_target_info = temp_markers[0]  # ถ้าเหลือ 1 ตัว
                else:
                    current_target_info = None



            if current_target_info:
                error_x = current_target_info.x - 0.5
                error_y = current_target_info.y - 0.5
                
                yaw_speed = -pid_yaw.update(current_target_info.x)
                pitch_speed = pid_pitch.update(current_target_info.y)
                
                ep_robot.gimbal.drive_speed(yaw_speed=yaw_speed, pitch_speed=pitch_speed)
                
                print(f"Tracking Marker ID 1 | Error X: {error_x:.3f}, Y: {error_y:.3f}")

                if abs(error_x) <= 0.05 and abs(error_y) <= 0.05:
                    print(f"เป้าหมาย {target_name.upper()} ล็อคแล้ว! ทำการยิงเลเซอร์")
                    ep_robot.gimbal.drive_speed(0, 0)
                    ep_robot.blaster.fire(fire_type=blaster.INFRARED_FIRE, times=1)
                    time.sleep(1) # Short delay after firing
                    locked = True
                    break
            else:
                print("Lost marker. Scanning...")
                #ep_gimbal.drive_speed(yaw_speed=0, pitch_speed=0)
                # Implement a scan if needed, but for this problem, we assume markers are mostly visible
                
            time.sleep(1.01) # Reduce delay for faster response
        
        if not locked:
            print(f"ไม่สามารถล็อคเป้าหมาย {target_name.upper()} ได้ทันเวลา")
            ep_robot.gimbal.drive_speed(0, 0)

    print("\n--- ลำดับการทำงานเสร็จสิ้น ---")
    ep_robot.vision.unsub_detect_info(name="marker")
    ep_robot.camera.stop_video_stream()
    ep_robot.close()