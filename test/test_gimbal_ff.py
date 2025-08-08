#  -*- coding:utf-8 -*-
import math
import time
import cv2
from robomaster import robot
from robomaster import gimbal, blaster, camera


# ---------- PID Controller ----------
class PID:
    def __init__(self, kp=3.0, ki=0.0, kd=0.4):  # ปรับ PID ให้ aggressive ขึ้น
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.prev_error = 0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


# ---------- คำนวณมุม yaw จากตำแหน่งเป้า ----------
def calculate_yaw(target_x, target_z=1.0):
    return math.degrees(math.atan2(target_x, target_z))


# ---------- หมุน gimbal และยิงเลเซอร์ ----------
def aim_and_fire(ep_gimbal, ep_blaster, pid, target_yaw, timeout=2.0, threshold=0.4):
    dt = 0.03
    start_time = time.time()

    while time.time() - start_time < timeout:
        yaw = ep_gimbal.get_angle()[1]  # [pitch, yaw]
        error = target_yaw - yaw

        if abs(error) < threshold:
            break

        speed = pid.compute(error, dt)
        speed = max(min(speed, 300), -300)  # ความเร็ว gimbal สูงสุด
        ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=speed)
        time.sleep(dt)

    # หยุดหมุน
    ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)

    # ยิง!
    print(f"[INFO] Aimed at {target_yaw:.2f}°, Firing!")
    ep_blaster.fire()
    time.sleep(0.3)  # หน่วงหลังยิงเพื่อความเสถียร


# ---------- Main ----------
if __name__ == '__main__':
    # เชื่อมต่อหุ่นยนต์
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")  # หรือ "ap" แล้วแต่การเชื่อมต่อ

    ep_gimbal = ep_robot.gimbal
    ep_blaster = ep_robot.blaster
    ep_camera = ep_robot.camera

    # เริ่มกล้อง (กรณีต้องการแสดงผล)
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)

    # สร้าง PID controller
    pid = PID(kp=3.0, ki=0.0, kd=0.4)

    # พิกัดเป้า: ซ้าย (-0.6m), กลาง (0), ขวา (+0.6m)
    x_targets = [-0.6, 0.0, 0.6]
    z_distance = 1.0  # ห่างจากหุ่นถึงเป้ากลาง
    yaw_targets = [calculate_yaw(x, z_distance) for x in x_targets]

    # ลำดับการยิง: ซ้าย → กลาง → ขวา → กลาง → ซ้าย
    sequence = [0, 1, 2, 1, 0]

    print("[START] Begin aiming and shooting sequence...")
    for i in sequence:
        target_yaw = yaw_targets[i]
        aim_and_fire(ep_gimbal, ep_blaster, pid, target_yaw)
        time.sleep(0.2)  # พักสั้นๆก่อนยิงรอบถัดไป

    # จบการทำงาน
    ep_camera.stop_video_stream()
    cv2.destroyAllWindows()
    ep_robot.close()
    print("[DONE] Mission complete: 5 shots fired.")