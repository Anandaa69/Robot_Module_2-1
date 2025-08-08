# -*-coding:utf-8-*-
# Lab PID Tuning Robomaster - Complete System (Simplified)
# นักศึกษา: [ชื่อ-นามสกุล] กลุ่ม: [เลขกลุ่ม]
# Copyright (c) 2020 DJI.

import cv2
import time
import numpy as np
import robomaster
from robomaster import robot
from robomaster import vision
from robomaster import blaster


class PIDController:
    """PID Controller สำหรับควบคุมมุม Gimbal"""
    def __init__(self, kp=1.0, ki=0.0, kd=0.0):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain  
        self.kd = kd  # Derivative gain
        
        self.previous_error = 0
        self.integral = 0
        self.previous_time = time.time()
        
    def compute(self, setpoint, current_value):
        """คำนวณค่า PID output"""
        current_time = time.time()
        dt = current_time - self.previous_time
        
        if dt <= 0:
            dt = 0.01
            
        error = setpoint - current_value
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        self.integral += error * dt
        i_term = self.ki * self.integral
        
        # Derivative term
        derivative = (error - self.previous_error) / dt
        d_term = self.kd * derivative
        
        # PID output
        output = p_term + i_term + d_term
        
        # จำกัดค่า output
        output = np.clip(output, -540, 540)
        
        # Update for next iteration
        self.previous_error = error
        self.previous_time = current_time
        
        return output, error


class MarkerInfo:
    """คลาสสำหรับเก็บข้อมูล Marker"""
    def __init__(self, x, y, w, h, info):
        self._x = x
        self._y = y
        self._w = w
        self._h = h
        self._info = info

    @property
    def pt1(self):
        return int((self._x - self._w / 2) * 1280), int((self._y - self._h / 2) * 720)

    @property
    def pt2(self):
        return int((self._x + self._w / 2) * 1280), int((self._y + self._h / 2) * 720)

    @property
    def center(self):
        return int(self._x * 1280), int(self._y * 720)

    @property
    def text(self):
        return self._info


class GimbalTargetingSystem:
    """ระบบควบคุม Gimbal แบบ PID พร้อม Vision"""
    def __init__(self):
        # PID Controllers สำหรับ Pitch และ Yaw (ปรับค่าตามการทดลอง)
        self.pid_yaw = PIDController(kp=3.0, ki=0.1, kd=1.0)    
        self.pid_pitch = PIDController(kp=2.5, ki=0.05, kd=0.8)
        
        # ข้อมูล Marker
        self.markers = []
        self.heart_detected = False
        self.current_target_name = ""
        
        # ข้อมูลมุมปัจจุบัน (จำลอง)
        self.current_yaw = 0
        self.current_pitch = 0

    def on_detect_marker(self, marker_info):
        """Callback function สำหรับตรวจจับ Marker"""
        number = len(marker_info)
        self.markers.clear()
        self.heart_detected = False
        
        for i in range(0, number):
            x, y, w, h, info = marker_info[i]
            self.markers.append(MarkerInfo(x, y, w, h, info))
            print(f"🎯 Marker detected: {info} at ({x:.3f}, {y:.3f})")
            
            # ตรวจสอบรูปหัวใจ (ปรับ marker ID ตามที่ใช้จริง)
            if "heart" in str(info).lower() or info == 1:
                self.heart_detected = True
                print(f"❤️  Heart detected at target: {self.current_target_name}")

    def sub_angle_handler(self, angle_info):
        """รับค่ามุมจาก gimbal sensor"""
        pitch_angle, yaw_angle, pitch_ground_angle, yaw_ground_angle = angle_info
        self.current_yaw = yaw_angle
        self.current_pitch = pitch_angle
        print(f"📐 Current angles - Yaw: {yaw_angle:.1f}°, Pitch: {pitch_angle:.1f}°")

    def pid_move_to_target(self, ep_gimbal, target_yaw, target_pitch, max_time=3.0):
        """เคลื่อนที่ไปยังเป้าด้วย PID Control"""
        print(f"🎯 PID Control: Moving to Yaw={target_yaw}°, Pitch={target_pitch}°")
        
        # Subscribe to angle data
        ep_gimbal.sub_angle(freq=10, callback=self.sub_angle_handler)
        
        start_time = time.time()
        tolerance = 2.0  # ความผิดพลาดที่ยอมรับได้ (องศา)
        
        # Reset PID controllers
        self.pid_yaw.integral = 0
        self.pid_pitch.integral = 0
        
        while (time.time() - start_time) < max_time:
            # คำนวณ PID output
            yaw_output, yaw_error = self.pid_yaw.compute(target_yaw, self.current_yaw)
            pitch_output, pitch_error = self.pid_pitch.compute(target_pitch, self.current_pitch)
            
            print(f"  Yaw: {self.current_yaw:.1f}° (err={yaw_error:.1f}°), "
                  f"Pitch: {self.current_pitch:.1f}° (err={pitch_error:.1f}°)")
            
            # ตรวจสอบว่าถึงเป้าหรือยัง
            if abs(yaw_error) < tolerance and abs(pitch_error) < tolerance:
                print(f"✅ Target reached! Yaw={self.current_yaw:.1f}°, Pitch={self.current_pitch:.1f}°")
                break
            
            # ส่งคำสั่งควบคุมความเร็ว
            ep_gimbal.drive_speed(pitch_speed=pitch_output, yaw_speed=yaw_output)
            
            time.sleep(0.1)  # 10Hz control loop
        
        # หยุดการเคลื่อนไหว
        ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
        time.sleep(0.2)
        
        # Unsubscribe angle data
        ep_gimbal.unsub_angle()

    def wait_for_heart_detection(self, timeout=3.0):
        """รอการตรวจจับรูปหัวใจ"""
        self.heart_detected = False
        start_time = time.time()
        
        print(f"🔍 Scanning for heart marker (timeout: {timeout}s)...")
        
        while not self.heart_detected and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        return self.heart_detected

    def shoot_with_feedback(self, ep_blaster, target_name):
        """ยิงพร้อมเอฟเฟกต์ LED"""
        print(f"🔥 FIRING at {target_name}!")
        
        # LED shooting effect - กะพริบเร็ว
        for _ in range(5):
            ep_blaster.set_led(brightness=255, effect=blaster.LED_ON)
            time.sleep(0.05)
            ep_blaster.set_led(brightness=0, effect=blaster.LED_OFF)
            time.sleep(0.05)
        
        # LED สีเขียวแสดงการยิงสำเร็จ
        ep_blaster.set_led(brightness=128, effect=blaster.LED_ON)
        time.sleep(0.5)
        
        print(f"✅ Shot fired at {target_name}!")


def main():
    """ฟังก์ชันหลักสำหรับการทำแลป"""
    print("🚀 LAB PID TUNING ROBOMASTER")
    print("🎯 Vision-Guided Shooting with PID Control")
    print("="*60)
    
    # สร้างระบบควบคุม
    targeting_system = GimbalTargetingSystem()
    
    # เชื่อมต่อหุ่นยนต์
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    
    ep_gimbal = ep_robot.gimbal
    ep_blaster = ep_robot.blaster
    ep_vision = ep_robot.vision
    ep_camera = ep_robot.camera
    
    try:
        # เริ่มระบบ Vision
        ep_camera.start_video_stream(display=False)
        ep_vision.sub_detect_info(name="marker", callback=targeting_system.on_detect_marker)
        
        # ตั้งค่า LED เริ่มต้น
        ep_blaster.set_led(brightness=64, effect=blaster.LED_ON)
        
        # กำหนดเป้า (ตามโจทย์แลป)
        targets = [
            ("เป้าซ้าย", 31, 0),      # (ชื่อเป้า, yaw, pitch)
            ("เป้ากลาง", 0, 0),
            ("เป้าขวา", -31, 0),
            ("เป้ากลาง", 0, 0),
            ("เป้าซ้าย", 31, 0)
        ]
        
        print("🎯 เริ่มภารกิจยิงเป้า")
        print("📋 ลำดับ: ซ้าย → กลาง → ขวา → กลาง → ซ้าย")
        print("🔧 PID Parameters:")
        print(f"   Yaw PID:   Kp={targeting_system.pid_yaw.kp}, Ki={targeting_system.pid_yaw.ki}, Kd={targeting_system.pid_yaw.kd}")
        print(f"   Pitch PID: Kp={targeting_system.pid_pitch.kp}, Ki={targeting_system.pid_pitch.ki}, Kd={targeting_system.pid_pitch.kd}")
        print()
        
        # Reset gimbal ไปตำแหน่งเริ่มต้น
        print("🔄 Resetting gimbal to center position...")
        ep_gimbal.recenter().wait_for_completed()
        time.sleep(1)
        
        shots_fired = 0
        targets_hit = 0
        
        for i, (target_name, yaw, pitch) in enumerate(targets, 1):
            targeting_system.current_target_name = target_name
            print(f"\n{'='*50}")
            print(f"🎯 TARGET {i}/5: {target_name}")
            print(f"📍 Target coordinates: Yaw={yaw}°, Pitch={pitch}°")
            
            # ใช้ PID Control เคลื่อนที่ไปเป้า
            targeting_system.pid_move_to_target(ep_gimbal, yaw, pitch)
            
            # ตรวจหา Heart Marker
            if targeting_system.wait_for_heart_detection(timeout=3.0):
                # พบหัวใจ - ยิง!
                targeting_system.shoot_with_feedback(ep_blaster, target_name)
                shots_fired += 1
                targets_hit += 1
                print(f"🎉 SUCCESS! Hit target: {target_name}")
            else:
                # ไม่พบหัวใจ - ข้าม
                print(f"❌ No heart detected at {target_name} - SKIP")
                ep_blaster.set_led(brightness=32, effect=blaster.LED_ON)
                time.sleep(0.3)
            
            # พักระหว่างเป้า
            time.sleep(0.5)
        
        print(f"\n{'='*60}")
        print("🏁 MISSION COMPLETED!")
        print(f"📊 RESULTS:")
        print(f"   • Total targets: 5")
        print(f"   • Shots fired: {shots_fired}")
        print(f"   • Targets hit: {targets_hit}")
        print(f"   • Accuracy: {(targets_hit/5)*100:.1f}%")
        
        # ประเมินผลการทำงาน
        if targets_hit == 5:
            print("🏆 EXCELLENT! Perfect accuracy!")
        elif targets_hit >= 3:
            print("👍 GOOD! Most targets hit!")
        else:
            print("💪 NEEDS IMPROVEMENT! Try tuning PID parameters.")
        
        # Reset gimbal กลับตำแหน่งเริ่มต้น
        print("\n🔄 Returning to center position...")
        ep_gimbal.recenter().wait_for_completed()
        
        # ปิด LED
        ep_blaster.set_led(brightness=0, effect=blaster.LED_OFF)
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        
    finally:
        # ปิดระบบทั้งหมด
        try:
            ep_vision.unsub_detect_info(name="marker")
            cv2.destroyAllWindows()
            ep_camera.stop_video_stream()
            ep_robot.close()
            print("🔚 System shutdown complete")
        except:
            pass


if __name__ == '__main__':
    main()