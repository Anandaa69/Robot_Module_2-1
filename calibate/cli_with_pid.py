# -*-coding:utf-8-*-
# Copyright (c) 2020 DJI.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License in the file LICENSE.txt or at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import robomaster
from robomaster import robot
import csv
from datetime import datetime
import time
import numpy as np
import math
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt

# --- CONFIG ---
file_name = 'tof_pid_control_data.csv'
FILTER_MODE = "low_pass"            # raw, moving_average, median_filter, low_pass
WINDOW_SIZE = 5                          # สำหรับ Moving Average และ Median

# --- ค่า Calibration Factors ---
CALIBRATION_SLOPE = 0.0894 
CALIBRATION_Y_INTERCEPT = 3.8409 

# --- Control Parameters ---
STOP_THRESHOLD = 50.0                    # cm - ระยะทางเป้าหมายที่ต้องการหยุด (เดิม)
MAX_SPEED = 0.5                          # m/s - ความเร็วสูงสุด (เพิ่มจาก 0.25)
MIN_SPEED = 0.25                         # m/s - ความเร็วต่ำสุดสำหรับการปรับแต่งละเอียด
CONTROL_ZONE = 100.0                     # cm - ระยะที่เริ่มใช้ PID control

# --- PID Parameters ---
KP = 0.01                                # Proportional gain
KI = 0.0008                              # Integral gain  
KD = 0.003                               # Derivative gain
PID_TOLERANCE = 2.0                      # cm - ความคลาดเคลื่อนที่ยอมรับได้

# --- Other Parameters ---
DURATION = 5                            # วินาที
SAMPLING_FREQ = 25                       # Hz สำหรับ ToF sensor
CUTOFF_FREQ = 2.0                        # Hz สำหรับ LPF
# -----------------

# --- GLOBALS ---
csv_writer = None
csv_file = None
tof_buffer = []
latest_filtered_tof = None
t_detect = None
t_stop_command = None
t_start = None
lpf_filter = None

# PID Variables
control_active = False
current_speed = MAX_SPEED

# === PID Controller Class ===
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None
        
    def update(self, error, current_time):
        if self.last_time is None:
            self.last_time = current_time
            dt = 0.04  # default dt สำหรับ 25Hz
        else:
            dt = current_time - self.last_time
            
        if dt <= 0:
            dt = 0.04
            
        # Proportional term
        proportional = self.kp * error
        
        # Integral term with anti-windup
        if abs(error) < 10.0:  # จำกัด integral เมื่อ error ไม่มากเกินไป
            self.integral += error * dt
            self.integral = max(min(self.integral, 100), -100)  # Anti-windup
        
        integral = self.ki * self.integral
        
        # Derivative term
        derivative = self.kd * (error - self.last_error) / dt
        
        # PID output
        output = proportional + integral + derivative
        
        # Update for next iteration
        self.last_error = error
        self.last_time = current_time
        
        return output, proportional, integral, derivative
    
    def reset(self):
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None

# === Calibration Function ===
def calibrate_tof_value(raw_tof_value_from_sensor):
    calibrated_value = (CALIBRATION_SLOPE * raw_tof_value_from_sensor) + CALIBRATION_Y_INTERCEPT
    return calibrated_value

# === Low-Pass Filter Class ===
class LowPassFilter:
    def __init__(self, cutoff_freq, sample_rate):
        self.dt = 1.0 / sample_rate
        self.alpha = (2 * math.pi * cutoff_freq * self.dt) / (2 * math.pi * cutoff_freq * self.dt + 1)
        self.last_output = None
    
    def filter(self, new_value):
        if self.last_output is None:
            self.last_output = new_value
        else:
            self.last_output = self.alpha * new_value + (1 - self.alpha) * self.last_output
        return self.last_output

# === Filter Functions ===
def moving_average(data, window_size):
    if len(data) == 0:
        return 0.0 
    if len(data) < window_size:
        return sum(data) / len(data)
    else:
        return sum(data[-window_size:]) / window_size

def apply_median_filter(data, window_size):
    if len(data) == 0:
        return 0.0 
    if len(data) < window_size:
        return data[-1] 
    else:
        filtered = median_filter(data[-window_size:], size=window_size)
        return filtered[-1]

def apply_lpf(data):
    global lpf_filter
    if lpf_filter is None:
        lpf_filter = LowPassFilter(CUTOFF_FREQ, SAMPLING_FREQ)
    
    return lpf_filter.filter(data[-1])

def filter_tof(calibrated_value):
    tof_buffer.append(calibrated_value)
    if FILTER_MODE == "moving_average":
        return moving_average(tof_buffer, WINDOW_SIZE)
    elif FILTER_MODE == "median_filter":
        return apply_median_filter(tof_buffer, WINDOW_SIZE)
    elif FILTER_MODE == "low_pass":
        return apply_lpf(tof_buffer)
    else:
        return calibrated_value 

# === CSV Logging ===
def write_tof_data_to_csv(raw_tof, calibrated_tof, filtered_tof, timestamp_iso, 
                         current_speed=None, pid_output=None, error=None, control_mode=None):
    global csv_writer
    if csv_writer:
        csv_writer.writerow([
            timestamp_iso,
            raw_tof,
            calibrated_tof, 
            filtered_tof,
            current_speed,
            pid_output,
            error,
            control_mode,
            t_detect,
            t_stop_command,
            t_stop_command - t_detect if t_detect and t_stop_command else None,
            t_detect - t_start if t_detect and t_start else None
        ])

# === Main Control Logic ===
pid_controller = PIDController(KP, KI, KD)

def tof_data_handler(sub_info):
    global latest_filtered_tof, t_detect, t_stop_command, control_active, current_speed

    raw_tof_mm = sub_info[0] 
    calibrated_tof_cm = calibrate_tof_value(raw_tof_mm) 
    filtered_tof_cm = filter_tof(calibrated_tof_cm) 
    latest_filtered_tof = filtered_tof_cm

    current_time = time.time()
    timestamp_iso = datetime.now().isoformat()
    
    # คำนวณ error (ระยะทางที่เหลือจาก STOP_THRESHOLD)
    error = filtered_tof_cm - STOP_THRESHOLD
    
    control_mode = "CRUISE"
    pid_output = 0
    
    # ตัดสินใจว่าจะใช้ PID control หรือไม่
    if filtered_tof_cm <= CONTROL_ZONE and abs(error) > PID_TOLERANCE:
        # เข้าสู่โหมด PID control
        if not control_active:
            control_active = True
            print(f"🎯 PID Control activated at distance: {filtered_tof_cm:.2f} cm")
        
        control_mode = "PID_ACTIVE"
        pid_output, p_term, i_term, d_term = pid_controller.update(error, current_time)
        
        # แปลง PID output เป็นความเร็ว
        if error > PID_TOLERANCE:  # ยังห่างจากเป้าหมาย
            # ความเร็วขึ้นอยู่กับ error และ PID output
            speed_from_error = (error / CONTROL_ZONE) * MAX_SPEED
            speed_adjustment = pid_output * 0.02  # ปรับ scaling factor ตามต้องการ
            current_speed = max(min(speed_from_error + speed_adjustment, MAX_SPEED), MIN_SPEED)
            
        elif error < -PID_TOLERANCE:  # ใกล้เกินไป (ถ้ามี overshoot)
            current_speed = MIN_SPEED * 0.5  # ชะลอมาก
            
        else:  # อยู่ในช่วง tolerance แล้ว
            current_speed = 0.0
            control_mode = "TARGET_REACHED"
        
        print(f"PID: Err={error:.2f}, P={p_term:.4f}, I={i_term:.4f}, D={d_term:.4f}, Out={pid_output:.4f}, Spd={current_speed:.3f}")
        
    elif filtered_tof_cm > CONTROL_ZONE:
        # โหมดความเร็วปกติ
        control_active = False
        current_speed = MAX_SPEED
        control_mode = "FULL_SPEED"
        
    else:
        # อยู่ในช่วง tolerance หรือถึงเป้าหมายแล้ว
        current_speed = 0.0
        control_mode = "STOPPED"
    
    # ส่งคำสั่งควบคุมความเร็ว
    ep_chassis.drive_speed(x=current_speed, y=0, z=0)
    
    # แสดงผลข้อมูล
    print(f"Raw: {raw_tof_mm:.1f}mm | Cal: {calibrated_tof_cm:.1f}cm | Filt: {filtered_tof_cm:.1f}cm | Speed: {current_speed:.3f}m/s | Mode: {control_mode}")
    
    # บันทึกข้อมูล
    write_tof_data_to_csv(raw_tof_mm, calibrated_tof_cm, filtered_tof_cm, timestamp_iso, 
                         current_speed, pid_output, error, control_mode)

    # เงื่อนไขการหยุดสุดท้าย - ใช้ STOP_THRESHOLD เดิม
    if abs(error) <= PID_TOLERANCE and current_speed == 0.0 and t_detect is None:
        t_detect = time.time()
        ep_chassis.drive_speed(x=0.0, y=0, z=0, timeout=0.75)
        t_stop_command = time.time()
        
        response_time = t_stop_command - t_detect
        total_time = t_detect - t_start if t_start else 0
        
        print(f"🛑 Target reached! Final distance: {filtered_tof_cm:.2f} cm")
        print(f"🎯 Target was: {STOP_THRESHOLD:.2f} cm")
        print(f"📏 Final error: {error:.2f} cm")
        print(f"⏱️ Detection time: {t_detect:.4f} sec")
        print(f"🚀 Stop command time: {t_stop_command:.4f} sec")
        print(f"⏳ Response time = {response_time:.4f} sec")
        print(f"🕐 Total time from start = {total_time:.4f} sec")

# === Main Script ===
if __name__ == "__main__":
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_sensor = ep_robot.sensor
    ep_chassis = ep_robot.chassis
    ep_gimbal = ep_robot.gimbal

    print("✅ Recalibrating gimbal...")
    ep_gimbal.recenter(pitch_speed=100, yaw_speed=100).wait_for_completed()
    print("✅ Gimbal recalibrated.")

    try:
        csv_file = open(f"J.Sahapong/lab4/data/{file_name}", "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "timestamp_iso",
            "raw_tof_mm",
            "calibrated_tof_cm",
            "filtered_tof_cm",
            "current_speed_ms",
            "pid_output",
            "error_cm",
            "control_mode",
            "t_detect",
            "t_stop_command",
            "response_time",
            "total_time_from_start"
        ])
    except IOError as e:
        print(f"Error opening CSV file: {e}")
        ep_robot.close()
        exit()

    print("✅ Start robot movement with PID feedback control...")
    print(f"🎯 Stop threshold: {STOP_THRESHOLD:.1f} cm")
    print(f"🚀 Max speed: {MAX_SPEED} m/s")
    print(f"🐌 Min speed: {MIN_SPEED} m/s") 
    print(f"🔍 Filter mode: {FILTER_MODE}")
    print(f"🎛️ PID gains: Kp={KP}, Ki={KI}, Kd={KD}")
    print(f"📏 Control zone: {CONTROL_ZONE} cm")
    print(f"✅ PID tolerance: ±{PID_TOLERANCE} cm")
    
    if FILTER_MODE == "low_pass":
        lpf_filter = LowPassFilter(CUTOFF_FREQ, SAMPLING_FREQ)
        print(f"📊 Low-Pass Filter: fc={CUTOFF_FREQ}Hz, fs={SAMPLING_FREQ}Hz")
    
    t_start = time.time()
    ep_chassis.drive_speed(x=current_speed, y=0, z=0)
    ep_sensor.sub_distance(freq=SAMPLING_FREQ, callback=tof_data_handler)

    time.sleep(DURATION)

    print("⏹ Stopping...")
    ep_sensor.unsub_distance()
    ep_chassis.drive_speed(x=0, y=0, z=0)
    ep_robot.close()

    if csv_file:
        csv_file.close()
        print("💾 Data saved. Robot stopped.")