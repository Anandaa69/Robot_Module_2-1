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

# BEST 25hz sensor distance + 10cm use median_filter window = 5

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
file_name = 'tof_calibrated_data.csv' # เปลี่ยนชื่อไฟล์ให้สื่อถึงการใช้ค่าสอบเทียบ
FILTER_MODE = "median_filter"            # raw, moving_average, median_filter, low_pass
WINDOW_SIZE = 5                     # สำหรับ Moving Average และ Median

# --- ค่า Calibration Factors ที่ได้จาก Linear Regression ---
# สมการของคุณคือ: ระยะทางจริง = (0.0894 * ค่าเซ็นเซอร์) + (3.8409)
# ค่าเซ็นเซอร์ (raw_tof) ที่ได้รับจาก RoboMaster EP ToF เป็นมิลลิเมตร (mm)
# ดังนั้น หากคุณต้องการให้ calibrated_tof เป็นเซนติเมตร (cm), 
# ค่า Slope (0.0894) และ Y-intercept (3.8409) จะต้องแปลงจาก mm -> cm
# หรือถ้าคุณต้องการให้ calibrated_tof เป็น mm, ค่า Slope ควรเป็น mm/mm และ Y-intercept เป็น mm
# **สมมติว่าคุณต้องการให้ผลลัพธ์ calibrated_tof เป็น cm:**
# จากข้อมูลที่คุณเคยให้มา (sensor_readings เป็นหน่วยดิบ 170-1062, actual_distances เป็น cm 20-100)
# ดังนั้น Slope 0.0894 คือ cm/raw_unit และ Y-intercept 3.8409 คือ cm
CALIBRATION_SLOPE = 0.0894 
CALIBRATION_Y_INTERCEPT = 3.8409 

# STOP_THRESHOLD จะใช้กับค่าที่ "สอบเทียบแล้ว" ซึ่งควรอยู่ในหน่วยเดียวกันกับ calibrated_tof
# ถ้า calibrated_tof เป็น cm, STOP_THRESHOLD ก็ควรเป็น cm
# ถ้า calibrated_tof เป็น mm, STOP_THRESHOLD ก็ควรเป็น mm
# ตัวอย่าง: ถ้าต้องการให้หยุดที่ 40 cm (จากเซ็นเซอร์)
STOP_THRESHOLD = 50 # cm (ค่านี้จะนำไปเปรียบเทียบกับ calibrated_tof)
# ถ้า 400.0 ในโค้ดเก่าหมายถึง 400 mm (40 cm) และ chod คือระยะชดเชย
# ตอนนี้เราไม่ต้องมี chod แล้ว เพราะค่าเซ็นเซอร์ถูกแปลงเป็นระยะจริงแล้ว
# และ STOP_THRESHOLD คือระยะจริงที่คุณต้องการให้หุ่นยนต์หยุด
# ถ้าคุณต้องการให้หยุดที่ 40 cm ให้ตั้งเป็น 40.0

SPEED = 0.25                  # m/s
DURATION = 30                 # วินาที
SAMPLING_FREQ = 25            # Hz สำหรับ LPF
CUTOFF_FREQ = 1.0             # Hz สำหรับ LPF
# -----------------

# --- GLOBALS ---
csv_writer = None
csv_file = None
tof_buffer = []
latest_filtered_tof = None
t_detect = None
t_stop_command = None
t_start = None  # เพิ่มตัวแปรเวลาเริ่มต้น
lpf_filter = None  # สำหรับ Low-Pass Filter instance

# === Calibration Function ===

def calibrate_tof_value(raw_tof_value_from_sensor):
    """
    แปลงค่า ToF ดิบ (ซึ่งมาจากเซ็นเซอร์, มักเป็นมิลลิเมตร) 
    ให้เป็นระยะทางที่สอบเทียบแล้ว โดยใช้สมการ Linear Regression.
    
    Args:
        raw_tof_value_from_sensor (float): ค่าระยะทางดิบที่อ่านได้จากเซ็นเซอร์ ToF (จาก SDK มักเป็นมิลลิเมตร).

    Returns:
        float: ค่าระยะทางที่สอบเทียบแล้ว (หน่วยตามที่คุณสอบเทียบ, ในกรณีนี้คือ cm).
    """
    # ใช้ค่า Slope และ Y-intercept ที่กำหนดไว้
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

def butterworth_lpf(cutoff_freq, sampling_freq, order=4): # ไม่ได้ใช้ในปัจจุบัน แต่เก็บไว้เผื่ออนาคต
    nyquist = sampling_freq / 2
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_lpf(data):
    """Simple Low-Pass Filter using LowPassFilter class"""
    global lpf_filter
    if lpf_filter is None:
        lpf_filter = LowPassFilter(CUTOFF_FREQ, SAMPLING_FREQ)
    
    return lpf_filter.filter(data[-1])

# === Filtering Handler ===

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

def write_tof_data_to_csv(raw_tof, calibrated_tof, filtered_tof, timestamp_iso): # เพิ่ม calibrated_tof
    global csv_writer
    if csv_writer:
        csv_writer.writerow([
            timestamp_iso,
            raw_tof,
            calibrated_tof, 
            filtered_tof,
            t_detect,
            t_stop_command,
            t_stop_command - t_detect if t_detect and t_stop_command else None,
            t_detect - t_start if t_detect and t_start else None
        ])

# === Callback ===

def tof_data_handler(sub_info):
    global latest_filtered_tof, t_detect, t_stop_command

    # sub_info[0] คือค่าดิบจากเซ็นเซอร์ ToF ด้านหน้า (เป็นมิลลิเมตร)
    raw_tof_mm = sub_info[0] 

    # --- ขั้นตอนสำคัญ: ใช้ฟังก์ชันสอบเทียบตรงนี้ ---
    # แปลงค่า raw_tof_mm ให้เป็น calibrated_tof_cm (ตามหน่วยที่คุณสอบเทียบ)
    calibrated_tof_cm = calibrate_tof_value(raw_tof_mm) 
    
    # นำค่าที่สอบเทียบแล้ว ไปเข้าฟิลเตอร์ต่อ
    filtered_tof_cm = filter_tof(calibrated_tof_cm) 
    latest_filtered_tof = filtered_tof_cm

    timestamp_iso = datetime.now().isoformat()
    # ปรับการแสดงผลให้เห็นทั้ง raw, calibrated, และ filtered
    print(f"Raw: {raw_tof_mm:.2f}mm | Calibrated: {calibrated_tof_cm:.2f}cm | Filtered: {filtered_tof_cm:.2f}cm")

    write_tof_data_to_csv(raw_tof_mm, calibrated_tof_cm, filtered_tof_cm, timestamp_iso) 

    # ตรวจสอบเงื่อนไขการหยุด โดยใช้ค่าที่กรองแล้ว (และสอบเทียบแล้ว)
    # ทั้ง filtered_tof_cm และ STOP_THRESHOLD ควรมีหน่วยเป็น cm
    if filtered_tof_cm <= STOP_THRESHOLD and t_detect is None:
        t_detect = time.time()
        ep_chassis.drive_speed(x=0.0, y=0, z=0,timeout=0.75)
        t_stop_command = time.time()
        
        response_time = t_stop_command - t_detect
        total_time = t_detect - t_start if t_start else 0
        
        print(f"🛑 Object detected! Filtered TOF: {filtered_tof_cm:.2f} cm")
        print(f"⏱️ Detection time: {t_detect:.4f} sec")
        print(f"🚀 Stop command time: {t_stop_command:.4f} sec")
        print(f"⏳ Response time = {response_time:.4f} sec")
        print(f"🕐 Total time from start = {total_time:.4f} sec")

# === Main Script ===

if __name__ == "__main__":
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap") # หรือ "sta" ตามที่คุณเชื่อมต่อ
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
            "raw_tof_mm", # เปลี่ยนชื่อคอลัมน์ให้ชัดเจนขึ้น
            "calibrated_tof_cm", # เปลี่ยนชื่อคอลัมน์ให้ชัดเจนขึ้น
            "filtered_tof_cm", # เปลี่ยนชื่อคอลัมน์ให้ชัดเจนขึ้น
            "t_detect",
            "t_stop_command",
            "response_time",
            "total_time_from_start"
        ])
    except IOError as e:
        print(f"Error opening CSV file: {e}")
        ep_robot.close()
        exit()

    print("✅ Start robot movement and ToF sensing...")
    # ตรวจสอบหน่วยของ STOP_THRESHOLD: ตอนนี้เราคาดว่ามันคือ cm
    print(f"🎯 Stop threshold: {STOP_THRESHOLD:.1f} cm") 
    print(f"🚀 Speed: {SPEED} m/s")
    print(f"🔍 Filter mode: {FILTER_MODE}")
    
    if FILTER_MODE == "low_pass":
        lpf_filter = LowPassFilter(CUTOFF_FREQ, SAMPLING_FREQ)
        print(f"📊 Low-Pass Filter: fc={CUTOFF_FREQ}Hz, fs={SAMPLING_FREQ}Hz, alpha={lpf_filter.alpha:.4f}")
    
    t_start = time.time()
    ep_chassis.drive_speed(x=SPEED, y=0, z=0)
    ep_sensor.sub_distance(freq=SAMPLING_FREQ, callback=tof_data_handler)

    time.sleep(DURATION)

    print("⏹ Stopping...")
    ep_sensor.unsub_distance()
    ep_chassis.drive_speed(x=0, y=0, z=0)
    ep_robot.close()

    if csv_file:
        csv_file.close()
        print("💾 Data saved. Robot stopped.")