import robomaster
from robomaster import robot
import csv
from datetime import datetime
import time
import numpy as np
import math
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt

# CONFIG
file_name = 'tof_40_raw.csv'
chod = 70 # 7 cm
FILTER_MODE = "low_pass"  # raw, moving_average, median_filter, low_pass
WINDOW_SIZE = 5              # สำหรับ Moving Average และ Median
STOP_THRESHOLD = 400.0 + chod        # cm
SPEED = 0.25                   # m/s
DURATION = 30                  # วินาที
SAMPLING_FREQ = 20            # Hz สำหรับ LPF
CUTOFF_FREQ = 1.0             # Hz สำหรับ LPF

# GLOBALS
csv_writer = None
csv_file = None
tof_buffer = []
latest_filtered_tof = None
t_detect = None
t_stop_command = None
t_start = None  # เพิ่มตัวแปรเวลาเริ่มต้น
lpf_filter = None  # สำหรับ Low-Pass Filter instance

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
    if len(data) < window_size:
        return sum(data) / len(data)
    else:
        return sum(data[-window_size:]) / window_size

def apply_median_filter(data, window_size):
    if len(data) < window_size:
        return data[-1]  # ค่าใหม่ล่าสุด
    else:
        filtered = median_filter(data[-window_size:], size=window_size)
        return filtered[-1]

def butterworth_lpf(cutoff_freq, sampling_freq, order=4):
    nyquist = sampling_freq / 2
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_lpf(data):
    """Simple Low-Pass Filter using LowPassFilter class"""
    global lpf_filter
    if lpf_filter is None:
        lpf_filter = LowPassFilter(CUTOFF_FREQ, SAMPLING_FREQ)
    
    # ใช้ค่าล่าสุดกับ filter
    return lpf_filter.filter(data[-1])

# === Filtering Handler ===

def filter_tof(raw_value):
    tof_buffer.append(raw_value)
    if FILTER_MODE == "moving_average":
        return moving_average(tof_buffer, WINDOW_SIZE)
    elif FILTER_MODE == "median_filter":
        return apply_median_filter(tof_buffer, WINDOW_SIZE)
    elif FILTER_MODE == "low_pass":
        return apply_lpf(tof_buffer)
    else:
        return raw_value

# === CSV Logging ===

def write_tof_data_to_csv(raw_tof, filtered_tof, timestamp_iso):
    global csv_writer
    if csv_writer:
        csv_writer.writerow([
            timestamp_iso,
            raw_tof,
            filtered_tof,
            t_detect,
            t_stop_command,
            t_stop_command - t_detect if t_detect and t_stop_command else None,
            t_detect - t_start if t_detect and t_start else None  # เวลารวมตั้งแต่เริ่มจนตรวจจับ
        ])

# === Callback ===

def tof_data_handler(sub_info):
    global latest_filtered_tof, t_detect, t_stop_command

    raw_tof = sub_info[0]
    filtered_tof = filter_tof(raw_tof)
    latest_filtered_tof = filtered_tof

    timestamp_iso = datetime.now().isoformat()
    print(f"Raw TOF: {raw_tof:.2f} | Filtered TOF: {filtered_tof:.2f}")

    write_tof_data_to_csv(raw_tof, filtered_tof, timestamp_iso)

    if filtered_tof <= STOP_THRESHOLD and t_detect is None:
        t_detect = time.time()
        ep_chassis.drive_speed(x=0.0, y=0, z=0)
        t_stop_command = time.time()
        
        response_time = t_stop_command - t_detect
        total_time = t_detect - t_start if t_start else 0
        
        print(f"🛑 Object detected! Filtered TOF: {filtered_tof:.2f} cm")
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
            "raw_tof",
            "filtered_tof",
            "t_detect",
            "t_stop_command",
            "response_time",
            "total_time_from_start"  # เพิ่มคอลัมน์เวลารวม
        ])
    except IOError as e:
        print(f"Error opening CSV file: {e}")
        ep_robot.close()
        exit()

    print("✅ Start robot movement and ToF sensing...")
    print(f"🎯 Stop threshold: {STOP_THRESHOLD:.1f} mm")
    print(f"🚀 Speed: {SPEED} m/s")
    print(f"🔍 Filter mode: {FILTER_MODE}")
    
    # สร้าง Low-Pass Filter instance
    if FILTER_MODE == "low_pass":
        lpf_filter = LowPassFilter(CUTOFF_FREQ, SAMPLING_FREQ)
        print(f"📊 Low-Pass Filter: fc={CUTOFF_FREQ}Hz, fs={SAMPLING_FREQ}Hz, alpha={lpf_filter.alpha:.4f}")
    
    t_start = time.time()  # บันทึกเวลาเริ่มต้น
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