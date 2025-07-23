import robomaster
from robomaster import robot
import csv
from datetime import datetime
import time
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt

# CONFIG
FILTER_MODE = "median_filter"  # raw, moving_average, median_filter, low_pass
WINDOW_SIZE = 5                # สำหรับ Moving Average และ Median
STOP_THRESHOLD = 40.0          # cm
SPEED = 0.25                   # m/s
DURATION = 30                  # วินาที
SAMPLING_FREQ = 20            # Hz สำหรับ LPF
CUTOFF_FREQ = 2.0             # Hz สำหรับ LPF

# GLOBALS
csv_writer = None
csv_file = None
tof_buffer = []
latest_filtered_tof = None
t_detect = None
t_stop_command = None

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
    if len(data) < WINDOW_SIZE:
        return data[-1]
    b, a = butterworth_lpf(CUTOFF_FREQ, SAMPLING_FREQ)
    filtered = filtfilt(b, a, data[-WINDOW_SIZE:])
    return filtered[-1]

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
            t_stop_command - t_detect if t_detect and t_stop_command else None
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
        print(f"🛑 Object detected! Response time = {t_stop_command - t_detect:.4f} sec")

# === Main Script ===

if __name__ == "__main__":
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_sensor = ep_robot.sensor
    ep_chassis = ep_robot.chassis

    try:
        csv_file = open("J.Sahapong/lab4/data/test_filtered.csv", "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "timestamp_iso",
            "raw_tof",
            "filtered_tof",
            "t_detect",
            "t_stop_command",
            "response_time"
        ])
    except IOError as e:
        print(f"Error opening CSV file: {e}")
        ep_robot.close()
        exit()

    print("✅ Start robot movement and ToF sensing...")
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
