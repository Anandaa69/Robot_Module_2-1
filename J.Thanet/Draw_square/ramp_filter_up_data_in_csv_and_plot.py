# -*- coding: utf-8 -*-
from robomaster import robot
import time
import csv
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# -------------------------
# เตรียมไฟล์ CSV สำหรับบันทึกข้อมูล
# -------------------------
# กำหนดค่าพารามิเตอร์สำหรับ PID Controller
KP = 2.1
KI = 0.3
KD = 10
RAMP_UP_TIME = 0.7
ROTATE = 2.115

# จัดการชื่อไฟล์เพื่อไม่ให้มีอักขระที่ไม่ถูกต้อง
KP_str = str(KP).replace('.', '-')
KI_str = str(KI).replace('.', '-')
KD_str = str(KD).replace('.', '-')
RAMP_UP_TIME_str = str(RAMP_UP_TIME).replace('.', '-')

# สร้างชื่อไฟล์ log
log_filename = f"robot_log_{datetime.now().strftime('%H_%M_%S')}_P{KP_str}_I{KI_str}_D{KD_str}_ramp{RAMP_UP_TIME_str}.csv"
with open(log_filename, mode='w', newline='') as f:
    writer = csv.writer(f)
    # เพิ่มคอลัมน์สำหรับข้อมูลที่ผ่านการฟิลเตอร์ทั้งหมด
    writer.writerow(["time", "x", "y", "z", "pid_output", "target_distance", "relative_position", 
                     "x_ma", "y_ma", "x_ema", "y_ema", "x_median", "y_median"])

# -------------------------
# คลาสสำหรับ PID Controller
# -------------------------
class PID:
    def __init__(self, Kp, Ki, Kd, setpoint=0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.prev_error = 0
        self.integral = 0
        self.integral_max = 1.0

    def compute(self, current, dt):
        error = self.setpoint - current
        self.integral += error * dt
        
        # Limit the integral to prevent wind-up
        if self.integral > self.integral_max:
            self.integral = self.integral_max
        elif self.integral < -self.integral_max:
            self.integral = -self.integral_max
            
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output
    
    def reset(self):
        self.prev_error = 0
        self.integral = 0

# -------------------------
# ตัวแปรสำหรับ Callback
# -------------------------
current_x = 0.0
current_y = 0.0
current_z = 0.0
last_pid_output = None
current_target = None
current_relative = None

# เก็บประวัติข้อมูลเพื่อใช้กับฟิลเตอร์แบบเรียลไทม์
position_history = pd.DataFrame(columns=['x', 'y'])
window_size = 50  # ขนาดของ Moving Average และ Median filter
ema_alpha = 0.2   # ค่าสำหรับ Exponential Moving Average

# -------------------------
# ฟังก์ชันสำหรับฟิลเตอร์ต่างๆ (ใช้งานแบบ Real-time)
# -------------------------
def apply_filters(df):
    # Moving Average
    df['x_ma'] = df['x'].rolling(window=window_size, min_periods=1).mean()
    df['y_ma'] = df['y'].rolling(window=window_size, min_periods=1).mean()

    # Exponential Moving Average
    df['x_ema'] = df['x'].ewm(alpha=ema_alpha, adjust=False).mean()
    df['y_ema'] = df['y'].ewm(alpha=ema_alpha, adjust=False).mean()

    # Median Filter
    df['x_median'] = df['x'].rolling(window=window_size, min_periods=1).median()
    df['y_median'] = df['y'].rolling(window=window_size, min_periods=1).median()

    return df

# -------------------------
# ฟังก์ชัน Callback สำหรับข้อมูลตำแหน่ง
# -------------------------
def sub_position_handler(position_info):
    global current_x, current_y, current_z, last_pid_output, current_target, current_relative, position_history
    
    current_x = position_info[0]
    current_y = position_info[1]
    current_z = position_info[2]

    # เพิ่มข้อมูลใหม่เข้าไปใน history
    new_data = pd.DataFrame([{'x': current_x, 'y': current_y}])
    position_history = pd.concat([position_history, new_data], ignore_index=True)
    
    # ใช้ฟิลเตอร์กับข้อมูลทั้งหมดใน history เพื่อบันทึกผลลัพธ์
    filtered_df = apply_filters(position_history)
    
    # ดึงค่าล่าสุดที่ถูกฟิลเตอร์แล้ว
    last_filtered = filtered_df.iloc[-1]
    
    # บันทึกลง CSV ทุกครั้งที่มีข้อมูลตำแหน่งใหม่
    with open(log_filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            time.time(),
            current_x,
            current_y,
            current_z,
            last_pid_output if last_pid_output is not None else "",
            current_target if current_target is not None else "",
            current_relative if current_relative is not None else "",
            last_filtered['x_ma'],
            last_filtered['y_ma'],
            last_filtered['x_ema'],
            last_filtered['y_ema'],
            last_filtered['x_median'],
            last_filtered['y_median']
        ])
    
    print("chassis position: x:{0}, y:{1}, z:{2}".format(current_x, current_y, current_z))

# -------------------------
# ฟังก์ชันสำหรับเคลื่อนที่ด้วย PID Controller
# -------------------------
def move_forward_with_pid(ep_chassis, target_distance, axis, direction=1):
    global last_pid_output, current_target, current_relative
    
    pid = PID(Kp=KP, Ki=KI, Kd=KD, setpoint=target_distance)
    start_time = time.time()
    last_time = start_time
    target_reached = False
    
    ramp_up_time = RAMP_UP_TIME
    min_speed = 0.1
    max_speed = 1.5
    
    if axis == 'x':
        start_position = current_x
    else:
        start_position = current_y

    current_target = target_distance

    print(f"Moving forward {target_distance}m (tracking {axis}-axis, direction: {direction}) from position {start_position:.2f}")
    
    try:
        while not target_reached:
            now = time.time()
            dt = now - last_time
            last_time = now
            
            elapsed_time = now - start_time
            
            if axis == 'x':
                current_position = current_x
            else:
                current_position = current_y
            
            relative_position = abs(current_position - start_position)
            current_relative = relative_position
            output = pid.compute(relative_position, dt)
            
            if elapsed_time < ramp_up_time:
                ramp_multiplier = min_speed + (elapsed_time / ramp_up_time) * (1.0 - min_speed)
            else:
                ramp_multiplier = 1.0
            
            ramped_output = output * ramp_multiplier
            speed = max(min(ramped_output, max_speed), -max_speed)
            last_pid_output = speed
            
            ep_chassis.drive_speed(x=speed * direction, y=0, z=0, timeout=1)

            if abs(relative_position - target_distance) < 0.02:
                print(f"Target reached. Final position: {current_position:.2f}")
                ep_chassis.drive_speed(x=0, y=0, z=0, timeout=1)
                target_reached = True
                break
                
    except KeyboardInterrupt:
        print("Program interrupted by user.")
        ep_chassis.drive_speed(x=0, y=0, z=0, timeout=1)

# -------------------------
# ฟังก์ชันสำหรับหมุน 90 องศา
# -------------------------
def rotate_90_degrees(ep_chassis):
    print("=== Rotation: 90 degrees ===")
    time.sleep(0.25)
    
    print("Now turning 90 degrees...")
    rotation_time = ROTATE
    ep_chassis.drive_speed(x=0, y=0, z=45, timeout=rotation_time)
    time.sleep(rotation_time + 0.5)
    
    ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.5)
    time.sleep(0.25)
    print("Rotation completed!")

# -------------------------
# ฟังก์ชัน Low-Pass Filter (สำหรับ Post-Processing)
# -------------------------
def lowpass_filter(data, cutoff, fs, order=4):
    """
    Applies a Butterworth Low-Pass Filter to the data.
    """
    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

# -------------------------
# บล็อกการทำงานหลัก
# -------------------------
if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_chassis = ep_robot.chassis
    ep_chassis.sub_position(freq=20, callback=sub_position_handler)
    time.sleep(0.25)
    
    try:
        # ส่วนของการเคลื่อนที่หุ่นยนต์เป็นรูปสี่เหลี่ยม
        print("=== เริ่มการเคลื่อนที่แบบสี่เหลี่ยม ===")
        
        print("=== ด้านที่ 1: เคลื่อนที่ตามแกน +X ===")
        move_forward_with_pid(ep_chassis, 0.6, 'x', direction=1)
        rotate_90_degrees(ep_chassis)
        
        print("=== ด้านที่ 2: เคลื่อนที่ตามแกน +Y ===")
        time.sleep(0.25)
        move_forward_with_pid(ep_chassis, 0.6, 'y', direction=1)
        rotate_90_degrees(ep_chassis)
        
        print("=== ด้านที่ 3: เคลื่อนที่ตามแกน -X ===")
        time.sleep(0.25)
        move_forward_with_pid(ep_chassis, 0.6, 'x', direction=1)
        rotate_90_degrees(ep_chassis)
        
        print("=== ด้านที่ 4: เคลื่อนที่ตามแกน -Y ===")
        time.sleep(0.25)
        move_forward_with_pid(ep_chassis, 0.6, 'y', direction=1)
        rotate_90_degrees(ep_chassis)

        print("=== เคลื่อนที่สี่เหลี่ยมเสร็จสมบูรณ์! ===")
        
    except KeyboardInterrupt:
        print("โปรแกรมถูกขัดจังหวะโดยผู้ใช้")
    
    finally:
        print("การเคลื่อนที่ทั้งหมดเสร็จสมบูรณ์แล้ว!")
        try:
            ep_chassis.unsub_position()
        except:
            pass
        ep_robot.close()
        print(f"ข้อมูลถูกบันทึกไว้ในไฟล์: {log_filename}")

        # ---------------------------------------------
        # กราฟสำหรับฟิลเตอร์แบบ Real-time (Post-Processing)
        # ---------------------------------------------
        print("\n=== กำลังสร้างกราฟสำหรับฟิลเตอร์แบบเรียลไทม์ ===")
        try:
            df = pd.read_csv(log_filename)
            
            # สร้างกราฟสำหรับ Moving Average Filter
            plt.figure(figsize=(12, 6))
            plt.plot(df['time'], df['x'], label='Original X', alpha=0.5)
            plt.plot(df['time'], df['y'], label='Original Y', alpha=0.5)
            plt.plot(df['time'], df['x_ma'], label=f'Moving Average X (window={window_size})')
            plt.plot(df['time'], df['y_ma'], label=f'Moving Average Y (window={window_size})')
            plt.xlabel('Time (s)')
            plt.ylabel('Position (m)')
            plt.title('Moving Average Filtered Robot Position')
            plt.legend()
            plt.grid(True)
            plt.show()

            # สร้างกราฟสำหรับ Exponential Moving Average Filter
            plt.figure(figsize=(12, 6))
            plt.plot(df['time'], df['x'], label='Original X', alpha=0.5)
            plt.plot(df['time'], df['y'], label='Original Y', alpha=0.5)
            plt.plot(df['time'], df['x_ema'], label=f'EMA X (alpha={ema_alpha})')
            plt.plot(df['time'], df['y_ema'], label=f'EMA Y (alpha={ema_alpha})')
            plt.xlabel('Time (s)')
            plt.ylabel('Position (m)')
            plt.title('Exponential Moving Average Filtered Robot Position')
            plt.legend()
            plt.grid(True)
            plt.show()

            # สร้างกราฟสำหรับ Median Filter
            plt.figure(figsize=(12, 6))
            plt.plot(df['time'], df['x'], label='Original X', alpha=0.5)
            plt.plot(df['time'], df['y'], label='Original Y', alpha=0.5)
            plt.plot(df['time'], df['x_median'], label=f'Median Filter X (window={window_size})')
            plt.plot(df['time'], df['y_median'], label=f'Median Filter Y (window={window_size})')
            plt.xlabel('Time (s)')
            plt.ylabel('Position (m)')
            plt.title('Median Filtered Robot Position')
            plt.legend()
            plt.grid(True)
            plt.show()

            print("แสดงผลกราฟของ Real-time Filters เสร็จสิ้นแล้ว")

        except FileNotFoundError:
            print(f"ข้อผิดพลาด: ไม่พบไฟล์ {log_filename}")
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการประมวลผลข้อมูล: {e}")

        # ---------------------------------------------
        # Low-Pass Filter Post-Processing
        # ---------------------------------------------
        print("\n=== กำลังใช้ Low-Pass Filter สำหรับการประมวลผลข้อมูลในภายหลัง ===")
        try:
            df = pd.read_csv(log_filename)
            
            # ตั้งค่าพารามิเตอร์ของฟิลเตอร์
            fs = 20.0  # Sampling frequency ของหุ่นยนต์ (20 Hz)
            cutoff = 2.5 # Cutoff frequency (สามารถปรับค่าได้ตามต้องการ)
            order = 4 # Order ของฟิลเตอร์

            # ใช้ฟิลเตอร์กับข้อมูลตำแหน่งแกน x และ y
            df['x_lp_filtered'] = lowpass_filter(df['x'].values, cutoff, fs, order)
            df['y_lp_filtered'] = lowpass_filter(df['y'].values, cutoff, fs, order)
            
            # บันทึกข้อมูลที่ผ่านการกรองแล้วในไฟล์ CSV ใหม่
            lp_log_filename = log_filename.replace('.csv', '_lp_filtered.csv')
            df.to_csv(lp_log_filename, index=False)
            print(f"ข้อมูลที่ผ่านการกรองถูกบันทึกไว้ในไฟล์: {lp_log_filename}")

            # สร้างกราฟเปรียบเทียบ
            plt.figure(figsize=(12, 6))
            plt.plot(df['time'], df['x'], label='Original X', alpha=0.5)
            plt.plot(df['time'], df['x_lp_filtered'], label=f'LPF X (cutoff={cutoff}Hz)')
            plt.plot(df['time'], df['y'], label='Original Y', alpha=0.5)
            plt.plot(df['time'], df['y_lp_filtered'], label=f'LPF Y (cutoff={cutoff}Hz)')
            plt.xlabel('Time (s)')
            plt.ylabel('Position (m)')
            plt.title('Low-Pass Filtered Robot Position')
            plt.legend()
            plt.grid(True)
            plt.show()

            print("แสดงผลกราฟของ Low-Pass Filter เสร็จสิ้นแล้ว")

        except FileNotFoundError:
            print(f"ข้อผิดพลาด: ไม่พบไฟล์ {log_filename}")
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการประมวลผลข้อมูล: {e}")