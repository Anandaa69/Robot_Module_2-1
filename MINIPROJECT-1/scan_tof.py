import time
import robomaster
from robomaster import robot
import numpy as np
from scipy.ndimage import median_filter
from datetime import datetime

class ToFSensorHandler:
    def __init__(self):
        # ค่า Calibration จาก Linear Regression
        self.CALIBRATION_SLOPE = 0.0894 
        self.CALIBRATION_Y_INTERCEPT = 3.8409
        
        # Median Filter settings
        self.WINDOW_SIZE = 5
        self.tof_buffer = []
        
        # เก็บค่าเซ็นเซอร์สำหรับแต่ละตำแหน่ง
        self.readings = {
            'left_90': [],
            'right_minus90': []
        }
        
        self.current_position = None
        self.collecting_data = False
        
    def calibrate_tof_value(self, raw_tof_mm):
        """แปลงค่า ToF ดิบ (mm) เป็นระยะทางที่สอบเทียบแล้ว (cm)"""
        calibrated_cm = (self.CALIBRATION_SLOPE * raw_tof_mm) + self.CALIBRATION_Y_INTERCEPT
        return calibrated_cm
    
    def apply_median_filter(self, data, window_size):
        """ใช้ Median Filter กับข้อมูล"""
        if len(data) == 0:
            return 0.0 
        if len(data) < window_size:
            return data[-1] 
        else:
            filtered = median_filter(data[-window_size:], size=window_size)
            return filtered[-1]
    
    def tof_data_handler(self, sub_info):
        """Callback สำหรับรับข้อมูลจากเซ็นเซอร์ ToF"""
        if not self.collecting_data:
            return
            
        raw_tof_mm = sub_info[0]  # ค่าดิบจากเซ็นเซอร์ (mm)
        
        # Calibrate ค่า
        calibrated_tof_cm = self.calibrate_tof_value(raw_tof_mm)
        
        # เพิ่มข้อมูลลง buffer
        self.tof_buffer.append(calibrated_tof_cm)
        
        # ใช้ Median Filter
        filtered_tof_cm = self.apply_median_filter(self.tof_buffer, self.WINDOW_SIZE)
        
        # เก็บข้อมูลตามตำแหน่งปัจจุบัน
        if self.current_position and len(self.tof_buffer) <= 25:  # เก็บ 25 ค่าต่อตำแหน่ง (1 วินาทีที่ 25Hz)
            self.readings[self.current_position].append({
                'raw_mm': raw_tof_mm,
                'calibrated_cm': calibrated_tof_cm,
                'filtered_cm': filtered_tof_cm,
                'timestamp': datetime.now().isoformat()
            })
        
        # แสดงผลแบบ real-time
        print(f"[{self.current_position}] Raw: {raw_tof_mm:.1f}mm | "
              f"Calibrated: {calibrated_tof_cm:.2f}cm | "
              f"Filtered: {filtered_tof_cm:.2f}cm")
    
    def start_data_collection(self, position):
        """เริ่มเก็บข้อมูลสำหรับตำแหน่งที่กำหนด"""
        self.current_position = position
        self.tof_buffer.clear()
        self.collecting_data = True
        
    def stop_data_collection(self, unsub_distance_func):
        """หยุดเก็บข้อมูล"""
        self.collecting_data = False
        self.current_position = None
        unsub_distance_func()
    
    def get_summary_stats(self, position):
        """คำนวณสถิติสรุปสำหรับตำแหน่งที่กำหนด"""
        if position not in self.readings or len(self.readings[position]) == 0:
            return None
            
        data = self.readings[position]
        filtered_values = [reading['filtered_cm'] for reading in data]
        
        return {
            'count': len(filtered_values),
            'mean': np.mean(filtered_values),
            'median': np.median(filtered_values),
            'std': np.std(filtered_values),
            'min': np.min(filtered_values),
            'max': np.max(filtered_values)
        }

def simple_movement_sequence_with_tof(gimbal, chassis, sensor, tof_handler):
    """ลำดับการเคลื่อนไหวพร้อมการเก็บข้อมูล ToF (รอบเดียว)"""
    print("\n=== เริ่มลำดับการเคลื่อนไหวและเก็บข้อมูล ToF ===")
    
    # ล็อคล้อ
    print("🔒 ล็อคล้อทั้ง 4 ล้อ...")
    chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
    time.sleep(0.5)
    
    speed = 540
    
    # 1. หมุนไปซ้าย 90° และเก็บข้อมูล ToF
    print("1. หมุนไปซ้าย 90° และเก็บข้อมูล ToF...")
    gimbal.moveto(pitch=0, yaw=90, pitch_speed=speed, yaw_speed=speed).wait_for_completed()
    time.sleep(0.2)  # รอให้นิ่ง
    
    print("📊 เริ่มเก็บข้อมูล ToF ที่ตำแหน่ง 90°...")
    tof_handler.start_data_collection('left_90')
    sensor.sub_distance(freq=25, callback=tof_handler.tof_data_handler)
    time.sleep(0.4) 
    tof_handler.stop_data_collection(sensor.unsub_distance)
    print("✅ เสร็จสิ้นการเก็บข้อมูลที่ 90°")
    
    # 2. หมุนไปขวา -90° และเก็บข้อมูล ToF
    print("2. หมุนไปขวา -90° และเก็บข้อมูล ToF...")
    gimbal.moveto(pitch=0, yaw=-90, pitch_speed=speed, yaw_speed=speed).wait_for_completed()
    time.sleep(0.2)  # รอให้นิ่ง
    
    print("📊 เริ่มเก็บข้อมูล ToF ที่ตำแหน่ง -90°...")
    tof_handler.start_data_collection('right_minus90')
    sensor.sub_distance(freq=25, callback=tof_handler.tof_data_handler)
    time.sleep(0.4) 
    tof_handler.stop_data_collection(sensor.unsub_distance)
    print("✅ เสร็จสิ้นการเก็บข้อมูลที่ -90°")
    
    # 3. กลับตำแหน่งเริ่มต้น
    print("3. กลับสู่ตำแหน่งเริ่มต้น...")
    gimbal.moveto(pitch=0, yaw=0, pitch_speed=speed, yaw_speed=speed).wait_for_completed()
    time.sleep(0.2)
    
    # ปลดล็อคล้อ
    print("🔓 ปลดล็อคล้อ...")
    chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0, timeout=0.1)
    time.sleep(0.2)
    
if __name__ == '__main__':
    print("กำลังเชื่อมต่อหุ่นยนต์...")
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    
    ep_gimbal = ep_robot.gimbal
    ep_chassis = ep_robot.chassis
    ep_sensor = ep_robot.sensor
    
    tof_handler = ToFSensorHandler()
    
    try:
        print("✅ Recalibrating gimbal...")
        ep_gimbal.recenter(pitch_speed=100, yaw_speed=100).wait_for_completed()
        print("✅ Gimbal recalibrated.")
        
        print("รีเซ็ตตำแหน่งเริ่มต้น...")
        ep_gimbal.moveto(pitch=0, yaw=0, pitch_speed=50, yaw_speed=50).wait_for_completed()
        time.sleep(1)
        
        print(f"🎯 ใช้ Calibration: slope={tof_handler.CALIBRATION_SLOPE}, intercept={tof_handler.CALIBRATION_Y_INTERCEPT}")
        print(f"🔍 ใช้ Median Filter: window size={tof_handler.WINDOW_SIZE}")
        
        simple_movement_sequence_with_tof(ep_gimbal, ep_chassis, ep_sensor, tof_handler)
            
    except KeyboardInterrupt:
        print("\nหยุดการทำงานโดยผู้ใช้")
    except Exception as e:
        print(f"\nเกิดข้อผิดพลาด: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            ep_sensor.unsub_distance()
        except:
            pass
        ep_robot.close()
        print("ปิดการเชื่อมต่อเรียบร้อย")