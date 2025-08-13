# โค้ดสำหรับสแกนหา marker เท่านั้น (เพิ่มการแสดงองศาทิศทาง)
import time
import robomaster
from robomaster import robot, vision
from datetime import datetime
import numpy as np
from scipy.ndimage import median_filter

# ===== Marker Detection Classes =====
class MarkerInfo:
    def __init__(self, x, y, w, h, marker_id):
        self._x = x
        self._y = y
        self._w = w
        self._h = h
        self._id = marker_id

    @property
    def id(self):
        return self._id

class MarkerVisionHandler:
    def __init__(self):
        self.markers = []
        self.marker_detected = False
        self.is_active = False
        self.detection_timeout = 2.0
    
    def on_detect_marker(self, marker_info):
        if not self.is_active:
            return
            
        if len(marker_info) > 0:
            valid_markers = []
            for i in range(len(marker_info)):
                x, y, w, h, marker_id = marker_info[i]
                marker = MarkerInfo(x, y, w, h, marker_id)
                valid_markers.append(marker)
            
            if valid_markers:
                self.marker_detected = True
                self.markers = valid_markers
    
    def wait_for_markers(self, timeout=None):
        if timeout is None:
            timeout = self.detection_timeout
        
        print(f"⏱️ Waiting {timeout} seconds for marker detection...")
        
        self.marker_detected = False
        self.markers.clear()
        
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            if self.marker_detected:
                print(f"✅ Marker detected after {time.time() - start_time:.1f}s")
                break
            time.sleep(0.05)
        
        return self.marker_detected
    
    def start_continuous_detection(self, vision):
        try:
            self.stop_continuous_detection(vision)
            time.sleep(0.3)
            
            result = vision.sub_detect_info(name="marker", callback=self.on_detect_marker)
            if result:
                self.is_active = True
                print("✅ Marker detection activated")
                return True
            else:
                print("❌ Failed to start marker detection")
                return False
        except Exception as e:
            print(f"❌ Error starting marker detection: {e}")
            return False
    
    def stop_continuous_detection(self, vision):
        try:
            self.is_active = False
            vision.unsub_detect_info(name="marker")
        except:
            pass
    
    def reset_detection(self):
        self.marker_detected = False
        self.markers.clear()

# ===== ToF Sensor Handler =====
class ToFSensorHandler:
    def __init__(self):
        self.CALIBRATION_SLOPE = 0.0894 
        self.CALIBRATION_Y_INTERCEPT = 3.8409
        self.WINDOW_SIZE = 5
        self.tof_buffer = []
        self.readings = []
        self.collecting_data = False
        
    def calibrate_tof_value(self, raw_tof_mm):
        calibrated_cm = (self.CALIBRATION_SLOPE * raw_tof_mm) + self.CALIBRATION_Y_INTERCEPT
        return calibrated_cm
    
    def apply_median_filter(self, data, window_size):
        if len(data) == 0:
            return 0.0 
        if len(data) < window_size:
            return data[-1] 
        else:
            filtered = median_filter(data[-window_size:], size=window_size)
            return filtered[-1]
    
    def tof_data_handler(self, sub_info):
        if not self.collecting_data:
            return
            
        raw_tof_mm = sub_info[0]
        
        if raw_tof_mm <= 0 or raw_tof_mm > 4000:
            return
            
        calibrated_tof_cm = self.calibrate_tof_value(raw_tof_mm)
        self.tof_buffer.append(calibrated_tof_cm)
        filtered_tof_cm = self.apply_median_filter(self.tof_buffer, self.WINDOW_SIZE)
        
        if len(self.tof_buffer) <= 20:
            self.readings.append(filtered_tof_cm)
    
    def start_scanning(self):
        self.tof_buffer.clear()
        self.readings.clear()
        self.collecting_data = True
        
    def stop_scanning(self, sensor):
        self.collecting_data = False
        try:
            sensor.unsub_distance()
        except:
            pass
    
    def get_average_distance(self):
        if len(self.readings) == 0:
            return 0.0
        
        # กรองค่าผิดปกติ
        if len(self.readings) > 4:
            q1 = np.percentile(self.readings, 25)
            q3 = np.percentile(self.readings, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            filtered_values = [x for x in self.readings if lower_bound <= x <= upper_bound]
            return np.mean(filtered_values) if filtered_values else np.mean(self.readings)
        
        return np.mean(self.readings)

# ===== Direction Helper Function =====
def get_direction_name(angle):
    """แปลงองศาเป็นชื่อทิศทาง"""
    direction_map = {
        0: "หน้า (Front)",
        -90: "ซ้าย (Left)", 
        90: "ขวา (Right)"
    }
    return direction_map.get(angle, f"องศา {angle}")

def get_compass_direction(angle):
    """แปลงองศาเป็นทิศทางเข็มทิศ"""
    # สำหรับ gimbal yaw: 0° = หน้า, -90° = ซ้าย, 90° = ขวา
    compass_map = {
        0: "เหนือ (N)",
        -90: "ตะวันตก (W)",
        90: "ตะวันออก (E)",
        180: "ใต้ (S)",
        -180: "ใต้ (S)"
    }
    return compass_map.get(angle, f"{angle}°")

# ===== Main Scanning Function =====
def scan_for_markers_all_directions(gimbal, chassis, sensor, marker_handler, tof_handler):
    """สแกนหา marker ในทุกทิศทาง พร้อมตรวจระยะทางและบอกองศา"""
    print(f"\n🔍 === SCANNING FOR MARKERS WITH DIRECTION ANGLES ===")
    
    # ล็อคล้อ
    chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
    time.sleep(0.5)
    
    speed = 150
    pitch_angle = -45
    directions = ['front', 'left', 'right']
    yaw_angles = {'front': 0, 'left': -90, 'right': 90}
    
    all_results = {}
    
    for direction in directions:
        current_angle = yaw_angles[direction]
        direction_name = get_direction_name(current_angle)
        compass_dir = get_compass_direction(current_angle)
        
        print(f"\n🧭 Scanning {direction_name} | Gimbal Yaw: {current_angle}° | Compass: {compass_dir}")
        
        # หมุน gimbal ไปยังทิศทางที่ต้องการ
        gimbal.moveto(pitch=pitch_angle, yaw=current_angle, 
                     pitch_speed=speed, yaw_speed=speed).wait_for_completed()
        time.sleep(0.8)  # รอให้เสถียร
        
        # วัดระยะทางก่อน
        print("📏 Measuring distance...")
        tof_handler.start_scanning()
        sensor.sub_distance(freq=50, callback=tof_handler.tof_data_handler)
        time.sleep(0.8)
        tof_handler.stop_scanning(sensor)
        
        distance = tof_handler.get_average_distance()
        print(f"   📐 Distance: {distance:.2f}cm at {current_angle}°")
        
        # ตรวจ marker เฉพาะถ้าระยะใกล้พอ
        if distance > 0 and distance <= 30.0:
            print("✅ Distance OK - Scanning for markers...")
            
            # รีเซ็ตและเริ่มสแกน marker
            marker_handler.reset_detection()
            detected = marker_handler.wait_for_markers(timeout=2.5)
            
            if detected and marker_handler.markers:
                marker_ids = [m.id for m in marker_handler.markers]
                all_results[direction] = {
                    'angle': current_angle,
                    'direction_name': direction_name,
                    'compass_direction': compass_dir,
                    'marker_ids': marker_ids,
                    'distance': distance,
                    'timestamp': datetime.now().isoformat()
                }
                
                print(f"🎯 FOUND MARKERS: {marker_ids}")
                print(f"   📍 Direction: {direction_name} ({current_angle}°)")
                print(f"   📏 Distance: {distance:.2f}cm")
                print(f"   🧭 Compass: {compass_dir}")
            else:
                print(f"❌ No markers found at {direction_name} ({current_angle}°)")
                all_results[direction] = None
        else:
            print(f"❌ Too far ({distance:.2f}cm > 30cm) at {current_angle}° - Skipping marker detection")
            all_results[direction] = None
        
        time.sleep(0.3)
    
    # กลับไปกึ่งกลาง
    print(f"\n↩️ Returning to center position (0°)...")
    gimbal.moveto(pitch=0, yaw=0, pitch_speed=speed, yaw_speed=speed).wait_for_completed()
    time.sleep(0.5)
    
    # ปลดล็อคล้อ
    chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0, timeout=0.1)
    
    return all_results

def print_detailed_results(results):
    """แสดงผลลัพธ์แบบละเอียด"""
    print(f"\n" + "="*60)
    print(f"🎯 DETAILED MARKER DETECTION RESULTS")
    print(f"="*60)
    
    total_markers = 0
    found_directions = []
    
    for direction, result in results.items():
        if result:
            total_markers += len(result['marker_ids'])
            found_directions.append(result)
            
            print(f"\n✅ {result['direction_name'].upper()}")
            print(f"   🧭 Gimbal Angle: {result['angle']:+4d}°")
            print(f"   🧭 Compass Direction: {result['compass_direction']}")
            print(f"   🎯 Marker IDs: {result['marker_ids']}")
            print(f"   📏 Distance: {result['distance']:.2f}cm")
            print(f"   ⏰ Time: {result['timestamp'][11:19]}")
        else:
            angle = {'front': 0, 'left': -90, 'right': 90}[direction]
            dir_name = get_direction_name(angle)
            compass = get_compass_direction(angle)
            
            print(f"\n❌ {dir_name.upper()}")
            print(f"   🧭 Gimbal Angle: {angle:+4d}°")
            print(f"   🧭 Compass Direction: {compass}")
            print(f"   🎯 Result: No markers detected")
    
    print(f"\n" + "="*60)
    print(f"📊 SUMMARY")
    print(f"="*60)
    print(f"🎯 Total markers found: {total_markers}")
    print(f"📍 Directions with markers: {len(found_directions)}/3")
    
    if found_directions:
        print(f"\n🧭 MARKER LOCATIONS BY ANGLE:")
        for result in found_directions:
            marker_list = ', '.join([f"ID{mid}" for mid in result['marker_ids']])
            print(f"   {result['angle']:+4d}° ({result['compass_direction']}): {marker_list}")
    
    print(f"="*60)

if __name__ == '__main__':
    print("🤖 Connecting to robot...")
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    
    ep_gimbal = ep_robot.gimbal
    ep_chassis = ep_robot.chassis
    ep_sensor = ep_robot.sensor
    ep_vision = ep_robot.vision
    
    marker_handler = MarkerVisionHandler()
    tof_handler = ToFSensorHandler()
    
    try:
        # ตั้งค่า gimbal
        ep_gimbal.recenter().wait_for_completed()
        time.sleep(0.5)
        
        # เปิดระบบตรวจจับ marker
        marker_handler.start_continuous_detection(ep_vision)
        
        # สแกนหา marker พร้อมตรวจระยะและองศา
        results = scan_for_markers_all_directions(ep_gimbal, ep_chassis, ep_sensor, 
                                                marker_handler, tof_handler)
        
        # แสดงผลลัพธ์แบบละเอียด
        print_detailed_results(results)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            ep_sensor.unsub_distance()
        except:
            pass
        marker_handler.stop_continuous_detection(ep_vision)
        ep_robot.close()
        print("🔌 Connection closed")