import cv2
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
        self.detection_timeout = 1.0
    
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
            time.sleep(0.02)
        
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
    print(f"🔄 NEW: Rotate first, then tilt for better stability!")
    
    # ล็อคล้อ
    chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
    time.sleep(0.1)
    
    speed = 480
    pitch_angle = -20
    directions = ['front', 'left', 'right']
    yaw_angles = {'front': 0, 'left': -90, 'right': 90}
    
    all_results = {}
    
    for direction in directions:
        current_angle = yaw_angles[direction]
        direction_name = get_direction_name(current_angle)
        compass_dir = get_compass_direction(current_angle)
        
        print(f"\n🧭 Scanning {direction_name} | Gimbal Yaw: {current_angle}° | Compass: {compass_dir}")
        print(f"   🎯 Target: {direction.upper()} direction")
        
        # แก้ไข: หมุน gimbal ก่อนแล้วค่อยก้ม (หมุนก่อน ก้มทีหลัง)
        print(f"   🔄 Step 1: Rotating gimbal to {current_angle}°...")
        gimbal.moveto(pitch=0, yaw=current_angle, 
                     pitch_speed=speed, yaw_speed=speed).wait_for_completed()
        time.sleep(0.2)  # รอให้การหมุนเสถียร
        print(f"      ✅ Rotation complete")
        
        print(f"   🎯 Step 2: Tilting gimbal to {pitch_angle}°...")
        gimbal.moveto(pitch=pitch_angle, yaw=current_angle, 
                     pitch_speed=speed, yaw_speed=speed).wait_for_completed()
        time.sleep(0.2)
        print(f"      ✅ Tilt complete")
        
        # วัดระยะทางก่อน
        print("📏 Measuring distance...")
        tof_handler.start_scanning()
        sensor.sub_distance(freq=50, callback=tof_handler.tof_data_handler)
        time.sleep(0.1)
        tof_handler.stop_scanning(sensor)
        
        distance = tof_handler.get_average_distance()
        print(f"   📐 Distance: {distance:.2f}cm at {current_angle}°")
        
        # ตรวจ marker เฉพาะถ้าระยะใกล้พอ และมี ToF reading ที่ถูกต้อง
        # เปลี่ยนเงื่อนไข: ต้องมีค่า distance > 0 และ <= 40.0cm
        if distance > 0 and distance <= 40.0:
            print("✅ Distance OK - Scanning for markers...")
            
            # รีเซ็ตและเริ่มสแกน marker
            marker_handler.reset_detection()
            detected = marker_handler.wait_for_markers(timeout=1.0)
            
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
                print(f"   ✅ {direction.upper()} scan complete with markers")
            else:
                print(f"❌ No markers found at {direction_name} ({current_angle}°)")
                all_results[direction] = None
                print(f"   ✅ {direction.upper()} scan complete (no markers)")
        else:
            # แก้ไขข้อความให้ถูกต้อง
            if distance <= 0:
                print(f"❌ Invalid ToF reading ({distance:.2f}cm) at {current_angle}° - Skipping marker detection")
                print(f"   ⚠️ Sensor may not be detecting properly or object too close/far")
            else:
                print(f"❌ Distance too far ({distance:.2f}cm > 40cm) at {current_angle}° - Skipping marker detection")
            
            all_results[direction] = None
            print(f"   ✅ {direction.upper()} scan complete (distance issue)")
        
        time.sleep(0.1)
    
    print(f"   🔄 Step 3: Returning gimbal to center (0°, 0°)...")
    gimbal.moveto(pitch=0, yaw=0, pitch_speed=speed, yaw_speed=speed).wait_for_completed()
    time.sleep(0.1)  # รอให้การก้มเสถียร
    print(f"      ✅ Center return complete")
    
    # ปลดล็อคล้อ
    chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0, timeout=0.1)
    
    # แสดงสรุปการสแกน
    print(f"\n🎯 === SCANNING COMPLETE ===")
    print(f"📊 Summary: {len([r for r in all_results.values() if r])} directions with markers found")
    
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
            
            # แสดงข้อมูลการสแกนเพิ่มเติม
            if result.get('extended_scan'):
                print(f"   🔄 Extended Scan: Yes (found at different angle)")
                print(f"   📍 Scan Type: Enhanced scanning for better detection")
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

# -------------------------------
# ฟังก์ชันตรวจจับสีแดง
# -------------------------------
def detect_red(ep_camera, threshold_area=100, attempts=3):
    """
    ตรวจจับสีแดงจากกล้องหน้า
    อ่าน frame หลายครั้ง (attempts) ป้องกัน queue empty
    คืนค่า True/False ถ้าเจอสีแดง
    """
    try:
        for _ in range(attempts):
            try:
                frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
                if frame is None:
                    continue
                    
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # ช่วงสีแดง (ปรับปรุงแล้วให้ครอบคลุมมากขึ้น)
                lower_red1 = np.array([0, 120, 70])
                upper_red1 = np.array([10, 255, 255])
                lower_red2 = np.array([170, 120, 70])
                upper_red2 = np.array([180, 255, 255])

                mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                mask = mask1 | mask2

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if cv2.contourArea(cnt) > threshold_area:
                        return True
                        
                # ถ้าไม่เจอสีแดงใน frame นี้ ให้ลองใหม่
                time.sleep(0.05)
                
            except Exception as e:
                print(f"❌ Error reading frame: {e}")
                time.sleep(0.1)
                
        return False
    except Exception as e:
        print(f"❌ detect_red error: {e}")
        return False

# -------------------------------
# ฟังก์ชันสแกนสีแดง + marker
# -------------------------------
def scan_red_then_marker_fixed(ep_robot, ep_gimbal, ep_chassis, ep_sensor, marker_handler, tof_handler):
    """
    สแกนหาสีแดงก่อน แล้วสแกน marker เฉพาะทิศทางที่เจอสีแดง
    """
    yaw_angles = [0, -90, 90]  # หน้า, ซ้าย, ขวา
    red_angles = []

    ep_camera = ep_robot.camera
    
    # เปิด Video Stream และรอให้เสถียร
    try:
        ep_camera.start_video_stream(display=False, resolution="720p")
        print("📹 Starting camera stream...")
        time.sleep(1.0)  # รอให้ frame มาเสถียร
    except Exception as e:
        print(f"❌ Error starting camera: {e}")
        return {}

    # ล็อคล้อหุ่นยนต์ ไม่ให้ chassis เคลื่อน
    ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)

    print("\n🔴 === SCANNING FOR RED COLOR ===")
    # ตรวจสีแดงในแต่ละทิศทาง
    for yaw in yaw_angles:
        direction_name = get_direction_name(yaw)
        print(f"\n🔄 หมุน Gimbal ไปที่ {direction_name} ({yaw}°) เพื่อตรวจจับสีแดง")
        
        ep_gimbal.moveto(pitch=0, yaw=yaw, pitch_speed=480, yaw_speed=480).wait_for_completed()
        time.sleep(0.3)  # รอให้กล้องเสถียร
        
        found_red = detect_red(ep_camera, threshold_area=100, attempts=5)
        if found_red:
            print(f"✅ เจอสีแดงที่ {direction_name} ({yaw}°)")
            red_angles.append(yaw)
        else:
            print(f"❌ ไม่เจอสีแดงที่ {direction_name} ({yaw}°)")

    # ปิด Video Stream หลังจากตรวจสีแดงเสร็จ
    try:
        ep_camera.stop_video_stream()
    except:
        pass

    results = {}

    if not red_angles:
        print("\n❌ ไม่เจอสีแดงในทิศทางใดเลย")
        # กลับไปตำแหน่งกลาง
        ep_gimbal.moveto(pitch=0, yaw=0, pitch_speed=480, yaw_speed=480).wait_for_completed()
        ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
        return results

    print(f"\n🎯 === SCANNING MARKERS (เฉพาะทิศทางที่เจอสีแดง) ===")
    print(f"🔴 เจอสีแดงใน {len(red_angles)} ทิศทาง: {red_angles}")

    # scan marker เฉพาะ yaw ที่เจอสีแดง
    for yaw in red_angles:
        direction_name = get_direction_name(yaw)
        print(f"\n🎯 สแกน Marker ที่ {direction_name} ({yaw}°)")
        
        # หมุนไป yaw นั้น
        ep_gimbal.moveto(pitch=-20, yaw=yaw, pitch_speed=480, yaw_speed=480).wait_for_completed()
        time.sleep(0.1)

        # วัดระยะด้วย ToF
        print("📏 วัดระยะทาง...")
        tof_handler.start_scanning()
        ep_sensor.sub_distance(freq=50, callback=tof_handler.tof_data_handler)
        time.sleep(0.25)  # รอให้ข้อมูล ToF เสถียร
        tof_handler.stop_scanning(ep_sensor)
        
        distance = tof_handler.get_average_distance()
        print(f"   📐 ระยะ: {distance:.2f} cm")

        # ตรวจ marker เฉพาะถ้าระยะใกล้พอ และมี ToF reading ที่ถูกต้อง
        if distance > 0 and distance <= 50.0:
            print("✅ ระยะใกล้พอ - ตรวจหา Marker...")
            marker_handler.reset_detection()
            detected = marker_handler.wait_for_markers(timeout=1.5)
            
            if detected and marker_handler.markers:
                marker_ids = [m.id for m in marker_handler.markers]
                results[yaw] = {
                    'direction_name': direction_name,
                    'marker_ids': marker_ids,
                    'distance': distance,
                    'found_red': True
                }
                print(f"🎯 เจอ Marker: {marker_ids} ที่ {direction_name} ({yaw}°)")
            else:
                results[yaw] = {
                    'direction_name': direction_name,
                    'marker_ids': [],
                    'distance': distance,
                    'found_red': True
                }
                print(f"❌ ไม่เจอ Marker ที่ {direction_name} ({yaw}°)")
        else:
            results[yaw] = {
                'direction_name': direction_name,
                'marker_ids': [],
                'distance': distance,
                'found_red': True,
                'reason': 'distance_issue'
            }
            # แก้ไขข้อความให้ถูกต้อง
            if distance <= 0:
                print(f"❌ ToF sensor ไม่ได้อ่านค่า ({distance:.2f}cm) ที่ {direction_name} ({yaw}°)")
                print(f"   ⚠️ เซนเซอร์อาจมีปัญหาหรือวัตถุใกล้/ไกลเกินไป")
            else:
                print(f"❌ ระยะไกลเกินไป ({distance:.2f}cm > 50cm) ที่ {direction_name} ({yaw}°)")

    # กลับไปตำแหน่งกลาง
    print(f"\n🔄 กลับสู่ตำแหน่งกลาง...")
    ep_gimbal.moveto(pitch=0, yaw=0, pitch_speed=480, yaw_speed=480).wait_for_completed()
    
    # ปลดล็อคล้อ
    ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0, timeout=0.1)

    return results

def print_red_scan_results(results):
    """แสดงผลลัพธ์การสแกนแบบสีแดง+marker"""
    print("\n" + "="*60)
    print("🔴 RED COLOR + MARKER DETECTION RESULTS")
    print("="*60)
    
    if not results:
        print("❌ ไม่พบสีแดงในทิศทางใดเลย")
        return
    
    total_markers = 0
    directions_with_markers = 0
    
    for yaw, info in results.items():
        if info:
            direction_name = info['direction_name']
            marker_ids = info['marker_ids']
            distance = info['distance']
            
            print(f"\n✅ {direction_name.upper()} ({yaw:+4d}°)")
            print(f"   🔴 พบสีแดง: ใช่")
            print(f"   📏 ระยะ: {distance:.2f} cm")
            
            if marker_ids:
                print(f"   🎯 Marker IDs: {marker_ids}")
                total_markers += len(marker_ids)
                directions_with_markers += 1
            else:
                reason = info.get('reason', 'not_found')
                if reason == 'distance_issue':
                    if distance <= 0:
                        print(f"   🎯 Marker: ไม่ตรวจ (ToF sensor มีปัญหา)")
                    else:
                        print(f"   🎯 Marker: ไม่ตรวจ (ระยะไกลเกินไป)")
                else:
                    print(f"   🎯 Marker: ไม่พบ")
    
    print(f"\n" + "="*60)
    print(f"📊 สรุปผลการสแกน")
    print(f"="*60)
    print(f"🔴 ทิศทางที่พบสีแดง: {len(results)}")
    print(f"🎯 ทิศทางที่พบ Marker: {directions_with_markers}")
    print(f"🎯 จำนวน Marker ทั้งหมด: {total_markers}")

if __name__ == "__main__":
    print("🤖 เชื่อมต่อหุ่นยนต์...")
    ep_robot = robot.Robot()
    
    try:
        ep_robot.initialize(conn_type="ap")
        print("✅ เชื่อมต่อสำเร็จ")
    except Exception as e:
        print(f"❌ เชื่อมต่อไม่สำเร็จ: {e}")
        exit(1)

    ep_gimbal = ep_robot.gimbal
    ep_chassis = ep_robot.chassis
    ep_sensor = ep_robot.sensor
    ep_vision = ep_robot.vision

    marker_handler = MarkerVisionHandler()
    tof_handler = ToFSensorHandler()

    try:
        # เริ่ม marker detection
        if not marker_handler.start_continuous_detection(ep_vision):
            print("❌ ไม่สามารถเริ่ม marker detection ได้")
            exit(1)
        
        # เริ่มสแกนสีแดงและ marker
        print("\n🚀 เริ่มการสแกนสีแดงและ Marker...")
        results = scan_red_then_marker_fixed(ep_robot, ep_gimbal, ep_chassis, ep_sensor, marker_handler, tof_handler)
        
        # แสดงผลลัพธ์
        print_red_scan_results(results)
        
    except KeyboardInterrupt:
        print("\n⚠️ หยุดการทำงานโดยผู้ใช้")
    except Exception as e:
        print(f"\n❌ เกิดข้อผิดพลาด: {e}")
    finally:
        # ปิดการเชื่อมต่อ
        try:
            marker_handler.stop_continuous_detection(ep_vision)
            ep_robot.close()
            print("🔌 ปิดการเชื่อมต่อเรียบร้อย")
        except:
            pass