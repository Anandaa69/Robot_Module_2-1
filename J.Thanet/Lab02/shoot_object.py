import cv2
import time
import robomaster
from robomaster import robot, blaster, vision
import threading
import math

# ===== Calibration Settings =====
YAW_CALIB_FACTOR = 1.35   # ค่ามากกว่า 1.0 = หมุนซ้าย/ขวามากขึ้น
PITCH_CALIB_FACTOR = 2 # ค่ามากกว่า 1.0 = หมุนขึ้น/ลงมากขึ้น

markers = []
detected_objects = {}
sorted_markers_by_x = []
window_size = 720
height_window = window_size
weight_window = 1280

shooting_enabled = False
current_target_index = 0
is_shooting = False
shoot_delay = 0.125
last_shoot_time = 0

camera_lock = threading.Lock()
current_frame = None
frame_ready = True

# ===== ตัวแปรใหม่สำหรับโหมดจับเป้า =====
capture_phase = True
capture_start_time = None
locked_markers = []

class MarkerInfo:
    def __init__(self, x, y, w, h, info):
        self._x = x
        self._y = y
        self._w = w
        self._h = h
        self._info = info

    @property
    def pt1(self):
        return int((self._x - self._w / 2) * weight_window), int((self._y - self._h / 2) * height_window)

    @property
    def pt2(self):
        return int((self._x + self._w / 2) * weight_window), int((self._y + self._h / 2) * height_window)

    @property
    def center(self):
        return int(self._x * weight_window), int(self._y * height_window)

    @property
    def center_pixel_x(self):
        return int(self._x * weight_window)
    
    @property
    def center_pixel_y(self):
        return int(self._y * height_window)

    @property
    def text(self):
        return self._info

    @property
    def center_x(self):
        return self._x
    
    @property
    def center_y(self):
        return self._y


def calculate_distance_from_marker_size(marker_width_normalized, marker_height_normalized):
    avg_size = (marker_width_normalized + marker_height_normalized) / 2
    if avg_size > 0.20:
        return 0.8
    elif avg_size > 0.15:
        return 1.2
    elif avg_size > 0.10:
        return 1.8
    elif avg_size > 0.08:
        return 2.5
    elif avg_size > 0.06:
        return 3.2
    elif avg_size > 0.04:
        return 4.0
    else:
        return 5.0


def pixel_to_gimbal_angle_precise(pixel_x, pixel_y, marker_width_norm, marker_height_norm):
    fov_x_deg = 70.0
    fov_y_deg = 40.0

    distance_m = calculate_distance_from_marker_size(marker_width_norm, marker_height_norm)

    center_x = weight_window / 2
    center_y = height_window / 2
    dx = pixel_x - center_x
    dy = center_y - pixel_y

    fov_x_rad = math.radians(fov_x_deg)
    fov_y_rad = math.radians(fov_y_deg)

    yaw_angle = math.degrees(math.atan(math.tan(fov_x_rad / 2) * (dx / (weight_window / 2))))
    pitch_angle = math.degrees(math.atan(math.tan(fov_y_rad / 2) * (dy / (height_window / 2))))

    # ===== Apply Calibration =====
    yaw_angle *= YAW_CALIB_FACTOR
    pitch_angle *= PITCH_CALIB_FACTOR

    # ===== Gravity Compensation =====
    gravity_compensation = 0
    if distance_m >= 1.5:
        gravity_compensation = (distance_m - 1.0) * 3.5
        if distance_m > 3.0:
            gravity_compensation += (distance_m - 3.0) * 2.0
    pitch_angle += gravity_compensation

    yaw_angle = max(-80, min(80, yaw_angle))
    pitch_angle = max(-25, min(35, pitch_angle))

    return yaw_angle, pitch_angle, distance_m


def shoot_at_target(ep_gimbal, ep_blaster, target_marker):
    global is_shooting, last_shoot_time
    if is_shooting:
        return False
    is_shooting = True
    print(f"\n🎯 กำลังเล็งเป้า Marker {target_marker.text}")
    yaw_angle, pitch_angle, distance = pixel_to_gimbal_angle_precise(
        target_marker.center_pixel_x, 
        target_marker.center_pixel_y,
        target_marker._w,
        target_marker._h
    )
    print(f"   📐 มุมเล็ง: Yaw={yaw_angle:.1f}°, Pitch={pitch_angle:.1f}°")
    print(f"   📏 ระยะประมาณ: {distance:.1f}m")
    try:
        ep_gimbal.moveto(pitch=pitch_angle, yaw=yaw_angle, pitch_speed=150, yaw_speed=150)
        time.sleep(0.2)
        print(f"   🔥 ยิง Marker {target_marker.text}!")
        for shot in range(1):
            ep_blaster.fire(fire_type=blaster.INFRARED_FIRE, times=1)
            if shot < 2:
                time.sleep(0.05)
        last_shoot_time = time.time()
        print(f"   ✅ ยิง Marker {target_marker.text} เสร็จ (3 นัด)")
        return True
    except Exception as e:
        print(f"   ❌ เกิดข้อผิดพลาดในการยิง: {e}")
        return False
    finally:
        is_shooting = False


def auto_shoot_sequence(ep_gimbal, ep_blaster):
    global current_target_index, shooting_enabled, last_shoot_time
    if not shooting_enabled or len(locked_markers) == 0:
        return
    if time.time() - last_shoot_time < shoot_delay:
        return
    if current_target_index >= len(locked_markers):
        print("\n🎉 ยิงครบทุก Marker แล้ว! กด 'r' เพื่อเริ่มใหม่")
        shooting_enabled = False
        current_target_index = 0
        ep_gimbal.moveto(pitch=0, yaw=0, pitch_speed=80, yaw_speed=80)
        return
    target_marker = locked_markers[current_target_index]
    def shoot_thread():
        global current_target_index
        success = shoot_at_target(ep_gimbal, ep_blaster, target_marker)
        if success:
            current_target_index += 1
    threading.Thread(target=shoot_thread, daemon=True).start()


def camera_thread(ep_camera):
    global current_frame
    while True:
        try:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.3)
            if img is not None:
                with camera_lock:
                    current_frame = img.copy()
        except Exception as e:
            print(f"Camera thread error: {e}")
            time.sleep(0.01)
        time.sleep(0.005)


def reset_gimbal(ep_gimbal):
    print("🔄 รีเซ็ต Gimbal กลับตำแหน่งกลาง...")
    ep_gimbal.moveto(pitch=0, yaw=0, pitch_speed=100, yaw_speed=100).wait_for_completed()
    print("✅ รีเซ็ต Gimbal เสร็จสิ้น")


def on_detect_marker(marker_info):
    global sorted_markers_by_x
    if not capture_phase:
        return
    markers.clear()
    for x, y, w, h, info in marker_info:
        markers.append(MarkerInfo(x, y, w, h, info))
        detected_objects[info] = {
            'center_normalized': (x, y),
            'center_pixel': (int(x * weight_window), int(y * height_window)),
            'size_normalized': (w, h),
            'size_pixel': (int(w * weight_window), int(h * height_window)),
            'timestamp': time.time()
        }
    sorted_markers_by_x = sorted(markers, key=lambda m: m.center_x)
    for idx, m in enumerate(sorted_markers_by_x):
        detected_objects[m.text]['order'] = idx + 1


def print_controls():
    print("\n" + "="*60)
    print("🎮 การควบคุม:")
    print("  's' หรือ 'SPACE' - เริ่มยิงอัตโนมัติ")
    print("  'x' - หยุดการยิง")
    print("  'r' - รีเซ็ตลำดับและ gimbal")
    print("  'q' - ออกจากโปรแกรม")
    print("="*60)


if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_blaster = ep_robot.blaster
    ep_gimbal = ep_robot.gimbal
    ep_vision = ep_robot.vision
    ep_camera = ep_robot.camera

    ep_gimbal.moveto(pitch=0, yaw=0).wait_for_completed()
    ep_camera.start_video_stream(display=False)
    ep_vision.sub_detect_info(name="marker", callback=on_detect_marker)

    capture_start_time = time.time()
    threading.Thread(target=camera_thread, args=(ep_camera,), daemon=True).start()
    print_controls()

    try:
        while True:
            if capture_phase and (time.time() - capture_start_time >= 2.0):
                capture_phase = False
                locked_markers = sorted_markers_by_x.copy()
                print(f"\n🔒 ล็อกเป้าหมาย {len(locked_markers)} ตัวเรียบร้อย")
                for i, m in enumerate(locked_markers):
                    print(f"  {i+1}. Marker {m.text} at {m.center_pixel_x},{m.center_pixel_y}")

            active_markers = locked_markers if not capture_phase else sorted_markers_by_x
            img = None
            with camera_lock:
                if current_frame is not None:
                    img = current_frame.copy()
            if img is None:
                time.sleep(0.005)
                continue

            for idx, marker in enumerate(active_markers):
                order_number = idx + 1
                color = (0, 255, 0)
                thickness = 3
                if shooting_enabled and idx == current_target_index:
                    color = (0, 0, 255)
                    thickness = 5
                elif shooting_enabled and idx < current_target_index:
                    color = (128, 128, 128)
                    thickness = 2
                cv2.rectangle(img, marker.pt1, marker.pt2, color, thickness)
                cv2.putText(img, f"{order_number}. {marker.text}", marker.center, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

            cv2.imshow("RoboMaster Auto Shooting", img)

            if shooting_enabled and not is_shooting and not capture_phase:
                if current_target_index < len(locked_markers):
                    auto_shoot_sequence(ep_gimbal, ep_blaster)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') or key == ord(' '):
                if not capture_phase and len(locked_markers) > 0:
                    shooting_enabled = True
                    current_target_index = 0
                    print("\n🚀 เริ่มยิง")
                elif capture_phase:
                    print("\n⏳ กำลังจับเป้า รอสักครู่...")
            elif key == ord('x'):
                shooting_enabled = False
                is_shooting = False
                print("\n⏹️ หยุดยิง")
            elif key == ord('r'):
                shooting_enabled = False
                is_shooting = False
                current_target_index = 0
                reset_gimbal(ep_gimbal)
                print("\n🔄 รีเซ็ตระบบ")

    finally:
        cv2.destroyAllWindows()
        ep_vision.unsub_detect_info(name="marker")
        ep_camera.stop_video_stream()
        ep_robot.close()
