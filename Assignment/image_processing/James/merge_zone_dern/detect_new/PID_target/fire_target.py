# detect_and_track_GPT_merged.py
import cv2
import numpy as np
import time
import math
import threading
import queue
from robomaster import robot, camera as r_camera, blaster

# =====================================================
# GPU Acceleration Check
# =====================================================
USE_GPU = False
try:
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print("✅ CUDA available, enabling GPU path")
        USE_GPU = True
    else:
        print("⚠️ CUDA not available, CPU path")
except Exception:
    print("⚠️ Skip CUDA check, CPU path")

# =====================================================
# PID Controller Class (เพิ่มใหม่)
# =====================================================
class PIDController:
    """
    คลาสสำหรับควบคุม PID ที่แยกส่วนของ Yaw และ Pitch
    """
    def __init__(self, Kp, Ki, Kd, integral_limit=200):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.p_error = 0.0
        self.acc_error = 0.0
        self.integral_limit = integral_limit
        self.last_time = time.time()

    def update(self, error):
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0.001: # ป้องกันการหารด้วยศูนย์และค่า dt ที่น้อยเกินไป
            return 0

        # Proportional
        P_out = self.Kp * error

        # Integral
        self.acc_error += error * dt
        self.acc_error = np.clip(self.acc_error, -self.integral_limit, self.integral_limit)
        I_out = self.Ki * self.acc_error

        # Derivative
        derivative = (error - self.p_error) / dt
        D_out = self.Kd * derivative

        # Update state
        self.p_error = error
        self.last_time = current_time

        return P_out + I_out + D_out

    def reset(self):
        self.p_error = 0.0
        self.acc_error = 0.0
        self.last_time = time.time()

# =====================================================
# Shared resources
# =====================================================
frame_queue = queue.Queue(maxsize=1)
# ขยาย output เพื่อรองรับข้อมูล PID และการยิง
processed_output = {
    "details": [],
    "speed_x": 0.0,
    "speed_y": 0.0,
    "target_locked": False,
    "target_error": (0, 0)
}
output_lock = threading.Lock()
stop_event = threading.Event()

# =====================================================
# AWB / Night enhance (จาก detect_GPT_BEST)
# =====================================================
def apply_awb(bgr):
    if hasattr(cv2, "xphoto") and hasattr(cv2.xphoto, "createLearningBasedWB"):
        wb = cv2.xphoto.createLearningBasedWB()
        try:
            wb.setSaturationThreshold(0.99)
        except Exception:
            pass
        return wb.balanceWhite(bgr)
    return bgr

def night_enhance_pipeline_cpu(bgr):
    return apply_awb(bgr)

# =====================================================
# Tracker/Detector (จาก detect_GPT_BEST)
# =====================================================
class ObjectTracker:
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        print(f"🖼️  ObjectTracker in {'GPU' if use_gpu else 'CPU'} mode")

    def _get_angle(self, pt1, pt2, pt0):
        dx1 = pt1[0] - pt0[0]; dy1 = pt1[1] - pt0[1]
        dx2 = pt2[0] - pt0[0]; dy2 = pt2[1] - pt0[1]
        dot = dx1*dx2 + dy1*dy2
        mag1 = (dx1*dx1 + dy1*dy1)**0.5
        mag2 = (dx2*dx2 + dy2*dy2)**0.5
        if mag1*mag2 == 0:
            return 0
        return math.degrees(math.acos(max(-1, min(1, dot/(mag1*mag2)))))

    def get_raw_detections(self, frame):
        enhanced = cv2.GaussianBlur(night_enhance_pipeline_cpu(frame), (5,5), 0)
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)

        ranges = {
            'Red': ([0,80,40],[10,255,255],[170,80,40],[180,255,255]),
            'Yellow': ([20,60,40],[35,255,255]),
            'Green': ([35,40,30],[85,255,255]),
            'Blue': ([90,40,30],[130,255,255])
        }
        masks = {}
        masks['Red'] = cv2.inRange(hsv, np.array(ranges['Red'][0]), np.array(ranges['Red'][1])) | \
                       cv2.inRange(hsv, np.array(ranges['Red'][2]), np.array(ranges['Red'][3]))
        for name in ['Yellow','Green','Blue']:
            masks[name] = cv2.inRange(hsv, np.array(ranges[name][0]), np.array(ranges[name][1]))

        combined = masks['Red'] | masks['Yellow'] | masks['Green'] | masks['Blue']
        kernel = np.ones((5,5), np.uint8)
        cleaned = cv2.morphologyEx(cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)

        contours,_ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        H,W = frame.shape[:2]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1500: continue
            x,y,w,h = cv2.boundingRect(cnt)
            if w==0 or h==0: continue
            ar = w/float(h)
            if ar>4.0 or ar<0.25: continue
            hull = cv2.convexHull(cnt); ha = cv2.contourArea(hull)
            if ha==0: continue
            solidity = area/ha
            if solidity < 0.85: continue
            if x<=2 or y<=2 or x+w>=W-2 or y+h>=H-2: continue

            contour_mask = np.zeros((H,W), np.uint8)
            cv2.drawContours(contour_mask, [cnt], -1, 255, -1)
            max_mean, found = 0, "Unknown"
            for cname, m in masks.items():
                mv = cv2.mean(m, mask=contour_mask)[0]
                if mv > max_mean:
                    max_mean, found = mv, cname
            if max_mean <= 20: continue

            shape = "Uncertain"
            peri = cv2.arcLength(cnt, True)
            circ = (4*math.pi*area)/(peri*peri) if peri>0 else 0
            if circ > 0.84:
                shape = "Circle"
            else:
                approx = cv2.approxPolyDP(cnt, 0.04*peri, True)
                if len(approx)==4 and solidity>0.9:
                    pts=[tuple(p[0]) for p in approx]
                    angs=[self._get_angle(pts[(i-1)%4], pts[(i+1)%4], p) for i,p in enumerate(pts)]
                    if all(75<=a<=105 for a in angs):
                        _,(rw,rh),_ = cv2.minAreaRect(cnt)
                        if min(rw,rh)>0:
                            ar2 = max(rw,rh)/min(rw,rh)
                            if 0.90<=ar2<=1.10: shape="Square"
                            elif w>h: shape="Rectangle_H"
                            else: shape="Rectangle_V"
            out.append({"contour":cnt,"shape":shape,"color":found,"box":(x,y,w,h)})
        return out

# =====================================================
# Connection Manager (จาก detect_GPT_BEST)
# =====================================================
class RMConnection:
    def __init__(self):
        self._lock = threading.Lock()
        self._robot = None
        self.connected = threading.Event()

    def connect(self):
        with self._lock:
            self._safe_close()
            print("🤖 Connecting to RoboMaster...")
            rb = robot.Robot()
            rb.initialize(conn_type="ap")
            # เพิ่มการ init gimbal และ blaster
            rb.gimbal.recenter(pitch_speed=200, yaw_speed=200).wait_for_completed()
            rb.blaster.set_led(brightness=255, effect=blaster.LED_ON)
            time.sleep(0.5)
            rb.camera.start_video_stream(display=False, resolution=r_camera.STREAM_720P)
            self._robot = rb
            self.connected.set()
            print("✅ RoboMaster connected, gimbal recentered & camera streaming")

    def _safe_close(self):
        if self._robot is not None:
            try:
                try: self._robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                except Exception: pass
                try: self._robot.camera.stop_video_stream()
                except Exception: pass
                try: self._robot.close()
                except Exception: pass
            finally:
                self._robot = None
                self.connected.clear()
                print("🔌 Connection closed")

    def drop_and_reconnect(self):
        with self._lock:
            self._safe_close()

    def get_robot_components(self):
        with self._lock:
            if self._robot is None:
                return None, None, None
            return self._robot.camera, self._robot.gimbal, self._robot.blaster

    def close(self):
        with self._lock:
            self._safe_close()

def reconnector_thread(manager: RMConnection):
    backoff = 1.0
    while not stop_event.is_set():
        if not manager.connected.is_set():
            try:
                manager.connect()
                backoff = 1.0
            except Exception as e:
                print(f"♻️ Reconnect failed: {e} (retry in {backoff:.1f}s)")
                time.sleep(backoff)
                backoff = min(backoff*1.6, 8.0)
                continue
        time.sleep(0.2)

# =====================================================
# Threads
# =====================================================
def capture_thread_func(manager: RMConnection, q: queue.Queue):
    print("🚀 Capture thread started")
    fail = 0
    while not stop_event.is_set():
        if not manager.connected.is_set():
            time.sleep(0.1)
            continue
        cam, _, _ = manager.get_robot_components()
        if cam is None:
            time.sleep(0.1)
            continue
        try:
            frame = cam.read_cv2_image(timeout=1.0)
            if frame is not None:
                if q.full():
                    try: q.get_nowait()
                    except queue.Empty: pass
                q.put(frame)
                fail = 0
            else:
                fail += 1
        except Exception as e:
            print(f"CRITICAL: camera read error: {e}")
            fail += 1

        if fail >= 10:
            print("⚠️ Too many camera errors → drop & reconnect")
            manager.drop_and_reconnect()
            try:
                while True: q.get_nowait()
            except queue.Empty:
                pass
            fail = 0
        time.sleep(0.005)
    print("🛑 Capture thread stopped")

def processing_thread_func(tracker: ObjectTracker, q: queue.Queue,
                           target_shape, target_color, roi_coords,
                           pid_yaw: PIDController, pid_pitch: PIDController,
                           is_detecting_func):
    """
    รวมการตรวจจับจาก detect_GPT_BEST และการคำนวณ PID
    """
    global processed_output
    print("🧠 Processing thread started.")
    ROI_X, ROI_Y, ROI_W, ROI_H = roi_coords

    # ค่าประมาณสำหรับกล้อง 720p (FOV~96° H, 54° V)
    # 1280x720 resolution
    FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
    PIXELS_PER_DEGREE_Y = FRAME_HEIGHT / 54.0
    # คำนวณ pixel offset สำหรับการเงย 3 องศา
    PITCH_OFFSET_PIXELS = 3.0 * PIXELS_PER_DEGREE_Y

    # จุดกึ่งกลางของภาพทั้งหมด (ไม่ใช่ ROI)
    CENTER_X = FRAME_WIDTH / 2
    CENTER_Y = FRAME_HEIGHT / 2 - PITCH_OFFSET_PIXELS # ปรับเป้าให้สูงขึ้น

    while not stop_event.is_set():
        if not is_detecting_func():
            # ถ้าไม่ได้อยู่ในโหมดตรวจจับ ให้ reset PID และตั้งค่า speed เป็น 0
            pid_yaw.reset()
            pid_pitch.reset()
            with output_lock:
                processed_output["speed_x"] = 0.0
                processed_output["speed_y"] = 0.0
                processed_output["target_locked"] = False
            time.sleep(0.1)
            continue

        try:
            frame_to_process = q.get(timeout=1.0)
            roi_frame = frame_to_process[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
            detections = tracker.get_raw_detections(roi_frame)

            detailed_results = []
            divider1 = int(ROI_W*0.33)
            divider2 = int(ROI_W*0.66)

            object_id_counter = 1
            target_found = None

            for d in detections:
                shape, color, (x,y,w,h) = d['shape'], d['color'], d['box']
                endx = x+w
                zone = "Center"
                if endx < divider1: zone = "Left"
                elif x >= divider2: zone = "Right"
                is_target = (shape == target_shape and color == target_color)

                # เก็บข้อมูลเป้าหมายแรกที่เจอ
                if is_target and target_found is None:
                    target_found = d

                detailed_results.append({
                    "id": object_id_counter, "color": color, "shape": shape,
                    "zone": zone, "is_target": is_target, "box": (x,y,w,h)
                })
                object_id_counter += 1

            # --- ส่วนคำนวณ PID ---
            speed_x, speed_y = 0.0, 0.0
            is_locked = False
            target_err = (0, 0)

            if target_found:
                x, y, w, h = target_found['box']
                # แปลงพิกัด box ของ ROI กลับเป็นพิกัดของเฟรมเต็ม
                abs_x, abs_y = x + ROI_X, y + ROI_Y
                target_center_x = abs_x + w / 2
                target_center_y = abs_y + h / 2

                # คำนวณ error (เป้าหมาย - ตำแหน่งจริง)
                err_x = CENTER_X - target_center_x
                err_y = CENTER_Y - target_center_y
                target_err = (err_x, err_y)

                speed_x = pid_yaw.update(err_x)
                speed_y = pid_pitch.update(err_y)

                # ตรวจสอบว่า lock เป้าได้หรือไม่ (error อยู่ในเกณฑ์ที่ยอมรับ)
                if abs(err_x) < 15 and abs(err_y) < 15:
                    is_locked = True
            else:
                # ถ้าไม่เจอเป้าหมาย ให้ reset PID
                pid_yaw.reset()
                pid_pitch.reset()

            with output_lock:
                processed_output = {
                    "details": detailed_results,
                    "speed_x": speed_x,
                    "speed_y": speed_y,
                    "target_locked": is_locked,
                    "target_error": target_err
                }

        except queue.Empty:
            continue
        except Exception as e:
            print(f"CRITICAL: Processing error: {e}")
            time.sleep(0.05)

    print("🛑 Processing thread stopped.")

# =====================================================
# Main
# =====================================================
if __name__ == "__main__":
    target_shape, target_color = "Circle", "Red"
    print(f"🎯 Target set to: {target_color} {target_shape}")

    tracker = ObjectTracker(use_gpu=USE_GPU)
    
    # --- ตั้งค่า PID ---
    # Kp: การตอบสนองต่อ error ปัจจุบัน (ยิ่งเยอะยิ่งเร็ว แต่อาจแกว่ง)
    # Ki: การแก้ error สะสม (ช่วยให้เข้าเป้านิ่ง แต่ถ้าเยอะไปจะ overshoot)
    # Kd: การต้านการเปลี่ยนแปลงที่รวดเร็ว (ช่วยลดการแกว่ง)
    pid_yaw_controller = PIDController(Kp=-0.3, Ki=-0.005, Kd=-0.05)
    pid_pitch_controller = PIDController(Kp=-0.3, Ki=-0.005, Kd=-0.05)


    # ROI (ปรับเป็นของ 720p ถ้าจำเป็น, ค่าเดิมน่าจะพอใช้ได้)
    ROI_Y, ROI_H, ROI_X, ROI_W = 264, 270, 10, 1260 # ขยาย ROI width ให้เต็ม

    manager = RMConnection()
    reconn = threading.Thread(target=reconnector_thread, args=(manager,), daemon=True)
    reconn.start()

    is_detecting_flag = {"v": False}
    def is_detecting(): return is_detecting_flag["v"]

    cap_t = threading.Thread(target=capture_thread_func, args=(manager, frame_queue), daemon=True)
    proc_t = threading.Thread(
        target=processing_thread_func,
        args=(tracker, frame_queue, target_shape, target_color,
              (ROI_X, ROI_Y, ROI_W, ROI_H), pid_yaw_controller, pid_pitch_controller, is_detecting),
        daemon=True
    )
    cap_t.start(); proc_t.start()

    print("\n--- Real-time Scanner & Tracker (Merged) ---")
    print("s: toggle detection/tracking, r: force reconnect, q: quit")

    display_frame = None
    last_fire_time = 0
    FIRE_COOLDOWN = 1.5 # วินาที

    try:
        while not stop_event.is_set():
            # --- รับข้อมูลจาก Threads ---
            with output_lock:
                details = processed_output["details"]
                s_x = processed_output["speed_x"]
                s_y = processed_output["speed_y"]
                is_locked = processed_output["target_locked"]
                target_error = processed_output["target_error"]

            # --- ควบคุม Gimbal และ Blaster ---
            cam, gimbal, blaster = manager.get_robot_components()
            if gimbal:
                # ส่งคำสั่งควบคุม gimbal (pitch_speed เป็นลบเพราะแกนกลับด้าน)
                gimbal.drive_speed(pitch_speed=-s_y, yaw_speed=s_x)

                # ตรรกะการยิง
                if is_locked and blaster and (time.time() - last_fire_time > FIRE_COOLDOWN):
                    print(f"🔥 Target locked! Firing! Error: {target_error}")
                    blaster.fire(fire_type=blaster.INFRARED_FIRE, times=1)
                    last_fire_time = time.time()

            # --- ส่วนแสดงผล (เหมือนเดิม) ---
            try:
                display_frame = frame_queue.get(timeout=0.5)
            except queue.Empty:
                if display_frame is None:
                    print("Waiting for first frame...")
                time.sleep(0.1)
                continue

            cv2.rectangle(display_frame, (ROI_X, ROI_Y), (ROI_X+ROI_W, ROI_Y+ROI_H), (255,0,0), 2)

            if is_detecting():
                status_text = "TRACKING" if any(d['is_target'] for d in details) else "DETECTING"
                status_color = (0, 165, 255) if status_text == "TRACKING" else (0, 255, 0)
                if is_locked:
                    status_text = "LOCKED"
                    status_color = (0, 0, 255)

                cv2.putText(display_frame, f"MODE: {status_text}", (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                
                with output_lock: details = processed_output["details"]
                d1_abs = ROI_X + int(ROI_W*0.33); d2_abs = ROI_X + int(ROI_W*0.66)
                cv2.line(display_frame, (d1_abs, ROI_Y), (d1_abs, ROI_Y+ROI_H), (255,255,0), 1)
                cv2.line(display_frame, (d2_abs, ROI_Y), (d2_abs, ROI_Y+ROI_H), (255,255,0), 1)

                for det in details:
                    x,y,w,h = det['box']
                    abs_x, abs_y = x + ROI_X, y + ROI_Y
                    box_color = (0,0,255) if det['is_target'] else ((0,255,255) if det['shape'] == 'Uncertain' else (0,255,0))
                    thickness = 4 if det['is_target'] else 2
                    cv2.rectangle(display_frame, (abs_x,abs_y), (abs_x+w, abs_y+h), box_color, thickness)
                    cv2.putText(display_frame, str(det['id']), (abs_x+5, abs_y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
                
                if details:
                    y_pos = 70
                    for obj in details:
                        target_str = " (TARGET!)" if obj['is_target'] else ""
                        line = f"ID {obj['id']}: {obj['color']} {obj['shape']}{target_str}"
                        cv2.putText(display_frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 4)
                        cv2.putText(display_frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                        y_pos += 25
            else:
                cv2.putText(display_frame, "MODE: VIEWING", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

            st = "CONNECTED" if manager.connected.is_set() else "RECONNECTING..."
            y_offset = 70 if not (is_detecting() and details) else y_pos
            cv2.putText(display_frame, f"SDK: {st}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            cv2.imshow("Robomaster Real-time Scan & Track", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('s'):
                is_detecting_flag["v"] = not is_detecting_flag["v"]
                print(f"Detection/Tracking {'ON' if is_detecting_flag['v'] else 'OFF'}")
            elif key == ord('r'):
                print("Manual reconnect requested")
                manager.drop_and_reconnect()
                try:
                    while True: frame_queue.get_nowait()
                except queue.Empty: pass

    except Exception as e:
        print(f"❌ Main loop error: {e}")
    finally:
        print("\n🔌 Shutting down...")
        stop_event.set()
        if manager.connected.is_set():
            _, gimbal, _ = manager.get_robot_components()
            if gimbal: gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
        try: cv2.destroyAllWindows()
        except Exception: pass
        manager.close()
        print("✅ Cleanup complete")