# -*- coding: utf-8 -*-
"""
fire_target_uncertain_followup.py

สิ่งที่สคริปต์นี้ทำ:
1) รัน detect_GPT_BEST + PID track (+pitch bias) + auto-ROI + fire-on-lock [เหมือน fire_target เดิม]
2) เมื่อ "ยิงแล้ว" และภายหลังพบวัตถุ "Uncertain" ทางซ้าย/ขวา:
   -> เดินหน้า 1 บล็อค (0.6 m) ด้วยฟังก์ชันเดินตรงจาก 3-10 (เฉพาะส่วนเดิน)
   -> ถึงบล็อคถัดไปแล้ว สั่งกิมบอล pitch = -14°, yaw = ข้างที่เจอ Uncertain (ถ้าเจอ 2 ข้างก็ตรวจทั้ง 2)
   -> เมื่อระบุรูปร่างได้ถูกต้องแล้ว ให้ print ว่าเจออะไร แล้วจบโปรแกรม

หมายเหตุ:
- โค้ดส่วนยิง/ตรวจจับ/โชว์จอ ถูกคง "ตรรกะเดิมทุกประการ" แล้วเพิ่ม hook ภายนอกโดยไม่แก้ขั้นตอนยิง
"""

import cv2
import numpy as np
import time
import math
import threading
import queue
from collections import deque

from robomaster import robot, camera as r_camera, blaster as r_blaster

# =========================
# CONFIG ของระบบยิง (เหมือนเดิม)
# =========================
TARGET_SHAPE = "Circle"
TARGET_COLOR = "Red"

PID_KP = -0.25
PID_KI = -0.01
PID_KD = -0.03
DERIV_LPF_ALPHA = 0.25

MAX_YAW_SPEED  = 220
MAX_PITCH_SPEED= 180
I_CLAMP = 2000.0

PIX_ERR_DEADZONE = 6
LOCK_TOL_X = 8
LOCK_TOL_Y = 8
LOCK_STABLE_COUNT = 6

FRAME_W, FRAME_H = 960, 540
VERTICAL_FOV_DEG = 54.0
PIXELS_PER_DEG_V = FRAME_H / VERTICAL_FOV_DEG

PITCH_BIAS_DEG = 2.0
PITCH_BIAS_PIX = +PITCH_BIAS_DEG * PIXELS_PER_DEG_V

ROI_Y0, ROI_H0, ROI_X0, ROI_W0 = 264, 270, 10, 911
ROI_SHIFT_PER_DEG = 6.0
ROI_Y_MIN, ROI_Y_MAX = 0, FRAME_H - 10

# =========================
# STATE & FLAGS (hook ภายนอก)
# =========================
frame_queue = queue.Queue(maxsize=1)
processed_output = {"details": []}  # [{id,color,shape,zone,is_target,box}]
output_lock = threading.Lock()
stop_event = threading.Event()

# กิมบอลมุมล่าสุด (pitch, yaw, pitch_ground, yaw_ground)
gimbal_angle_lock = threading.Lock()
gimbal_angles = (0.0, 0.0, 0.0, 0.0)

def sub_angle_cb(angle_info):
    global gimbal_angles
    with gimbal_angle_lock:
        gimbal_angles = tuple(angle_info)

# --- Hook สำหรับ flow หลังยิง ---
post_fire_state = {
    "just_fired": False,           # ตั้ง True ตอนที่ยิงสำเร็จ
    "last_fire_ts": 0.0,           # ไทม์สแตมป์ยิงล่าสุด
    "await_uncertain": True,       # หลังยิงรอพบ Uncertain
    "uncertain_sides": set(),      # {"Left", "Right"}
    "followup_done": False,        # ทำ flow เดิน/หมุน/สแกน เสร็จแล้ว
}

# ===================
# AWB / Night (เดิม)
# ===================
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

# ==============================
# Detector (ตรรกะเดิมทั้งหมด)
# ==============================
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
        if mag1*mag2 == 0: return 0
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
            if circ > 0.82:
                shape = "Circle"
            else:
                approx = cv2.approxPolyDP(cnt, 0.04*peri, True)
                if len(approx)==4 and solidity>0.88:
                    pts=[tuple(p[0]) for p in approx]
                    angs=[self._get_angle(pts[(i-1)%4], pts[(i+1)%4], p) for i,p in enumerate(pts)]
                    if all(70<=a<=110 for a in angs):
                        _,(rw,rh),_ = cv2.minAreaRect(cnt)
                        if min(rw,rh)>0:
                            ar2 = max(rw,rh)/min(rw,rh)
                            if 0.88<=ar2<=1.12: shape="Square"
                            elif w>h: shape="Rectangle_H"
                            else: shape="Rectangle_V"
            out.append({"contour":cnt,"shape":shape,"color":found,"box":(x,y,w,h)})
        return out
# (อ้างอิงส่วนตรวจจับ/โชว์/ยิงมาจากไฟล์ fire_target เดิม)  # :contentReference[oaicite:2]{index=2}

# ======================================
# Connection manager (เพิ่ม get_chassis)
# ======================================
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
            rb.camera.start_video_stream(display=False, resolution=r_camera.STREAM_540P)
            try:
                rb.gimbal.sub_angle(freq=50, callback=sub_angle_cb)
            except Exception as e:
                print("Gimbal sub_angle error:", e)
            self._robot = rb
            self.connected.set()
            print("✅ RoboMaster connected & camera streaming")
            try:
                rb.gimbal.recenter(pitch_speed=200, yaw_speed=200).wait_for_completed()
            except Exception as e:
                print("Recenter error:", e)

    def _safe_close(self):
        if self._robot is not None:
            try:
                try: self._robot.camera.stop_video_stream()
                except Exception: pass
                try:
                    try: self._robot.gimbal.unsub_angle()
                    except Exception: pass
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

    def get_camera(self):
        with self._lock:
            return None if self._robot is None else self._robot.camera

    def get_gimbal(self):
        with self._lock:
            return None if self._robot is None else self._robot.gimbal

    def get_blaster(self):
        with self._lock:
            return None if self._robot is None else self._robot.blaster

    def get_chassis(self):  # เพิ่มเพื่อใช้เดินตรง
        with self._lock:
            return None if self._robot is None else self._robot.chassis

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

# =========================
# Threads: capture, detect
# =========================
def capture_thread_func(manager: RMConnection, q: queue.Queue):
    print("🚀 Capture thread started")
    fail = 0
    while not stop_event.is_set():
        if not manager.connected.is_set():
            time.sleep(0.1); continue
        cam = manager.get_camera()
        if cam is None:
            time.sleep(0.1); continue
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
                           target_shape, target_color,
                           roi_state,
                           is_detecting_func):
    global processed_output
    print("🧠 Processing thread started.")

    while not stop_event.is_set():
        if not is_detecting_func():
            time.sleep(0.05); continue
        try:
            frame_to_process = q.get(timeout=1.0)

            with gimbal_angle_lock:
                pitch_deg = gimbal_angles[0]  # + ขึ้น, - ลง
            roi_y_dynamic = int(ROI_Y0 - (max(0.0, -pitch_deg) * ROI_SHIFT_PER_DEG))
            roi_y_dynamic = max(ROI_Y_MIN, min(ROI_Y_MAX, roi_y_dynamic))

            ROI_X, ROI_W = roi_state["x"], roi_state["w"]
            ROI_H = roi_state["h"]
            roi_state["y"] = roi_y_dynamic

            roi_frame = frame_to_process[roi_y_dynamic:roi_y_dynamic+ROI_H, ROI_X:ROI_X+ROI_W]
            detections = tracker.get_raw_detections(roi_frame)

            detailed_results = []
            divider1 = int(ROI_W*0.33)
            divider2 = int(ROI_W*0.66)

            object_id_counter = 1
            sides_seen_uncertain = set()

            for d in detections:
                shape, color, (x,y,w,h) = d['shape'], d['color'], d['box']
                endx = x+w
                zone = "Center"
                if endx < divider1: zone = "Left"
                elif x >= divider2: zone = "Right"
                is_target = (shape == target_shape and color == target_color)

                detailed_results.append({
                    "id": object_id_counter,
                    "color": color,
                    "shape": shape,
                    "zone": zone,
                    "is_target": is_target,
                    "box": (x,y,w,h)
                })

                # --- Hook: เก็บข้างที่พบ "Uncertain" เฉพาะหลังยิงแล้ว ---
                if post_fire_state["just_fired"] and shape == "Uncertain" and zone in ("Left","Right"):
                    sides_seen_uncertain.add(zone)

                object_id_counter += 1

            with output_lock:
                processed_output = {"details": detailed_results}

            # อัปเดต state ว่าหลังยิงเจอ Uncertain ฝั่งไหนบ้าง
            if sides_seen_uncertain:
                post_fire_state["uncertain_sides"].update(sides_seen_uncertain)

        except queue.Empty:
            continue
        except Exception as e:
            print(f"CRITICAL: Processing error: {e}")
            time.sleep(0.02)

    print("🛑 Processing thread stopped.")

# ==========================================
# Control thread (PID drive + fire on lock)
# ==========================================
def control_thread_func(manager: RMConnection, roi_state, is_detecting_func):
    print("🎯 Control thread started.")
    prev_time = None
    err_x_prev_f = 0.0
    err_y_prev_f = 0.0
    integ_x = 0.0
    integ_y = 0.0
    lock_queue = deque(maxlen=LOCK_STABLE_COUNT)

    while not stop_event.is_set():
        if not (is_detecting_func() and manager.connected.is_set()):
            time.sleep(0.02); continue

        with output_lock:
            dets = list(processed_output["details"])

        # หาเป้าหมาย (TARGET)
        target_box = None
        max_area = -1
        for det in dets:
            if det.get("is_target", False):
                x,y,w,h = det["box"]
                area = w*h
                if area > max_area:
                    max_area = area
                    target_box = (x,y,w,h)

        gimbal = manager.get_gimbal()
        blaster = manager.get_blaster()
        if (gimbal is None) or (blaster is None):
            time.sleep(0.02); continue

        if target_box is not None:
            x,y,w,h = target_box
            cx_roi = x + w/2.0
            cy_roi = y + h/2.0
            ROI_X, ROI_Y, ROI_W, ROI_H = roi_state["x"], roi_state["y"], roi_state["w"], roi_state["h"]
            cx = ROI_X + cx_roi
            cy = ROI_Y + cy_roi
            center_x = FRAME_W/2.0
            center_y = FRAME_H/2.0

            err_x = (center_x - cx)
            err_y = (center_y - cy) + PITCH_BIAS_PIX

            if abs(err_x) < PIX_ERR_DEADZONE: err_x = 0.0
            if abs(err_y) < PIX_ERR_DEADZONE: err_y = 0.0

            now = time.time()
            if prev_time is None:
                prev_time = now
                err_x_prev_f = err_x
                err_y_prev_f = err_y
                time.sleep(0.005)
                continue
            dt = max(1e-3, now - prev_time)
            prev_time = now

            err_x_f = err_x_prev_f + DERIV_LPF_ALPHA*(err_x - err_x_prev_f)
            err_y_f = err_y_prev_f + DERIV_LPF_ALPHA*(err_y - err_y_prev_f)
            dx = (err_x_f - err_x_prev_f)/dt
            dy = (err_y_f - err_y_prev_f)/dt
            err_x_prev_f = err_x_f
            err_y_prev_f = err_y_f

            integ_x = np.clip(integ_x + err_x*dt, -I_CLAMP, I_CLAMP)
            integ_y = np.clip(integ_y + err_y*dt, -I_CLAMP, I_CLAMP)

            u_x = PID_KP*err_x + PID_KI*integ_x + PID_KD*dx
            u_y = PID_KP*err_y + PID_KI*integ_y + PID_KD*dy

            u_x = float(np.clip(u_x, -MAX_YAW_SPEED, MAX_YAW_SPEED))
            u_y = float(np.clip(u_y, -MAX_PITCH_SPEED, MAX_PITCH_SPEED))

            try:
                gimbal.drive_speed(pitch_speed=-u_y, yaw_speed=u_x)
            except Exception as e:
                print("drive_speed error:", e)

            locked = (abs(err_x) <= LOCK_TOL_X) and (abs(err_y) <= LOCK_TOL_Y)
            lock_queue.append(1 if locked else 0)

            if len(lock_queue) == LOCK_STABLE_COUNT and sum(lock_queue) == LOCK_STABLE_COUNT:
                try:
                    blaster.fire(fire_type=r_blaster.WATER_FIRE)
                    time.sleep(0.1)
                    lock_queue.clear()

                    # ---- Hook: Mark fired ----
                    post_fire_state["just_fired"] = True
                    post_fire_state["last_fire_ts"] = time.time()
                    post_fire_state["await_uncertain"] = True
                    post_fire_state["uncertain_sides"].clear()
                    print("💥 Fired! Awaiting 'Uncertain' then follow-up flow...")

                except Exception as e:
                    print("fire error:", e)
        else:
            try:
                gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
            except Exception:
                pass
            lock_queue.clear()
            integ_x *= 0.98
            integ_y *= 0.98

        time.sleep(0.005)

    print("🛑 Control thread stopped.")
# (ตรรกะยิงคงเดิมจาก fire_target)  # :contentReference[oaicite:3]{index=3}

# =====================================================
# ส่วน "เดินหน้า 1 บล็อค" เอามาจาก 3-10 เฉพาะการเดินตรง
# =====================================================
class AttitudeHandler:
    """ย่อส่วนจาก 3-10 สำหรับใช้ lock yaw เบาๆ ตอนเดิน"""
    def __init__(self):
        self.current_yaw, self.yaw_tolerance, self.is_monitoring = 0.0, 3.0, False
    def attitude_handler(self, attitude_info):
        if self.is_monitoring: self.current_yaw = attitude_info[0]
    def start_monitoring(self, chassis):
        self.is_monitoring = True; chassis.sub_attitude(freq=20, callback=self.attitude_handler)
    def stop_monitoring(self, chassis):
        self.is_monitoring = False
        try: chassis.unsub_attitude()
        except Exception: pass
    def normalize_angle(self, angle):
        while angle > 180: angle -= 360
        while angle <= -180: angle += 360
        return angle
    def correct_yaw_to_target(self, chassis, target_yaw=0.0):
        normalized_target = self.normalize_angle(target_yaw); time.sleep(0.1)
        robot_rotation = -self.normalize_angle(normalized_target - self.current_yaw)
        if abs(robot_rotation) > self.yaw_tolerance:
            chassis.move(x=0, y=0, z=robot_rotation, z_speed=60).wait_for_completed(timeout=2)
            time.sleep(0.1)
        # fine tune (ย่อ)
        remaining = -self.normalize_angle(normalized_target - self.current_yaw)
        if 0.5 < abs(remaining) < 20:
            chassis.move(x=0, y=0, z=remaining, z_speed=40).wait_for_completed(timeout=2)
            time.sleep(0.1)

def get_compensated_target_yaw():
    """ใน flow นี้ตั้งให้ 0° (เดินไปข้างหน้าแนวแกนโลกเดิม)"""
    return 0.0

class PID_1D:
    def __init__(self, Kp, Ki, Kd, setpoint=0):
        self.Kp, self.Ki, self.Kd, self.setpoint = Kp, Ki, Kd, setpoint
        self.prev_error, self.integral, self.integral_max = 0, 0, 1.0
    def compute(self, current, dt):
        error = self.setpoint - current
        self.integral += error * dt
        self.integral = max(min(self.integral, self.integral_max), -self.integral_max)
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

class MovementController:
    """ดึงจาก 3-10 เฉพาะ move_forward_one_grid() (0.6 m)"""  # :contentReference[oaicite:4]{index=4}
    def __init__(self, chassis):
        self.chassis = chassis
        self.current_x_pos, self.current_y_pos = 0.0, 0.0
        self.chassis.sub_position(freq=20, callback=self.position_handler)
    def position_handler(self, position_info):
        self.current_x_pos, self.current_y_pos = position_info[0], position_info[1]
    def _calculate_yaw_correction(self, attitude_handler, target_yaw):
        KP_YAW = 1.8; MAX_YAW_SPEED = 25
        yaw_error = attitude_handler.normalize_angle(target_yaw - attitude_handler.current_yaw)
        speed = KP_YAW * yaw_error
        return max(min(speed, MAX_YAW_SPEED), -MAX_YAW_SPEED)
    def move_forward_one_grid(self, axis, attitude_handler):
        attitude_handler.correct_yaw_to_target(self.chassis, get_compensated_target_yaw())
        target_distance = 0.6
        pid = PID_1D(Kp=1.8, Ki=0.25, Kd=12, setpoint=target_distance)
        start_time, last_time = time.time(), time.time()
        start_position = self.current_x_pos if axis == 'x' else self.current_y_pos
        print(f"🚀 Moving FORWARD 0.6m, monitoring GLOBAL AXIS '{axis}'")
        while time.time() - start_time < 3.5:
            now = time.time(); dt = now - last_time; last_time = now
            current_position = self.current_x_pos if axis == 'x' else self.current_y_pos
            relative_position = abs(current_position - start_position)
            if abs(relative_position - target_distance) < 0.03:
                print("✅ Move complete!"); break
            output = pid.compute(relative_position, dt)
            ramp_multiplier = min(1.0, 0.1 + ((now - start_time) / 1.0) * 0.9)
            speed = max(-1.0, min(1.0, output * ramp_multiplier))
            yaw_correction = self._calculate_yaw_correction(attitude_handler, get_compensated_target_yaw())
            self.chassis.drive_speed(x=speed, y=0, z=yaw_correction, timeout=1)
            print(f"Moving... Dist: {relative_position:.3f}/{target_distance:.2f} m", end='\r')
        self.chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0); time.sleep(0.5)

# ===========================================================
# Follow-up orchestrator: เดิน 1 บล็อค -> กิมบอล -14° + สแกน
# ===========================================================
def followup_flow_if_ready(manager: RMConnection, is_detecting_func):
    """
    เรียกวนจาก main loop: ถ้าหลังยิงพบ Uncertain แล้ว จะ:
      1) ปิดตรวจจับชั่วคราว
      2) เดินหน้า 0.6 m (แกนโลก x)
      3) หยุด -> หมุนกิมบอล pitch=-14, yaw ไปฝั่งที่มี 'Uncertain' (ซ้าย/ขวา)
         - ถ้ามี 2 ฝั่งก็ตรวจทีละฝั่ง: Left(-90°) แล้ว Right(+90°)
      4) เปิดตรวจจับเฉพาะตอนส่องฝั่งนั้นจนพบวัตถุที่ 'shape != Uncertain'
         -> print "พบ <color> <shape>" แล้ว stop_event.set() เพื่อจบโปรแกรม
    """
    if post_fire_state["followup_done"]:
        return

    # เงื่อนไขเข้า flow: ยิงไปแล้ว + รอพบ Uncertain + มีรายชื่อข้างอย่างน้อย 1
    if not post_fire_state["just_fired"]: return
    # ให้เวลาหลังยิงนิดหน่อยเผื่อเฟรมค้าง
    if time.time() - post_fire_state["last_fire_ts"] < 0.2: return
    if post_fire_state["await_uncertain"] and not post_fire_state["uncertain_sides"]:
        return

    # ---- เริ่ม follow-up ----
    print("\n🧭 FOLLOW-UP: Move 1 block forward then side-scan at pitch -14°")
    # ปิดตรวจจับชั่วคราวระหว่างเดิน
    detect_prev = is_detecting_func()
    is_detecting_flag["v"] = False
    time.sleep(0.1)

    # เดิน 0.6 m
    chassis = manager.get_chassis()
    if chassis is None:
        print("❌ No chassis from SDK; cannot move.")
    else:
        att = AttitudeHandler(); att.start_monitoring(chassis)
        mover = MovementController(chassis)
        try:
            mover.move_forward_one_grid(axis='x', attitude_handler=att)   # 0.6 m
        finally:
            att.stop_monitoring(chassis)

    # เตรียมลิสต์ฝั่งเรียงลำดับตรวจ
    sides = list(sorted(post_fire_state["uncertain_sides"], key=lambda s: 0 if s=="Left" else 1))
    if len(sides)==0:
        # fallback ถ้าไม่มีบันทึก ให้ลองทั้งซ้ายและขวา
        sides = ["Left","Right"]

    gimbal = manager.get_gimbal()
    if gimbal is None:
        print("❌ No gimbal; cannot scan.")
        is_detecting_flag["v"] = detect_prev
        post_fire_state["followup_done"] = True
        return

    # ฟังก์ชันช่วย: เปิด detect ชั่วคราวเพื่อยืนยัน shape
    def scan_one_side(side_label):
        yaw_target = -90 if side_label=="Left" else 90
        try:
            gimbal.moveto(pitch=-14, yaw=yaw_target, yaw_speed=180).wait_for_completed()
        except Exception as e:
            print("gimbal moveto error:", e)
            return False

        # เปิดตรวจจับ
        is_detecting_flag["v"] = True

        # รอจนกว่าจะเห็น shape != "Uncertain" (timeout ~4s)
        t0 = time.time()
        while time.time() - t0 < 4.0 and not stop_event.is_set():
            # อ่าน snapshot ผลตรวจ
            with output_lock:
                dets = list(processed_output["details"])
            # กรองเฉพาะกล่องในโซนฝั่งที่ส่อง (ตีความง่าย ๆ: x ครึ่งซ้าย/ขวา)
            # แต่เราใช้ "zone" ที่คำนวณไว้แล้ว
            for d in dets:
                if d["zone"] == ("Left" if yaw_target==-90 else "Right"):
                    if d["shape"] != "Uncertain":
                        print(f"✅ พบ {d['color']} {d['shape']} ทาง{side_label}")
                        # จบโปรแกรม
                        stop_event.set()
                        return True
            time.sleep(0.05)

        # ไม่เจอ
        is_detecting_flag["v"] = False
        return False

    # ตรวจทุกฝั่งที่มี
    found_any = False
    for side in sides:
        found_any = scan_one_side(side)
        if found_any: break

    if not found_any:
        # ลองอีกฝั่งที่เหลือ (กรณีเริ่มจากขวาแล้วไม่เจอ)
        for side in (["Left","Right"] if sides!=["Left","Right"] else []):
            if side not in sides:
                found_any = scan_one_side(side)
                if found_any: break

    # ถ้ายังไม่เจอ ก็พิมพ์สรุปแล้วปิด
    if not found_any:
        print("⚠️ ไม่พบวัตถุที่ระบุรูปร่างชัดเจนจากฝั่งที่สแกน")

    post_fire_state["followup_done"] = True

# =========
# Main UI
# =========
USE_GPU = False
try:
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print("✅ CUDA available, enabling GPU path")
        USE_GPU = True
    else:
        print("⚠️ CUDA not available, CPU path")
except Exception:
    print("⚠️ Skip CUDA check, CPU path")

if __name__ == "__main__":
    print(f"🎯 Target set to: {TARGET_COLOR} {TARGET_SHAPE}")

    tracker = ObjectTracker(use_gpu=USE_GPU)

    roi_state = {"x": ROI_X0, "y": ROI_Y0, "w": ROI_W0, "h": ROI_H0}

    manager = RMConnection()
    reconn = threading.Thread(target=reconnector_thread, args=(manager,), daemon=True)
    reconn.start()

    is_detecting_flag = {"v": True}
    def is_detecting(): return is_detecting_flag["v"]

    cap_t  = threading.Thread(target=capture_thread_func, args=(manager, frame_queue), daemon=True)
    proc_t = threading.Thread(target=processing_thread_func,
                              args=(tracker, frame_queue, TARGET_SHAPE, TARGET_COLOR, roi_state, is_detecting),
                              daemon=True)
    ctrl_t = threading.Thread(target=control_thread_func, args=(manager, roi_state, is_detecting), daemon=True)

    cap_t.start(); proc_t.start(); ctrl_t.start()

    print("\n--- Real-time Scanner + PID Track (+pitch bias) + Follow-up Uncertain Flow ---")
    print("s: toggle detection, r: force reconnect, q: quit")

    display_frame = None
    try:
        while not stop_event.is_set():
            # main loop: แสดงภาพ + ติดตามสถานะ + เรียก follow-up ถ้าพร้อม
            try:
                display_frame = frame_queue.get(timeout=1.0)
            except queue.Empty:
                if display_frame is None:
                    print("Waiting for first frame.")
                time.sleep(0.15)
                # แม้ไม่มีเฟรม ก็เช็ค follow-up ได้
                followup_flow_if_ready(manager, is_detecting)
                continue

            # วาด ROI + Line + รายการตรวจจับ เหมือนเดิม
            ROI_X, ROI_Y, ROI_W, ROI_H = roi_state["x"], roi_state["y"], roi_state["w"], roi_state["h"]
            cv2.rectangle(display_frame, (ROI_X, ROI_Y), (ROI_X+ROI_W, ROI_Y+ROI_H), (0,255,0), 2)
            cv2.line(display_frame, (ROI_X + ROI_W//3, ROI_Y), (ROI_X + ROI_W//3, ROI_Y+ROI_H), (255,255,0), 1)
            cv2.line(display_frame, (ROI_X + 2*ROI_W//3, ROI_Y), (ROI_X + 2*ROI_W//3, ROI_Y+ROI_H), (255,255,0), 1)

            with output_lock:
                details = list(processed_output["details"])
            for det in details:
                x,y,w,h = det["box"]
                abs_x, abs_y = ROI_X + x, ROI_Y + y
                color_str = det['color']; shape_str = det['shape']
                zone_str = det['zone']
                box_color = (0,0,255) if det['is_target'] else (255,255,255)
                thickness = 3 if det['is_target'] else 1
                cv2.rectangle(display_frame, (abs_x,abs_y), (abs_x+w,abs_y+h), box_color, thickness)
                cv2.putText(display_frame, f"{color_str} {shape_str} [{zone_str}]",
                            (abs_x, abs_y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

            # รายการซ้ายบน + เส้น crosshair และเส้น bias
            if details:
                y_pos = 70
                for obj in details:
                    target_str = " (TARGET!)" if obj['is_target'] else ""
                    line = f"ID {obj['id']}: {obj['color']} {obj['shape']} [{obj['zone']}]"+target_str
                    cv2.putText(display_frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 4)
                    cv2.putText(display_frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                    y_pos += 22

            cy_bias = int(FRAME_H/2 - PITCH_BIAS_PIX)
            cv2.line(display_frame, (FRAME_W//2 - 20, FRAME_H//2), (FRAME_W//2 + 20, FRAME_H//2), (255,255,255), 1)
            cv2.line(display_frame, (FRAME_W//2, FRAME_H//2 - 20), (FRAME_W//2, FRAME_H//2 + 20), (255,255,255), 1)
            cv2.line(display_frame, (0, cy_bias), (FRAME_W, cy_bias), (0, 128, 255), 1)

            # สถานะ SDK
            st = "CONNECTED" if manager.connected.is_set() else "RECONNECTING."
            cv2.putText(display_frame, f"SDK: {st}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            cv2.imshow("Robomaster: Track+Fire + Uncertain Follow-up", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                is_detecting_flag["v"] = not is_detecting_flag["v"]
                print(f"Detection {'ON' if is_detecting_flag['v'] else 'OFF'}")
            elif key == ord('r'):
                print("Manual reconnect requested")
                manager.drop_and_reconnect()
                try:
                    while True: frame_queue.get_nowait()
                except queue.Empty:
                    pass

            # เรียก follow-up ถ้าพร้อม
            followup_flow_if_ready(manager, is_detecting)

    except Exception as e:
        print(f"❌ Main loop error: {e}")
    finally:
        print("\n🔌 Shutting down.")
        stop_event.set()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        manager.close()
        print("✅ Cleanup complete")
