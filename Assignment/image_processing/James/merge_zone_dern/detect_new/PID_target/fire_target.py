# detect_GPT_BEST + PID gimbal track (+3° pitch bias) + auto-ROI shift + fire-on-lock
# ตรรกะตรวจจับ/โชว์จอ ยังคงรูปแบบไฟล์เดิมทั้งหมด
# ปรับเพิ่ม: กิมบอล, PID, ยิง, ROI dynamics, Reconnect ครบ

import cv2
import numpy as np
import time
import math
import threading
import queue
from collections import deque

from robomaster import robot, camera as r_camera, blaster as r_blaster

# =========================
# CONFIG (ปรับได้ตามเครื่อง)
# =========================
TARGET_SHAPE = "Circle"
TARGET_COLOR = "Red"

# PID ที่ "ไม่หลอน" (เริ่มค่าปลอดภัย)
PID_KP = -0.25   # สเกลกับ error (pixels)
PID_KI = -0.01  # เริ่ม 0 กันหลอนสะสม
PID_KD = -0.03   # derivative (ใช้บน error ที่ผ่าน LPF)
DERIV_LPF_ALPHA = 0.25  # 0..1  (ต่ำ = หน่วงมาก ลด noise)

MAX_YAW_SPEED  = 220    # deg/s ตาม SDK จะ map ให้เองใน drive_speed
MAX_PITCH_SPEED= 180
I_CLAMP = 2000.0        # ลิมิตค่า integral เพื่อตัด windup

PIX_ERR_DEADZONE = 6     # พิกเซล ถือว่าเล็งแล้ว (แก้ได้)
LOCK_TOL_X = 8           # เกณฑ์ยิง
LOCK_TOL_Y = 8
LOCK_STABLE_COUNT = 6    # เฟรมติดกันที่เข้าเกณฑ์ก่อนยิง

# กล้อง 1280x720 จากโค้ดเดิม; สมมุติ vFOV 54° (แก้ได้ถ้าทราบจริง)
FRAME_W, FRAME_H = 960, 540
VERTICAL_FOV_DEG = 54.0
PIXELS_PER_DEG_V = FRAME_H / VERTICAL_FOV_DEG

# ต้องการเล็งเชิดขึ้น +3°
PITCH_BIAS_DEG = 2.5
PITCH_BIAS_PIX = +PITCH_BIAS_DEG * PIXELS_PER_DEG_V  # บวกที่ error_y (เล็งสูงขึ้น)

# ROI เริ่มต้นตามไฟล์เดิม
ROI_Y0, ROI_H0, ROI_X0, ROI_W0 = 264, 270, 10, 911

# สัมประสิทธิ์เลื่อน ROI ตาม pitch (พิกเซล/องศา)
ROI_SHIFT_PER_DEG = 6.0  # pitch ลง 1° -> ROI_Y ขยับขึ้น ~6 px (ปรับได้)
ROI_Y_MIN, ROI_Y_MAX = 0, FRAME_H - 10

# ================
# GPU check (เดิม)
# ================
USE_GPU = False
try:
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print("✅ CUDA available, enabling GPU path")
        USE_GPU = True
    else:
        print("⚠️ CUDA not available, CPU path")
except Exception:
    print("⚠️ Skip CUDA check, CPU path")

# ======================
# Shared & Thread flags
# ======================
frame_queue = queue.Queue(maxsize=1)
processed_output = {"details": []}  # [{id,color,shape,zone,is_target,box}]
output_lock = threading.Lock()
stop_event = threading.Event()

# มุมกิมบอล (pitch, yaw, pitch_g, yaw_g) ล่าสุด
gimbal_angle_lock = threading.Lock()
gimbal_angles = (0.0, 0.0, 0.0, 0.0)

def sub_angle_cb(angle_info):
    global gimbal_angles
    with gimbal_angle_lock:
        gimbal_angles = tuple(angle_info)  # (pitch, yaw, pitch_ground, yaw_ground)

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
        if mag1*mag2 == 0:
            return 0
        return math.degrees(math.acos(max(-1, min(1, dot/(mag1*mag2)))) )

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

# ======================================
# Connection manager + (เพิ่ม get gimbal)
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
            # subscribe angles
            try:
                rb.gimbal.sub_angle(freq=50, callback=sub_angle_cb)
            except Exception as e:
                print("Gimbal sub_angle error:", e)
            self._robot = rb
            self.connected.set()
            print("✅ RoboMaster connected & camera streaming")

            # recenter gimbal on start
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
                           roi_state,  # dict: {x,y,w,h} (อัปเดต y แบบ dynamic)
                           is_detecting_func):
    global processed_output
    print("🧠 Processing thread started.")

    while not stop_event.is_set():
        if not is_detecting_func():
            time.sleep(0.05); continue
        try:
            frame_to_process = q.get(timeout=1.0)

            # เลื่อน ROI ตาม pitch ปัจจุบัน
            with gimbal_angle_lock:
                pitch_deg = gimbal_angles[0]  # + ขึ้น, - ลง (ตาม SDK)
            # ถ้าก้มลง (pitch < 0) => ขยับ ROI_Y ขึ้น
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
                object_id_counter += 1

            with output_lock:
                processed_output = {"details": detailed_results}

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
    # PID states
    prev_time = None
    err_x_prev_f = 0.0
    err_y_prev_f = 0.0
    integ_x = 0.0
    integ_y = 0.0

    lock_queue = deque(maxlen=LOCK_STABLE_COUNT)

    while not stop_event.is_set():
        if not (is_detecting_func() and manager.connected.is_set()):
            time.sleep(0.02); continue

        # หา target ล่าสุด (เลือกกล่องที่ใหญ่สุดในหมวด is_target)
        with output_lock:
            dets = list(processed_output["details"])

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

        # เปิดเฉพาะเมื่อมีเป้า
        if target_box is not None:
            x,y,w,h = target_box

            # center เป้าภายใน ROI
            cx_roi = x + w/2.0
            cy_roi = y + h/2.0

            # แปลงเป็นพิกเซลในทั้งเฟรม
            ROI_X, ROI_Y, ROI_W, ROI_H = roi_state["x"], roi_state["y"], roi_state["w"], roi_state["h"]
            cx = ROI_X + cx_roi
            cy = ROI_Y + cy_roi

            # ศูนย์ภาพ
            center_x = FRAME_W/2.0
            center_y = FRAME_H/2.0

            # Error (ภาพ → กิมบอล): yaw ใช้ x, pitch ใช้ y
            # เพิ่ม PITCH_BIAS_PIX เพื่อเล็งสูงขึ้น ~ +3°
            err_x = (center_x - cx)
            err_y = (center_y - cy) + PITCH_BIAS_PIX

            # deadzone ลด jitter
            if abs(err_x) < PIX_ERR_DEADZONE: err_x = 0.0
            if abs(err_y) < PIX_ERR_DEADZONE: err_y = 0.0

            # dt
            now = time.time()
            if prev_time is None:
                prev_time = now
                err_x_prev_f = err_x
                err_y_prev_f = err_y
                time.sleep(0.005)
                continue
            dt = max(1e-3, now - prev_time)
            prev_time = now

            # Low-pass derivative (บน error)
            err_x_f = err_x_prev_f + DERIV_LPF_ALPHA*(err_x - err_x_prev_f)
            err_y_f = err_y_prev_f + DERIV_LPF_ALPHA*(err_y - err_y_prev_f)
            dx = (err_x_f - err_x_prev_f)/dt
            dy = (err_y_f - err_y_prev_f)/dt
            err_x_prev_f = err_x_f
            err_y_prev_f = err_y_f

            # anti-windup
            integ_x = np.clip(integ_x + err_x*dt, -I_CLAMP, I_CLAMP)
            integ_y = np.clip(integ_y + err_y*dt, -I_CLAMP, I_CLAMP)

            # PID → ความเร็วกิมบอล
            u_x = PID_KP*err_x + PID_KI*integ_x + PID_KD*dx   # map → yaw_speed
            u_y = PID_KP*err_y + PID_KI*integ_y + PID_KD*dy   # map → pitch_speed

            # clamp
            u_x = float(np.clip(u_x, -MAX_YAW_SPEED, MAX_YAW_SPEED))
            u_y = float(np.clip(u_y, -MAX_PITCH_SPEED, MAX_PITCH_SPEED))

            try:
                # หมายเหตุ: pitch_speed ใน SDK แกนกลับกับภาพ (บนลงลบ/บวกแล้วแต่ SDK)
                # จากโค้ดเดิมใช้ ep_gimbal.drive_speed(pitch=-speed_y, yaw=+speed_x)
                gimbal.drive_speed(pitch_speed=-u_y, yaw_speed=u_x)
            except Exception as e:
                print("drive_speed error:", e)

            # ตรวจ lock เพื่อยิง
            locked = (abs(err_x) <= LOCK_TOL_X) and (abs(err_y) <= LOCK_TOL_Y)
            lock_queue.append(1 if locked else 0)

            if len(lock_queue) == LOCK_STABLE_COUNT and sum(lock_queue) == LOCK_STABLE_COUNT:
                try:
                    blaster.fire(fire_type=r_blaster.WATER_FIRE)
                    # กันยิงรัว: เว้นเล็กน้อยและรีเซ็ตคิว lock
                    time.sleep(0.1)
                    lock_queue.clear()
                except Exception as e:
                    print("fire error:", e)
        else:
            # ไม่มีเป้า → ค่อยๆหยุด
            try:
                gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
            except Exception:
                pass
            lock_queue.clear()
            # ลด integral ให้หายหลอน
            integ_x *= 0.98
            integ_y *= 0.98

        time.sleep(0.005)

    print("🛑 Control thread stopped.")

# =========
# Main UI
# =========
if __name__ == "__main__":
    print(f"🎯 Target set to: {TARGET_COLOR} {TARGET_SHAPE}")

    tracker = ObjectTracker(use_gpu=USE_GPU)

    # ROI state (dynamic Y)
    roi_state = {"x": ROI_X0, "y": ROI_Y0, "w": ROI_W0, "h": ROI_H0}

    manager = RMConnection()
    reconn = threading.Thread(target=reconnector_thread, args=(manager,), daemon=True)
    reconn.start()

    is_detecting_flag = {"v": True}  # เริ่มตรวจจับ ON เลย
    def is_detecting(): return is_detecting_flag["v"]

    cap_t  = threading.Thread(target=capture_thread_func, args=(manager, frame_queue), daemon=True)
    proc_t = threading.Thread(target=processing_thread_func,
                              args=(tracker, frame_queue, TARGET_SHAPE, TARGET_COLOR, roi_state, is_detecting),
                              daemon=True)
    ctrl_t = threading.Thread(target=control_thread_func, args=(manager, roi_state, is_detecting), daemon=True)

    cap_t.start(); proc_t.start(); ctrl_t.start()

    print("\n--- Real-time Scanner + PID Track (+3°) (Auto-Reconnect, Full Display) ---")
    print("s: toggle detection, r: force reconnect, q: quit")

    display_frame = None
    try:
        while not stop_event.is_set():
            try:
                display_frame = frame_queue.get(timeout=1.0)
            except queue.Empty:
                if display_frame is None:
                    print("Waiting for first frame...")
                time.sleep(0.2)
                continue

            # วาด ROI และเส้นแบ่งซ้าย/กลาง/ขวา (ยึดสไตล์เดิม)
            ROI_X, ROI_Y, ROI_W, ROI_H = roi_state["x"], roi_state["y"], roi_state["w"], roi_state["h"]
            cv2.rectangle(display_frame, (ROI_X, ROI_Y), (ROI_X+ROI_W, ROI_Y+ROI_H), (255,0,0), 2)

            if is_detecting():
                cv2.putText(display_frame, "MODE: DETECTING", (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                with output_lock:
                    details = processed_output["details"]

                d1_abs = ROI_X + int(ROI_W*0.33)
                d2_abs = ROI_X + int(ROI_W*0.66)
                cv2.line(display_frame, (d1_abs, ROI_Y), (d1_abs, ROI_Y+ROI_H), (255,255,0), 1)
                cv2.line(display_frame, (d2_abs, ROI_Y), (d2_abs, ROI_Y+ROI_H), (255,255,0), 1)

                # กล่องวัตถุ: สี/ความหนาแบบเดิม (แดง=target, เหลือง=Uncertain, เขียว=อื่น)
                for det in details:
                    x,y,w,h = det['box']
                    abs_x, abs_y = x + ROI_X, y + ROI_Y
                    if det['is_target']:
                        box_color = (0,0,255)
                    elif det['shape'] == 'Uncertain':
                        box_color = (0,255,255)
                    else:
                        box_color = (0,255,0)
                    thickness = 4 if det['is_target'] else 2
                    cv2.rectangle(display_frame, (abs_x,abs_y), (abs_x+w, abs_y+h), box_color, thickness)
                    cv2.putText(display_frame, str(det['id']), (abs_x+5, abs_y+25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

                # รายการซ้ายบน
                if details:
                    y_pos = 70
                    for obj in details:
                        target_str = " (TARGET!)" if obj['is_target'] else ""
                        line = f"ID {obj['id']}: {obj['color']} {obj['shape']}{target_str}"
                        # shadow
                        cv2.putText(display_frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 4)
                        cv2.putText(display_frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                        y_pos += 25

                # วาด crosshair กลางภาพ และเส้น bias +3° (แนวนอน)
                cy_bias = int(FRAME_H/2 - PITCH_BIAS_PIX)
                cv2.line(display_frame, (FRAME_W//2 - 20, FRAME_H//2), (FRAME_W//2 + 20, FRAME_H//2), (255,255,255), 1)
                cv2.line(display_frame, (FRAME_W//2, FRAME_H//2 - 20), (FRAME_W//2, FRAME_H//2 + 20), (255,255,255), 1)
                cv2.line(display_frame, (0, cy_bias), (FRAME_W, cy_bias), (0, 128, 255), 1)  # เส้นเป้า +3°

            else:
                cv2.putText(display_frame, "MODE: VIEWING", (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

            # สถานะ SDK
            st = "CONNECTED" if manager.connected.is_set() else "RECONNECTING..."
            cv2.putText(display_frame, f"SDK: {st}", (20, 70 if not is_detecting() else 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            cv2.imshow("Robomaster Real-time Scan + PID Track (+3°)", display_frame)
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

    except Exception as e:
        print(f"❌ Main loop error: {e}")
    finally:
        print("\n🔌 Shutting down...")
        stop_event.set()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        manager.close()
        print("✅ Cleanup complete")
