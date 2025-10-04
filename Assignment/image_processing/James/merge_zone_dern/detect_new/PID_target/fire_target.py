"""
merge_with_PID_fire.py

Patch/controller that reads detection results (processed_output)
and runs a PID-based gimbal tracker + firing controller.

Assumptions:
- The detection script (detect_GPT_BEST or similar) produces a global:
    processed_output = {"details": [ { "id", "color", "shape", "zone", "is_target", "box" }, ... ]}
  and an output_lock (threading.Lock) to read it safely.
- The RoboMaster SDK robot is available (robot.Robot()).

If you embed this into the same file as the detection script, do NOT change the detection logic.
If you run as a separate module, you must enable sharing of processed_output (e.g., by importing the module
that defines it). This example expects SAME-PROCESS usage (put after the detect code).
"""

import time
import threading
import numpy as np
from robomaster import robot

# ---------- CONFIG ----------
# PID gains (tweak on real robot)
YAW_P = 0.02
YAW_I = 0.0005
YAW_D = 0.004

PITCH_P = 0.018
PITCH_I = 0.0004
PITCH_D = 0.0035

# how many consecutive stable frames before firing
FIRE_STABLE_FRAMES = 5

# acceptance: error in pixels (or convert to degrees) - we'll use pixel-based tolerance then map to small speed = 0
PIXEL_TOLERANCE = 10.0  # if center error < this, treat as zero

# additional pitch offset in degrees to aim slightly higher (positive = up, adjust if your gimbal uses opposite sign)
PITCH_AIM_OFFSET_DEG = 3.0

# max angular speeds to cap (SDK expects speed units; tune per your SDK)
MAX_YAW_SPEED = 1200.0
MAX_PITCH_SPEED = 1200.0

# map pixel error -> yaw/pitch speed scale (tweak)
SCALE_YAW = 1.0
SCALE_PITCH = 1.0

# Whether to actually call the fire() API (set False for dry-run)
ARMED_TO_FIRE = True

# ---------- END CONFIG ----------

class PIDController:
    def __init__(self, P, I, D, integral_limit=10000.0):
        self.P = P; self.I = I; self.D = D
        self.prev_err = 0.0
        self.integral = 0.0
        self.prev_t = None
        self.integral_limit = integral_limit

    def reset(self):
        self.prev_err = 0.0
        self.integral = 0.0
        self.prev_t = None

    def step(self, err):
        t = time.time()
        if self.prev_t is None:
            dt = 1.0/30.0
        else:
            dt = max(1e-3, t - self.prev_t)
        de = (err - self.prev_err) / dt if dt>0 else 0.0

        self.integral += err * dt
        # anti-windup
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))

        out = self.P * err + self.I * self.integral + self.D * de

        self.prev_err = err
        self.prev_t = t
        return out

# ---------- Controller thread ----------
def gimbal_pid_fire_loop(ep_robot, get_frame_dims_func, read_processed_output_func, read_output_lock_func):
    """
    ep_robot: initialized robot.Robot() instance
    get_frame_dims_func: () -> (frame_w, frame_h) (image center used)
    read_processed_output_func: () -> processed_output dict (thread-safe)
    read_output_lock_func: () -> lock (if needed)
    """

    ep_gimbal = ep_robot.gimbal
    ep_chassis = getattr(ep_robot, "chassis", None)
    ep_sentry = getattr(ep_robot, "sentry", None)  # if any fire API from other submodule

    yaw_pid = PIDController(YAW_P, YAW_I, YAW_D, integral_limit=20000.0)
    pitch_pid = PIDController(PITCH_P, PITCH_I, PITCH_D, integral_limit=20000.0)

    stable_count = 0
    last_target_id = None
    last_fire_time = 0.0
    FIRE_COOLDOWN = 1.0  # seconds between fire actions (tweak for magazine / rearm)

    print("[PID-FIRE] Controller started")
    try:
        while True:
            # get frame center
            frame_w, frame_h = get_frame_dims_func()
            cx_img = frame_w / 2.0
            cy_img = frame_h / 2.0

            # read processed_output safely
            # EXPECTED format: {"details":[{id,color,shape,zone,is_target,box},...]}
            processed = read_processed_output_func()

            target = None
            if processed and "details" in processed:
                # choose first target or highest-priority (here: first is fine)
                # better: choose the target with largest area or closest to center
                best_score = None
                for d in processed["details"]:
                    if not d.get("is_target", False):
                        continue
                    x,y,w,h = d.get("box", (0,0,0,0))
                    center_x = x + w/2.0
                    center_y = y + h/2.0
                    # score = area - distance penalty
                    area = w*h
                    dist = abs(center_x - cx_img) + abs(center_y - cy_img)
                    score = area - 0.5*dist
                    if best_score is None or score > best_score[0]:
                        best_score = (score, d, center_x, center_y, w, h)
                if best_score is not None:
                    _, d, center_x, center_y, w, h = best_score
                    target = {"meta": d, "cx": center_x, "cy": center_y, "w": w, "h": h}

            if target is None:
                # no target found: drive speeds to zero and reset PID integral slowly
                ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                yaw_pid.reset(); pitch_pid.reset()
                stable_count = 0
                time.sleep(0.03)
                continue

            # compute pixel error (image center - object center)
            err_x = cx_img - target["cx"]   # positive -> need yaw positive (depends SDK sign)
            err_y = cy_img - target["cy"]   # positive -> need pitch positive (depends SDK sign)

            # Normalize small errors to zero (deadband)
            if abs(err_x) < PIXEL_TOLERANCE: err_x = 0.0
            if abs(err_y) < PIXEL_TOLERANCE: err_y = 0.0

            # PID -> speed commands (these numbers are in SDK speed units; tune as needed)
            yaw_speed = yaw_pid.step(err_x) * SCALE_YAW
            pitch_speed = pitch_pid.step(err_y) * SCALE_PITCH

            # clip speeds
            yaw_speed = max(-MAX_YAW_SPEED, min(MAX_YAW_SPEED, yaw_speed))
            pitch_speed = max(-MAX_PITCH_SPEED, min(MAX_PITCH_SPEED, pitch_speed))

            # Send speed to gimbal. NOTE: SDK uses pitch_speed negative to move up/down in some versions.
            # You may need to invert sign of pitch_speed if your gimbal moves opposite.
            ep_gimbal.drive_speed(pitch_speed=-pitch_speed, yaw_speed=yaw_speed)

            # Check stability: if both errors small, increment stable counter
            if abs(err_x) <= PIXEL_TOLERANCE and abs(err_y) <= PIXEL_TOLERANCE:
                stable_count += 1
            else:
                stable_count = 0

            # Convert pitch offset deg -> small feedforward by adjusting target pitch using gimbal APIs
            # We will apply pitch offset when firing: do a short move to add PITCH_AIM_OFFSET_DEG
            # Determine fire condition
            now = time.time()
            if stable_count >= FIRE_STABLE_FRAMES and (now - last_fire_time) > FIRE_COOLDOWN:
                # Apply pitch offset before firing: move gimbal relative by small angle then fire
                try:
                    # read current pitch/yaw (if SDK supports reading absolute angles)
                    # If SDK doesn't provide get_angle, we just send a short move (non-blocking) and then fire.
                    # Use move: ep_gimbal.rotate(pitch=PITCH_AIM_OFFSET_DEG, yaw=0).wait_for_completed()  <-- example
                    # Many SDKs support re-center or set_angle. Adjust below if needed.
                    print(f"[PID-FIRE] Target stable -> firing (id={target['meta']['id']})")
                    if ARMED_TO_FIRE:
                        # small upward nudge then fire
                        # some RoboMaster SDKs: ep_gimbal.move(x=..., y=..., z=..., timeout=...) — replace with API you have
                        # We'll try to use ep_gimbal.move_* or ep_gimbal.recenter + pitch offset depending on availability.
                        try:
                            # preferred: relative pitch rotation API
                            ep_gimbal.move(pitch=PITCH_AIM_OFFSET_DEG).wait_for_completed()
                        except Exception:
                            # fallback: try set gimbal to absolute angle + offset (if API exists)
                            try:
                                cur = ep_gimbal.get_pitch()  # placeholder, depends on SDK
                                ep_gimbal.set_pitch(cur + PITCH_AIM_OFFSET_DEG).wait_for_completed()
                            except Exception:
                                # last resort: short sleep to let PID hold then call fire
                                time.sleep(0.06)

                        # fire api (this is just an illustrative name; replace with your SDK's fire call)
                        try:
                            ep_robot.shooting.fire(mode="single")  # replace with actual API (or ep_robot.fire())
                        except Exception:
                            # some SDKs: ep_robot.fire() or ep_robot.gimbal.fire() — adjust to your SDK
                            try:
                                ep_robot.fire()   # fallback
                            except Exception:
                                print("[PID-FIRE] WARNING: couldn't call fire() - no known API on ep_robot")

                        # optional: return gimbal back by moving -PITCH_AIM_OFFSET_DEG
                        try:
                            ep_gimbal.move(pitch=-PITCH_AIM_OFFSET_DEG).wait_for_completed()
                        except Exception:
                            pass
                        last_fire_time = time.time()
                    else:
                        print("[PID-FIRE] DRY RUN: would fire now (not armed).")
                        last_fire_time = time.time()
                except Exception as e:
                    print(f"[PID-FIRE] error during firing: {e}")

            # loop rate
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("[PID-FIRE] terminated by user")
    except Exception as e:
        print(f"[PID-FIRE] Exception: {e}")
    finally:
        try:
            ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
        except Exception:
            pass
        print("[PID-FIRE] stopped")

# ---------- Helpers to integrate with your detection script ----------
# The following two helper functions assume you run this file appended to your detect script
# where `processed_output` and `output_lock` are defined as in your detect program.

def _get_frame_dims_from_existing():
    # If your detect script knows IMAGE width/height, return them here.
    # Fallback: use a conservative default - change to actual values.
    try:
        # if your detect file stores ROI dims as ROI_W/ROI_H or global frame size, read them
        return (1280, 720)
    except Exception:
        return (1280, 720)

def _read_processed_output_safe():
    # expects `processed_output` and `output_lock` to exist in the same module (global)
    global processed_output, output_lock
    try:
        with output_lock:
            # shallow copy to avoid race conditions
            return dict(processed_output)
    except Exception:
        return {"details": []}

# ---------- Example bootstrap ----------
def start_pid_fire_controller(ep_robot):
    t = threading.Thread(
        target=gimbal_pid_fire_loop,
        args=(ep_robot, _get_frame_dims_from_existing, _read_processed_output_safe, lambda: output_lock),
        daemon=True
    )
    t.start()
    return t

# ---------- USAGE (example when appended to detect script) ----------
if __name__ == "__main__":
    # This block used when you append this file to your detect script AFTER the detect thread is started.
    # Initialize Robot if not already initialized by detect script.
    try:
        ep_robot = robot.Robot()
        ep_robot.initialize(conn_type="ap")
    except Exception as e:
        print("Could not init robot in PID wrapper:", e)
        ep_robot = None

    if ep_robot:
        print("Starting PID+Fire controller (arm state: {})".format("ARMED" if ARMED_TO_FIRE else "DRYRUN"))
        pid_thread = start_pid_fire_controller(ep_robot)
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("Exiting PID wrapper")
        finally:
            try:
                ep_robot.close()
            except:
                pass
# s1_detect_track_fire.py
# Single-file: Detection + PID Gimbal Tracking (aim +3 deg) + Fire on target only
# Keys:  s = toggle detect, r = force reconnect, f = arm/disarm firing, q = quit
# Safety: ทดลองแบบ ARMED=False ก่อนเสมอ

import cv2, numpy as np, time, math, threading, queue
from threading import Thread
from robomaster import robot, camera

# =========================
# Global shared state
# =========================
frame_queue = queue.Queue(maxsize=3)
output_lock = threading.Lock()
processed_output = {"details": []}  # [{"id", "color", "shape", "zone", "is_target", "box"}]
stop_event = threading.Event()

# =========================
# Connection Manager (Auto-reconnect)
# =========================
class RMConnection:
    def __init__(self):
        self._robot = None
        self.connected = threading.Event()
        self._lock = threading.Lock()

    def open(self):
        with self._lock:
            if self._robot:
                return self._robot
            r = robot.Robot()
            r.initialize(conn_type="ap")
            r.camera.start_video_stream(display=False, resolution=camera.STREAM_720P)
            r.gimbal.recenter(pitch_speed=200, yaw_speed=200).wait_for_completed()
            self._robot = r
            self.connected.set()
            print("[RM] Connected.")
            return self._robot

    def get_robot(self):
        with self._lock:
            return self._robot

    def close(self):
        with self._lock:
            if self._robot:
                try:
                    self._robot.camera.stop_video_stream()
                except Exception:
                    pass
                try:
                    self._robot.close()
                except Exception:
                    pass
                self._robot = None
                self.connected.clear()
                print("[RM] Closed.")

    def drop_and_reconnect(self):
        print("[RM] Reconnecting...")
        self.close()
        for _ in range(3):
            try:
                self.open()
                return True
            except Exception as e:
                print("[RM] reconnect failed:", e)
                time.sleep(1.0)
        return False

def reconnector_thread(manager: RMConnection):
    while not stop_event.is_set():
        r = manager.get_robot()
        if r is None:
            try:
                manager.open()
            except Exception as e:
                print("[RM] initial connect error:", e)
        time.sleep(1.5)

# =========================
# Detection (HSV + morphology + contour filters)
# =========================
class ObjectTracker:
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu  # placeholder (masking on CPU)
    def _night_enhance(self, bgr):
        # ใช้ Gaussian blur + ปรับ HSV เล็กน้อยสำหรับไฟน้อย
        return bgr

    def get_raw_detections(self, frame):
        # -- โครงเหมือนเดิม: GaussianBlur -> HSV -> mask 4 สี -> morphology -> contour filter
        enhanced = cv2.GaussianBlur(self._night_enhance(frame), (5,5), 0)
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
            if ar > 4.0 or ar < 0.25: continue
            hull = cv2.convexHull(cnt); ha = cv2.contourArea(hull)
            if ha == 0: continue
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

            # shape heuristics (circle-ish vs other)
            shape = "Circle"
            out.append({"color": found, "shape": shape, "box": (x,y,w,h)})
        return out

# =========================
# Threads: Capture & Processing
# =========================
def capture_thread_func(manager: RMConnection, frame_q: queue.Queue):
    print("[CAP] started")
    last_ok = 0
    while not stop_event.is_set():
        r = manager.get_robot()
        if r is None:
            time.sleep(0.1); continue
        try:
            frame = r.camera.read_cv2_image(timeout=0.5)
            if frame is not None:
                if frame_q.full():
                    try: frame_q.get_nowait()
                    except queue.Empty: pass
                frame_q.put_nowait(frame)
                last_ok = time.time()
            else:
                # no frame
                if time.time() - last_ok > 2.0:
                    manager.drop_and_reconnect()
        except Exception:
            time.sleep(0.1)
    print("[CAP] stopped")

def processing_thread_func(tracker: ObjectTracker, frame_q: queue.Queue,
                           target_shape, target_color, roi_rect, is_detecting_fn):
    print("[PROC] started")
    ROI_X, ROI_Y, ROI_W, ROI_H = roi_rect
    while not stop_event.is_set():
        try:
            frame = frame_q.get(timeout=0.5)
        except queue.Empty:
            continue

        if not is_detecting_fn():
            with output_lock:
                processed_output["details"] = []
            continue

        # Crop ROI
        H, W = frame.shape[:2]
        rx1, ry1 = max(0,ROI_X), max(0,ROI_Y)
        rx2, ry2 = min(W, ROI_X+ROI_W), min(H, ROI_Y+ROI_H)
        roi = frame[ry1:ry2, rx1:rx2].copy()

        try:
            detections = tracker.get_raw_detections(roi)
        except Exception as e:
            print("[PROC] detection error:", e)
            detections = []

        detailed_results = []
        divider1 = int(ROI_W*0.33)
        divider2 = int(ROI_W*0.66)
        oid = 1
        for d in detections:
            shape, color, (x,y,w,h) = d['shape'], d['color'], d['box']
            endx = x + w
            zone = "Center"
            if endx < divider1: zone = "Left"
            elif x >= divider2: zone = "Right"
            is_target = (shape == target_shape and color == target_color)
            detailed_results.append({
                "id": oid, "color": color, "shape": shape, "zone": zone,
                "is_target": is_target, "box": (x,y,w,h)
            })
            oid += 1

        with output_lock:
            processed_output["details"] = detailed_results
    print("[PROC] stopped")

# =========================
# PID + FIRE Controller
# =========================
class PID:
    def __init__(self, P,I,D, integral_limit=20000):
        self.P,self.I,self.D = P,I,D
        self.i=0.0; self.prev_e=0.0; self.prev_t=None
        self.limit = integral_limit
    def step(self, e):
        t = time.time()
        dt = 1/30 if self.prev_t is None else max(1e-3, t-self.prev_t)
        de = (e - self.prev_e)/dt
        self.i += e*dt
        self.i = max(-self.limit, min(self.limit, self.i))
        out = self.P*e + self.I*self.i + self.D*de
        self.prev_e, self.prev_t = e, t
        return out
    def reset(self):
        self.i=0.0; self.prev_e=0.0; self.prev_t=None

def fire_once(ep_robot):
    # รองรับหลาย SDK ชื่อฟังก์ชันต่างกัน
    try:
        ep_robot.gun.fire()               # บางเฟิร์มแวร์: gun
        return True
    except Exception:
        pass
    try:
        ep_robot.blaster.fire()           # อีกชื่อที่พบได้
        return True
    except Exception:
        pass
    try:
        ep_robot.shooting.fire(mode="single")  # บาง lib
        return True
    except Exception:
        pass
    print("[FIRE] No known fire() API on this SDK build.")
    return False

def gimbal_pid_fire_loop(manager: RMConnection, get_frame_dims_fn, is_detecting_fn):
    # ---- Config ----
    # PID gain ตั้งอ่อนๆ ก่อน
    YP, YI, YD = 0.020, 0.0005, 0.004   # yaw
    PP, PI, PD = 0.018, 0.0004, 0.0035  # pitch
    PIX_TOL = 10.0
    SCALE_YAW = 1.0
    SCALE_PITCH = 1.0
    MAX_SPEED = 1200.0
    AIM_OFFSET_DEG = 3.0     # เล็งเผื่อ “ขึ้น” 3°
    STABLE_N = 5
    FIRE_COOLDOWN = 1.0
    # toggle ยิง
    armed = {"v": False}

    yaw = PID(YP,YI,YD)
    pit = PID(PP,PI,PD)
    stable = 0
    last_fire = 0

    print("[CTRL] started. Press 'f' in UI window to arm/disarm firing.")

    def set_gimbal_speed(ep, ps, ys):
        try:
            # หมายเหตุ: หลาย SDK pitch sign กลับทิศ—เริ่มด้วยสลับ pitch
            ep.gimbal.drive_speed(pitch_speed=-ps, yaw_speed=ys)
        except Exception as e:
            pass

    while not stop_event.is_set():
        r = manager.get_robot()
        if r is None:
            time.sleep(0.1); continue

        W,H = get_frame_dims_fn()
        cx, cy = W/2, H/2

        # เลือก target: is_target == True ตัวที่ใกล้ center/พื้นที่ใหญ่สุด
        with output_lock:
            details = list(processed_output.get("details", []))

        best = None
        if details:
            best_score = None
            for d in details:
                if not d.get("is_target", False): continue
                x,y,w,h = d["box"]
                ccx, ccy = x+w/2, y+h/2
                area = w*h
                dist = abs(ccx - cx) + abs(ccy - cy)
                score = area - 0.5*dist
                if best_score is None or score > best_score:
                    best_score = score
                    best = (d, ccx, ccy, w, h)

        if best is None or not is_detecting_fn():
            yaw.reset(); pit.reset(); stable = 0
            set_gimbal_speed(r, 0, 0)
            time.sleep(0.02)
            continue

        _, tx, ty, tw, th = best
        ex, ey = cx - tx, cy - ty
        if abs(ex) < PIX_TOL: ex = 0.0
        if abs(ey) < PIX_TOL: ey = 0.0

        yaw_spd = max(-MAX_SPEED, min(MAX_SPEED, yaw.step(ex) * SCALE_YAW))
        pit_spd = max(-MAX_SPEED, min(MAX_SPEED, pit.step(ey) * SCALE_PITCH))
        set_gimbal_speed(r, pit_spd, yaw_spd)

        if ex == 0 and ey == 0:
            stable += 1
        else:
            stable = 0

        # ยิงเมื่อ: ตรงนิ่ง N เฟรม, อนุญาตยิง, คูลดาวน์ผ่าน
        if stable >= STABLE_N and armed["v"] and (time.time()-last_fire) > FIRE_COOLDOWN:
            try:
                # เอียงขึ้น 3° ชั่วครู่แล้วยิง
                try:
                    r.gimbal.move(pitch=AIM_OFFSET_DEG).wait_for_completed()
                except Exception:
                    # fallback: ไม่มีก็ปล่อยให้ PID ค้าง แล้วชดเชยด้วย sleep
                    time.sleep(0.05)
                ok = fire_once(r)
                try:
                    r.gimbal.move(pitch=-AIM_OFFSET_DEG).wait_for_completed()
                except Exception:
                    pass
                if ok: print("[FIRE] bang!")
            except Exception as e:
                print("[FIRE] error:", e)
            last_fire = time.time()
            stable = 0

        # อ่านคีย์ toggle arm ผ่าน shared flag ที่ main จะเขียน
        # (ตัวแปร armed จะถูกปรับใน main loop ผ่าน closure หรือ event – เราจะปล่อยให้ main แก้ armed["v"])
        time.sleep(0.02)

    print("[CTRL] stopped")

# =========================
# Main (UI + hotkeys)
# =========================
def main():
    # --- Target definition ---
    target_shape, target_color = "Circle", "Red"
    print(f"[MAIN] Target: {target_color} {target_shape}")

    tracker = ObjectTracker(use_gpu=False)

    # ROI เช่นในเดิม: X,Y,W,H
    ROI_Y, ROI_H, ROI_X, ROI_W = 264, 270, 10, 911  # ปรับตามเฟรมจริง
    manager = RMConnection()
    reconn = threading.Thread(target=reconnector_thread, args=(manager,), daemon=True); reconn.start()

    is_detecting_flag = {"v": False}
    firing_armed_flag = {"v": False}  # toggle ด้วยคีย์ 'f'

    def is_detecting(): return is_detecting_flag["v"]

    cap_t = threading.Thread(target=capture_thread_func, args=(manager, frame_queue), daemon=True); cap_t.start()
    proc_t = threading.Thread(
        target=processing_thread_func,
        args=(tracker, frame_queue, target_shape, target_color, (ROI_X, ROI_Y, ROI_W, ROI_H), is_detecting),
        daemon=True
    ); proc_t.start()

    # เริ่ม controller
    def get_dims():
        # อิงจาก stream 1280x720 ของ S1 โดยปกติ (ปรับได้หาก stream อื่น)
        return (ROI_W, ROI_H)
    ctrl_t = threading.Thread(
        target=gimbal_pid_fire_loop, args=(manager, get_dims, is_detecting), daemon=True
    ); ctrl_t.start()

    # ---- UI Loop ----
    print("\n--- Robomaster Scan + Track + Fire ---")
    print("s: toggle detection, r: force reconnect, f: arm/disarm firing, q: quit")

    display_frame = None
    try:
        while not stop_event.is_set():
            try:
                base = frame_queue.get(timeout=1.0)
            except queue.Empty:
                if display_frame is None:
                    print("Waiting for first frame.")
                time.sleep(0.2)
                continue

            display_frame = base.copy()

            # วาด ROI
            cv2.rectangle(display_frame, (ROI_X, ROI_Y), (ROI_X+ROI_W, ROI_Y+ROI_H), (255,0,0), 2)

            if is_detecting():
                cv2.putText(display_frame, "MODE: DETECTING", (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                with output_lock:
                    details = list(processed_output.get("details", []))

                d1_abs = ROI_X + int(ROI_W*0.33)
                d2_abs = ROI_X + int(ROI_W*0.66)
                cv2.line(display_frame, (d1_abs, ROI_Y), (d1_abs, ROI_Y+ROI_H), (255,255,0), 1)
                cv2.line(display_frame, (d2_abs, ROI_Y), (d2_abs, ROI_Y+ROI_H), (255,255,0), 1)

                # วาดกรอบวัตถุ (แดง=target, เหลือง=uncertain—ที่นี่ไม่มี uncertain flag จึงเป็นเขียว default)
                for det in details:
                    x,y,w,h = det['box']
                    abs_x, abs_y = x + ROI_X, y + ROI_Y
                    if det['is_target']:
                        box_color = (0,0,255)
                    else:
                        box_color = (0,255,0)
                    thickness = 4 if det['is_target'] else 2
                    cv2.rectangle(display_frame, (abs_x,abs_y), (abs_x+w, abs_y+h), box_color, thickness)
                    cv2.putText(display_frame, str(det['id']), (abs_x+5, abs_y+25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

                # รายการข้อความด้านซ้ายบน
                if details:
                    y_pos = 70
                    for obj in details:
                        target_str = " (TARGET!)" if obj['is_target'] else ""
                        line = f"ID {obj['id']}: {obj['color']} {obj['shape']}{target_str}"
                        cv2.putText(display_frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 4)
                        cv2.putText(display_frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                        y_pos += 25
            else:
                cv2.putText(display_frame, "MODE: VIEWING", (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

            # สถานะ SDK + Armed
            st = "CONNECTED" if manager.connected.is_set() else "RECONNECTING..."
            arm = "ARMED" if firing_armed_flag["v"] else "SAFE"
            arm_color = (0,0,255) if firing_armed_flag["v"] else (0,255,0)
            cv2.putText(display_frame, f"SDK: {st}", (20, 70 if not is_detecting() else 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(display_frame, f"FIRE: {arm}", (20, 95 if not is_detecting() else 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, arm_color, 2)

            cv2.imshow("Robomaster S1 - Detect/Track/Fire", display_frame)
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
            elif key == ord('f'):
                firing_armed_flag["v"] = not firing_armed_flag["v"]
                print(f"Firing {'ARMED' if firing_armed_flag['v'] else 'SAFE'}")

        # end while
    except Exception as e:
        print("Main loop error:", e)
    finally:
        print("\n[MAIN] Shutdown...")
        stop_event.set()
        try: cv2.destroyAllWindows()
        except Exception: pass
        manager.close()
        print("[MAIN] Cleanup complete.")

if __name__ == "__main__":
    main()
