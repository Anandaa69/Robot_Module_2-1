# -*- coding: utf-8 -*-
"""
detect_map_merged_s1_safe.py  (RoboMaster S1 safe version)

รวม:
- ระบบตรวจจับ (threads, ROI, detect) จาก fire_target.py
- ระบบเดินสำรวจ/OGM/วาดแผนที่ จาก test_cur_time_copy.py
- โหมดสแกน "s" เปิด detect 1 วิ/สเต็ป แล้วบันทึกวัตถุลง "บล็อคถัดไป" + JSON
- ปรับให้เข้ากับ S1: getter อุปกรณ์ทั้งหมดใช้ getattr ปลอดภัย (ไม่มีก็ได้ None)
  -> ถ้าไม่พบเซ็นเซอร์ระยะ (ToF/IR) จะถือว่า front = inf (ไม่บล็อค) ตาม fallback
"""

import cv2
import numpy as np
import time
import math
import json
import threading
import queue
from collections import deque
import os

from robomaster import robot, camera as r_camera, blaster as r_blaster

# =========================
# ====== fire_target ======
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

frame_queue = queue.Queue(maxsize=1)
processed_output = {"details": []}  # [{id,color,shape,zone,is_target,box}]
output_lock = threading.Lock()
stop_event = threading.Event()

gimbal_angle_lock = threading.Lock()
gimbal_angles = (0.0, 0.0, 0.0, 0.0)

def sub_angle_cb(angle_info):
    global gimbal_angles
    with gimbal_angle_lock:
        gimbal_angles = tuple(angle_info)

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

class RMConnection:
    """Safe getters for S1: ถ้าไม่มีอุปกรณ์ คืน None (ไม่ throw)"""
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

            # show optional modules (for debugging what S1 exposes)
            optional = ["tof", "sensor_adapter", "ir_distance_sensor", "vision", "armor", "led"]
            available = [name for name in optional if hasattr(rb, name)]
            if available:
                print("ℹ️ Optional modules detected:", ", ".join(available))
            else:
                print("ℹ️ No optional distance/adapter modules detected (fallbacks will be used).")

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

    # --- safe getters ---
    def get_camera(self):
        with self._lock:
            return getattr(self._robot, "camera", None) if self._robot else None
    def get_gimbal(self):
        with self._lock:
            return getattr(self._robot, "gimbal", None) if self._robot else None
    def get_blaster(self):
        with self._lock:
            return getattr(self._robot, "blaster", None) if self._robot else None
    def get_chassis(self):
        with self._lock:
            return getattr(self._robot, "chassis", None) if self._robot else None
    def get_tof(self):
        with self._lock:
            return getattr(self._robot, "tof", None) if self._robot else None
    def get_sensor_adapter(self):
        with self._lock:
            return getattr(self._robot, "sensor_adapter", None) if self._robot else None
    def get_ir_distance_sensor(self):
        with self._lock:
            return getattr(self._robot, "ir_distance_sensor", None) if self._robot else None

    def get_attr(self, name: str):
        with self._lock:
            return getattr(self._robot, name, None) if self._robot else None

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
            time.sleep(0.02)
            try:
                q.get_nowait()  # ทิ้งเฟรมเก่าเพื่อไม่ให้คิวตัน
            except queue.Empty:
                pass
            continue
        try:
            frame_to_process = q.get(timeout=1.0)
            with gimbal_angle_lock:
                pitch_deg = gimbal_angles[0]
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

# =========================
# ====== สำรวจ/OGM ========
# =========================
SPEED_ROTATE = 480
OCCUPANCY_THRESHOLD = 0.7
FREE_THRESHOLD = 0.3

CURRENT_POSITION = (3,2)  # (row,col)
CURRENT_DIRECTION = 0     # 0:N,1:E,2:S,3:W
CURRENT_TARGET_YAW = 0.0
ROBOT_FACE = 1

IMU_DRIFT_COMPENSATION_DEG = 0.0

def get_compensated_target_yaw():
    return CURRENT_TARGET_YAW + IMU_DRIFT_COMPENSATION_DEG

class WallBelief:
    def __init__(self): self.log_odds = 0.0
    def update(self, occ, _sensor): self.log_odds = max(min(self.log_odds + (1.5 if occ else -1.5), 10), -10)
    def get_probability(self): return 1.0 - 1.0 / (1.0 + math.exp(self.log_odds))
    def is_occupied(self): return self.get_probability() > OCCUPANCY_THRESHOLD

class OGMCell:
    def __init__(self):
        self.log_odds_occupied = 0.0
        self.walls = {'N': None, 'E': None, 'S': None, 'W': None}
        self.objects = {"Left": [], "Center": [], "Right": []}
    def get_node_probability(self): return 1.0 - 1.0 / (1.0 + math.exp(self.log_odds_occupied))
    def is_node_occupied(self): return self.get_node_probability() > OCCUPANCY_THRESHOLD

class OccupancyGridMap:
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.grid = [[OGMCell() for _ in range(width)] for _ in range(height)]
        self._link_walls()
    def _link_walls(self):
        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r][c].walls['N'] is None:
                    wall = WallBelief(); self.grid[r][c].walls['N'] = wall
                    if r>0: self.grid[r-1][c].walls['S'] = wall
                if self.grid[r][c].walls['W'] is None:
                    wall = WallBelief(); self.grid[r][c].walls['W'] = wall
                    if c>0: self.grid[r][c-1].walls['E'] = wall
                if self.grid[r][c].walls['S'] is None: self.grid[r][c].walls['S'] = WallBelief()
                if self.grid[r][c].walls['E'] is None: self.grid[r][c].walls['E'] = WallBelief()

import matplotlib.pyplot as plt
class RealTimeVisualizer:
    def __init__(self, grid_size):
        plt.ion()
        self.grid_size = grid_size
        self.fig, self.ax = plt.subplots(figsize=(8,7))
        self.colors = {"robot":"#0000FF", "wall":"#000000"}
    def update_plot(self, occupancy_map, robot_pos, path=None):
        self.ax.clear()
        self.ax.set_title("Real-time Hybrid Belief Map (Nodes & Walls + Objects)")
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.ax.set_xlim(-0.5, self.grid_size-0.5); self.ax.set_ylim(self.grid_size-0.5, -0.5)
        # nodes
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                prob = occupancy_map.grid[r][c].get_node_probability()
                color = '#8B0000' if prob>OCCUPANCY_THRESHOLD else ('#D3D3D3' if prob<FREE_THRESHOLD else '#90EE90')
                self.ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1,1, facecolor=color, edgecolor='k', lw=0.5))
        # walls
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell = occupancy_map.grid[r][c]
                if cell.walls['N'].is_occupied(): self.ax.plot([c-0.5, c+0.5],[r-0.5,r-0.5], color=self.colors['wall'], lw=4)
                if cell.walls['W'].is_occupied(): self.ax.plot([c-0.5, c-0.5],[r-0.5,r+0.5], color=self.colors['wall'], lw=4)
                if r==self.grid_size-1 and cell.walls['S'].is_occupied(): self.ax.plot([c-0.5, c+0.5],[r+0.5,r+0.5], color=self.colors['wall'], lw=4)
                if c==self.grid_size-1 and cell.walls['E'].is_occupied(): self.ax.plot([c+0.5, c+0.5],[r-0.5,r+0.5], color=self.colors['wall'], lw=4)
        # objects (Left/Center/Right)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                objs = occupancy_map.grid[r][c].objects
                base_x, base_y = c, r
                if objs["Left"]:
                    self.ax.scatter([base_x-0.25],[base_y], s=40, marker='s')
                if objs["Center"]:
                    self.ax.scatter([base_x],[base_y-0.25], s=40, marker='o')
                if objs["Right"]:
                    self.ax.scatter([base_x+0.25],[base_y], s=40, marker='^')
        # robot
        if robot_pos:
            rr, cc = robot_pos
            self.ax.add_patch(plt.Rectangle((cc-0.5, rr-0.5),1,1, facecolor=self.colors['robot'], edgecolor='k', lw=2, alpha=0.6))
        self.fig.tight_layout()
        self.fig.canvas.draw(); self.fig.canvas.flush_events(); plt.pause(0.01)

# ===== Movement/Attitude (ย่อ) =====
class AttitudeHandler:
    def __init__(self):
        self.current_yaw, self.yaw_tolerance, self.is_monitoring = 0.0, 3.0, False
    def attitude_handler(self, attitude_info):
        if self.is_monitoring: self.current_yaw = attitude_info[0]
    def start_monitoring(self, chassis):
        if chassis is None: return
        self.is_monitoring = True
        try:
            chassis.sub_attitude(freq=20, callback=self.attitude_handler)
        except Exception as e:
            print("sub_attitude error:", e)
    def stop_monitoring(self, chassis):
        self.is_monitoring = False
        if chassis is None: return
        try: chassis.unsub_attitude()
        except Exception: pass
    def normalize_angle(self, a):
        while a>180: a-=360
        while a<=-180: a+=360
        return a
    def correct_yaw_to_target(self, chassis, target_yaw=0.0):
        if chassis is None: return
        normalized = self.normalize_angle(target_yaw); time.sleep(0.1)
        robot_rot = -self.normalize_angle(normalized - self.current_yaw)
        if abs(robot_rot) > self.yaw_tolerance:
            try:
                chassis.move(x=0,y=0,z=robot_rot,z_speed=60).wait_for_completed(timeout=2)
            except Exception as e:
                print("chassis.move rotate error:", e)
            time.sleep(0.1)

class PID_1D:
    def __init__(self, Kp, Ki, Kd, setpoint=0):
        self.Kp, self.Ki, self.Kd, self.setpoint = Kp, Ki, Kd, setpoint
        self.prev_error, self.integral, self.integral_max = 0, 0, 1.0
    def compute(self, current, dt):
        error = self.setpoint - current
        self.integral += error*dt; self.integral = max(min(self.integral, self.integral_max), -self.integral_max)
        deriv = (error - self.prev_error)/dt if dt>0 else 0
        self.prev_error = error
        return self.Kp*error + self.Ki*self.integral + self.Kd*deriv

class MovementController:
    def __init__(self, chassis):
        self.chassis = chassis
        self.current_x_pos, self.current_y_pos = 0.0, 0.0
        if self.chassis is not None:
            try:
                self.chassis.sub_position(freq=20, callback=self.position_handler)
            except Exception as e:
                print("sub_position error:", e)
    def position_handler(self, pos_info):
        self.current_x_pos, self.current_y_pos = pos_info[0], pos_info[1]
    def _yaw_correction(self, att, target_yaw):
        KP_YAW, MAX_YAW_SPEED = 1.8, 25
        yaw_error = att.normalize_angle(target_yaw - att.current_yaw)
        spd = KP_YAW * yaw_error
        return max(min(spd, MAX_YAW_SPEED), -MAX_YAW_SPEED)
    def move_forward_one_grid(self, axis, att):
        if self.chassis is None: 
            print("⚠️ No chassis; skip move"); return
        att.correct_yaw_to_target(self.chassis, get_compensated_target_yaw())
        target_dist = 0.6
        pid = PID_1D(1.8, 0.25, 12, setpoint=target_dist)
        start, last = time.time(), time.time()
        start_pos = self.current_x_pos if axis=='x' else self.current_y_pos
        print(f"🚀 Move forward 0.6m on {axis}-axis")
        while time.time()-start < 3.5:
            now=time.time(); dt=now-last; last=now
            cur = self.current_x_pos if axis=='x' else self.current_y_pos
            rel = abs(cur-start_pos)
            if abs(rel-target_dist) < 0.03:
                print("✅ Move complete"); break
            out = pid.compute(rel, dt)
            ramp = min(1.0, 0.1 + ((now-start)/1.0)*0.9)
            speed = max(-1.0, min(1.0, out*ramp))
            yaw_corr = self._yaw_correction(att, get_compensated_target_yaw())
            try:
                self.chassis.drive_speed(x=speed,y=0,z=yaw_corr,timeout=1)
            except Exception as e:
                print("drive_speed error:", e)
                break
        try:
            self.chassis.drive_wheels(w1=0,w2=0,w3=0,w4=0)
        except Exception: pass
        time.sleep(0.2)

class EnvironmentScanner:
    """
    ปลอดภัยต่อ S1: ถ้าไม่มีเซ็นเซอร์ระยะ จะคืน inf (ถือว่าไม่บล็อค)
    """
    def __init__(self, tof_sensor, ir_sensor, gimbal):
        self.tof_sensor = tof_sensor
        self.ir_sensor = ir_sensor
        self.gimbal = gimbal
        self.last_front_cm = float('inf')
        self.is_gimbal_forward = True

        # พยายาม subscribe แบบปลอดภัย (ถ้า SDK รองรับ)
        try:
            if self.tof_sensor and hasattr(self.tof_sensor, "sub_distance"):
                self.tof_sensor.sub_distance(freq=10, callback=self._tof_cb)
                print("ℹ️ Subscribed ToF distance.")
        except Exception as e:
            print("ToF subscribe not available:", e)
        try:
            if self.ir_sensor and hasattr(self.ir_sensor, "sub_distance"):
                self.ir_sensor.sub_distance(freq=10, callback=self._ir_cb)
                print("ℹ️ Subscribed IR distance.")
        except Exception as e:
            pass  # ไม่เป็นไร

    def _tof_cb(self, info):
        # บาง SDK ให้ mm, บางตัว cm — เราจะแปลงให้เป็น cm ถ้าดูเหมือน mm
        v = float(info[0]) if isinstance(info, (list,tuple)) else float(info)
        cm = v/10.0 if v>100 else v
        if self.is_gimbal_forward:
            self.last_front_cm = cm

    def _ir_cb(self, info):
        v = float(info[0]) if isinstance(info, (list,tuple)) else float(info)
        cm = v/10.0 if v>100 else v
        if self.is_gimbal_forward:
            # ใช้ค่าที่ “สั้นกว่า” เพื่อ safety
            self.last_front_cm = min(self.last_front_cm, cm)

    def _instant_read(self):
        # พยายามอ่านแบบโพลล์ หากมีเมธอดให้เรียก
        for sensor in [self.tof_sensor, self.ir_sensor]:
            if sensor is None: continue
            for name in ["get_distance", "get", "distance"]:
                try:
                    fn = getattr(sensor, name, None)
                    if callable(fn):
                        val = fn()
                        v = float(val[0]) if isinstance(val, (list,tuple)) else float(val)
                        return v/10.0 if v>100 else v
                    elif isinstance(fn, (int,float)):
                        v = float(fn); return v/10.0 if v>100 else v
                except Exception:
                    continue
        return None

    def get_front_tof_cm(self):
        try:
            if self.gimbal:
                self.gimbal.moveto(pitch=0, yaw=0, yaw_speed=SPEED_ROTATE).wait_for_completed()
        except Exception: pass
        self.is_gimbal_forward = True
        time.sleep(0.05)
        inst = self._instant_read()
        if inst is not None:
            self.last_front_cm = inst
        return float(self.last_front_cm)

    def cleanup(self):
        try:
            if self.tof_sensor and hasattr(self.tof_sensor, "unsub_distance"):
                self.tof_sensor.unsub_distance()
        except Exception: pass
        try:
            if self.ir_sensor and hasattr(self.ir_sensor, "unsub_distance"):
                self.ir_sensor.unsub_distance()
        except Exception: pass

# =========================
# ===== JSON LOGGING  =====
# =========================
DATA_DIR = "./runtime_logs"
os.makedirs(DATA_DIR, exist_ok=True)
OBJ_JSON_PATH = os.path.join(DATA_DIR, "detected_objects_log.json")

def append_objects_json(step_idx, next_rc, zone_to_objs, front_blocked, raw_list):
    data = {
        "timestamp": time.time(),
        "step": step_idx,
        "next_cell": {"row": next_rc[0], "col": next_rc[1]},
        "front_blocked_le_60cm": bool(front_blocked),
        "zones": zone_to_objs,      # dict: {"Left":[...], "Center":[...], "Right":[...]}
        "raw_detections": raw_list  # list รายละเอียด det ทุกกล่อง (id,color,shape,zone,box)
    }
    try:
        if os.path.exists(OBJ_JSON_PATH):
            with open(OBJ_JSON_PATH, "r", encoding="utf-8") as f:
                arr = json.load(f)
        else:
            arr = []
        arr.append(data)
        with open(OBJ_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(arr, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("⚠️ JSON write error:", e)

# =========================
# ===== Main Program  =====
# =========================
if __name__ == "__main__":
    print("🎯 Detect+Map merged (S1-safe) start")

    tracker = ObjectTracker(use_gpu=False)
    roi_state = {"x": ROI_X0, "y": ROI_Y0, "w": ROI_W0, "h": ROI_H0}
    manager = RMConnection()
    reconn = threading.Thread(target=reconnector_thread, args=(manager,), daemon=True)
    reconn.start()

    is_detecting_flag = {"v": False}  # เริ่มยังไม่เข้าโหมด s
    def is_detecting(): return is_detecting_flag["v"]

    cap_t  = threading.Thread(target=capture_thread_func, args=(manager, frame_queue), daemon=True)
    proc_t = threading.Thread(target=processing_thread_func,
                              args=(tracker, frame_queue, TARGET_SHAPE, TARGET_COLOR, roi_state, is_detecting),
                              daemon=True)
    cap_t.start(); proc_t.start()

    # ---- เตรียมโมดูลฮาร์ดแวร์ (safe getters) ----
    time.sleep(0.5)
    gimbal = manager.get_gimbal()
    chassis = manager.get_chassis()
    tof = manager.get_tof()                             # อาจ None
    sensor_adaptor = manager.get_sensor_adapter()       # อาจ None
    ir_sensor = manager.get_ir_distance_sensor()        # อาจ None

    att = AttitudeHandler()
    if chassis is not None:
        att.start_monitoring(chassis)
    mover = MovementController(chassis) if chassis is not None else None
    scanner = EnvironmentScanner(tof_sensor=tof, ir_sensor=ir_sensor, gimbal=gimbal)

    ogm = OccupancyGridMap(width=8, height=8)

    import matplotlib.pyplot as plt
    vis = RealTimeVisualizer(grid_size=8)

    # ====== Exploration Loop ======
    step = 0
    try:
        while not stop_event.is_set() and step < 40:
            step += 1
            print(f"\n========== STEP {step} at {CURRENT_POSITION} facing {CURRENT_DIRECTION} ==========")

            # 1) ปรับกลางช่อง/มุมเบื้องต้น (ย่อ)
            if mover is not None and chassis is not None and gimbal is not None:
                att.correct_yaw_to_target(chassis, get_compensated_target_yaw())

            # 2) บล็อคถัดไป
            drdc_by_dir = {0:(-1,0), 1:(0,1), 2:(1,0), 3:(0,-1)}
            dr, dc = drdc_by_dir[CURRENT_DIRECTION]
            next_cell = (CURRENT_POSITION[0]+dr, CURRENT_POSITION[1]+dc)
            axis_to_monitor = 'x' if (ROBOT_FACE % 2 != 0) else 'y'

            # 3) เช็ค front ≤ 60 cm (fallback: ไม่มีเซนเซอร์ -> inf)
            front_cm = scanner.get_front_tof_cm() if scanner is not None else float('inf')
            print(f"[Distance] Front ≈ {front_cm:.1f} cm")
            front_blocked = (front_cm < 60.0)

            # 4) เปิดโหมด s (detect) 1 วินาที
            is_detecting_flag["v"] = True
            time.sleep(1.0)
            is_detecting_flag["v"] = False

            # 5) ดึงผล detect และกรองโซน
            with output_lock:
                dets = list(processed_output["details"])
            allowed_zones = {"Left","Right","Center"} if front_blocked else {"Left","Right"}
            dets_use = [d for d in dets if d["zone"] in allowed_zones]

            # 6) บันทึกลง OGM ที่ "บล็อคถัดไป"
            if 0 <= next_cell[0] < ogm.height and 0 <= next_cell[1] < ogm.width:
                cell_obj = ogm.grid[next_cell[0]][next_cell[1]].objects
                zone_bag = {"Left": [], "Center": [], "Right": []}
                for d in dets_use:
                    one = {"color": d["color"], "shape": d["shape"], "zone": d["zone"], "box": d["box"]}
                    cell_obj[d["zone"]].append(one)
                    zone_bag[d["zone"]].append(one)
                vis.update_plot(ogm, CURRENT_POSITION)
                append_objects_json(step, next_cell, zone_bag, front_blocked, dets)

            # 7) เดินไปบล็อคถัดไปถ้าไม่บล็อค
            if mover is not None and chassis is not None:
                if not front_blocked:
                    mover.move_forward_one_grid(axis_to_monitor, att)
                    CURRENT_POSITION = next_cell
                else:
                    print("⛔ Front blocked (<=60cm). Skip moving forward.")

            # 8) อัปเดตแผนที่หลังเคลื่อน
            vis.update_plot(ogm, CURRENT_POSITION)

            # 9) รักษายอว์เบาๆ
            att.correct_yaw_to_target(chassis, get_compensated_target_yaw())
            time.sleep(0.1)

        print("\n✅ Exploration loop finished.")
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        stop_event.set()
        try: att.stop_monitoring(chassis)
        except Exception: pass
        try: scanner.cleanup()
        except Exception: pass
        try: cv2.destroyAllWindows()
        except Exception: pass
        manager.close()
        print("🔚 Shutdown complete.")
