# -*- coding: utf-8 -*-
"""
detect_map_merged.py

รวม:
- ระบบตรวจจับวัตถุ + เธรดกล้อง/ประมวลผลจาก fire_target.py (คง logic เดิม)
- ระบบเดินสำรวจ/OGM แมพ/ToF/ชดเชย yaw จาก test_cur_time_copy.py (คง logic เดิมที่จำเป็น)
- เพิ่ม flow: เปิดโหมด s (detect) 1 วินาที ณ แต่ละสเต็ป, กรอง zone ตาม ToF หน้า 60ซม.,
  เซฟผลตรวจลง "บล็อคถัดไป" ทั้ง overlay บนแผนที่ และบันทึก JSON

ข้อสังเกต:
- ผมเพิ่ม get_chassis()/get_tof()/get_sensor_adapter() ให้กับ RMConnection เพื่อเรียกฮาร์ดแวร์จากที่เดียว
- โครงเธรด detect ของ fire_target ถูกยกทั้งชุด (capture+processing) และสามารถเปิด/ปิดด้วย flag is_detecting_flag (โหมด s)
- การวาดแผนที่ต่อยอดจาก RealTimeVisualizer: เพิ่มการแปะ "สัญลักษณ์ออปเจกต์" ในช่องบล็อคถัดไป แยกซ้าย/หน้า/ขวา
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
import statistics

from robomaster import robot, camera as r_camera, blaster as r_blaster

# =========================
# ====== fire_target ======
# (นำแกนหลักมาจากไฟล์เดิม: tracker, capture/processing threads, PID constants, ROI dynamics)
# =========================
# --- CONFIG detect ---
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

# NOTE: ใน flow นี้เรา "ไม่ใช้ยิง" ระหว่างสำรวจ จึงไม่ใช้ pitch bias/ควบคุมยิง
# (เก็บไว้หากภายหลังต้องการ re-use)
PITCH_BIAS_DEG = 2.0
PITCH_BIAS_PIX = +PITCH_BIAS_DEG * PIXELS_PER_DEG_V

ROI_Y0, ROI_H0, ROI_X0, ROI_W0 = 264, 270, 10, 911
ROI_SHIFT_PER_DEG = 6.0
ROI_Y_MIN, ROI_Y_MAX = 0, FRAME_H - 10

# --- Shared/thread states (from fire_target) ---
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

# --- White balance (เดิม) ---
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
    """ยกจาก fire_target + เพิ่ม access อุปกรณ์ที่ฝั่งสำรวจต้องใช้"""
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
    # ---- เพิ่มสำหรับฝั่งสำรวจ ----
    def get_chassis(self):
        with self._lock:
            return None if self._robot is None else self._robot.chassis
    # แทนที่เมธอดเดิม
    def get_tof(self):
        with self._lock:
            # ถ้าไม่มี .tof ใน Robot ให้ได้ None แทนการ throw AttributeError
            return getattr(self._robot, "tof", None)
    def get_sensor_adapter(self):
        with self._lock:
            # สำหรับ Sharp/IR (ADC/IO)
            return None if self._robot is None else self._robot.sensor_adapter

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
    """ยกตรรกะจาก fire_target: อ่านเฟรม -> crop ROI ตาม pitch -> detect -> อัปเดต processed_output"""
    global processed_output
    print("🧠 Processing thread started.")
    while not stop_event.is_set():
        if not is_detecting_func():
            time.sleep(0.02);  # โหมด s off: ไม่ประมวลผล
            # แต่เรายังเปิดกล้องตลอด (เฟรมในคิวไหลอยู่)
            try:
                q.get_nowait()
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
# (อ้างอิงโครงจาก fire_target.py)  # :contentReference[oaicite:2]{index=2}

# =========================
# ====== สำรวจ/OGM ========
# (ยกเฉพาะส่วนที่จำเป็นจาก test_cur_time_copy สำหรับเดินทีละบล็อค, จัดศูนย์, วัดผนัง, สร้างแผนที่)
# =========================
SPEED_ROTATE = 480
SHARP_WALL_THRESHOLD_CM = 60.0
SHARP_STDEV_THRESHOLD = 0.3
TOF_ADJUST_SPEED = 0.1
TOF_CALIBRATION_SLOPE = 0.0894
TOF_CALIBRATION_Y_INTERCEPT = 3.8409

CURRENT_POSITION = (3,2)  # (row,col)
CURRENT_DIRECTION = 0     # 0:N,1:E,2:S,3:W
CURRENT_TARGET_YAW = 0.0
ROBOT_FACE = 1

IMU_DRIFT_COMPENSATION_DEG = 0.0

PROB_OCC_GIVEN_OCC = {'tof': 0.95, 'sharp': 0.90}
PROB_OCC_GIVEN_FREE = {'tof': 0.05, 'sharp': 0.10}
LOG_ODDS_OCC = {
    'tof': math.log(PROB_OCC_GIVEN_OCC['tof'] / (1 - PROB_OCC_GIVEN_OCC['tof'])),
    'sharp': math.log(PROB_OCC_GIVEN_OCC['sharp'] / (1 - PROB_OCC_GIVEN_OCC['sharp']))
}
LOG_ODDS_FREE = {
    'tof': math.log(PROB_OCC_GIVEN_FREE['tof'] / (1 - PROB_OCC_GIVEN_FREE['tof'])),
    'sharp': math.log(PROB_OCC_GIVEN_FREE['sharp'] / (1 - PROB_OCC_GIVEN_FREE['sharp']))
}
OCCUPANCY_THRESHOLD = 0.7
FREE_THRESHOLD = 0.3

def calibrate_tof_value(raw_mm):
    if raw_mm is None or raw_mm <= 0:
        return float('inf')
    return (TOF_CALIBRATION_SLOPE * raw_mm) + TOF_CALIBRATION_Y_INTERCEPT

def get_compensated_target_yaw():
    return CURRENT_TARGET_YAW + IMU_DRIFT_COMPENSATION_DEG

class WallBelief:
    def __init__(self): self.log_odds = 0.0
    def update(self, occ, sensor): self.log_odds = max(min(self.log_odds + (LOG_ODDS_OCC[sensor] if occ else LOG_ODDS_FREE[sensor]), 10), -10)
    def get_probability(self): return 1.0 - 1.0 / (1.0 + math.exp(self.log_odds))
    def is_occupied(self): return self.get_probability() > OCCUPANCY_THRESHOLD

class OGMCell:
    def __init__(self):
        self.log_odds_occupied = 0.0
        self.walls = {'N': None, 'E': None, 'S': None, 'W': None}
        # เพิ่มที่เก็บ “ไอคอนออปเจกต์ในเซลล์” แยกตำแหน่ง Left/Center/Right
        self.objects = {"Left": [], "Center": [], "Right": []}  # รายการ dict ของวัตถุในช่อง

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
    def update_wall(self, r,c,dir_char,occ, sensor): 
        if 0<=r<self.height and 0<=c<self.width:
            w = self.grid[r][c].walls.get(dir_char); 
            if w: w.update(occ, sensor)
    def update_node(self, r,c,occ, sensor='tof'):
        if 0<=r<self.height and 0<=c<self.width:
            self.grid[r][c].log_odds_occupied += (LOG_ODDS_OCC[sensor] if occ else LOG_ODDS_FREE[sensor])
    def is_path_clear(self, r1,c1, r2,c2):
        dr, dc = r2-r1, c2-c1
        if abs(dr)+abs(dc)!=1: return False
        if dr==-1: wall_char='N'
        elif dr==1: wall_char='S'
        elif dc==1: wall_char='E'
        else: wall_char='W'
        w = self.grid[r1][c1].walls.get(wall_char)
        if w and w.is_occupied(): return False
        if 0<=r2<self.height and 0<=c2<self.width:
            if self.grid[r2][c2].is_node_occupied(): return False
        else: return False
        return True

import matplotlib.pyplot as plt
class RealTimeVisualizer:
    def __init__(self, grid_size):
        plt.ion()
        self.grid_size = grid_size
        self.fig, self.ax = plt.subplots(figsize=(8,7))
        self.colors = {"robot":"#0000FF", "target":"#FFD700", "wall":"#000000"}
    def update_plot(self, occupancy_map, robot_pos, path=None):
        self.ax.clear()
        self.ax.set_title("Real-time Hybrid Belief Map (Nodes & Walls + Objects)")
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.ax.set_xlim(-0.5, self.grid_size-0.5); self.ax.set_ylim(self.grid_size-0.5, -0.5)
        # วาด node prob
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                prob = occupancy_map.grid[r][c].get_node_probability()
                color = '#8B0000' if prob>OCCUPANCY_THRESHOLD else ('#D3D3D3' if prob<FREE_THRESHOLD else '#90EE90')
                self.ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1,1, facecolor=color, edgecolor='k', lw=0.5))
        # วาด wall
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell = occupancy_map.grid[r][c]
                if cell.walls['N'].is_occupied(): self.ax.plot([c-0.5, c+0.5],[r-0.5,r-0.5], color=self.colors['wall'], lw=4)
                if cell.walls['W'].is_occupied(): self.ax.plot([c-0.5, c-0.5],[r-0.5,r+0.5], color=self.colors['wall'], lw=4)
                if r==self.grid_size-1 and cell.walls['S'].is_occupied(): self.ax.plot([c-0.5, c+0.5],[r+0.5,r+0.5], color=self.colors['wall'], lw=4)
                if c==self.grid_size-1 and cell.walls['E'].is_occupied(): self.ax.plot([c+0.5, c+0.5],[r-0.5,r+0.5], color=self.colors['wall'], lw=4)
        # วางไอคอน objects ในเซลล์: จุด 3 ตำแหน่ง Left/Center/Right
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                objs = occupancy_map.grid[r][c].objects
                # กำหนดตำแหน่งย่อยในช่อง (offset)
                base_x, base_y = c, r
                if objs["Left"]:
                    self.ax.scatter([base_x-0.25],[base_y], s=40, marker='s')
                if objs["Center"]:
                    self.ax.scatter([base_x],[base_y-0.25], s=40, marker='o')
                if objs["Right"]:
                    self.ax.scatter([base_x+0.25],[base_y], s=40, marker='^')
        # วางตำแหน่งหุ่น
        if robot_pos:
            rr, cc = robot_pos
            self.ax.add_patch(plt.Rectangle((cc-0.5, rr-0.5),1,1, facecolor=self.colors['robot'], edgecolor='k', lw=2, alpha=0.6))
        self.fig.tight_layout()
        self.fig.canvas.draw(); self.fig.canvas.flush_events(); plt.pause(0.01)

# ===== Movement/Attitude/Scanner (ย่อส่วนจาก test_cur_time_copy) =====
class AttitudeHandler:
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
    def normalize_angle(self, a):
        while a>180: a-=360
        while a<=-180: a+=360
        return a
    def correct_yaw_to_target(self, chassis, target_yaw=0.0):
        normalized = self.normalize_angle(target_yaw); time.sleep(0.1)
        robot_rot = -self.normalize_angle(normalized - self.current_yaw)
        if abs(robot_rot) > self.yaw_tolerance:
            chassis.move(x=0,y=0,z=robot_rot,z_speed=60).wait_for_completed(timeout=2)
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
        self.chassis.sub_position(freq=20, callback=self.position_handler)
    def position_handler(self, pos_info):
        self.current_x_pos, self.current_y_pos = pos_info[0], pos_info[1]
    def _yaw_correction(self, att, target_yaw):
        KP_YAW, MAX_YAW_SPEED = 1.8, 25
        yaw_error = att.normalize_angle(target_yaw - att.current_yaw)
        spd = KP_YAW * yaw_error
        return max(min(spd, MAX_YAW_SPEED), -MAX_YAW_SPEED)
    def move_forward_one_grid(self, axis, att):
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
            self.chassis.drive_speed(x=speed,y=0,z=yaw_corr,timeout=1)
        self.chassis.drive_wheels(w1=0,w2=0,w3=0,w4=0); time.sleep(0.2)

class EnvironmentScanner:
    """ใช้ ToF หน้าตรง + Sharp/IR ข้าง (ย่อส่วน)"""
    def __init__(self, sensor_adaptor, tof_sensor, gimbal):
        self.sensor_adaptor, self.tof_sensor, self.gimbal = sensor_adaptor, tof_sensor, gimbal
        self.tof_wall_threshold_cm = 60.0
        self.last_front_cm = float('inf')
        self.is_gimbal_forward = True
        if self.tof_sensor is not None:
            self.tof_sensor.sub_distance(freq=10, callback=self._tof_cb)
    def _tof_cb(self, sub_info):
        cm = calibrate_tof_value(sub_info[0])
        if self.is_gimbal_forward:
            self.last_front_cm = cm
    def get_front_tof_cm(self):
        try:
            self.gimbal.moveto(pitch=0, yaw=0, yaw_speed=SPEED_ROTATE).wait_for_completed()
        except Exception: pass
        self.is_gimbal_forward = True
        time.sleep(0.1)
        return self.last_front_cm
    def cleanup(self):
        try: self.tof_sensor.unsub_distance()
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
    print("🎯 Detect+Map merged start")

    # --- INIT manager & threads (fire_target style) ---
    tracker = ObjectTracker(use_gpu=False)
    roi_state = {"x": ROI_X0, "y": ROI_Y0, "w": ROI_W0, "h": ROI_H0}
    manager = RMConnection()
    reconn = threading.Thread(target=reconnector_thread, args=(manager,), daemon=True)
    reconn.start()

    is_detecting_flag = {"v": False}  # เริ่ม "ยังไม่เข้าโหมด s"
    def is_detecting(): return is_detecting_flag["v"]

    cap_t  = threading.Thread(target=capture_thread_func, args=(manager, frame_queue), daemon=True)
    proc_t = threading.Thread(target=processing_thread_func,
                              args=(tracker, frame_queue, TARGET_SHAPE, TARGET_COLOR, roi_state, is_detecting),
                              daemon=True)
    cap_t.start(); proc_t.start()

    # --- INIT exploration parts (test_cur_time_copy style minimal) ---
    # อุปกรณ์ชั้นล่าง (อาจต่างตามเครื่องจริง โปรดตรวจชื่อ attr ของ SDK คุณ)
    # ถ้า None บางส่วน, ฟังก์ชันที่อาศัยเซ็นเซอร์นั้นจะข้ามอย่างปลอดภัย
    time.sleep(0.5)  # เผื่อให้เชื่อมต่อทัน
    gimbal = manager.get_gimbal()
    chassis = manager.get_chassis()
    tof = manager.get_tof()
    sensor_adaptor = manager.get_sensor_adapter()

    # ตัวช่วยเดิน + มุม
    att = AttitudeHandler()
    if chassis is not None:
        att.start_monitoring(chassis)
    mover = MovementController(chassis) if chassis is not None else None
    scanner = EnvironmentScanner(sensor_adaptor, tof, gimbal)
    ogm = OccupancyGridMap(width=8, height=8)
    vis = RealTimeVisualizer(grid_size=8)

    # ====== Exploration Loop ======
    step = 0
    try:
        while not stop_event.is_set() and step < 40:
            step += 1
            print(f"\n========== STEP {step} at {CURRENT_POSITION} facing {CURRENT_DIRECTION} ==========")

            # 1) ปรับกลางช่องด้วย ToF (ถ้าพร้อม)
            if mover is not None and chassis is not None and gimbal is not None:
                # (เรียกเวอร์ชันย่อ: รักษายอว์และค่อยๆขยับด้วยสปีดเล็ก)
                # ขอย่อ: ถ้าระยะมากกว่า 50 ซม. ให้ข้ามการ center เพื่อไม่เสียเวลา
                front_cm = scanner.get_front_tof_cm() if scanner is not None else float('inf')
                print(f"[ToF] Front distance ~ {front_cm:.1f} cm")
                if front_cm < 50:
                    att.correct_yaw_to_target(chassis, get_compensated_target_yaw())

            # 2) ตัดสิน “บล็อคถัดไป” และ “แกน monitor” จากหน้า/ทิศ
            drdc_by_dir = {0:(-1,0), 1:(0,1), 2:(1,0), 3:(0,-1)}
            dr, dc = drdc_by_dir[CURRENT_DIRECTION]
            next_cell = (CURRENT_POSITION[0]+dr, CURRENT_POSITION[1]+dc)
            axis_to_monitor = 'x' if (ROBOT_FACE % 2 != 0) else 'y'  # จากโค้ดเดิม

            # 3) เช็ค ToF 60cm ข้างหน้า
            front_cm = scanner.get_front_tof_cm() if scanner is not None else float('inf')
            front_blocked = (front_cm < 60.0)
            print(f"[ToF] Front <=60cm? {'YES' if front_blocked else 'NO'}  ({front_cm:.1f} cm)")

            # 4) เปิดโหมด s (detect) 1 วินาที
            is_detecting_flag["v"] = True
            t0 = time.time()
            time.sleep(1.0)  # ช่วงหน้าต่างตรวจ 1 วินาที (ปล่อย processing thread เก็บผล)
            is_detecting_flag["v"] = False

            # 5) ดึงผลตรวจและ “กรอง zone ตามกฎ ToF”
            with output_lock:
                dets = list(processed_output["details"])

            # ถ้า front_blocked -> เก็บทุก zone, else -> เก็บเฉพาะ Left/Right
            allowed_zones = {"Left","Right","Center"} if front_blocked else {"Left","Right"}
            dets_use = [d for d in dets if d["zone"] in allowed_zones]

            # 6) บันทึกลง OGM “บล็อคถัดไป” (วัตถุ/สัญลักษณ์)
            if 0 <= next_cell[0] < ogm.height and 0 <= next_cell[1] < ogm.width:
                cell_obj = ogm.grid[next_cell[0]][next_cell[1]].objects
                zone_bag = {"Left": [], "Center": [], "Right": []}
                for d in dets_use:
                    one = {"color": d["color"], "shape": d["shape"], "zone": d["zone"], "box": d["box"]}
                    cell_obj[d["zone"]].append(one)
                    zone_bag[d["zone"]].append(one)
                # วาดแผนที่ (overlay สัญลักษณ์)
                vis.update_plot(ogm, CURRENT_POSITION)
                # 7) บันทึก JSON รายละเอียดครบ
                append_objects_json(step, next_cell, zone_bag, front_blocked, dets)

            # 8) เดินไปบล็อคถัดไปถ้าทางโล่ง (ดูจาก ToF จริง ณ ตอนนี้)
            if mover is not None and chassis is not None:
                if not front_blocked:
                    mover.move_forward_one_grid(axis_to_monitor, att)
                    CURRENT_POSITION = next_cell
                else:
                    print("⛔ Front blocked (<=60cm). Skip moving forward this step.")

            # 9) อัปเดตภาพแผนที่หลังเคลื่อน
            vis.update_plot(ogm, CURRENT_POSITION)

            # 10) หมุน/จัดข้าง (ย่อ) — คุณสามารถเสียบฟังก์ชัน align+sharp/IR จากไฟล์เดิมเพิ่มได้
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
