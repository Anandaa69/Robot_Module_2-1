# zone_multithread_reconnect_smoothing.py
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time
import math
import threading
import queue
import os
import csv
from collections import deque, Counter
from datetime import datetime

# RoboMaster SDK
from robomaster import robot, camera as r_camera

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
# Shared resources
# =====================================================
frame_queue = queue.Queue(maxsize=1)
processed_output = {"details": []}
output_lock = threading.Lock()
stop_event = threading.Event()

# Target config (ปรับได้)
TARGET_SHAPE = "Circle"
TARGET_COLOR = "Red"

# ROI (พิกัดบนเฟรมเต็ม)
ROI_Y, ROI_H, ROI_X, ROI_W = 264, 270, 10, 911

# =====================================================
# AWB / Night enhance (เบาๆ รวดเร็ว)
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
    # ปรับสมดุลสีให้กล้องกลางคืน/ในที่มืดดีขึ้น
    return apply_awb(bgr)

# =====================================================
# Detector/Tracker ขั้นพื้นฐาน (รูปร่าง/สี)
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
        """
        คืนค่า list ของ dict: {'contour', 'shape', 'color', 'box':(x,y,w,h)}
        พิกัด box อ้างอิงใน 'เฟรม ROI' (ไม่ใช่เฟรมเต็ม)
        """
        enhanced = cv2.GaussianBlur(night_enhance_pipeline_cpu(frame), (5,5), 0)
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)

        # ช่วงค่าสี (ปรับตามสภาพแสงจริงได้)
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
            if area < 1500: 
                continue
            x,y,w,h = cv2.boundingRect(cnt)
            if w==0 or h==0: 
                continue
            ar = w/float(h)
            if ar>4.0 or ar<0.25: 
                continue
            hull = cv2.convexHull(cnt); ha = cv2.contourArea(hull)
            if ha==0: 
                continue
            solidity = area/ha
            if solidity < 0.85: 
                continue
            # กันติดขอบ
            if x<=2 or y<=2 or x+w>=W-2 or y+h>=H-2: 
                continue

            # ตัดสินสีแบบเฉลี่ยบน mask ของ contour
            contour_mask = np.zeros((H,W), np.uint8)
            cv2.drawContours(contour_mask, [cnt], -1, 255, -1)
            max_mean, found = 0, "Unknown"
            for cname, m in masks.items():
                mv = cv2.mean(m, mask=contour_mask)[0]
                if mv > max_mean:
                    max_mean, found = mv, cname
            if max_mean <= 20:
                continue

            # ตัดสินรูปร่าง
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
# Track/Hysteresis Manager (แก้แกว่ง Uncertain ↔ รูปร่างจริง + Log flip)
# =====================================================
class Track:
    def __init__(self, tid, cx, cy, maxlen=8, lock_frames_default=20):
        self.id = tid
        self.cx = cx
        self.cy = cy
        self.history = deque(maxlen=maxlen)   # เก็บชื่อ shape
        self.locked_shape = None
        self.lock_frames = 0
        self.lock_frames_default = lock_frames_default
        self.last_shape = None
        self.flip_count = 0
        self.missed = 0

    def update_pos(self, cx, cy):
        self.cx, self.cy = cx, cy
        self.missed = 0

    def push_shape(self, shape):
        if self.last_shape is not None and self.last_shape != shape:
            self.flip_count += 1
        self.last_shape = shape
        self.history.append(shape)

    def decide_shape(self):
        # ถ้าถูกล็อก → ใช้ล็อกจนหมดอายุ
        if self.locked_shape is not None:
            self.lock_frames -= 1
            if self.lock_frames <= 0:
                self.locked_shape = None
            return self.locked_shape

        if not self.history:
            return "Uncertain"

        cnt = Counter(self.history)
        # หาคะแนนแบบไม่รวม Uncertain
        cnt_no_unc = {k:v for k,v in cnt.items() if k != "Uncertain"}
        if cnt_no_unc:
            shape_major, votes = max(cnt_no_unc.items(), key=lambda kv: kv[1])
            # เกณฑ์เดบาวน์: อย่างน้อยครึ่งหนึ่ง + ปัดขึ้น
            if votes >= max(3, len(self.history)//2 + 1):
                # ถ้ามี Uncertain ผสมหลายครั้ง → ล็อกชั่วคราว
                if cnt.get("Uncertain", 0) >= 2 and votes >= 3:
                    self.locked_shape = shape_major
                    self.lock_frames = self.lock_frames_default
                return shape_major

        # ไม่ผ่านเงื่อนไข → คืนเสียงส่วนมากรวม Uncertain
        return max(cnt.items(), key=lambda kv: kv[1])[0]


class TrackManager:
    def __init__(self, max_dist=80.0, maxlen=8, lock_frames=20, log_path="shape_flips.csv"):
        self.tracks = {}          # tid -> Track
        self.next_id = 1
        self.max_dist = max_dist
        self.maxlen = maxlen
        self.lock_frames = lock_frames
        self.log_path = log_path
        self._init_log()

    def _init_log(self):
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["timestamp","track_id","prev_shape","new_shape","note"])

    def _log_flip(self, tid, prev_shape, new_shape, note=""):
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([datetime.now().isoformat(timespec="seconds"), tid, prev_shape, new_shape, note])

    @staticmethod
    def _dist(a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def associate(self, detections):
        """
        detections: list of {'box':(x,y,w,h), 'shape', 'color', 'zone'} (พิกัดใน ROI)
        return: list ของ dict เดิม + เพิ่ม 'track_id', 'shape_smoothed'
        """
        results = []

        # เตรียมเซนทรอยด์เพื่อจับคู่
        det_centroids = []
        for i, d in enumerate(detections):
            x,y,w,h = d["box"]
            det_centroids.append((i, (x + w/2.0, y + h/2.0)))

        # จับคู่ nearest-neighbor
        for i, (cx, cy) in det_centroids:
            best_tid, best_dist = None, 1e9
            for tid, tr in self.tracks.items():
                if getattr(tr, "_matched", False):
                    continue
                d = self._dist((cx,cy), (tr.cx, tr.cy))
                if d < best_dist and d <= self.max_dist:
                    best_dist, best_tid = d, tid

            if best_tid is None:
                # สร้างแทร็กใหม่
                tid = self.next_id; self.next_id += 1
                tr = Track(tid, cx, cy, maxlen=self.maxlen, lock_frames_default=self.lock_frames)
                self.tracks[tid] = tr
                best_tid = tid

            tr = self.tracks[best_tid]
            tr.update_pos(cx, cy)
            prev = tr.last_shape
            tr.push_shape(detections[i]["shape"])
            decided = tr.decide_shape()
            tr._matched = True

            # log flip เมื่อเปลี่ยน (รวมกรณี Uncertain<->จริง)
            if prev is not None and prev != decided:
                note = "uncertain-flip" if (prev == "Uncertain" or decided == "Uncertain") else ""
                self._log_flip(tr.id, prev, decided, note=note)

            out = detections[i].copy()
            out["track_id"] = tr.id
            out["shape_smoothed"] = decided
            results.append(out)

        # อัปเดตแทร็กที่หาย
        for tid, tr in list(self.tracks.items()):
            if not getattr(tr, "_matched", False):
                tr.missed += 1
                if tr.missed > 30:  # หายไปนาน ลบทิ้ง
                    del self.tracks[tid]
            if hasattr(tr, "_matched"):
                delattr(tr, "_matched")

        return results

# =====================================================
# Connection Manager + Reconnector (ไม่ปิดโปรแกรม)
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
            rb.camera.start_video_stream(display=False, resolution=r_camera.STREAM_540P)
            self._robot = rb
            self.connected.set()
            print("✅ RoboMaster connected & camera streaming")

    def _safe_close(self):
        if self._robot is not None:
            try:
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
            self._safe_close()  # ให้ reconnector มองเห็น connected=False

    def get_camera(self):
        with self._lock:
            return None if self._robot is None else self._robot.camera

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
        cam = manager.get_camera()
        if cam is None:
            time.sleep(0.1); 
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
            # ล้างคิวรอเฟรมใหม่หลังเชื่อมต่อสำเร็จ
            try:
                while True: q.get_nowait()
            except queue.Empty:
                pass
            fail = 0
        time.sleep(0.005)
    print("🛑 Capture thread stopped")

def processing_thread_func(tracker: ObjectTracker, q: queue.Queue,
                           target_shape, target_color, roi_coords,
                           is_detecting_func, trackman: TrackManager):
    """
    รักษารูปแบบผลลัพธ์แบบเดิม:
    processed_output = {"details":[{id,color,shape,zone,is_target,box}]}
    โดย shape ใช้ค่า 'shape_smoothed' ที่ผ่านเดบาวน์แล้ว
    """
    global processed_output
    print("🧠 Processing thread started.")
    ROI_X, ROI_Y, ROI_W, ROI_H = roi_coords

    while not stop_event.is_set():
        if not is_detecting_func():
            time.sleep(0.1)
            continue
        try:
            frame_to_process = q.get(timeout=1.0)
            roi_frame = frame_to_process[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
            detections = tracker.get_raw_detections(roi_frame)

            # ตีเส้นแบ่งโซนในพิกัด ROI
            divider1 = int(ROI_W*0.33)
            divider2 = int(ROI_W*0.66)
            for d in detections:
                x,y,w,h = d["box"]
                endx = x + w
                zone = "Center"
                if endx < divider1: zone = "Left"
                elif x >= divider2: zone = "Right"
                d["zone"] = zone

            # อัปเดตแทร็ก + เดบาวน์
            tracked = trackman.associate(detections)

            # สร้างผลลัพธ์สำหรับการแสดงผล
            detailed_results = []
            for d in tracked:
                shape_raw = d["shape"]
                shape_smoothed = d["shape_smoothed"]
                color = d["color"]
                x,y,w,h = d["box"]
                tid = d["track_id"]
                is_target = (shape_smoothed == target_shape and color == target_color)
                detailed_results.append({
                    "id": tid,                   # แสดง Track-ID (นิ่งกว่า index)
                    "color": color,
                    "shape": shape_smoothed,     # ใช้ค่าหลังเดบาวน์
                    "shape_raw": shape_raw,      # เก็บไว้ตรวจสอบ
                    "zone": d["zone"],
                    "is_target": is_target,
                    "box": (x,y,w,h)
                })

            with output_lock:
                processed_output = {"details": sorted(detailed_results, key=lambda a: a["id"])}

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
    print(f"🎯 Target set to: {TARGET_COLOR} {TARGET_SHAPE}")

    tracker = ObjectTracker(use_gpu=USE_GPU)
    manager = RMConnection()
    reconn = threading.Thread(target=reconnector_thread, args=(manager,), daemon=True)
    reconn.start()

    # ค่าปรับเพื่อลดการแกว่ง (ปรับได้ตามสนามจริง)
    # max_dist: รัศมี (พิกเซล) สำหรับจับคู่ดีเท็คกับแทร็กเดิม (ในพิกัด ROI)
    # maxlen: ขนาดหน้าต่าง history (มาก = นิ่งขึ้น แต่ช้าขึ้น)
    # lock_frames: จำนวนเฟรมที่ล็อกรูปร่างเมื่อพบ flip Uncertain บ่อยๆ
    trackman = TrackManager(max_dist=80.0, maxlen=8, lock_frames=20, log_path="shape_flips.csv")

    is_detecting_flag = {"v": False}
    def is_detecting(): return is_detecting_flag["v"]

    cap_t = threading.Thread(target=capture_thread_func, args=(manager, frame_queue), daemon=True)
    proc_t = threading.Thread(
        target=processing_thread_func,
        args=(tracker, frame_queue, TARGET_SHAPE, TARGET_COLOR, (ROI_X, ROI_Y, ROI_W, ROI_H),
              is_detecting, trackman),
        daemon=True
    )
    cap_t.start(); proc_t.start()

    print("\n--- Real-time Scan (Auto-Reconnect + Debounce + Full Display) ---")
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

            # วาด ROI กรอบนอก
            cv2.rectangle(display_frame, (ROI_X, ROI_Y), (ROI_X+ROI_W, ROI_Y+ROI_H), (255,0,0), 2)

            if is_detecting():
                cv2.putText(display_frame, "MODE: DETECTING", (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                with output_lock:
                    details = processed_output["details"]

                # เส้นแบ่งโซน (absolute บนเฟรมเต็ม)
                d1_abs = ROI_X + int(ROI_W*0.33)
                d2_abs = ROI_X + int(ROI_W*0.66)
                cv2.line(display_frame, (d1_abs, ROI_Y), (d1_abs, ROI_Y+ROI_H), (255,255,0), 1)
                cv2.line(display_frame, (d2_abs, ROI_Y), (d2_abs, ROI_Y+ROI_H), (255,255,0), 1)

                # วาดกล่อง + หมายเลข track-id + สีกรอบ
                for det in details:
                    x,y,w,h = det['box']
                    abs_x, abs_y = x + ROI_X, y + ROI_Y
                    shape = det['shape']
                    is_target = det['is_target']

                    if is_target:
                        box_color = (0,0,255)      # แดง: เป้าหมาย
                    elif shape == 'Uncertain':
                        box_color = (0,255,255)    # เหลือง: ไม่แน่ใจ
                    else:
                        box_color = (0,255,0)      # เขียว: รูปร่างปกติ

                    thickness = 4 if is_target else 2
                    cv2.rectangle(display_frame, (abs_x,abs_y), (abs_x+w, abs_y+h), box_color, thickness)
                    cv2.putText(display_frame, f"#{det['id']}", (abs_x+5, abs_y+25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

                # รายการข้อความด้านซ้ายบน (ID, สี, รูปร่าง, TARGET!)
                if details:
                    y_pos = 70
                    for obj in details:
                        tgt = " (TARGET!)" if obj['is_target'] else ""
                        line = f"ID {obj['id']}: {obj['color']} {obj['shape']}{tgt}"
                        # เผื่ออยากดูค่า raw:
                        # line = f"ID {obj['id']}: {obj['color']} {obj['shape']} [raw:{obj.get('shape_raw','?')}]"
                        cv2.putText(display_frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 4)
                        cv2.putText(display_frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                        y_pos += 25
            else:
                cv2.putText(display_frame, "MODE: VIEWING", (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

            # สถานะการเชื่อมต่อ SDK
            st = "CONNECTED" if manager.connected.is_set() else "RECONNECTING..."
            cv2.putText(display_frame, f"SDK: {st}", (20, 70 if not is_detecting() else 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            cv2.imshow("Robomaster Real-time Scan (Reconnect+Debounce)", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                is_detecting_flag["v"] = not is_detecting_flag["v"]
                print(f"Detection {'ON' if is_detecting_flag['v'] else 'OFF'}")
            elif key == ord('r'):
                print("Manual reconnect requested")
                manager.drop_and_reconnect()
                # ล้างคิวเฟรมเพื่อรอเฟรมใหม่จากกล้อง
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
