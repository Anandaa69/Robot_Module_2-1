# -*-coding:utf-8-*-

"""
Round 2: Complete Target Shooting System
(FINAL VERSION: 3-Way ToF Scan + Precision Turning + Adv. Camera)
- การเดินใช้ PID ควบคุมระยะทาง + Speed Ramp-Up
- จัดตำแหน่งกลางด้วย ToF Scan 3 ทิศทาง (หน้า-ซ้าย-ขวา)
- การเลี้ยวใช้ Real-time Feedback จาก Attitude Sensor เพื่อความแม่นยำ
- ระบบกล้องทำงานใน Background, กรอบ ROI เคลื่อนที่ตามมุม Gimbal
"""

import time
import robomaster
from robomaster import robot, camera as r_camera, blaster as r_blaster
import numpy as np
import math
import json
import matplotlib.pyplot as plt
import threading
import queue
from collections import deque
import traceback
import os
import cv2

# =============================================================================
# ===== CONFIGURATION & PARAMETERS ============================================
# =============================================================================

SHOW_WINDOW = True
DATA_FOLDER = r"./Assignment/data_2"

CURRENT_POSITION = (5, 0); CURRENT_DIRECTION = 1; CURRENT_TARGET_YAW = 0.0; ROBOT_FACE = 1
IMU_DRIFT_COMPENSATION_DEG = 0.0

PID_AIM_KP = -0.15; PID_AIM_KI = -0.005; PID_AIM_KD = -0.02; DERIV_LPF_ALPHA = 0.25; I_CLAMP = 2000.0
PIX_ERR_DEADZONE = 8; LOCK_TOL_X = 12; LOCK_TOL_Y = 12; LOCK_STABLE_COUNT = 6
MAX_YAW_SPEED = 120; MAX_PITCH_SPEED = 100

TOF_ADJUST_SPEED = 0.08; TOF_CALIBRATION_SLOPE = 0.0894; TOF_CALIBRATION_Y_INTERCEPT = 3.8409
TOF_TARGET_CM_FRONT = 19.0  # ระยะห่างเป้าหมายด้านหน้า
TOF_TARGET_CM_SIDE = 17.5   # ระยะห่างเป้าหมายด้านซ้าย-ขวา
TOF_TARGET_CM_BACK = 17.0   # ระยะห่างเป้าหมายด้านหลัง

FRAME_W, FRAME_H = 960, 540; VERTICAL_FOV_DEG = 54.0; PIXELS_PER_DEG_V = FRAME_H / VERTICAL_FOV_DEG
PITCH_BIAS_DEG = 2; PITCH_BIAS_PIX = +PITCH_BIAS_DEG * PIXELS_PER_DEG_V

ROI_Y0, ROI_H0, ROI_X0, ROI_W0 = 264, 270, 10, 911
ROI_SHIFT_PER_DEG = 6.0; ROI_Y_MIN, ROI_Y_MAX = 0, FRAME_H - 10

SPEED_ROTATE = 480; FIRE_SHOTS_COUNT = 5

is_tracking_mode = False; is_detecting_flag = {"v": True}; fired_targets = set()
shots_fired = 0; gimbal_angle_lock = threading.Lock(); gimbal_angles = (0.0, 0.0, 0.0, 0.0)
frame_queue = queue.Queue(maxsize=1); processed_output = {"details": []}; output_lock = threading.Lock()
stop_event = threading.Event(); last_frame_received_ts = 0.0

# =============================================================================
# ===== HELPER FUNCTIONS, DATA LOADING, PATHFINDING ===========================
# =============================================================================
def calibrate_tof_value(raw_tof_value):
    try:
        if raw_tof_value is None or raw_tof_value <= 0: return float('inf')
        return (TOF_CALIBRATION_SLOPE * raw_tof_value) + TOF_CALIBRATION_Y_INTERCEPT
    except Exception: return float('inf')

class PID:
    def __init__(self, Kp, Ki, Kd, setpoint=0):
        self.Kp, self.Ki, self.Kd, self.setpoint = Kp, Ki, Kd, setpoint
        self.prev_error, self.integral = 0, 0
    def compute(self, current, dt):
        error = self.setpoint - current; self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error; return output

def load_data():
    print("📂 Loading data...");
    try:
        map_file=os.path.join(DATA_FOLDER,"Mapping_Top.json");objects_file=os.path.join(DATA_FOLDER,"Detected_Objects.json")
        with open(map_file,"r",encoding="utf-8") as f: map_data=json.load(f)
        with open(objects_file,"r",encoding="utf-8") as f: objects_data=json.load(f)
        print("✅ Data loaded."); return map_data,objects_data
    except Exception as e: print(f"❌ Error loading data: {e}"); return None,None

def create_grid_from_map(map_data):
    print("🗺️ Creating grid..."); max_row=max(n['coordinate']['row'] for n in map_data['nodes']); max_col=max(n['coordinate']['col'] for n in map_data['nodes'])
    width,height=max_col+1,max_row+1; print(f"📊 Grid size: {width}x{height}"); grid=[[0 for _ in range(width)] for _ in range(height)]
    for node in map_data['nodes']:
        if node['is_occupied']: grid[node['coordinate']['row']][node['coordinate']['col']]=1
    return grid,width,height

def find_path_bfs(grid,start,end,width,height,map_data=None):
    if start==end: return [start]
    q,v=deque([[start]]),{start}; m=[(-1,0),(0,1),(1,0),(0,-1)]; mn=['N','E','S','W']
    while q:
        p=q.popleft(); r,c=p[-1]
        if (r,c)==end: return p
        for i,(dr,dc) in enumerate(m):
            nr,nc=r+dr,c+dc
            if not (0<=nr<height and 0<=nc<width and (nr,nc) not in v and grid[nr][nc]==0): continue
            if map_data and not can_move_between_cells(r,c,mn[i],map_data): continue
            v.add((nr,nc)); np=list(p); np.append((nr,nc)); q.append(np)
    return None

def can_move_between_cells(r,c,d,map_data):
    for node in map_data['nodes']:
        if node['coordinate']['row']==r and node['coordinate']['col']==c:
            w=node['walls']
            if d=='N' and w.get('north',False): return False
            if d=='S' and w.get('south',False): return False
            if d=='E' and w.get('east',False): return False
            if d=='W' and w.get('west',False): return False
            return True
    return False

def solve_tsp_greedy(grid,start,targets,width,height,map_data=None):
    rem,cur,order=targets.copy(),start,[]
    while rem:
        paths={t: find_path_bfs(grid,cur,t,width,height,map_data) for t in rem}
        valid={t:p for t,p in paths.items() if p}
        if not valid: print("⚠️ Cannot reach remaining targets."); break
        near=min(valid,key=lambda t:len(valid[t]))
        order.append(near); rem.remove(near); cur=near
    return order
    
# ... (OBJECT DETECTION, RMConnection, CAMERA THREADS - Unchanged) ...
def apply_awb(bgr):
    if hasattr(cv2,"xphoto") and hasattr(cv2.xphoto,"createLearningBasedWB"):
        wb=cv2.xphoto.createLearningBasedWB(); wb.setSaturationThreshold(0.99); return wb.balanceWhite(bgr)
    return bgr
def night_enhance_pipeline_cpu(bgr): return apply_awb(bgr)
class ObjectTracker:
    def __init__(self,use_gpu=False): self.use_gpu=use_gpu
    def _get_angle(self,pt1,pt2,pt0):
        dx1=pt1[0]-pt0[0]; dy1=pt1[1]-pt0[1]; dx2=pt2[0]-pt0[0]; dy2=pt2[1]-pt0[1]
        dot=dx1*dx2+dy1*dy2; mag1=(dx1*dx1+dy1*dy1)**0.5; mag2=(dx2*dx2+dy2*dy2)**0.5
        if mag1*mag2==0: return 0
        return math.degrees(math.acos(max(-1,min(1,dot/(mag1*mag2)))))
    def get_raw_detections(self,frame):
        enhanced=cv2.GaussianBlur(night_enhance_pipeline_cpu(frame),(5,5),0); hsv=cv2.cvtColor(enhanced,cv2.COLOR_BGR2HSV)
        ranges={'Red':([0,80,40],[10,255,255],[170,80,40],[180,255,255]),'Yellow':([20,60,40],[35,255,255]),'Green':([35,40,30],[85,255,255]),'Blue':([90,40,30],[130,255,255])}
        masks={}; masks['Red']=cv2.inRange(hsv,np.array(ranges['Red'][0]),np.array(ranges['Red'][1]))|cv2.inRange(hsv,np.array(ranges['Red'][2]),np.array(ranges['Red'][3]))
        for name in ['Yellow','Green','Blue']: masks[name]=cv2.inRange(hsv,np.array(ranges[name][0]),np.array(ranges[name][1]))
        combined=masks['Red']|masks['Yellow']|masks['Green']|masks['Blue']; kernel=np.ones((5,5),np.uint8); cleaned=cv2.morphologyEx(cv2.morphologyEx(combined,cv2.MORPH_OPEN,kernel),cv2.MORPH_CLOSE,kernel)
        contours,_=cv2.findContours(cleaned,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); out=[]; H,W=frame.shape[:2]
        for cnt in contours:
            area=cv2.contourArea(cnt)
            if area<1500: continue
            x,y,w,h=cv2.boundingRect(cnt)
            if w==0 or h==0: continue
            ar=w/float(h);
            try: solidity=area/cv2.contourArea(cv2.convexHull(cnt))
            except: solidity=0
            if ar>4.0 or ar<0.25 or solidity<0.85 or x<=2 or y<=2 or x+w>=W-2 or y+h>=H-2: continue
            contour_mask=np.zeros((H,W),np.uint8); cv2.drawContours(contour_mask,[cnt],-1,255,-1); max_mean,found=0,"Unknown"
            for cname,m in masks.items():
                mv=cv2.mean(m,mask=contour_mask)[0]
                if mv>max_mean: max_mean,found=mv,cname
            if max_mean<=20: continue
            shape="Uncertain"; peri=cv2.arcLength(cnt,True); circ=(4*math.pi*area)/(peri*peri) if peri>0 else 0
            if circ>0.82: shape="Circle"
            else:
                approx=cv2.approxPolyDP(cnt,0.04*peri,True)
                if len(approx)==4 and solidity>0.88:
                    pts=[tuple(p[0]) for p in approx]; angs=[self._get_angle(pts[(i-1)%4],pts[(i+1)%4],p) for i,p in enumerate(pts)]
                    if all(70<=a<=110 for a in angs): shape="Square" if 0.88<=(w/h)<=1.12 else "Rectangle_H" if w>h else "Rectangle_V"
            out.append({"shape":shape,"color":found,"box":(x,y,w,h),"is_target":False})
        return out
class RMConnection:
    def __init__(self): self._lock=threading.Lock(); self._robot=None; self.connected=threading.Event()
    def connect(self):
        with self._lock:
            self._safe_close(); print("🤖 Connecting..."); rb=robot.Robot(); rb.initialize(conn_type="ap"); rb.camera.start_video_stream(display=False,resolution=r_camera.STREAM_540P)
            try: rb.gimbal.sub_angle(freq=20,callback=sub_angle_cb)
            except Exception as e: print("Gimbal sub error:",e)
            self._robot=rb; self.connected.set(); print("✅ Connected")
            try: rb.gimbal.recenter(pitch_speed=200,yaw_speed=200).wait_for_completed()
            except Exception as e: print("Recenter error:",e)
    def _safe_close(self):
        if self._robot is not None:
            try:
                self._robot.camera.stop_video_stream(); self._robot.gimbal.unsub_angle(); self._robot.close()
            except Exception: pass
            finally: self._robot=None; self.connected.clear(); print("🔌 Closed")
    def get_robot(self):
        with self._lock: return self._robot
    def close(self):
        with self._lock: self._safe_close()
def sub_angle_cb(angle_info):
    global gimbal_angles
    with gimbal_angle_lock: gimbal_angles=tuple(angle_info)
def get_compensated_target_yaw(): return CURRENT_TARGET_YAW+IMU_DRIFT_COMPENSATION_DEG
def capture_thread_func(manager:RMConnection, q:queue.Queue):
    global last_frame_received_ts
    print("🚀 Capture thread started"); fail=0; last_success_time=time.time()
    while not stop_event.is_set():
        if not manager.connected.is_set(): time.sleep(0.1); continue
        cam=manager.get_robot().camera
        if cam is None: time.sleep(0.1); continue
        try:
            frame=cam.read_cv2_image(timeout=0.5)
            if frame is not None and frame.size>0:
                if q.full():
                    try: q.get_nowait()
                    except queue.Empty: pass
                q.put(frame); last_frame_received_ts=time.time(); fail=0; last_success_time=time.time()
            else: fail+=1
        except Exception as e: fail+=1; print(f"Capture error: {e}")
        if fail>=5 and (time.time()-last_success_time)>2.0:
            print("⚠️ Camera errors -> reconnect"); manager.close(); time.sleep(1); manager.connect(); fail=0; last_success_time=time.time()
        time.sleep(0.01)
def processing_thread_func(tracker:ObjectTracker, q:queue.Queue, roi_state):
    global processed_output
    def is_detecting(): return is_detecting_flag["v"]
    while not stop_event.is_set():
        if not is_detecting(): time.sleep(0.2); continue
        try:
            frame=q.get(timeout=0.5); pitch=gimbal_angles[0]
            roi_y=int(ROI_Y0-(max(0.0,-pitch)*ROI_SHIFT_PER_DEG)); roi_y=max(ROI_Y_MIN,min(ROI_Y_MAX,roi_y))
            roi_frame=frame[roi_y:roi_y+roi_state["h"],roi_state["x"]:roi_state["x"]+roi_state["w"]]
            detections=tracker.get_raw_detections(roi_frame)
            with output_lock: processed_output={"details":detections,"dynamic_y":roi_y}
        except queue.Empty: continue
        except Exception as e: print(f"Processing error: {e}")
def camera_display_thread(roi_state):
    print("📹 Display thread started")
    while not stop_event.is_set():
        try:
            frame=frame_queue.get(timeout=0.5).copy()
            with output_lock: details=processed_output.get("details",[]); dy=processed_output.get("dynamic_y",roi_state["y"])
            cv2.rectangle(frame,(roi_state["x"],dy),(roi_state["x"]+roi_state["w"],dy+roi_state["h"]),(255,255,0),2)
            for det in details:
                x,y,w,h=det['box']; ax,ay=x+roi_state["x"],y+dy; color=(0,0,255) if det.get('is_target',False) else (0,255,255)
                thick=3 if det.get('is_target',False) else 2; cv2.rectangle(frame,(ax,ay),(ax+w,ay+h),color,thick)
                cv2.putText(frame,f"{det['color']} {det['shape']}",(ax,ay-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
            mode="TRACKING" if is_tracking_mode else "NAVIGATING"; cv2.putText(frame,f"MODE: {mode}",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            cv2.imshow("Robot Camera Feed",frame)
            if cv2.waitKey(1)&0xFF==ord('q'): stop_event.set(); break
        except queue.Empty: continue
        except Exception as e: print(f"Display error: {e}")
    cv2.destroyAllWindows(); print("📹 Display thread stopped")

# =============================================================================
# ===== SENSOR & MOVEMENT CONTROL (MODIFIED) ==================================
# =============================================================================

class EnvironmentScanner:
    def __init__(self, tof_sensor):
        self.tof_sensor = tof_sensor; self.last_tof_distance_cm = float('inf')
        self.tof_sensor.sub_distance(freq=20, callback=self._tof_data_handler)
    def _tof_data_handler(self, sub_info): self.last_tof_distance_cm = calibrate_tof_value(sub_info[0])
    def get_tof_distance_cm(self): return self.last_tof_distance_cm
    def cleanup(self): self.tof_sensor.unsub_distance()

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
    def normalize_angle(self, angle):
        while angle > 180: angle -= 360
        while angle <= -180: angle += 360
        return angle
    
    def correct_yaw_to_target(self, chassis, target_yaw=0.0):
        normalized_target = self.normalize_angle(target_yaw); time.sleep(0.05)
        robot_rotation = -self.normalize_angle(normalized_target - self.current_yaw)
        print(f"\n🔧 Correcting Yaw: {self.current_yaw:.1f}° -> {target_yaw:.1f}°. Rotating: {robot_rotation:.1f}°")
        
        if abs(robot_rotation) > self.yaw_tolerance:
            # ปรับปรุงการหมุนให้แม่นยำขึ้น
            if abs(robot_rotation) > 45:  # สำหรับการหมุน 90 องศา
                chassis.move(x=0, y=0, z=robot_rotation, z_speed=80).wait_for_completed(timeout=2)
            else:  # สำหรับการปรับแต่งเล็กน้อย
                chassis.move(x=0, y=0, z=robot_rotation, z_speed=60).wait_for_completed(timeout=1)
            time.sleep(0.1)  # เพิ่มเวลารอให้การหมุนเสถียร
        
        final_error = abs(self.normalize_angle(normalized_target - self.current_yaw))
        if final_error <= self.yaw_tolerance: 
            print(f"✅ Yaw Correction Success: {self.current_yaw:.1f}°"); 
            return True
        
        print(f"⚠️ First attempt incomplete. Current: {self.current_yaw:.1f}°. Fine-tuning...")
        remaining_rotation = -self.normalize_angle(normalized_target - self.current_yaw)
        
        # ปรับปรุงการปรับแต่งให้แม่นยำขึ้น
        if abs(remaining_rotation) > 0.5 and abs(remaining_rotation) < 30:
            chassis.move(x=0, y=0, z=remaining_rotation, z_speed=30).wait_for_completed(timeout=3)
            time.sleep(0.15)  # เพิ่มเวลารอ
        
        final_error = abs(self.normalize_angle(normalized_target - self.current_yaw))
        if final_error <= self.yaw_tolerance: 
            print(f"✅ Yaw Fine-tuning Success: {self.current_yaw:.1f}°"); 
            return True
        else:
            print(f"❌ Yaw Correction Failed. Final error: {final_error:.2f}°")
            return False

class MovementController:
    def __init__(self, chassis, scanner, attitude_handler, gimbal):
        self.chassis,self.scanner,self.attitude_handler,self.gimbal=chassis,scanner,attitude_handler,gimbal
        self.current_x_pos,self.current_y_pos=0.0,0.0
        self.chassis.sub_position(freq=20,callback=self.position_handler)
    def position_handler(self,p_info): self.current_x_pos,self.current_y_pos=p_info[0],p_info[1]
    def _calculate_yaw_correction(self, attitude_handler, target_yaw):
        KP_YAW = 1.8; MAX_YAW_SPEED = 25
        yaw_error = attitude_handler.normalize_angle(target_yaw - attitude_handler.current_yaw)
        speed = KP_YAW * yaw_error
        return max(min(speed, MAX_YAW_SPEED), -MAX_YAW_SPEED)
    
    def move_forward_one_grid(self, axis, target_yaw):
        self.attitude_handler.correct_yaw_to_target(self.chassis, target_yaw)
        pid = PID(Kp=1.0, Ki=0.25, Kd=8, setpoint=0.6)
        start_time, last_time = time.time(), time.time()
        start_position = self.current_x_pos if axis == 'x' else self.current_y_pos
        print(f"🚀 Moving 0.6m, AXIS '{axis}'")
        while time.time() - start_time < 3.5:  # Increased timeout
            now = time.time(); dt = now - last_time; last_time = now
            if dt <= 0: continue
            current_position = self.current_x_pos if axis == 'x' else self.current_y_pos
            relative_position = abs(current_position - start_position)
            if abs(relative_position - 0.6) < 0.03: print("\n✅ Move complete!"); break
            output = pid.compute(relative_position, dt)
            ramp_multiplier = min(1.0, 0.1 + ((now - start_time) / 1.0) * 0.9)
            speed = max(-1.0, min(1.0, output * ramp_multiplier))
            yaw_correction = self._calculate_yaw_correction(self.attitude_handler, target_yaw)
            self.chassis.drive_speed(x=speed, y=0, z=yaw_correction, timeout=1)
            print(f"Dist: {relative_position:.3f}/0.60 m", end='\r')
        self.chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0); time.sleep(0.25)
        
    def perform_3_way_tof_centering(self, wall_threshold_cm=50.0, tol_cm=2.0, max_adjust_time=3.0):
        print("\n--- Performing 4-Way ToF Centering (Front-Left-Right-Back) ---")
        comp_yaw = get_compensated_target_yaw()
        
        # ตรวจสอบด้านหน้า
        self.gimbal.moveto(pitch=0, yaw=0, yaw_speed=SPEED_ROTATE).wait_for_completed()
        time.sleep(0.3); front_dist = self.scanner.get_tof_distance_cm()
        print(f"[Front] Initial: {front_dist:.1f} cm")
        
        if front_dist < wall_threshold_cm: 
            self._adjust_axis('x', front_dist, TOF_TARGET_CM_FRONT, tol_cm, max_adjust_time, comp_yaw)
        else: 
            print("[Front] ✅ Open space. Checking back side...")
            # ถ้าไม่มีด้านหน้า ให้ตรวจสอบด้านหลัง
            self.gimbal.moveto(pitch=0, yaw=180, yaw_speed=SPEED_ROTATE).wait_for_completed()
            time.sleep(0.3); back_dist = self.scanner.get_tof_distance_cm()
            print(f"[Back] Initial: {back_dist:.1f} cm")
            if back_dist < wall_threshold_cm:
                self._adjust_axis('x', back_dist, TOF_TARGET_CM_BACK, tol_cm, max_adjust_time, comp_yaw)
            else:
                print("[Back] ✅ Open space. Skipping back adjustment.")
            # กลับมาด้านหน้า
            self.gimbal.moveto(pitch=0, yaw=0, yaw_speed=SPEED_ROTATE).wait_for_completed()
            time.sleep(0.3)
        
        # ตรวจสอบด้านซ้าย
        self.gimbal.moveto(pitch=0, yaw=-90, yaw_speed=SPEED_ROTATE).wait_for_completed()
        time.sleep(0.3); left_dist = self.scanner.get_tof_distance_cm()
        print(f"[Left] Initial: {left_dist:.1f} cm")
        if left_dist < wall_threshold_cm: 
            self._adjust_axis('y', left_dist, TOF_TARGET_CM_SIDE, tol_cm, max_adjust_time, comp_yaw)
        else: 
            print("[Left] ✅ Open space. Skipping.")
        
        # ตรวจสอบด้านขวา
        self.gimbal.moveto(pitch=0, yaw=90, yaw_speed=SPEED_ROTATE).wait_for_completed()
        time.sleep(0.3); right_dist = self.scanner.get_tof_distance_cm()
        print(f"[Right] Initial: {right_dist:.1f} cm")
        if right_dist < wall_threshold_cm: 
            self._adjust_axis('y', right_dist, TOF_TARGET_CM_SIDE, tol_cm, max_adjust_time, comp_yaw, is_right_side=True)
        else: 
            print("[Right] ✅ Open space. Skipping.")
        
        # กลับมาด้านหน้า
        self.gimbal.moveto(pitch=0, yaw=0, yaw_speed=SPEED_ROTATE).wait_for_completed()
        print("--- 4-Way ToF Centering Complete ---")

    def _adjust_axis(self, axis, initial_dist, target_cm, tol_cm, max_time, comp_yaw, is_right_side=False):
        start_t = time.time(); error = initial_dist - target_cm
        if abs(error) <= tol_cm: print(f"✅ Already centered."); return
        
        # แก้ไขการคำนวณทิศทางการเคลื่อนที่
        if axis == 'x': 
            # สำหรับแกน X (หน้า-หลัง): ถ้าไกลเกินไป (error > 0) ให้เข้าใกล้ (direction > 0)
            # ถ้าใกล้เกินไป (error < 0) ให้ถอยออก (direction < 0)
            direction = TOF_ADJUST_SPEED if error > 0 else -TOF_ADJUST_SPEED
        else:
            # สำหรับแกน Y (ซ้าย-ขวา): ต้องคำนึงถึงทิศทางของหุ่นยนต์
            # เมื่อหุ่นยนต์หันหน้าไปทางทิศเหนือ (yaw=0°):
            # - ซ้าย (yaw=-90°): ถ้าไกลเกินไป ให้เข้าใกล้กำแพงซ้าย (y=-)
            # - ขวา (yaw=+90°): ถ้าไกลเกินไป ให้เข้าใกล้กำแพงขวา (y=+)
            if is_right_side:
                # ด้านขวา: ถ้าไกลเกินไป (error > 0) ให้เข้าใกล้กำแพงขวา (y=+)
                # ถ้าใกล้เกินไป (error < 0) ให้ถอยออกจากกำแพงขวา (y=-)
                direction = TOF_ADJUST_SPEED if error > 0 else -TOF_ADJUST_SPEED
            else:
                # ด้านซ้าย: ถ้าไกลเกินไป (error > 0) ให้เข้าใกล้กำแพงซ้าย (y=-)
                # ถ้าใกล้เกินไป (error < 0) ให้ถอยออกจากกำแพงซ้าย (y=+)
                direction = -TOF_ADJUST_SPEED if error > 0 else TOF_ADJUST_SPEED
        
        print(f"🎯 Target: {target_cm}cm, Current: {initial_dist:.1f}cm, Error: {error:.1f}cm, Direction: {direction:.3f}")
        print(f"   📍 Axis: {axis.upper()}, Side: {'Right' if is_right_side else 'Left'}")
        if axis == 'y':
            if is_right_side:
                direction_text = "Toward Right Wall" if direction > 0 else "Away from Right Wall"
            else:
                direction_text = "Toward Left Wall" if direction < 0 else "Away from Left Wall"
            print(f"   🚀 Moving: {direction_text} (y={direction:.3f})")
        else:
            direction_text = "Toward Front Wall" if direction > 0 else "Away from Front Wall"
            print(f"   🚀 Moving: {direction_text} (x={direction:.3f})")
        
        while time.time() - start_t < max_time:
            cur_dist = self.scanner.get_tof_distance_cm()
            if cur_dist is None or math.isinf(cur_dist): time.sleep(0.05); continue
            error = cur_dist - target_cm
            
            # ตรวจสอบความปลอดภัย - ห้ามเข้าใกล้กำแพงเกินไป
            if cur_dist < 10.0:  # ถ้าใกล้กำแพงเกิน 10cm ให้หยุด
                print(f"\n⚠️ Too close to wall ({cur_dist:.1f}cm)! Stopping for safety.")
                break
            
            print(f"Adjusting {axis.upper()}-axis... Cur:{cur_dist:.1f} cm, Err:{error:.1f} cm", end="\r")
            if abs(error) <= tol_cm: print(f"\n✅ Centered OK. Final: {cur_dist:.1f} cm"); break
            
            yaw_cor = self._calculate_yaw_correction(self.attitude_handler, comp_yaw)
            if axis == 'x': self.chassis.drive_speed(x=direction, y=0, z=yaw_cor, timeout=0.2)
            else: self.chassis.drive_speed(x=0, y=direction, z=yaw_cor, timeout=0.2)
            time.sleep(0.1)
        self.chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0); time.sleep(0.1)

    def rotate_to_direction(self, target_direction):
        global CURRENT_DIRECTION
        if CURRENT_DIRECTION == target_direction: return
        
        diff = (target_direction - CURRENT_DIRECTION + 4) % 4
        if diff == 1: 
            self.rotate_90_degrees_right()
        elif diff == 3: 
            self.rotate_90_degrees_left()
        elif diff == 2: 
            self.rotate_90_degrees_right(); 
            self.rotate_90_degrees_right()
        
        # รอให้การเลี้ยวเสร็จสิ้นและเสถียร
        print("   -> Waiting for robot rotation to complete...")
        time.sleep(0.2)  # รอให้การเลี้ยวเสร็จสิ้นและเสถียร
        
        # ปรับ gimbal ให้ตรงกับทิศทางใหม่หลังจากหุ่นหมุนเสร็จ
        try:
            print("   -> Adjusting gimbal to match new robot direction...")
            self.gimbal.moveto(pitch=0, yaw=0, yaw_speed=SPEED_ROTATE).wait_for_completed()
            time.sleep(0.2)  # เพิ่มเวลารอให้ gimbal settle
            print("   -> Gimbal adjusted to match robot direction")
        except Exception as e:
            print(f"⚠️ Gimbal centering error: {e}")
            time.sleep(0.3)  # รอให้ gimbal settle หลัง error
            print("   -> Continuing without gimbal adjustment...")

    def rotate_90_degrees_right(self):
        global CURRENT_TARGET_YAW, CURRENT_DIRECTION, ROBOT_FACE
        print("🔄 Rotating 90° RIGHT...")
        CURRENT_TARGET_YAW = self.attitude_handler.normalize_angle(CURRENT_TARGET_YAW + 90)
        
        # ปรับปรุงการหมุนให้แม่นยำขึ้น
        success = self.attitude_handler.correct_yaw_to_target(self.chassis, get_compensated_target_yaw())
        if not success:
            print("⚠️ Primary rotation failed, attempting alternative method...")
            # ลองใช้วิธีหมุนแบบอื่น
            self.attitude_handler.correct_yaw_to_target(self.chassis, get_compensated_target_yaw())
        
        CURRENT_DIRECTION = (CURRENT_DIRECTION + 1) % 4; ROBOT_FACE += 1

    def rotate_90_degrees_left(self):
        global CURRENT_TARGET_YAW, CURRENT_DIRECTION, ROBOT_FACE
        print("🔄 Rotating 90° LEFT...")
        CURRENT_TARGET_YAW = self.attitude_handler.normalize_angle(CURRENT_TARGET_YAW - 90)
        
        # ปรับปรุงการหมุนให้แม่นยำขึ้น
        success = self.attitude_handler.correct_yaw_to_target(self.chassis, get_compensated_target_yaw())
        if not success:
            print("⚠️ Primary rotation failed, attempting alternative method...")
            # ลองใช้วิธีหมุนแบบอื่น
            self.attitude_handler.correct_yaw_to_target(self.chassis, get_compensated_target_yaw())
        
        CURRENT_DIRECTION = (CURRENT_DIRECTION - 1 + 4) % 4; ROBOT_FACE -= 1
        if ROBOT_FACE < 1: ROBOT_FACE += 4
    def cleanup(self):
        try: self.chassis.unsub_position()
        except: pass

# ... (TARGETING & SHOOTING - Unchanged) ...
class PIDController:
    def __init__(self,kp,ki,kd,i_clamp=1000.0):self.kp,self.ki,self.kd=kp,ki,kd;self.i_clamp=i_clamp;self.prev_error,self.integral,self.prev_deriv=0,0,0
    def compute(self,error,dt):
        self.integral+=error*dt;self.integral=max(min(self.integral,self.i_clamp),-self.i_clamp);derivative=(error-self.prev_error)/dt if dt>0 else 0;self.prev_deriv=DERIV_LPF_ALPHA*derivative+(1-DERIV_LPF_ALPHA)*self.prev_deriv;output=self.kp*error+self.ki*self.integral+self.kd*self.prev_deriv;self.prev_error=error;return output
    def reset(self): self.prev_error,self.integral,self.prev_deriv=0,0,0
def aim_and_shoot_target(manager, target_shape, target_color, roi_state, max_lock_time=10.0):
    global is_tracking_mode, fired_targets
    print(f"🎯 Targeting: {target_color} {target_shape}");
    gimbal=manager.get_robot().gimbal; blaster=manager.get_robot().blaster
    if gimbal is None or blaster is None: print("⚠️ Gimbal/blaster not available");return False
    
    yaw_pid=PIDController(PID_AIM_KP,PID_AIM_KI,PID_AIM_KD,I_CLAMP)
    pitch_pid=PIDController(PID_AIM_KP,PID_AIM_KI,PID_AIM_KD,I_CLAMP)
    
    lock_count=0; start_time=time.time(); is_tracking_mode=True; shots_fired_this_target=0
    
    while time.time()-start_time<max_lock_time and shots_fired_this_target<FIRE_SHOTS_COUNT:
        with output_lock:
            details=processed_output.get("details",[]); dynamic_y=processed_output.get("dynamic_y",roi_state["y"])
            target_box=None
            for det in details:
                if det['shape']==target_shape and det['color']==target_color:
                    det['is_target'] = True; target_box=det['box']
                else: det['is_target'] = False
        
        if target_box is None: gimbal.drive_speed(pitch_speed=0,yaw_speed=0);time.sleep(0.1);continue
        
        x,y,w,h=target_box; cx=roi_state["x"]+x+w/2.0; cy=dynamic_y+y+h/2.0
        err_x=(FRAME_W/2.0-cx); err_y=(FRAME_H/2.0-cy)+PITCH_BIAS_PIX
        if abs(err_x)<PIX_ERR_DEADZONE: err_x=0.0
        if abs(err_y)<PIX_ERR_DEADZONE: err_y=0.0
        
        dt=0.01; u_x=float(np.clip(yaw_pid.compute(err_x,dt),-MAX_YAW_SPEED,MAX_YAW_SPEED)); u_y=float(np.clip(pitch_pid.compute(err_y,dt),-MAX_PITCH_SPEED,MAX_PITCH_SPEED))
        gimbal.drive_speed(pitch_speed=-u_y,yaw_speed=u_x)
        
        if abs(err_x)<=LOCK_TOL_X and abs(err_y)<=LOCK_TOL_Y:
            lock_count+=1
            if lock_count>=LOCK_STABLE_COUNT:
                blaster.fire(fire_type=r_blaster.WATER_FIRE); shots_fired_this_target+=1; print(f"🔥 Fired shot {shots_fired_this_target}/{FIRE_SHOTS_COUNT}"); time.sleep(0.5); lock_count=0
        else: lock_count=0
        time.sleep(dt)

    is_tracking_mode=False
    gimbal.recenter(pitch_speed=200,yaw_speed=200).wait_for_completed()
    if shots_fired_this_target>=FIRE_SHOTS_COUNT: fired_targets.add(f"{target_color}_{target_shape}"); return True
    else: print(f"⏰ Targeting timeout for {target_color} {target_shape}"); return False

# =============================================================================
# ===== MISSION EXECUTION (MODIFIED) ==========================================
# =============================================================================
def find_targets_by_type(objects_data,target_shape,target_color):
    matching_targets=[];target_color_clean=target_color.strip().lower();target_shape_clean=target_shape.strip().lower()
    for obj in objects_data['detected_objects']:
        if obj['shape'].strip().lower()==target_shape_clean and obj['color'].strip().lower()==target_color_clean:
            matching_targets.append({'target_position':tuple(obj['cell_position'].values()),'shooting_position':tuple(obj['detected_from_node']),'shape':obj['shape'],'color':obj['color'],'zone':obj['zone']})
    return matching_targets

def execute_shooting_mission(grid,width,height,target_sequence,movement_controller,attitude_handler,manager,roi_state,map_data=None):
    global CURRENT_POSITION
    print(f"🚀 Starting mission with {len(target_sequence)} targets")
    for i,target_info in enumerate(target_sequence):
        shooting_pos,target_pos=target_info['shooting_position'],target_info['target_position']
        target_shape,target_color=target_info['shape'],target_info['color']
        print(f"\n🎯 Target {i+1}/{len(target_sequence)}: {target_color} {target_shape} at {target_pos}");print(f"    Navigating from {CURRENT_POSITION} to {shooting_pos}")
        path=find_path_bfs(grid,CURRENT_POSITION,shooting_pos,width,height,map_data)
        if not path: print(f"❌ Cannot reach {shooting_pos}. Skipping."); continue
        print(f"📍 Path: {path}")
        
        for j in range(len(path) - 1):
            next_pos=path[j+1]; dr,dc=next_pos[0]-path[j][0],next_pos[1]-path[j][1]
            if dr==-1: target_dir=0
            elif dr==1: target_dir=2
            elif dc==1: target_dir=1
            else: target_dir=3
            movement_controller.rotate_to_direction(target_dir)
            
            axis='x' if ROBOT_FACE%2!=0 else 'y'
            movement_controller.move_forward_one_grid(axis,get_compensated_target_yaw())
            
            # --- MODIFIED: Call the 4-way centering function ---
            movement_controller.perform_3_way_tof_centering()
            
            CURRENT_POSITION=next_pos
            print(f"📍 Arrived at {CURRENT_POSITION}")
        
        print(f"🎯 At {CURRENT_POSITION}, turning to face {target_pos}")
        dr,dc=target_pos[0]-shooting_pos[0],target_pos[1]-shooting_pos[1]
        if dr==-1: target_dir=0
        elif dr==1: target_dir=2
        elif dc==1: target_dir=1
        elif dc==-1: target_dir=3
        else: print(f"❌ Invalid direction. Skipping."); continue
        movement_controller.rotate_to_direction(target_dir)
        manager.get_robot().gimbal.recenter(pitch_speed=200,yaw_speed=200).wait_for_completed()
        print(f"🎯 Facing target, starting detection...")
        success=aim_and_shoot_target(manager,target_shape,target_color,roi_state)
        if success: print(f"✅ Target {i+1} complete")
        else: print(f"❌ Failed target {i+1}")
        time.sleep(1.0)
    print("🎉 Mission completed!")

# =============================================================================
# ===== MAIN FUNCTION (MODIFIED) ===============================================
# =============================================================================
def main():
    print("🎯 Round 2: Complete System (3-Way ToF Scan + Precision Turning)")
    map_data,objects_data=load_data()
    if not all([map_data,objects_data]): print("❌ Failed to load data."); return
    grid,width,height=create_grid_from_map(map_data)
    print("\n🎯 Target Specification:")
    target_shape=input("Enter shape: ").strip(); target_color=input("Enter color: ").strip()
    targets=find_targets_by_type(objects_data,target_shape,target_color)
    if not targets: print(f"❌ No targets '{target_color} {target_shape}'."); return
    print(f"\n✅ Found {len(targets)} targets. Solving path...")
    shooting_positions=list(set([t['shooting_position'] for t in targets]))
    optimal_order=solve_tsp_greedy(grid,CURRENT_POSITION,shooting_positions,width,height,map_data)
    target_sequence=[]
    for pos in optimal_order:
        for target in targets:
            if target['shooting_position']==pos and target not in target_sequence: target_sequence.append(target)
    print("\n🎯 Optimal sequence:"); [print(f"  {i+1}. Shoot {t['color']} {t['shape']} from {t['shooting_position']}") for i,t in enumerate(target_sequence)]
    print("\n🤖 Initializing robot..."); manager=RMConnection(); ep_robot=None; display_t=None
    try:
        manager.connect()
        if not manager.connected.wait(timeout=10.0): print("❌ Robot connection failed."); return
        ep_robot=manager.get_robot(); ep_chassis=ep_robot.chassis; ep_gimbal=ep_robot.gimbal; ep_tof_sensor=ep_robot.sensor

        attitude_handler=AttitudeHandler(); scanner=EnvironmentScanner(ep_tof_sensor)
        movement_controller=MovementController(ep_chassis,scanner,attitude_handler, ep_gimbal)
        
        attitude_handler.start_monitoring(ep_chassis)
        roi_state={"x":ROI_X0,"y":ROI_Y0,"w":ROI_W0,"h":ROI_H0}

        tracker=ObjectTracker(); cap_t=threading.Thread(target=capture_thread_func,args=(manager,frame_queue),daemon=True)
        proc_t=threading.Thread(target=processing_thread_func,args=(tracker,frame_queue,roi_state),daemon=True)
        cap_t.start(); proc_t.start()
        if SHOW_WINDOW:
            display_t=threading.Thread(target=camera_display_thread,args=(roi_state,),daemon=True); display_t.start()
        
        time.sleep(2)

        execute_shooting_mission(grid,width,height,target_sequence,movement_controller,attitude_handler,manager,roi_state,map_data)

    except KeyboardInterrupt: print("\n⚠️ Mission interrupted.")
    except Exception as e: print(f"\n❌ An error occurred: {e}"); traceback.print_exc()
    finally:
        print("\n🧹 Cleaning up..."); stop_event.set()
        if ep_robot:
            if 'attitude_handler' in locals(): attitude_handler.stop_monitoring(ep_robot.chassis)
            if 'movement_controller' in locals(): movement_controller.cleanup()
            if 'scanner' in locals(): scanner.cleanup()
        if 'display_t' in locals() and display_t and display_t.is_alive(): display_t.join(timeout=1.0)
        manager.close(); print("✅ Mission finished.")

if __name__ == '__main__':
    main()