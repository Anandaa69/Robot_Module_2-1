# -*-coding:utf-8-*-

import time
import robomaster
from robomaster import robot
import numpy as np
import math
import json
import matplotlib.pyplot as plt
from collections import deque
import traceback
import statistics
import cv2
import sys
import threading

# =============================================================================
# ===== CONFIGURATION & PARAMETERS ============================================
# =============================================================================

# --- Robot Movement ---
SPEED_ROTATE = 480

# --- Sharp Distance Sensor Configuration ---
LEFT_SHARP_SENSOR_ID = 1
LEFT_SHARP_SENSOR_PORT = 1
LEFT_TARGET_CM = 13.0
RIGHT_SHARP_SENSOR_ID = 2
RIGHT_SHARP_SENSOR_PORT = 1
RIGHT_TARGET_CM = 13.0

# --- IR Sensor Configuration ---
LEFT_IR_SENSOR_ID = 1
LEFT_IR_SENSOR_PORT = 2
RIGHT_IR_SENSOR_ID = 2
RIGHT_IR_SENSOR_PORT = 2

# --- Sensor Detection Thresholds ---
SHARP_WALL_THRESHOLD_CM = 60.0
SHARP_STDEV_THRESHOLD = 0.3

# --- ToF Centering Configuration ---
TOF_ADJUST_SPEED = 0.1
TOF_CALIBRATION_SLOPE = 0.0894
TOF_CALIBRATION_Y_INTERCEPT = 3.8409

# --- Logical state for the grid map ---
CURRENT_POSITION = (5, 3)
CURRENT_DIRECTION = 0  # 0:North, 1:East, 2:South, 3:West
TARGET_DESTINATION = (5, 3)

# --- Physical state for the robot ---
CURRENT_TARGET_YAW = 0.0
ROBOT_FACE = 1  # 1,3,5.. = X axis, 2,4,6.. = Y axis

# --- IMU Drift Compensation Parameters ---
IMU_COMPENSATION_START_NODE_COUNT = 7
IMU_COMPENSATION_NODE_INTERVAL = 10
IMU_COMPENSATION_DEG_PER_INTERVAL = -2.0
IMU_DRIFT_COMPENSATION_DEG = 0.0

# --- Occupancy Grid Parameters ---
PROB_OCC_GIVEN_OCC = {'tof': 0.95, 'sharp': 0.90}
PROB_OCC_GIVEN_FREE = {'tof': 0.05, 'sharp': 0.10}
LOG_ODDS_OCC = { 'tof': math.log(PROB_OCC_GIVEN_OCC['tof'] / (1-PROB_OCC_GIVEN_OCC['tof'])), 'sharp': math.log(PROB_OCC_GIVEN_OCC['sharp'] / (1-PROB_OCC_GIVEN_OCC['sharp'])) }
LOG_ODDS_FREE = { 'tof': math.log(PROB_OCC_GIVEN_FREE['tof'] / (1-PROB_OCC_GIVEN_FREE['tof'])), 'sharp': math.log(PROB_OCC_GIVEN_FREE['sharp'] / (1-PROB_OCC_GIVEN_FREE['sharp'])) }
OCCUPANCY_THRESHOLD = 0.7
FREE_THRESHOLD = 0.3

# --- Object Detection Configuration ---
TARGET_OBJECT_SHAPE = "Circle"
TARGET_OBJECT_COLOR = "Red"
NUMBER_OF_TARGETS_TO_FIND = 1
FRONT_WALL_SCAN_THRESHOLD_CM = 80.0
FOUND_OBJECTS = []

# =============================================================================
# ===== HELPER & IMAGE PROCESSING FUNCTIONS ===================================
# =============================================================================

def convert_adc_to_cm(adc_value):
    if adc_value <= 0: return float('inf')
    return 30263 * (adc_value ** -1.352)

def calibrate_tof_value(raw_tof_value):
    try:
        if raw_tof_value is None or raw_tof_value <= 0: return float('inf')
        return (TOF_CALIBRATION_SLOPE * raw_tof_value) + TOF_CALIBRATION_Y_INTERCEPT
    except Exception: return float('inf')

def get_compensated_target_yaw():
    return CURRENT_TARGET_YAW + IMU_DRIFT_COMPENSATION_DEG

def apply_awb(bgr):
    if hasattr(cv2, "xphoto") and hasattr(cv2.xphoto, "createLearningBasedWB"):
        wb = cv2.xphoto.createLearningBasedWB(); wb.setSaturationThreshold(0.99)
        return wb.balanceWhite(bgr)
    return bgr

def retinex_msrcp(bgr, sigmas=(15, 80, 250), eps=1e-6):
    img = bgr.astype(np.float32) + 1.0; intensity = img.mean(axis=2)
    log_I = np.log(intensity + eps); msr = np.zeros_like(intensity, dtype=np.float32)
    for s in sigmas: msr += (log_I - np.log(cv2.GaussianBlur(intensity, (0, 0), s) + eps))
    msr /= float(len(sigmas)); msr -= msr.min(); msr /= (msr.max() + eps)
    msr = (msr * 255.0).astype(np.float32); scale = (msr + 1.0) / (intensity + eps)
    return np.clip(img * scale[..., None], 0, 255).astype(np.uint8)

def night_enhance_pipeline(bgr):
    den = cv2.fastNlMeansDenoisingColored(bgr, None, 7, 7, 7, 21)
    awb = apply_awb(den)
    return retinex_msrcp(awb)

# =============================================================================
# ===== OCCUPANCY GRID MAP & VISUALIZATION (RESTORED & ENHANCED) ==============
# =============================================================================

class WallBelief:
    def __init__(self): self.log_odds = 0.0
    def update(self, is_occupied_reading, sensor_type):
        if is_occupied_reading: self.log_odds += LOG_ODDS_OCC[sensor_type]
        else: self.log_odds += LOG_ODDS_FREE[sensor_type]
        self.log_odds = max(min(self.log_odds, 10), -10)
    def get_probability(self): return 1.0 - 1.0 / (1.0 + math.exp(self.log_odds))
    def is_occupied(self): return self.get_probability() > OCCUPANCY_THRESHOLD

class OGMCell:
    def __init__(self):
        self.log_odds_occupied = 0.0; self.walls = {'N': None, 'E': None, 'S': None, 'W': None}
    def get_node_probability(self): return 1.0 - 1.0 / (1.0 + math.exp(self.log_odds_occupied))
    def is_node_occupied(self): return self.get_node_probability() > OCCUPANCY_THRESHOLD

class OccupancyGridMap:
    # (Unchanged from debug_img_3-10.py)
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.grid = [[OGMCell() for _ in range(width)] for _ in range(height)]
        self._link_walls()
    def _link_walls(self):
        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r][c].walls['N'] is None:
                    wall = WallBelief(); self.grid[r][c].walls['N'] = wall
                    if r > 0: self.grid[r-1][c].walls['S'] = wall
                if self.grid[r][c].walls['W'] is None:
                    wall = WallBelief(); self.grid[r][c].walls['W'] = wall
                    if c > 0: self.grid[r][c-1].walls['E'] = wall
                if self.grid[r][c].walls['S'] is None: self.grid[r][c].walls['S'] = WallBelief()
                if self.grid[r][c].walls['E'] is None: self.grid[r][c].walls['E'] = WallBelief()
    def update_wall(self, r, c, d_char, is_occ, s_type):
        if 0<=r<self.height and 0<=c<self.width:
            if wall:=self.grid[r][c].walls.get(d_char): wall.update(is_occ, s_type)
    def update_node(self, r, c, is_occ, s_type='tof'):
        if 0<=r<self.height and 0<=c<self.width:
            if is_occ: self.grid[r][c].log_odds_occupied += LOG_ODDS_OCC[s_type]
            else: self.grid[r][c].log_odds_occupied += LOG_ODDS_FREE[s_type]
    def is_path_clear(self, r1, c1, r2, c2):
        dr, dc = r2-r1, c2-c1
        if abs(dr)+abs(dc)!=1: return False
        if dr==-1: wall_char='N'
        elif dr==1: wall_char='S'
        elif dc==1: wall_char='E'
        elif dc==-1: wall_char='W'
        else: return False
        if wall:=self.grid[r1][c1].walls.get(wall_char):
            if wall.is_occupied(): return False
        if 0<=r2<self.height and 0<=c2<self.width:
            if self.grid[r2][c2].is_node_occupied(): return False
        else: return False
        return True

class RealTimeVisualizer:
    ### FIX 3: RESTORED FULL VISUALIZATION + ADDED OBJECTS ###
    def __init__(self, grid_size, target_dest=None):
        self.grid_size = grid_size
        self.target_dest = target_dest
        plt.ion()
        self.fig, (self.ax_map, self.ax_img) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [1.2, 1]})
        self.colors = {"robot": "#0000FF", "target_dest": "#FFD700", "path": "#FFFF00", "wall": "#000000", "wall_prob": "#000080"}
        self.last_image = None

    def update_image(self, image_bgr):
        self.ax_img.clear()
        if image_bgr is not None:
            self.last_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            self.ax_img.imshow(self.last_image)
        elif self.last_image is not None:
             self.ax_img.imshow(self.last_image)
        self.ax_img.set_title("Last Processed Camera View")
        self.ax_img.axis('off')

    def update_plot(self, occupancy_map, robot_pos, found_objects, path=None):
        self.ax_map.clear()
        self.ax_map.set_title("Real-time Hybrid Belief Map (Nodes & Walls)")
        self.ax_map.set_xticks([]); self.ax_map.set_yticks([])
        self.ax_map.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax_map.set_ylim(self.grid_size - 0.5, -0.5)
        
        # Draw Nodes with Probability Text (Restored from 3-10)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                prob = occupancy_map.grid[r][c].get_node_probability()
                if prob > OCCUPANCY_THRESHOLD: color = '#8B0000'
                elif prob < FREE_THRESHOLD: color = '#D3D3D3'
                else: color = '#90EE90'
                self.ax_map.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=color, edgecolor='k', lw=0.5))
                self.ax_map.text(c, r, f"{prob:.2f}", ha="center", va="center", color="black", fontsize=8)

        # Draw Walls with Probability Text (Restored from 3-10)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell = occupancy_map.grid[r][c]
                prob_n = cell.walls['N'].get_probability()
                if abs(prob_n - 0.5) > 0.01: self.ax_map.text(c, r - 0.5, f"{prob_n:.2f}", ha="center", va="center", color=self.colors['wall_prob'], fontsize=6, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1', edgecolor='none'))
                prob_w = cell.walls['W'].get_probability()
                if abs(prob_w - 0.5) > 0.01: self.ax_map.text(c - 0.5, r, f"{prob_w:.2f}", ha="center", va="center", color=self.colors['wall_prob'], fontsize=6, rotation=90, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1', edgecolor='none'))
        # Draw edge walls
        for c in range(self.grid_size):
            r_edge = self.grid_size - 1; prob_s = occupancy_map.grid[r_edge][c].walls['S'].get_probability()
            if abs(prob_s - 0.5) > 0.01: self.ax_map.text(c, r_edge + 0.5, f"{prob_s:.2f}", ha="center", va="center", color=self.colors['wall_prob'], fontsize=6, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1', edgecolor='none'))
        for r in range(self.grid_size):
            c_edge = self.grid_size - 1; prob_e = occupancy_map.grid[r][c_edge].walls['E'].get_probability()
            if abs(prob_e - 0.5) > 0.01: self.ax_map.text(c_edge + 0.5, r, f"{prob_e:.2f}", ha="center", va="center", color=self.colors['wall_prob'], fontsize=6, rotation=90, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1', edgecolor='none'))
        
        # Draw Occupied Walls (Restored from 3-10)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell = occupancy_map.grid[r][c]
                if cell.walls['N'].is_occupied(): self.ax_map.plot([c-0.5, c+0.5], [r-0.5, r-0.5], color=self.colors['wall'], linewidth=4)
                if cell.walls['W'].is_occupied(): self.ax_map.plot([c-0.5, c-0.5], [r-0.5, r+0.5], color=self.colors['wall'], linewidth=4)
                if r==self.grid_size-1 and cell.walls['S'].is_occupied(): self.ax_map.plot([c-0.5,c+0.5],[r+0.5,r+0.5],color=self.colors['wall'],linewidth=4)
                if c==self.grid_size-1 and cell.walls['E'].is_occupied(): self.ax_map.plot([c+0.5,c+0.5],[r-0.5,r+0.5],color=self.colors['wall'],linewidth=4)

        if self.target_dest: self.ax_map.add_patch(plt.Rectangle((self.target_dest[1]-0.5, self.target_dest[0]-0.5), 1, 1, facecolor=self.colors['target_dest'], edgecolor='k', lw=2, alpha=0.8))
        if path:
            for r_p, c_p in path: self.ax_map.add_patch(plt.Rectangle((c_p-0.5, r_p-0.5), 1, 1, facecolor=self.colors['path'], edgecolor='k', lw=0.5, alpha=0.7))

        # Draw found objects on the map (Enhanced)
        for obj in found_objects:
            r_o, c_o = obj['pos']
            if obj['type'] == 'target': self.ax_map.plot(c_o, r_o, marker='*', markersize=15, color='magenta', markeredgecolor='black')
            elif obj['type'] == 'confident': self.ax_map.plot(c_o, r_o, marker='o', markersize=10, color=obj['color'].lower(), markeredgecolor='black')
            elif obj['type'] == 'uncertain': self.ax_map.text(c_o, r_o, '?', ha='center', va='center', fontsize=12, color='orange', weight='bold')

        if robot_pos: self.ax_map.add_patch(plt.Rectangle((robot_pos[1]-0.5, robot_pos[0]-0.5), 1, 1, facecolor=self.colors['robot'], edgecolor='k', lw=2))

        # Full Legend (Restored and Enhanced)
        legend_elements = [
            plt.Rectangle((0,0),1,1,facecolor='#8B0000',label=f'Node Occupied (P>{OCCUPANCY_THRESHOLD})'),
            plt.Rectangle((0,0),1,1,facecolor='#90EE90',label=f'Node Unknown'),
            plt.Rectangle((0,0),1,1,facecolor='#D3D3D3',label=f'Node Free (P<{FREE_THRESHOLD})'),
            plt.Line2D([0],[0],color=self.colors['wall'],lw=4,label='Wall Occupied'),
            plt.Rectangle((0,0),1,1,facecolor=self.colors['robot'],label='Robot'),
            plt.Rectangle((0,0),1,1,facecolor=self.colors['target_dest'],label='Final Destination'),
            plt.Line2D([0],[0],marker='*',color='w',label=f'Target Object ({TARGET_OBJECT_COLOR} {TARGET_OBJECT_SHAPE})',markersize=15,markerfacecolor='magenta'),
            plt.Line2D([0],[0],marker='o',color='w',label='Confident Object (by color)',markersize=10,markerfacecolor='gray'),
            plt.Line2D([0],[0],marker=r'$?$',color='w',label='Uncertain Object',markersize=15,markerfacecolor='orange'),
        ]
        self.ax_map.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1.0))
        self.fig.tight_layout()
        self.fig.canvas.draw(); self.fig.canvas.flush_events(); plt.pause(0.01)


# =============================================================================
# ===== CORE ROBOT CONTROL CLASSES (UNCHANGED from 3-10) ======================
# =============================================================================
class AttitudeHandler:
    def __init__(self): self.current_yaw, self.yaw_tolerance, self.is_monitoring = 0.0, 3.0, False
    def attitude_handler(self, attitude_info):
        if self.is_monitoring: self.current_yaw = attitude_info[0]
    def start_monitoring(self, chassis): self.is_monitoring=True; chassis.sub_attitude(freq=20, callback=self.attitude_handler)
    def stop_monitoring(self, chassis): self.is_monitoring=False;
        try: 
            chassis.unsub_attitude()
        except Exception: pass
    def normalize_angle(self, angle):
        while angle>180: angle-=360
        while angle<=-180: angle+=360
        return angle
    def correct_yaw_to_target(self, chassis, target_yaw=0.0):
        normalized_target=self.normalize_angle(target_yaw); time.sleep(0.1)
        robot_rotation=-self.normalize_angle(normalized_target - self.current_yaw)
        print(f"\n🔧 Correcting Yaw: {self.current_yaw:.1f}° -> {target_yaw:.1f}°. Rotating: {robot_rotation:.1f}°")
        if abs(robot_rotation)>self.yaw_tolerance: chassis.move(x=0,y=0,z=robot_rotation,z_speed=60).wait_for_completed(timeout=2); time.sleep(0.1)
        final_error=abs(self.normalize_angle(normalized_target-self.current_yaw))
        if final_error<=self.yaw_tolerance: print(f"✅ Yaw Correction Success: {self.current_yaw:.1f}°"); return True
        print(f"⚠️ First attempt incomplete. Current: {self.current_yaw:.1f}°. Fine-tuning...")
        remaining_rotation=-self.normalize_angle(normalized_target-self.current_yaw)
        if abs(remaining_rotation)>0.5 and abs(remaining_rotation)<20: chassis.move(x=0,y=0,z=remaining_rotation,z_speed=40).wait_for_completed(timeout=2); time.sleep(0.1)
        final_error=abs(self.normalize_angle(normalized_target-self.current_yaw))
        if final_error<=self.yaw_tolerance: print(f"✅ Yaw Fine-tuning Success: {self.current_yaw:.1f}°"); return True
        else: print(f"🔥🔥 Yaw Correction FAILED. Final Yaw: {self.current_yaw:.1f}°"); return False

class PID:
    def __init__(self, Kp, Ki, Kd, setpoint=0): self.Kp, self.Ki, self.Kd, self.setpoint = Kp, Ki, Kd, setpoint; self.prev_error, self.integral, self.integral_max = 0, 0, 1.0
    def compute(self, current, dt):
        error = self.setpoint-current; self.integral += error*dt; self.integral=max(min(self.integral,self.integral_max),-self.integral_max)
        derivative=(error-self.prev_error)/dt if dt>0 else 0; output=self.Kp*error+self.Ki*self.integral+self.Kd*derivative
        self.prev_error=error; return output

class MovementController:
    def __init__(self, chassis): self.chassis=chassis; self.current_x_pos,self.current_y_pos=0.0,0.0; self.chassis.sub_position(freq=20,callback=self.position_handler)
    def position_handler(self, position_info): self.current_x_pos, self.current_y_pos = position_info[0], position_info[1]
    def _calculate_yaw_correction(self, attitude_handler, target_yaw):
        KP_YAW=1.8; MAX_YAW_SPEED=25; yaw_error=attitude_handler.normalize_angle(target_yaw-attitude_handler.current_yaw)
        speed=KP_YAW*yaw_error; return max(min(speed, MAX_YAW_SPEED),-MAX_YAW_SPEED)
    def move_forward_one_grid(self, axis, attitude_handler):
        attitude_handler.correct_yaw_to_target(self.chassis,get_compensated_target_yaw())
        target_distance=0.6; pid=PID(Kp=1.8,Ki=0.25,Kd=12,setpoint=target_distance); start_time,last_time=time.time(),time.time()
        start_position=self.current_x_pos if axis=='x' else self.current_y_pos
        print(f"🚀 Moving FORWARD 0.6m, monitoring GLOBAL AXIS '{axis}'")
        while time.time()-start_time<3.5:
            now=time.time(); dt=now-last_time; last_time=now
            current_position=self.current_x_pos if axis=='x' else self.current_y_pos
            relative_position=abs(current_position-start_position)
            if abs(relative_position-target_distance)<0.03: print("\n✅ Move complete!"); break
            output=pid.compute(relative_position,dt); ramp_multiplier=min(1.0,0.1+((now-start_time)/1.0)*0.9)
            speed=max(-1.0,min(1.0,output*ramp_multiplier))
            yaw_correction=self._calculate_yaw_correction(attitude_handler,get_compensated_target_yaw())
            self.chassis.drive_speed(x=speed,y=0,z=yaw_correction,timeout=1)
            print(f"Moving... Dist: {relative_position:.3f}/{target_distance:.2f} m",end='\r')
        self.chassis.drive_wheels(w1=0,w2=0,w3=0,w4=0); time.sleep(0.5)
    def adjust_position_to_wall(self, sensor_adaptor, attitude_handler, side, sensor_config, target_distance_cm, direction_multiplier):
        compensated_yaw=get_compensated_target_yaw(); print(f"\n--- Adjusting {side} Side (Yaw locked at {compensated_yaw:.2f}°) ---")
        TOLERANCE_CM,MAX_EXEC_TIME,KP_SLIDE,MAX_SLIDE_SPEED=0.8,8,0.045,0.18; start_time=time.time()
        while time.time()-start_time<MAX_EXEC_TIME:
            adc_val=sensor_adaptor.get_adc(id=sensor_config["sharp_id"],port=sensor_config["sharp_port"]); current_dist=convert_adc_to_cm(adc_val)
            dist_error=target_distance_cm-current_dist
            if abs(dist_error)<=TOLERANCE_CM: print(f"\n[{side}] Target distance reached! Final distance: {current_dist:.2f} cm"); break
            slide_speed=max(min(direction_multiplier*KP_SLIDE*dist_error,MAX_SLIDE_SPEED),-MAX_SLIDE_SPEED)
            yaw_correction=self._calculate_yaw_correction(attitude_handler,compensated_yaw)
            self.chassis.drive_speed(x=0,y=slide_speed,z=yaw_correction)
            print(f"Adjusting {side}... Current: {current_dist:5.2f}cm, Target: {target_distance_cm:4.1f}cm, Error: {dist_error:5.2f}cm, Speed: {slide_speed:5.3f}",end='\r')
            time.sleep(0.02)
        else: print(f"\n[{side}] Movement timed out!")
        self.chassis.drive_wheels(w1=0,w2=0,w3=0,w4=0); time.sleep(0.1)
    def center_in_node_with_tof(self, scanner, attitude_handler, target_cm=19, tol_cm=1.0, max_adjust_time=6.0):
        if scanner.is_performing_full_scan: print("[ToF Centering] SKIPPED: A critical side-scan is in progress."); return
        print("\n--- Stage: Centering in Node with ToF ---"); time.sleep(0.1)
        tof_dist=scanner.last_tof_distance_cm
        if tof_dist is None or math.isinf(tof_dist): print("[ToF] ❌ No valid ToF data available. Skipping."); return
        print(f"[ToF] Initial front distance: {tof_dist:.2f} cm")
        if tof_dist>=50: print("[ToF] Distance >= 50cm, likely open space. Skipping centering."); return
        direction=0
        if tof_dist>target_cm+tol_cm: direction=abs(TOF_ADJUST_SPEED)
        elif tof_dist<22: direction=-abs(TOF_ADJUST_SPEED)
        else: direction=abs(TOF_ADJUST_SPEED)
        if direction==0: print(f"[ToF] Already centered ({tof_dist:.2f} cm). Skipping."); return
        start_time=time.time(); compensated_yaw=get_compensated_target_yaw()
        while time.time()-start_time<max_adjust_time:
            yaw_correction=self._calculate_yaw_correction(attitude_handler,compensated_yaw)
            self.chassis.drive_speed(x=direction,y=0,z=yaw_correction,timeout=0.1); time.sleep(0.12)
            self.chassis.drive_wheels(w1=0,w2=0,w3=0,w4=0); time.sleep(0.08)
            current_tof=scanner.last_tof_distance_cm
            if current_tof is None or math.isinf(current_tof): continue
            print(f"[ToF] Adjusting... Current Distance: {current_tof:.2f} cm",end="\r")
            if abs(current_tof-target_cm)<=tol_cm: print(f"\n[ToF] ✅ Centering complete. Final distance: {current_tof:.2f} cm"); break
            if (direction>0 and current_tof<target_cm-tol_cm) or (direction<0 and current_tof>target_cm+tol_cm): direction*=-1; print("\n[ToF] 🔄 Overshot target. Reversing direction.")
        else: print(f"\n[ToF] ⚠️ Centering timed out. Final distance: {scanner.last_tof_distance_cm:.2f} cm")
        self.chassis.drive_wheels(w1=0,w2=0,w3=0,w4=0); time.sleep(0.1)
    def rotate_to_direction(self, target_direction, attitude_handler):
        global CURRENT_DIRECTION
        if CURRENT_DIRECTION==target_direction: return
        diff=(target_direction-CURRENT_DIRECTION+4)%4
        if diff==1: self.rotate_90_degrees_right(attitude_handler)
        elif diff==3: self.rotate_90_degrees_left(attitude_handler)
        elif diff==2: self.rotate_90_degrees_right(attitude_handler); self.rotate_90_degrees_right(attitude_handler)
    def rotate_90_degrees_right(self, attitude_handler):
        global CURRENT_TARGET_YAW,CURRENT_DIRECTION,ROBOT_FACE
        print("🔄 Rotating 90° RIGHT..."); CURRENT_TARGET_YAW=attitude_handler.normalize_angle(CURRENT_TARGET_YAW+90)
        attitude_handler.correct_yaw_to_target(self.chassis,get_compensated_target_yaw())
        CURRENT_DIRECTION=(CURRENT_DIRECTION+1)%4; ROBOT_FACE+=1
    def rotate_90_degrees_left(self, attitude_handler):
        global CURRENT_TARGET_YAW,CURRENT_DIRECTION,ROBOT_FACE
        print("🔄 Rotating 90° LEFT..."); CURRENT_TARGET_YAW=attitude_handler.normalize_angle(CURRENT_TARGET_YAW-90)
        attitude_handler.correct_yaw_to_target(self.chassis,get_compensated_target_yaw())
        CURRENT_DIRECTION=(CURRENT_DIRECTION-1+4)%4; ROBOT_FACE-=1
        if ROBOT_FACE<1: ROBOT_FACE+=4
    def cleanup(self):
        try: self.chassis.unsub_position()
        except Exception: pass

class EnvironmentScanner:
    def __init__(self, sensor_adaptor, tof_sensor, gimbal, chassis):
        self.sensor_adaptor,self.tof_sensor,self.gimbal,self.chassis = sensor_adaptor,tof_sensor,gimbal,chassis
        self.tof_wall_threshold_cm=60.0; self.last_tof_distance_cm=float('inf'); self.side_tof_reading_cm=float('inf')
        self.is_gimbal_centered=True; self.is_performing_full_scan=False
        self.tof_sensor.sub_distance(freq=10,callback=self._tof_data_handler)
        self.side_sensors={"Left":{"sharp_id":1,"sharp_port":1,"ir_id":1,"ir_port":2},"Right":{"sharp_id":2,"sharp_port":1,"ir_id":2,"ir_port":2}}
    def _tof_data_handler(self, sub_info):
        calibrated_cm=calibrate_tof_value(sub_info[0])
        if self.is_gimbal_centered: self.last_tof_distance_cm=calibrated_cm
        else: self.side_tof_reading_cm=calibrated_cm
    def _get_stable_reading_cm(self, side, duration=0.35):
        sensor_info=self.side_sensors.get(side); readings=[]
        start_time=time.time()
        while time.time()-start_time<duration:
            adc=self.sensor_adaptor.get_adc(id=sensor_info["sharp_id"],port=sensor_info["sharp_port"])
            readings.append(convert_adc_to_cm(adc)); time.sleep(0.05)
        if len(readings)<5: return None,None
        return statistics.mean(readings),statistics.stdev(readings)
    def get_sensor_readings(self):
        self.is_performing_full_scan=True
        try:
            self.gimbal.moveto(pitch=0,yaw=0,yaw_speed=SPEED_ROTATE).wait_for_completed(); time.sleep(0.15)
            readings={}; readings['front']=(self.last_tof_distance_cm<self.tof_wall_threshold_cm)
            print(f"[SCAN] Front (ToF): {self.last_tof_distance_cm:.1f}cm -> {'OCCUPIED' if readings['front'] else 'FREE'}")
            for side in ["Left","Right"]:
                avg_dist,std_dev=self._get_stable_reading_cm(side)
                if avg_dist is None: readings[side.lower()]=False; continue
                is_sharp=(avg_dist<SHARP_WALL_THRESHOLD_CM and std_dev<SHARP_STDEV_THRESHOLD)
                is_ir=(self.sensor_adaptor.get_io(id=self.side_sensors[side]["ir_id"],port=self.side_sensors[side]["ir_port"])==0)
                if is_sharp==is_ir: is_wall=is_sharp
                else:
                    target_gimbal_yaw=-90 if side=="Left" else 90
                    try:
                        self.is_gimbal_centered=False
                        self.gimbal.moveto(pitch=0,yaw=target_gimbal_yaw,yaw_speed=SPEED_ROTATE).wait_for_completed(); time.sleep(0.1)
                        is_wall=(self.side_tof_reading_cm<self.tof_wall_threshold_cm)
                    finally: self.gimbal.moveto(pitch=0,yaw=0,yaw_speed=SPEED_ROTATE).wait_for_completed(); self.is_gimbal_centered=True; time.sleep(0.1)
                readings[side.lower()]=is_wall
            return readings
        finally: self.is_performing_full_scan=False
    def get_front_tof_cm(self):
        self.gimbal.moveto(pitch=0,yaw=0,yaw_speed=SPEED_ROTATE).wait_for_completed(); time.sleep(0.1)
        return self.last_tof_distance_cm
    def cleanup(self):
        try: self.tof_sensor.unsub_distance()
        except Exception: pass


# =============================================================================
# ===== OBJECT DETECTION CLASSES ==============================================
# =============================================================================

class ObjectTracker:
    def __init__(self):
        print("🖼️  Loading and processing template images...")
        ### FIX 1: CORRECTED TEMPLATE FILE PATHS ###
        template_paths = {
            "Circle": "./Assignment/image_processing/template/circle1.png",
            "Square": "./Assignment/image_processing/template/square.png",
            "Rectangle_H": "./Assignment/image_processing/template/rec_h.png",
            "Rectangle_V": "./Assignment/image_processing/template/rec_v.png",
        }
        self.templates = self._load_templates(template_paths)
        if not self.templates: print("❌ Could not load template files. Shape matching will be degraded.")
        else: print("✅ Templates loaded successfully.")
    def _load_templates(self, template_paths):
        processed_templates={};
        for shape_name,path in template_paths.items():
            template_img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            if template_img is None: continue
            _,thresh=cv2.threshold(cv2.GaussianBlur(template_img,(5,5),0),127,255,cv2.THRESH_BINARY_INV)
            contours,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            if contours: processed_templates[shape_name]=max(contours,key=cv2.contourArea)
        return processed_templates
    def _get_angle(self, pt1, pt2, pt0):
        dx1=pt1[0]-pt0[0]; dy1=pt1[1]-pt0[1]; dx2=pt2[0]-pt0[0]; dy2=pt2[1]-pt0[1]; dot=dx1*dx2+dy1*dy2
        mag1=math.sqrt(dx1*dx1+dy1*dy1); mag2=math.sqrt(dx2*dx2+dy2*dy2)
        if mag1*mag2==0: return 0
        return math.degrees(math.acos(dot/(mag1*mag2)))
    def _get_raw_detections(self, frame):
        enhanced=night_enhance_pipeline(frame)
        hsv=cv2.cvtColor(cv2.GaussianBlur(enhanced,(5,5),0),cv2.COLOR_BGR2HSV); h,s,v=cv2.split(hsv)
        v_eq=cv2.createCLAHE(clipLimit=2.5,tileGridSize=(8,8)).apply(v); normalized_hsv=cv2.merge([h,s,v_eq])
        ranges={'Red':[np.array([0,80,40]),np.array([10,255,255]),np.array([170,80,40]),np.array([180,255,255])],'Yellow':[np.array([20,60,40]),np.array([35,255,255])],'Green':[np.array([35,40,30]),np.array([85,255,255])],'Blue':[np.array([90,40,30]),np.array([130,255,255])]}
        masks={}; l1,u1,l2,u2=ranges['Red']; masks['Red']=cv2.inRange(normalized_hsv,l1,u1)|cv2.inRange(normalized_hsv,l2,u2)
        for name in ['Yellow','Green','Blue']: masks[name]=cv2.inRange(normalized_hsv,ranges[name][0],ranges[name][1])
        combined=masks['Red']|masks['Yellow']|masks['Green']|masks['Blue']
        kernel=np.ones((5,5),np.uint8); cleaned=cv2.morphologyEx(cv2.morphologyEx(combined,cv2.MORPH_OPEN,kernel,iterations=1),cv2.MORPH_CLOSE,kernel,iterations=2)
        contours,_=cv2.findContours(cleaned,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); raw_detections=[]
        for cnt in contours:
            if cv2.contourArea(cnt)<1500: continue
            contour_mask=np.zeros(frame.shape[:2],dtype="uint8"); cv2.drawContours(contour_mask,[cnt],-1,255,-1)
            max_mean,found_color=0,"Unknown"
            for color_name,m in masks.items():
                mean_val=cv2.mean(m,mask=contour_mask)[0]
                if mean_val>max_mean: max_mean,found_color=mean_val,color_name
            if max_mean<=20: continue
            shape="Uncertain"; peri=cv2.arcLength(cnt,True); approx=cv2.approxPolyDP(cnt,0.04*peri,True)
            if (4*np.pi*cv2.contourArea(cnt))/(peri*peri if peri>0 else 1)>0.84: shape="Circle"
            elif len(approx)==4:
                pts=[tuple(p[0]) for p in approx]; angles=[self._get_angle(pts[(i-1)%4],pts[(i+1)%4],p) for i,p in enumerate(pts)]
                if all(75<=a<=105 for a in angles):
                    _,(w,h),_=cv2.minAreaRect(cnt); aspect=max(w,h)/min(w,h) if min(w,h)>0 else 0
                    shape="Square" if 0.9<=aspect<=1.1 else "Rectangle_H" if w<h else "Rectangle_V"
            raw_detections.append({'contour':cnt,'shape':shape,'color':found_color})
        return raw_detections

class ObjectDetectionHandler:
    def __init__(self):
        self.tracker=ObjectTracker(); self.thread=None; self.last_results=None; self.last_processed_image=None; self.is_running=False
    def process_image_async(self, frame, tof_distance):
        if self.is_running: print("📸 Detection thread busy. Skipping frame."); return
        self.is_running=True; self.thread=threading.Thread(target=self._worker_process_image,args=(frame,tof_distance)); self.thread.start()
    def _worker_process_image(self, frame, tof_distance):
        print(f"🧠 [Thread] Starting detection. ToF: {tof_distance:.1f} cm")
        try:
            processed_frame=frame[352:352+360,14:14+1215]
            output_image,detections=self._analyze_frame(processed_frame,tof_distance)
            self.last_processed_image=output_image; self.last_results=detections
            print(f"🧠 [Thread] Detection complete. Found {len(detections)} potential objects.")
        except Exception as e: print(f"Error in detection thread: {e}"); self.last_processed_image=frame; self.last_results=[]
        finally: self.is_running=False
    def _analyze_frame(self, frame, tof_distance):
        output_image=frame.copy(); height,width,_=frame.shape
        DIVIDER_X1,DIVIDER_X2=400,800
        cv2.line(output_image,(DIVIDER_X1,0),(DIVIDER_X1,height),(255,255,0),2); cv2.line(output_image,(DIVIDER_X2,0),(DIVIDER_X2,height),(255,255,0),2)
        raw_detections=self.tracker._get_raw_detections(frame); final_detections=[]
        ignore_center=(tof_distance>=FRONT_WALL_SCAN_THRESHOLD_CM)
        if ignore_center: print("   -> ToF >= 80cm. IGNORING center detections.")
        for det in raw_detections:
            shape,color,contour=det['shape'],det['color'],det['contour']; x,y,w,h=cv2.boundingRect(contour)
            obj_start_x,obj_end_x=x,x+w
            zone_label="Unknown"
            if obj_end_x<DIVIDER_X1: zone_label="Left"
            elif obj_start_x>=DIVIDER_X2: zone_label="Right"
            elif obj_start_x>=DIVIDER_X1 and obj_end_x<DIVIDER_X2: zone_label="Center"
            if ignore_center and zone_label=="Center": print(f"      -> Skipping {color} {shape} in Center zone."); continue
            is_target=(shape==TARGET_OBJECT_SHAPE and color==TARGET_OBJECT_COLOR)
            box_color=(0,0,255) if is_target else (0,255,255) if shape=="Uncertain" else (0,255,0)
            cv2.rectangle(output_image,(x,y),(x+w,y+h),box_color,3)
            cv2.putText(output_image,f"{color} {shape}",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,box_color,2)
            final_detections.append({'shape':shape,'color':color,'zone':zone_label})
        return output_image,final_detections
    def get_latest_results(self):
        if not self.is_running and self.thread is not None and not self.thread.is_alive():
            results=(self.last_processed_image,self.last_results); self.last_processed_image=None; self.last_results=None; return results
        return None,None

# =============================================================================
# ===== PATHFINDING & EXPLORATION LOGIC =======================================
# =============================================================================

def find_path_bfs(occupancy_map, start, end):
    queue=deque([[start]]); visited={start}; moves=[(-1,0),(0,1),(1,0),(0,-1)]
    while queue:
        path=queue.popleft(); r,c=path[-1]
        if (r,c)==end: return path
        for dr,dc in moves:
            nr,nc=r+dr,c+dc
            if 0<=nr<occupancy_map.height and 0<=nc<occupancy_map.width:
                if occupancy_map.is_path_clear(r,c,nr,nc) and (nr,nc) not in visited:
                    visited.add((nr,nc)); new_path=list(path); new_path.append((nr,nc)); queue.append(new_path)
    return None

def find_nearest_unvisited_path(occupancy_map, start_pos, visited_cells):
    h,w=occupancy_map.height,occupancy_map.width
    unvisited=[(r,c) for r in range(h) for c in range(w) if (r,c) not in visited_cells and not occupancy_map.grid[r][c].is_node_occupied()]
    if not unvisited: return None
    shortest_path=None
    for target in unvisited:
        path=find_path_bfs(occupancy_map,start_pos,target)
        if path and (shortest_path is None or len(path)<len(shortest_path)): shortest_path=path
    return shortest_path

def execute_path(path, movement_controller, attitude_handler, scanner, visualizer, occupancy_map, path_name="Backtrack"):
    global CURRENT_POSITION
    print(f"🎯 Executing {path_name} Path: {path}"); dir_map={(-1,0):0,(0,1):1,(1,0):2,(0,-1):3}; dir_char={0:'N',1:'E',2:'S',3:'W'}
    for i in range(len(path)-1):
        visualizer.update_plot(occupancy_map,path[i],FOUND_OBJECTS,path)
        cr,cc=path[i]; nr,nc=path[i+1]; dr,dc=nr-cr,nc-cc
        target_dir=dir_map[(dr,dc)]; movement_controller.rotate_to_direction(target_dir,attitude_handler)
        is_blocked=scanner.get_front_tof_cm()<scanner.tof_wall_threshold_cm
        occupancy_map.update_wall(cr,cc,dir_char[CURRENT_DIRECTION],is_blocked,'tof')
        visualizer.update_plot(occupancy_map,CURRENT_POSITION,FOUND_OBJECTS)
        if is_blocked: print(f"   -> 🔥 [{path_name}] OBSTACLE DETECTED. Aborting path."); break
        axis='x' if ROBOT_FACE%2!=0 else 'y'
        movement_controller.move_forward_one_grid(axis=axis,attitude_handler=attitude_handler)
        movement_controller.center_in_node_with_tof(scanner,attitude_handler)
        CURRENT_POSITION=(nr,nc)
        visualizer.update_plot(occupancy_map,CURRENT_POSITION,FOUND_OBJECTS,path)
        perform_side_alignment_and_mapping(movement_controller,scanner,attitude_handler,occupancy_map,visualizer)
    print(f"✅ {path_name} complete.")

def perform_side_alignment_and_mapping(movement_controller, scanner, attitude_handler, occupancy_map, visualizer):
    print("\n--- Stage: Wall Detection & Side Alignment ---"); r,c=CURRENT_POSITION; d_char={0:'N',1:'E',2:'S',3:'W'}
    side_walls=scanner.get_sensor_readings()
    left_dir=(CURRENT_DIRECTION-1+4)%4; occupancy_map.update_wall(r,c,d_char[left_dir],side_walls['left'],'sharp')
    visualizer.update_plot(occupancy_map,CURRENT_POSITION,FOUND_OBJECTS)
    if side_walls['left']: movement_controller.adjust_position_to_wall(scanner.sensor_adaptor,attitude_handler,"Left",scanner.side_sensors["Left"],LEFT_TARGET_CM,1)
    right_dir=(CURRENT_DIRECTION+1)%4; occupancy_map.update_wall(r,c,d_char[right_dir],side_walls['right'],'sharp')
    visualizer.update_plot(occupancy_map,CURRENT_POSITION,FOUND_OBJECTS)
    if side_walls['right']: movement_controller.adjust_position_to_wall(scanner.sensor_adaptor,attitude_handler,"Right",scanner.side_sensors["Right"],RIGHT_TARGET_CM,-1)
    if not side_walls['left'] and not side_walls['right']: print("\n⚠️ No side walls detected. Skipping alignment.")
    attitude_handler.correct_yaw_to_target(movement_controller.chassis,get_compensated_target_yaw()); time.sleep(0.1)

def explore_with_ogm(ep_robot, scanner, movement_controller, attitude_handler, occupancy_map, visualizer, max_steps=40):
    global CURRENT_POSITION, CURRENT_DIRECTION, IMU_DRIFT_COMPENSATION_DEG, FOUND_OBJECTS
    detection_handler=ObjectDetectionHandler(); visited_cells=set()
    
    for step in range(max_steps):
        r,c=CURRENT_POSITION
        print(f"\n--- Step {step+1} at {CURRENT_POSITION}, Facing: {['N','E','S','W'][CURRENT_DIRECTION]} ---")
        
        # 1. Process Results from PREVIOUS step's detection
        processed_image,new_detections=detection_handler.get_latest_results()
        if new_detections is not None:
            visualizer.update_image(processed_image)
            
            ### FIX 2: CORRECTED OBJECT MAPPING LOGIC ###
            print(f"   -> Mapping {len(new_detections)} detections relative to {CURRENT_POSITION}")
            dir_vectors = [(-1,0), (0,1), (1,0), (0,-1)] # N, E, S, W
            
            front_vec = dir_vectors[CURRENT_DIRECTION]
            left_vec = dir_vectors[(CURRENT_DIRECTION - 1 + 4) % 4]
            right_vec = dir_vectors[(CURRENT_DIRECTION + 1) % 4]

            for det in new_detections:
                target_pos = None
                if det['zone'] == 'Center': target_pos = (r + front_vec[0], c + front_vec[1])
                elif det['zone'] == 'Left': target_pos = (r + left_vec[0], c + left_vec[1])
                elif det['zone'] == 'Right': target_pos = (r + right_vec[0], c + right_vec[1])
                
                if target_pos:
                    obj_type = "uncertain" if det['shape'] == "Uncertain" else "confident"
                    if det['shape'] == TARGET_OBJECT_SHAPE and det['color'] == TARGET_OBJECT_COLOR: obj_type = "target"
                    
                    if obj_type == 'uncertain' and det['zone'] == 'Center':
                        print(f"      -> Discarding 'Uncertain' object in Center zone.")
                        continue
                    
                    new_obj={'pos':target_pos,'shape':det['shape'],'color':det['color'],'type':obj_type}
                    is_duplicate=any(d['pos']==new_obj['pos'] and d['shape']==new_obj['shape'] and d['color']==new_obj['color'] for d in FOUND_OBJECTS)
                    if not is_duplicate:
                        FOUND_OBJECTS.append(new_obj)
                        print(f"      -> Added {obj_type} object: {det['color']} {det['shape']} at {target_pos}")

        # 2. Standard movement and mapping logic
        attitude_handler.correct_yaw_to_target(movement_controller.chassis,get_compensated_target_yaw())
        perform_side_alignment_and_mapping(movement_controller,scanner,attitude_handler,occupancy_map,visualizer)
        is_front_occupied=scanner.get_front_tof_cm()<scanner.tof_wall_threshold_cm
        occupancy_map.update_wall(r,c,{0:'N',1:'E',2:'S',3:'W'}[CURRENT_DIRECTION],is_front_occupied,'tof')
        occupancy_map.update_node(r,c,False,'tof'); visited_cells.add((r,c))
        visualizer.update_plot(occupancy_map,CURRENT_POSITION,FOUND_OBJECTS)

        # 3. Start Object Detection for the CURRENT view
        num_targets_found=sum(1 for obj in FOUND_OBJECTS if obj['type']=='target')
        if num_targets_found<NUMBER_OF_TARGETS_TO_FIND:
            if not detection_handler.is_running:
                print("\n--- Stage: Object Detection Scan ---"); front_tof=scanner.get_front_tof_cm()
                print("   -> Capturing frame for analysis...")
                frame=ep_robot.camera.read_cv2_image(timeout=5)
                if frame is not None: detection_handler.process_image_async(frame,front_tof)
                else: print("   -> ❌ Failed to capture frame.")
            else: print("\n📸 Skipping detection, previous still processing.")
        else: print("\n🎉 All target objects found! Disabling camera scans.")
        
        # 4. Decide next move
        priority_dirs=[(CURRENT_DIRECTION+1)%4,CURRENT_DIRECTION,(CURRENT_DIRECTION-1+4)%4]; moved=False
        for target_dir in priority_dirs:
            tr,tc=r+dir_vectors[target_dir][0],c+dir_vectors[target_dir][1]
            if occupancy_map.is_path_clear(r,c,tr,tc) and (tr,tc) not in visited_cells:
                movement_controller.rotate_to_direction(target_dir,attitude_handler)
                scanner.gimbal.moveto(pitch=0,yaw=0,yaw_speed=SPEED_ROTATE).wait_for_completed(); time.sleep(0.1)
                is_blocked=scanner.get_front_tof_cm()<scanner.tof_wall_threshold_cm
                occupancy_map.update_wall(r,c,{0:'N',1:'E',2:'S',3:'W'}[CURRENT_DIRECTION],is_blocked,'tof')
                visualizer.update_plot(occupancy_map,CURRENT_POSITION,FOUND_OBJECTS)
                if occupancy_map.is_path_clear(r,c,tr,tc):
                    axis='x' if ROBOT_FACE%2!=0 else 'y'
                    movement_controller.move_forward_one_grid(axis=axis,attitude_handler=attitude_handler)
                    movement_controller.center_in_node_with_tof(scanner,attitude_handler)
                    CURRENT_POSITION=(tr,tc); moved=True; break
                else: print(f"    Confirmation failed. Path to {['N','E','S','W'][target_dir]} is blocked.")
        
        if not moved:
            print("No immediate unvisited path. Initiating backtrack...")
            backtrack_path=find_nearest_unvisited_path(occupancy_map,CURRENT_POSITION,visited_cells)
            if backtrack_path and len(backtrack_path)>1:
                execute_path(backtrack_path,movement_controller,attitude_handler,scanner,visualizer,occupancy_map)
                print("Backtrack complete. Resuming exploration."); continue
            else: print("🎉 EXPLORATION COMPLETE!"); break
    
    print("\n🎉 === EXPLORATION PHASE FINISHED ===")

# =============================================================================
# ===== MAIN EXECUTION BLOCK ==================================================
# =============================================================================
if __name__ == '__main__':
    ep_robot=None; occupancy_map=OccupancyGridMap(width=6,height=6)
    attitude_handler=AttitudeHandler(); movement_controller=None; scanner=None
    
    try:
        visualizer=RealTimeVisualizer(grid_size=6,target_dest=TARGET_DESTINATION)
        print("🤖 Connecting to robot..."); ep_robot=robot.Robot(); ep_robot.initialize(conn_type="ap")
        ep_chassis,ep_gimbal,ep_camera=ep_robot.chassis,ep_robot.gimbal,ep_robot.camera
        ep_tof_sensor,ep_sensor_adaptor=ep_robot.sensor,ep_robot.sensor_adaptor
        
        print("📷 Starting camera stream..."); ep_camera.start_video_stream(display=False,resolution="720p")
        print(" GIMBAL: Centering gimbal..."); ep_gimbal.moveto(pitch=0,yaw=0,yaw_speed=SPEED_ROTATE).wait_for_completed()
        time.sleep(2)

        scanner=EnvironmentScanner(ep_sensor_adaptor,ep_tof_sensor,ep_gimbal,ep_chassis)
        movement_controller=MovementController(ep_chassis); attitude_handler.start_monitoring(ep_chassis)
        
        explore_with_ogm(ep_robot,scanner,movement_controller,attitude_handler,occupancy_map,visualizer)
        
        print(f"\n\n--- NAVIGATION TO TARGET PHASE: From {CURRENT_POSITION} to {TARGET_DESTINATION} ---")
        if CURRENT_POSITION==TARGET_DESTINATION: print("🎉 Robot is already at the target!")
        else:
            path_to_target=find_path_bfs(occupancy_map,CURRENT_POSITION,TARGET_DESTINATION)
            if path_to_target and len(path_to_target)>1:
                execute_path(path_to_target,movement_controller,attitude_handler,scanner,visualizer,occupancy_map,path_name="Final Navigation")
                print(f"🎉🎉 Robot has arrived at the target destination: {TARGET_DESTINATION}!")
            else: print(f"⚠️ Could not find a path from {CURRENT_POSITION} to {TARGET_DESTINATION}.")
        
    except KeyboardInterrupt: print("\n⚠️ User interrupted exploration.")
    except Exception as e: print(f"\n⚌ An error occurred: {e}"); traceback.print_exc()
    finally:
        if ep_robot:
            print("🔌 Cleaning up and closing connection...")
            if scanner: scanner.cleanup()
            if attitude_handler and attitude_handler.is_monitoring: attitude_handler.stop_monitoring(ep_chassis)
            if movement_controller: movement_controller.cleanup()
            ep_camera.stop_video_stream(); ep_robot.close(); print("🔌 Connection closed.")
        
        plt.ioff(); print("... You can close the plot window now ..."); plt.show()