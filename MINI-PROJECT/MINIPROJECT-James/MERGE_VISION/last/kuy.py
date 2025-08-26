"""
Enhanced Robot Exploration with Integrated Marker Detection

This file combines the best exploration logic from best_code.py with marker detection
capabilities from marker_red_part.py. Key features:

🎯 MARKER DETECTION INTEGRATION:
- Automatic marker scanning after each gimbal ToF scan (-90°, 0°, 90°)
- Special back direction scanning (180°) for initial node (0,0)
- MarkerVisionHandler for continuous marker detection
- Comprehensive marker tracking per node and direction

🔍 SCANNING PATTERN:
- Primary: ToF wall detection + RED color filtering → marker scanning only where red detected
- Red detection during ToF scan at 0°, -90°, 90° (and 180° for initial node)
- Marker scanning with gimbal tilt (-20°) only in red-detected directions
- Enhanced GraphNode with marker properties (markersFound, markerScanResults, hasMarkers)

📊 REPORTING:
- Marker statistics in exploration reports
- Markers found by direction and node
- Marker detection rate calculations
"""

import time
import robomaster
from robomaster import robot, vision
import numpy as np
from scipy.ndimage import median_filter
from datetime import datetime
import json
from collections import deque
import cv2
import threading
import signal
import sys

ROBOT_FACE = 1 # 0 1
CURRENT_TARGET_YAW = 0.0

# ===== Marker Detection Classes =====
class MarkerInfo:
    def __init__(self, x, y, w, h, marker_id):
        self._x = x
        self._y = y
        self._w = w
        self._h = h
        self._id = marker_id

    @property
    def id(self):
        return self._id

class MarkerVisionHandler:
    def __init__(self):
        self.markers = []
        self.marker_detected = False
        self.is_active = False
        self.detection_timeout = 1.0
    
    def on_detect_marker(self, marker_info):
        if not self.is_active:
            return
            
        if len(marker_info) > 0:
            valid_markers = []
            for i in range(len(marker_info)):
                x, y, w, h, marker_id = marker_info[i]
                marker = MarkerInfo(x, y, w, h, marker_id)
                valid_markers.append(marker)
            
            if valid_markers:
                self.marker_detected = True
                self.markers = valid_markers
    
    def wait_for_markers(self, timeout=None):
        if timeout is None:
            timeout = self.detection_timeout
        
        print(f"⏱️ Waiting {timeout} seconds for marker detection...")
        
        self.marker_detected = False
        self.markers.clear()
        
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            if self.marker_detected:
                print(f"✅ Marker detected after {time.time() - start_time:.1f}s")
                break
            time.sleep(0.02)
        
        return self.marker_detected
    
    def start_continuous_detection(self, vision):
        try:
            self.stop_continuous_detection(vision)
            time.sleep(0.3)
            
            result = vision.sub_detect_info(name="marker", callback=self.on_detect_marker)
            if result:
                self.is_active = True
                print("✅ Marker detection activated")
                return True
            else:
                print("❌ Failed to start marker detection")
                return False
        except Exception as e:
            print(f"❌ Error starting marker detection: {e}")
            return False
    
    def stop_continuous_detection(self, vision):
        try:
            self.is_active = False
            vision.unsub_detect_info(name="marker")
        except:
            pass
    
    def reset_detection(self):
        self.marker_detected = False
        self.markers.clear()

# ===== Direction Helper Functions =====
def get_direction_name(angle):
    """แปลงองศาเป็นชื่อทิศทาง"""
    direction_map = {
        0: "หน้า (Front)",
        -90: "ซ้าย (Left)", 
        90: "ขวา (Right)",
        180: "หลัง (Back)",
        -180: "หลัง (Back)"
    }
    return direction_map.get(angle, f"องศา {angle}")

def get_compass_direction(angle):
    """แปลงองศาเป็นทิศทางเข็มทิศ"""
    # สำหรับ gimbal yaw: 0° = หน้า, -90° = ซ้าย, 90° = ขวา
    compass_map = {
        0: "เหนือ (N)",
        -90: "ตะวันตก (W)",
        90: "ตะวันออก (E)",
        180: "ใต้ (S)",
        -180: "ใต้ (S)"
    }
    return compass_map.get(angle, f"{angle}°")

# เพิ่มคลาสใหม่สำหรับติดตามการเคลื่อนไหว
class MovementTracker:
    def __init__(self):
        self.consecutive_forward_moves = 0
        self.consecutive_backward_moves = 0
        self.movement_history = []  # เก็บประวัติการเคลื่อนไหว
        self.last_movement_type = None  # 'forward', 'backward', 'rotation'
        
    def record_movement(self, movement_type):
        """บันทึกการเคลื่อนไหวและอัพเดทตัวนับ"""
        self.movement_history.append(movement_type)
        
        if movement_type == 'forward':
            if self.last_movement_type == 'forward':
                self.consecutive_forward_moves += 1
            else:
                self.consecutive_forward_moves = 1
            # รีเซ็ตตัวนับถอยหลัง
            self.consecutive_backward_moves = 0
            
        elif movement_type == 'backward':
            if self.last_movement_type == 'backward':
                self.consecutive_backward_moves += 1
            else:
                self.consecutive_backward_moves = 1
            # รีเซ็ตตัวนับเดินหน้า
            self.consecutive_forward_moves = 0
            
        elif movement_type == 'rotation':
            # การหมุนจะรีเซ็ตทั้งสองตัวนับ
            self.consecutive_forward_moves = 0
            self.consecutive_backward_moves = 0
        
        self.last_movement_type = movement_type
        
        print(f"📊 Movement recorded: {movement_type}")
        print(f"   🔄 Consecutive forward: {self.consecutive_forward_moves}")
        print(f"   ↩️ Consecutive backward: {self.consecutive_backward_moves}")
    
    def has_consecutive_forward_moves(self, threshold=2):
        """เช็คว่ามีการเดินหน้าติดกันตามจำนวนที่กำหนดหรือไม่"""
        return self.consecutive_forward_moves >= threshold
    
    def has_consecutive_backward_moves(self, threshold=2):
        """เช็คว่ามีการถอยหลังติดกันตามจำนวนที่กำหนดหรือไม่"""
        return self.consecutive_backward_moves >= threshold
    
    def reset_counters(self):
        """รีเซ็ตตัวนับทั้งหมด"""
        self.consecutive_forward_moves = 0
        self.consecutive_backward_moves = 0
        self.last_movement_type = None
        print("🔄 Movement counters reset!")
    
    def get_movement_status(self):
        """ส่งคืนสถานะการเคลื่อนไหวปัจจุบัน"""
        return {
            'consecutive_forward': self.consecutive_forward_moves,
            'consecutive_backward': self.consecutive_backward_moves,
            'last_movement': self.last_movement_type,
            'history_length': len(self.movement_history)
        }

class AttitudeHandler:
    def __init__(self):
        self.current_yaw = 0.0
        self.current_pitch = 0.0
        self.current_roll = 0.0
        self.target_yaw = 0.0
        self.yaw_tolerance = 3
        self.is_monitoring = False
        
    def attitude_handler(self, attitude_info):
        if not self.is_monitoring:
            return
            
        yaw, pitch, roll = attitude_info
        self.current_yaw = yaw
        self.current_pitch = pitch
        self.current_roll = roll
        print(f"\r🧭 Current chassis attitude: yaw={yaw:.1f}°, pitch={pitch:.1f}°, roll={roll:.1f}°", end="", flush=True)
        
    def start_monitoring(self, chassis):
        self.is_monitoring = True
        chassis.sub_attitude(freq=20, callback=self.attitude_handler)
        
    def stop_monitoring(self, chassis):
        self.is_monitoring = False
        try:
            chassis.unsub_attitude()
        except:
            pass
            
    def normalize_angle(self, angle):
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle
        
    def is_at_target_yaw(self, target_yaw=0.0):
        if abs(target_yaw) == 180:
            diff_180 = abs(self.normalize_angle(self.current_yaw - 180))
            diff_neg180 = abs(self.normalize_angle(self.current_yaw - (-180)))
            diff = min(diff_180, diff_neg180)
            target_display = f"±180"
        else:
            diff = abs(self.normalize_angle(self.current_yaw - target_yaw))
            target_display = f"{target_yaw}"
            
        is_correct = diff <= self.yaw_tolerance
        print(f"\n🎯 Yaw check: current={self.current_yaw:.1f}°, target={target_display}°, diff={diff:.1f}°, correct={is_correct}")
        return is_correct
        
    def correct_yaw_to_target(self, chassis, target_yaw=0.0):
        if self.is_at_target_yaw(target_yaw):
            print(f"✅ Chassis already at correct yaw: {self.current_yaw:.1f}° (target: {target_yaw}°)")
            return True
            
        gimbal_to_target = target_yaw - self.current_yaw
        gimbal_diff = self.normalize_angle(gimbal_to_target)
        robot_rotation = -gimbal_diff
        
        print(f"🔧 Correcting chassis yaw: from {self.current_yaw:.1f}° to {target_yaw}°")
        print(f"📐 Gimbal needs to change: {gimbal_diff:.1f}°")
        print(f"📐 Robot will rotate: {robot_rotation:.1f}°")
        
        try:
            if abs(robot_rotation) > self.yaw_tolerance:
                correction_speed = 60
                
                print(f"🔄 Rotating robot {robot_rotation:.1f}°")
                chassis.move(x=0, y=0, z=robot_rotation, z_speed=correction_speed).wait_for_completed()
                time.sleep(0.3)
            
            final_check = self.is_at_target_yaw(target_yaw)
            
            if final_check:
                print(f"✅ Successfully corrected chassis yaw to {self.current_yaw:.1f}°")
                return True
            else:
                print(f"⚠️ Chassis yaw correction incomplete: {self.current_yaw:.1f}° (target: {target_yaw}°)")
                
                remaining_gimbal = target_yaw - self.current_yaw
                remaining_diff = self.normalize_angle(remaining_gimbal)
                remaining_robot = -remaining_diff
                print(f"📐 Remaining gimbal difference: {remaining_diff:.1f}°")
                print(f"📐 Additional robot rotation needed: {remaining_robot:.1f}°")
                
                if abs(remaining_robot) > self.yaw_tolerance and abs(remaining_robot) < 45:
                    print(f"🔧 Fine-tuning robot with additional {remaining_robot:.1f}°")
                    chassis.move(x=0, y=0, z=remaining_robot, z_speed=60).wait_for_completed()
                    time.sleep(0.3)
                    return self.is_at_target_yaw(target_yaw)
                else:
                    print(f"⚠️ Remaining rotation too large ({remaining_robot:.1f}°), may need multiple corrections")
                return False
                
        except Exception as e:
            print(f"❌ Failed to correct chassis yaw: {e}")
            return False

# ===== PID Controller =====
class PID:
    def __init__(self, Kp, Ki, Kd, setpoint=0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.prev_error = 0
        self.integral = 0
        self.integral_max = 1.0

    def compute(self, current, dt):
        error = self.setpoint - current
        self.integral += error * dt
        
        if self.integral > self.integral_max:
            self.integral = self.integral_max
        elif self.integral < -self.integral_max:
            self.integral = -self.integral_max
            
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

# ===== Movement Controller =====
class MovementController:
    def __init__(self, chassis):
        self.chassis = chassis
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        
        # PID Parameters
        self.KP = 2.08
        self.KI = 0.25
        self.KD = 10
        self.RAMP_UP_TIME = 0.7
        self.ROTATE_TIME = 2.11  # Right turn
        self.ROTATE_LEFT_TIME = 1.9  # Left turn

        # เพิ่ม MovementTracker
        self.movement_tracker = MovementTracker()

        # เพิ่มระบบติดตาม attitude drift correction
        self.nodes_visited_count = 0  # นับจำนวนโหนดที่ผ่านไป
        self.DRIFT_CORRECTION_INTERVAL = 10  # ทุก 9 โหนด
        self.DRIFT_CORRECTION_ANGLE = 2  # หมุนเพิ่ม 3 องศาไปทางขวา
        self.total_drift_corrections = 0  # นับจำนวนครั้งที่แก้ไข
        
        # *** เพิ่มตัวแปรใหม่สำหรับป้องกัน drift correction ซ้ำ ***
        self.last_correction_at = 0  # เก็บ node count ครั้งสุดท้ายที่ทำ correction

        # Subscribe to position updates
        self.chassis.sub_position(freq=20, callback=self.position_handler)
        time.sleep(0.25)
    
    def position_handler(self, position_info):
        self.current_x = position_info[0]
        self.current_y = position_info[1]
        self.current_z = position_info[2]

    def increment_node_visit_for_backtrack_with_correction(self, attitude_handler):
        """เพิ่มจำนวนโหนดที่เยียมชมสำหรับการ backtrack และ trigger drift correction ถ้าถึงเวลา"""
        self.nodes_visited_count += 1
        print(f"📊 Backtrack node visit count: {self.nodes_visited_count}")
        
        # เช็คว่าต้องทำ drift correction หรือไม่ และยังไม่เคยทำที่ count นี้
        if (self.nodes_visited_count % self.DRIFT_CORRECTION_INTERVAL == 0 and 
            self.nodes_visited_count != self.last_correction_at):
            
            print(f"🔧 BACKTRACK DRIFT CORRECTION TRIGGERED!")
            print(f"   📍 After {self.nodes_visited_count} total nodes visited (including backtrack)")
            print(f"   🔄 Correction #{self.total_drift_corrections + 1}")
            
            # ทำ drift correction
            success = self.perform_attitude_drift_correction(attitude_handler)
            self.last_correction_at = self.nodes_visited_count  # บันทึกว่าทำ correction แล้ว
            
            if success:
                print(f"✅ Backtrack drift correction completed!")
                return True
            else:
                print(f"⚠️ Backtrack drift correction had issues, but continuing...")
                return False
                
        return False

    def increment_node_visit_main_exploration(self, attitude_handler):
        """เพิ่มจำนวนโหนดที่เยียมชมสำหรับ main exploration loop และ trigger drift correction ถ้าถึงเวลา"""
        self.nodes_visited_count += 1
        print(f"📊 Main exploration node visit count: {self.nodes_visited_count}")
        
        # เช็คว่าต้องทำ drift correction หรือไม่ และยังไม่เคยทำที่ count นี้
        if (self.nodes_visited_count % self.DRIFT_CORRECTION_INTERVAL == 0 and 
            self.nodes_visited_count != self.last_correction_at):
            
            print(f"🔧 MAIN EXPLORATION DRIFT CORRECTION TRIGGERED!")
            print(f"   📍 After {self.nodes_visited_count} total nodes visited")
            print(f"   🔄 Correction #{self.total_drift_corrections + 1}")
            
            # ทำ drift correction
            success = self.perform_attitude_drift_correction(attitude_handler)
            self.last_correction_at = self.nodes_visited_count  # บันทึกว่าทำ correction แล้ว
            
            if success:
                print(f"✅ Main exploration drift correction completed!")
                return True
            else:
                print(f"⚠️ Main exploration drift correction had issues, but continuing...")
                return False
                
        return False

    def perform_attitude_drift_correction(self, attitude_handler):
        """ทำการแก้ไข attitude drift โดยหมุนขวาเพิ่มตามมุมที่กำหนด"""
        global CURRENT_TARGET_YAW
        
        print(f"⚙️ === PERFORMING ATTITUDE DRIFT CORRECTION ===")
        print(f"🔧 Correcting attitude drift after {self.nodes_visited_count} nodes")
        print(f"📐 Adding {self.DRIFT_CORRECTION_ANGLE}° clockwise correction")

        # ใช้ yaw ปัจจุบันเป็นฐาน ไม่ใช่บวกสะสม
        current_yaw_before = attitude_handler.current_yaw
        target_after_correction = attitude_handler.normalize_angle(
            current_yaw_before + self.DRIFT_CORRECTION_ANGLE
        )

        # อัปเดต CURRENT_TARGET_YAW ให้ตรงกับเป้าหมายล่าสุด
        CURRENT_TARGET_YAW = target_after_correction

        print(f"📊 Drift correction details:")
        print(f"   🧭 Current yaw before: {current_yaw_before:.1f}°")
        print(f"   🎯 New target: {target_after_correction:.1f}°")

        try:
            success = attitude_handler.correct_yaw_to_target(self.chassis, target_after_correction)

            if success:
                self.total_drift_corrections += 1
                current_yaw_after = attitude_handler.current_yaw

                print(f"✅ Attitude drift correction completed!")
                print(f"   🧭 Final yaw: {current_yaw_after:.1f}°")
                print(f"   📈 Total corrections performed: {self.total_drift_corrections}")
                print(f"   📐 Total accumulated correction: {self.total_drift_corrections * self.DRIFT_CORRECTION_ANGLE}°")

                self.movement_tracker.record_movement('rotation')

                # ป้องกัน trigger ซ้ำ — บันทึก node ล่าสุดที่แก้ไปแล้ว
                self.last_correction_at = self.nodes_visited_count

                return True
            else:
                print(f"⚠️ Attitude drift correction may be incomplete!")
                return False

        except Exception as e:
            print(f"❌ Error during attitude drift correction: {e}")
            return False

        finally:
            print(f"⚙️ === ATTITUDE DRIFT CORRECTION END ===")
            time.sleep(0.3)

    def move_forward_with_pid(self, target_distance, axis, direction=1, allow_yaw_correction=True):
        """Move forward using PID control with movement tracking"""
        # บันทึกการเคลื่อนไหว
        movement_type = 'forward' if direction == 1 else 'backward'
        self.movement_tracker.record_movement(movement_type)
        
        # เช็คว่ามีการเคลื่อนไหวติดกันหรือไม่ (เฉพาะกรณีอนุญาตแก้ yaw)
        if allow_yaw_correction:
            if self.movement_tracker.has_consecutive_forward_moves(2):
                print("⚠️ DETECTED: 2 consecutive forward moves!")
                target_angle = attitude_handler.normalize_angle(CURRENT_TARGET_YAW)
                print(f"🎯 Target yaw: {target_angle}°")
                attitude_handler.correct_yaw_to_target(self.chassis, target_angle)
                
            if self.movement_tracker.has_consecutive_backward_moves(2):
                print("⚠️ DETECTED: 2 consecutive backward moves!")
                target_angle = attitude_handler.normalize_angle(CURRENT_TARGET_YAW)
                print(f"🎯 Target yaw: {target_angle}°")
                attitude_handler.correct_yaw_to_target(self.chassis, target_angle)
        
        pid = PID(Kp=self.KP, Ki=self.KI, Kd=self.KD, setpoint=target_distance)
        
        start_time = time.time()
        last_time = start_time
        target_reached = False
        
        # Ramp-up parameters
        min_speed = 0.1
        max_speed = 1.5
        
        if axis == 'x':
            start_position = self.current_x
        else:
            start_position = self.current_y

        direction_text = "FORWARD" if direction == 1 else "BACKWARD"
        print(f"🚀 Moving {direction_text} {target_distance}m on {axis}-axis, direction: {direction}")
        
        try:
            while not target_reached:
                now = time.time()
                dt = now - last_time
                last_time = now
                elapsed_time = now - start_time
                
                if axis == 'x':
                    current_position = self.current_x
                else:
                    current_position = self.current_y
                
                relative_position = abs(current_position - start_position)

                # PID calculation
                output = pid.compute(relative_position, dt)
                
                # Ramp-up logic
                if elapsed_time < self.RAMP_UP_TIME:
                    ramp_multiplier = min_speed + (elapsed_time / self.RAMP_UP_TIME) * (1.0 - min_speed)
                else:
                    ramp_multiplier = 1.0
                
                ramped_output = output * ramp_multiplier
                speed = max(min(ramped_output, max_speed), -max_speed)
                
                if axis == 'x':
                    self.chassis.drive_speed(x=speed * direction, y=0, z=0, timeout=1)
                else:
                    self.chassis.drive_speed(x=speed * direction, y=0, z=0, timeout=1)

                # Stop condition
                if abs(relative_position - target_distance) < 0.02:
                    print(f"✅ Target reached! Final position: {current_position:.3f}")
                    self.chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
                    target_reached = True
                    break
                    
        except KeyboardInterrupt:
            print("Movement interrupted by user.")
    
    def rotate_90_degrees_right(self, attitude_handler=None):
        global CURRENT_TARGET_YAW
        print("🔄 Rotating 90° RIGHT...")
        # บันทึกการหมุน
        self.movement_tracker.record_movement('rotation')
        time.sleep(0.2)
        
        CURRENT_TARGET_YAW += 90
        target_angle = attitude_handler.normalize_angle(CURRENT_TARGET_YAW)
        
        print(f"🎯 Target yaw: {target_angle}°")
        success = attitude_handler.correct_yaw_to_target(self.chassis, target_angle)
        
        if success:
            print("✅ Right rotation completed!")
        else:
            print("⚠️ Right rotation may be incomplete")
            
        time.sleep(0.2)

    def rotate_90_degrees_left(self, attitude_handler=None):
        global CURRENT_TARGET_YAW
        print("🔄 Rotating 90° LEFT...")
        # บันทึกการหมุน
        self.movement_tracker.record_movement('rotation')
        time.sleep(0.2)
        
        CURRENT_TARGET_YAW -= 90
        target_angle = attitude_handler.normalize_angle(CURRENT_TARGET_YAW)
        
        print(f"🎯 Target yaw: {target_angle}°")
        success = attitude_handler.correct_yaw_to_target(self.chassis, target_angle)
        
        if success:
            print("✅ Left rotation completed!")
        else:
            print("⚠️ Left rotation may be incomplete")
            
        time.sleep(0.2)
    
    def reverse_from_dead_end(self):
        """Reverse robot from dead end position"""
        global ROBOT_FACE
        print("🔙 DEAD END DETECTED - Reversing...")
        # บันทึกการถอยหลัง
        self.movement_tracker.record_movement('backward')
        # Determine current axis based on robot face
        axis_test = 'x'
        if ROBOT_FACE % 2 == 0:
            axis_test = 'y'
        elif ROBOT_FACE % 2 == 1:
            axis_test = 'x'
        
        print(f"🔙 Reversing 0.6m on {axis_test}-axis")
        
        # Move backward using negative direction
        self.move_forward_with_pid(0.6, axis_test, direction=-1)
        
        print("✅ Reverse from dead end completed!")

    def reverse_to_previous_node(self):
        """NEW: Reverse 0.6m to go back to previous node without rotating"""
        self.movement_tracker.record_movement('backward')

        global ROBOT_FACE
        print("🔙 BACKTRACKING - Reversing to previous node...")

        axis_test = 'x'
        if ROBOT_FACE % 2 == 0:
            axis_test = 'y'
        elif ROBOT_FACE % 2 == 1:
            axis_test = 'x'
        
        print(f"🔙 Reversing 0.6m on {axis_test}-axis for backtrack")
        
        # ❌ ปิด yaw correction ระหว่าง backtracking
        self.move_forward_with_pid(0.6, axis_test, direction=-1, allow_yaw_correction=False)
        
        print("✅ Reverse backtrack completed!")
    
    def cleanup(self):
        """Clean up position subscription"""
        try:
            self.chassis.unsub_position()
        except:
            pass

    def get_drift_correction_status(self):
        """ส่งคืนสถานะการแก้ไข drift ปัจจุบัน"""
        return {
            'nodes_visited': self.nodes_visited_count,
            'next_correction_at': ((self.nodes_visited_count // self.DRIFT_CORRECTION_INTERVAL) + 1) * self.DRIFT_CORRECTION_INTERVAL,
            'nodes_until_correction': self.DRIFT_CORRECTION_INTERVAL - (self.nodes_visited_count % self.DRIFT_CORRECTION_INTERVAL),
            'total_corrections': self.total_drift_corrections,
            'correction_interval': self.DRIFT_CORRECTION_INTERVAL,
            'correction_angle': self.DRIFT_CORRECTION_ANGLE,
            'last_correction_at': self.last_correction_at
        }
    
    def reset_drift_correction(self):
        """รีเซ็ตระบบแก้ไข drift (สำหรับ testing หรือเริ่มใหม่)"""
        self.nodes_visited_count = 0
        self.total_drift_corrections = 0
        print("🔄 Drift correction system reset!")

    def get_movement_status(self):
        """ส่งคืนสถานะการเคลื่อนไหวปัจจุบัน"""
        return self.movement_tracker.get_movement_status()
    
    def reset_movement_tracking(self):
        """รีเซ็ตการติดตามการเคลื่อนไหว"""
        self.movement_tracker.reset_counters()

# ===== Graph Node =====
class GraphNode:
    def __init__(self, node_id, position):
        self.id = node_id
        self.position = position  # (x, y)
        self.outOfBoundsExits = []  # list ทิศทางที่เกินแมพ
        self.outOfBoundsCount = 0   # จำนวนทางออกนอกแมพ

        # Wall detection - NOW STORES ABSOLUTE DIRECTIONS
        self.walls = {
            'north': False,
            'south': False,
            'east': False,
            'west': False
        }

        # Legacy support (will be removed)
        self.wallLeft = False
        self.wallRight = False
        self.wallFront = False
        self.wallBack = False

        # Neighbors (connected nodes)
        self.neighbors = {
            'north': None,
            'south': None,
            'east': None,
            'west': None
        }

        # Exploration state
        self.visited = True
        self.visitCount = 1
        self.exploredDirections = []
        self.unexploredExits = []
        self.isDeadEnd = False

        # NEW: Add flag to track if node has been fully scanned
        self.fullyScanned = False
        self.scanTimestamp = None

        # Additional info
        self.marker = False
        self.lastVisited = datetime.now().isoformat()
        self.sensorReadings = {}
        
        # NEW: Marker detection information
        self.markersFound = {}  # Dictionary: {direction: [marker_ids]}
        self.markerScanResults = {}  # Dictionary: {direction: {'marker_ids': [], 'distance': float, 'timestamp': str}}
        self.hasMarkers = False  # Boolean flag if any markers found in any direction
        
        # IMPORTANT: Store the ABSOLUTE direction robot was facing when first scanned
        self.initialScanDirection = None

# ===== Graph Mapper =====
class GraphMapper:
    def __init__(self, min_x=-3, min_y=-3, max_x=3, max_y=3):
        self.nodes = {}
        self.currentPosition = (0, 0)
        self.currentDirection = 'north'
        self.frontierQueue = []
        self.pathStack = []
        self.visitedNodes = set()
        self.previous_node = None

        # === Border limits ===
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y

        # Override methods เพื่อใช้ priority-based exploration
        self.find_next_exploration_direction = self.find_next_exploration_direction_with_priority
        # ✅ KEEP the absolute version for proper boundary checking
        # self.update_unexplored_exits_absolute = self.update_unexplored_exits_with_priority

    # 3. เพิ่มฟังก์ชันใหม่สำหรับตรวจสอบ boundary
    def is_position_within_boundaries(self, position):
        """Check if position is within map boundaries"""
        x, y = position
        return (self.min_x <= x <= self.max_x and 
                self.min_y <= y <= self.max_y)

    def get_boundary_status(self):
        """Get current boundary configuration"""
        return {
            'min_x': self.min_x,
            'max_x': self.max_x,
            'min_y': self.min_y,
            'max_y': self.max_y,
            'width': self.max_x - self.min_x + 1,
            'height': self.max_y - self.min_y + 1,
            'total_cells': (self.max_x - self.min_x + 1) * (self.max_y - self.min_y + 1)
        }

    def get_node_id(self, position):
        return f"{position[0]}_{position[1]}"
    
    def create_node(self, position):
        node_id = self.get_node_id(position)
        if node_id not in self.nodes:
            node = GraphNode(node_id, position)
            # Store the absolute direction robot was facing when node was first created
            node.initialScanDirection = self.currentDirection
            self.nodes[node_id] = node
            self.visitedNodes.add(node_id)
        return self.nodes[node_id]

    def get_current_node(self):
        node_id = self.get_node_id(self.currentPosition)
        return self.nodes.get(node_id)
    
    def update_current_node_walls_absolute(self, left_wall, right_wall, front_wall):
        """NEW: Update walls using ABSOLUTE directions"""
        current_node = self.get_current_node()
        if current_node:
            # Map relative sensor readings to absolute directions
            direction_map = {
                'north': {'front': 'north', 'left': 'west', 'right': 'east'},
                'south': {'front': 'south', 'left': 'east', 'right': 'west'},
                'east': {'front': 'east', 'left': 'north', 'right': 'south'},
                'west': {'front': 'west', 'left': 'south', 'right': 'north'}
            }
            
            current_mapping = direction_map[self.currentDirection]
            
            # Update absolute wall information
            current_node.walls[current_mapping['front']] = front_wall
            current_node.walls[current_mapping['left']] = left_wall
            current_node.walls[current_mapping['right']] = right_wall
            
            # Legacy support - update old format too
            current_node.wallFront = front_wall
            current_node.wallLeft = left_wall
            current_node.wallRight = right_wall
            
            current_node.lastVisited = datetime.now().isoformat()
            
            # NEW: Mark node as fully scanned
            current_node.fullyScanned = True
            current_node.scanTimestamp = datetime.now().isoformat()
            
            self.update_unexplored_exits_absolute(current_node)
            self.build_connections()

    # 1. แก้ไขฟังก์ชัน update_unexplored_exits_absolute
    def update_unexplored_exits_absolute(self, node):
        """Update unexplored exits using ABSOLUTE directions + outer border check"""
        try:
            node.unexploredExits = []
            node.outOfBoundsExits = []
            node.outOfBoundsCount = 0

            x, y = node.position
            
            possible_directions = {
                'north': (x, y + 1),
                'south': (x, y - 1),
                'east':  (x + 1, y),
                'west':  (x - 1, y)
            }

            print(f"🧭 Updating unexplored exits for {node.id} at {node.position}")
            print(f"🔍 Wall status: {node.walls}")
            print(f"🗺️ Map boundaries: x[{self.min_x},{self.max_x}], y[{self.min_y},{self.max_y}]")

            for direction, (target_x, target_y) in possible_directions.items():
                try:
                    target_node_id = self.get_node_id((target_x, target_y))
                    
                    # Check if target is outside map boundaries
                    is_outer_boundary = (
                        target_x < self.min_x or target_x > self.max_x or
                        target_y < self.min_y or target_y > self.max_y
                    )
                    
                    # Additional debugging for boundary conditions
                    if is_outer_boundary:
                        print(f"      🌐 Boundary check: {(target_x, target_y)} outside [{self.min_x},{self.max_x}] x [{self.min_y},{self.max_y}]")

                                # เช็ค wall / explored / target exist
                    is_blocked = node.walls.get(direction, True)
                    already_explored = direction in node.exploredDirections
                    target_exists = target_node_id in self.nodes
                    target_fully_explored = False
                    target_has_unexplored_exits = False
                    
                    if target_exists:
                        target_node = self.nodes[target_node_id]
                        target_fully_explored = target_node.fullyScanned
                        # ✅ NEW: Check if target node still has unexplored exits
                        target_has_unexplored_exits = len(target_node.unexploredExits) > 0

                    print(f"   🔍 Direction {direction}:")
                    print(f"      🚧 Blocked: {is_blocked}")
                    print(f"      ✅ Already explored: {already_explored}")
                    print(f"      🗃️  Target exists: {target_exists}")
                    print(f"      🔍 Target fully explored: {target_fully_explored}")
                    print(f"      🎯 Target has unexplored exits: {target_has_unexplored_exits}")
                    print(f"      🌐 Is outer boundary: {is_outer_boundary}")

                    # ✅ FIXED: More flexible exploration logic
                    should_explore = (
                        not is_blocked and
                        not already_explored and
                        (not target_exists or not target_fully_explored or target_has_unexplored_exits)
                    )

                    if should_explore:
                        if is_outer_boundary:
                            node.outOfBoundsExits.append(direction)
                            node.outOfBoundsCount = len(node.outOfBoundsExits)
                            print(f"      🚫 OUTER BOUNDARY! Added to outOfBoundsExits, NO exploration.")
                        else:
                            node.unexploredExits.append(direction)
                            print(f"      ✅ ADDED to unexplored exits!")
                    else:
                        print(f"      ❌ NOT added to unexplored exits")

                except Exception as e:
                    print(f"      ❌ Error processing direction {direction}: {e}")
                    continue

            print(f"🎯 Final unexplored exits for {node.id}: {node.unexploredExits}")
            print(f"🌐 Out-of-bounds exits: {node.outOfBoundsExits} (count: {node.outOfBoundsCount})")

            # Frontier queue update with error handling
            try:
                has_unexplored = len(node.unexploredExits) > 0
                if has_unexplored and node.id not in self.frontierQueue:
                    self.frontierQueue.append(node.id)
                    print(f"🚀 Added {node.id} to frontier queue")
                elif not has_unexplored and node.id in self.frontierQueue:
                    self.frontierQueue.remove(node.id)
                    print(f"🧹 Removed {node.id} from frontier queue")

                # Dead end detection
                blocked_count = sum(1 for blocked in node.walls.values() if blocked)
                node.isDeadEnd = blocked_count >= 3
                if node.isDeadEnd:
                    print(f"🚫 DEAD END CONFIRMED at {node.id} - {blocked_count} walls detected!")
                    if node.id in self.frontierQueue:
                        self.frontierQueue.remove(node.id)
                        print(f"🧹 Removed dead end {node.id} from frontier queue")
            except Exception as e:
                print(f"❌ Error updating frontier queue: {e}")
                
        except Exception as e:
            print(f"❌ Critical error in update_unexplored_exits_absolute: {e}")
            # Ensure node has at least empty lists to prevent crashes
            if not hasattr(node, 'unexploredExits'):
                node.unexploredExits = []
            if not hasattr(node, 'outOfBoundsExits'):
                node.outOfBoundsExits = []
                node.outOfBoundsCount = 0


    
    def build_connections(self):
        """Build connections between adjacent nodes"""
        for node_id, node in self.nodes.items():
            x, y = node.position
            
            # Check all four directions
            directions = {
                'north': (x, y + 1),
                'south': (x, y - 1),
                'east': (x + 1, y),
                'west': (x - 1, y)
            }
            
            for direction, neighbor_pos in directions.items():
                neighbor_id = self.get_node_id(neighbor_pos)
                if neighbor_id in self.nodes:
                    node.neighbors[direction] = self.nodes[neighbor_id]
    
    def get_next_position_from(self, position, direction):
        """Helper method to calculate next position from given position and direction"""
        x, y = position
        if direction == 'north':
            return (x, y + 1)
        elif direction == 'south':
            return (x, y - 1)
        elif direction == 'east':
            return (x + 1, y)
        elif direction == 'west':
            return (x - 1, y)
        return position
    
    def is_dead_end(self, node=None):
        """Check if current node or given node is a dead end"""
        if node is None:
            node = self.get_current_node()
        
        if not node:
            return False
        
        return node.isDeadEnd
    
    def get_next_position(self, direction):
        """Calculate next position based on current position and ABSOLUTE direction"""
        x, y = self.currentPosition
        if direction == 'north':
            return (x, y + 1)
        elif direction == 'south':
            return (x, y - 1)
        elif direction == 'east':
            return (x + 1, y)
        elif direction == 'west':
            return (x - 1, y)
        return self.currentPosition
    
    def get_previous_position(self, direction):
        """Calculate previous position based on current position and direction robot came from"""
        x, y = self.currentPosition
        if direction == 'north':
            return (x, y - 1)  # came from south
        elif direction == 'south':
            return (x, y + 1)  # came from north
        elif direction == 'east':
            return (x - 1, y)  # came from west
        elif direction == 'west':
            return (x + 1, y)  # came from east
        return self.currentPosition
    
    def can_move_to_direction_absolute(self, target_direction):
        """Check if robot can move to target ABSOLUTE direction"""
        current_node = self.get_current_node()
        if not current_node:
            return False
        
        # Check if target direction has a wall using absolute direction
        is_blocked = current_node.walls.get(target_direction, True)
        return not is_blocked
    
    def rotate_to_absolute_direction(self, target_direction, movement_controller, attitude_handler):
        """NEW: Rotate robot to face target ABSOLUTE direction"""
        global ROBOT_FACE
        global CURRENT_TARGET_YAW
        print(f"🎯 Rotating from {self.currentDirection} to {target_direction}")
        
        if self.currentDirection == target_direction:
            print(f"✅ Already facing {target_direction}")
            return
        
        direction_order = ['north', 'east', 'south', 'west']
        current_idx = direction_order.index(self.currentDirection)
        target_idx = direction_order.index(target_direction)
        
        # Calculate shortest rotation
        diff = (target_idx - current_idx) % 4
        
        if diff == 1:  # Turn right
            movement_controller.rotate_90_degrees_right(attitude_handler)
            ROBOT_FACE += 1
        elif diff == 3:  # Turn left
            movement_controller.rotate_90_degrees_left(attitude_handler)
            ROBOT_FACE += 1
        elif diff == 2:  # Turn around (180°)
            movement_controller.rotate_90_degrees_right(attitude_handler)
            movement_controller.rotate_90_degrees_right(attitude_handler)
            ROBOT_FACE += 2
        
        # Update current direction
        self.currentDirection = target_direction
        print(f"✅ Now facing {self.currentDirection}")

    def handle_dead_end(self, movement_controller):
        """Handle dead end situation by reversing"""
        print(f"🚫 === DEAD END HANDLER ACTIVATED ===")
        current_node = self.get_current_node()

        if current_node:
            print(f"📍 Dead end at position: {current_node.position}")
            print(f"🧱 Walls: {current_node.walls}")

        # Use the reverse method
        movement_controller.reverse_from_dead_end()

        # Update position after reversing (move back in opposite direction)
        reverse_direction_map = {
            'north': 'south',
            'south': 'north',
            'east': 'west',
            'west': 'east'
        }

        reverse_direction = reverse_direction_map[self.currentDirection]
        self.currentPosition = self.get_next_position(reverse_direction)

        print(f"🔙 Reversed to position: {self.currentPosition}")
        print(f"🧭 Still facing: {self.currentDirection}")

        return True
    
    # 2. เพิ่มการตรวจสอบในฟังก์ชัน move_to_absolute_direction
    def move_to_absolute_direction(self, target_direction, movement_controller, attitude_handler):
        """NEW: Move to target ABSOLUTE direction with proper rotation and border check"""
        global ROBOT_FACE

        current_node = self.get_current_node()
        if not current_node:
            print("❌ No current node - cannot move")
            return False

        # === ✅ เพิ่มการตรวจสอบ border แบบ double-check ===
        target_pos = self.get_next_position(target_direction)
        target_x, target_y = target_pos
        
        is_outside_map = (
            target_x < self.min_x or target_x > self.max_x or
            target_y < self.min_y or target_y > self.max_y
        )
        
        if is_outside_map:
            print(f"🚫 TARGET POSITION {target_pos} IS OUTSIDE MAP BOUNDARIES!")
            print(f"🗺️ Map boundaries: x[{self.min_x},{self.max_x}], y[{self.min_y},{self.max_y}]")
            print(f"🚫 Movement to {target_direction} CANCELLED!")
            return False

        # Prevent movement outside border (original check)
        if target_direction in current_node.outOfBoundsExits:
            print(f"🚫 Target direction {target_direction} is OUT OF BOUNDS! Movement cancelled.")
            return False

        print(f"🎯 Moving to ABSOLUTE direction: {target_direction}")
        
        # Check if movement is possible
        if not self.can_move_to_direction_absolute(target_direction):
            print(f"❌ BLOCKED! Cannot move to {target_direction} - wall detected!")
            return False
        
        # First rotate to face the target direction
        self.rotate_to_absolute_direction(target_direction, movement_controller, attitude_handler)

        # เช็คสถานะการเคลื่อนไหวก่อนเคลื่อนที่
        movement_status = movement_controller.get_movement_status()
        print(f"📊 Current movement status: {movement_status}")

        # Determine axis for movement
        axis_test = 'x'
        if ROBOT_FACE % 2 == 0:
            axis_test = 'y'
        elif ROBOT_FACE % 2 == 1:
            axis_test = 'x'
        
        print(f"🚀 Moving forward on {axis_test}-axis")

        # Move forward (จะบันทึกเป็น 'forward' ใน MovementTracker)
        movement_controller.move_forward_with_pid(0.6, axis_test, direction=1)
        
        # Update position
        self.currentPosition = self.get_next_position(target_direction)
        
        # ✅ FIXED: Don't mark direction as explored immediately
        # Only remove from unexplored exits when we actually move
        if hasattr(self, 'previous_node') and self.previous_node:
            # Remove from unexplored exits since we're moving in this direction
            if target_direction in self.previous_node.unexploredExits:
                self.previous_node.unexploredExits.remove(target_direction)
                print(f"🔄 Removed {target_direction} from unexplored exits of {self.previous_node.id}")
            
            # ✅ NEW: Mark as explored only after we scan the target node
            # This will be done in the main exploration loop after scanning
        
        print(f"✅ Successfully moved to {self.currentPosition}")
        return True

    def reverse_to_absolute_direction(self, target_direction, movement_controller, attitude_handler):
        """NEW: Reverse to target ABSOLUTE direction for backtracking"""
        global ROBOT_FACE
        print(f"🔙 BACKTRACK: Reversing to ABSOLUTE direction: {target_direction}")
        
        # Calculate what direction we need to reverse to
        reverse_direction_map = {
            'north': 'south',
            'south': 'north',
            'east': 'west',
            'west': 'east'
        }
        
        required_facing_direction = reverse_direction_map[target_direction]
        
        # First rotate to face the direction OPPOSITE to where we want to go
        self.rotate_to_absolute_direction(required_facing_direction, movement_controller, attitude_handler)
        
        # Now reverse (which will move us in the target direction)
        movement_controller.reverse_to_previous_node()
        
        # Update position
        self.currentPosition = self.get_next_position(target_direction)
        
        print(f"✅ Successfully reversed to {self.currentPosition}, still facing {self.currentDirection}")
        return True

    def find_next_exploration_direction_with_priority(self):
        """Find next exploration direction with LEFT-first priority, skipping out-of-bounds exits"""
        current_node = self.get_current_node()
        if not current_node:
            return None

        if self.is_dead_end(current_node):
            print(f"🚫 Current node is a dead end - no exploration directions available")
            return None

        print(f"🧭 Current robot facing: {self.currentDirection}")
        print(f"🔍 Available unexplored exits: {current_node.unexploredExits}")
        print(f"🌐 Out-of-bounds exits: {current_node.outOfBoundsExits}")

        # กำหนด mapping ทิศสัมพัทธ์
        direction_map = {
            'north': {'front': 'north', 'left': 'west', 'right': 'east', 'back': 'south'},
            'south': {'front': 'south', 'left': 'east', 'right': 'west', 'back': 'north'},
            'east': {'front': 'east', 'left': 'north', 'right': 'south', 'back': 'west'},
            'west': {'front': 'west', 'left': 'south', 'right': 'north', 'back': 'east'}
        }
        current_mapping = direction_map[self.currentDirection]

        # ลำดับความสำคัญ
        priority_order = ['left', 'front', 'right', 'back']
        print(f"🎯 Checking exploration priority order: {priority_order}")

        for relative_direction in priority_order:
            absolute_direction = current_mapping.get(relative_direction)
            if not absolute_direction:
                continue

            # ข้ามทิศที่อยู่นอก border
            if absolute_direction in current_node.outOfBoundsExits:
                print(f"🚫 {relative_direction} ({absolute_direction}) is OUT OF BOUNDS! Skipping...")
                continue

            # เลือกทิศที่อยู่ใน unexplored exits และสามารถเดินได้
            if absolute_direction in current_node.unexploredExits:
                if self.can_move_to_direction_absolute(absolute_direction):
                    print(f"✅ Selected direction: {relative_direction} ({absolute_direction})")
                    return absolute_direction
                else:
                    print(f"❌ {relative_direction} ({absolute_direction}) is blocked by wall!")
                    current_node.unexploredExits.remove(absolute_direction)

        print(f"❌ No valid exploration direction found")
        return None

    def update_unexplored_exits_with_priority(self, node):
        """Update unexplored exits with priority ordering"""
        node.unexploredExits = []
        
        x, y = node.position
        
        # กำหนดลำดับการตรวจสอบตามความสำคัญ LEFT-FIRST
        # แต่เก็บเป็นทิศทางสัมบูรณ์
        direction_map = {
            'north': {'front': 'north', 'left': 'west', 'right': 'east', 'back': 'south'},
            'south': {'front': 'south', 'left': 'east', 'right': 'west', 'back': 'north'},
            'east': {'front': 'east', 'left': 'north', 'right': 'south', 'back': 'west'},
            'west': {'front': 'west', 'left': 'south', 'right': 'north', 'back': 'east'}
        }
        
        current_mapping = direction_map[self.currentDirection]
        
        # ลำดับความสำคัญ
        priority_order = ['left', 'front', 'right', 'back']
        
        possible_directions = {
            'north': (x, y + 1),
            'south': (x, y - 1),
            'east': (x + 1, y),
            'west': (x - 1, y)
        }
        
        print(f"🧭 Updating unexplored exits for {node.id} at {node.position}")
        print(f"🔍 Wall status: {node.walls}")
        print(f"🤖 Robot facing: {self.currentDirection}")
        
        # ตรวจสอบตามลำดับความสำคัญและเพิ่มเข้า unexploredExits
        for relative_dir in priority_order:
            absolute_dir = current_mapping[relative_dir]
            target_pos = possible_directions[absolute_dir]
            target_node_id = self.get_node_id(target_pos)
            
            # ตรวจสอบเงื่อนไขเหมือนเดิม
            is_blocked = node.walls.get(absolute_dir, True)
            already_explored = absolute_dir in node.exploredDirections
            target_exists = target_node_id in self.nodes
            target_fully_explored = False
            target_has_unexplored_exits = False
            
            if target_exists:
                target_node = self.nodes[target_node_id]
                target_fully_explored = target_node.fullyScanned
                # ✅ NEW: Check if target node still has unexplored exits
                target_has_unexplored_exits = len(target_node.unexploredExits) > 0
            
            print(f"   📍 {relative_dir} ({absolute_dir}):")
            print(f"      🚧 Blocked: {is_blocked}")
            print(f"      ✅ Already explored: {already_explored}")
            print(f"      🏗️  Target exists: {target_exists}")
            print(f"      🔍 Target fully explored: {target_fully_explored}")
            print(f"      🎯 Target has unexplored exits: {target_has_unexplored_exits}")
            
            # ✅ FIXED: More flexible exploration logic
            should_explore = (not is_blocked and 
                            not already_explored and 
                            (not target_exists or not target_fully_explored or target_has_unexplored_exits))
            
            if should_explore:
                node.unexploredExits.append(absolute_dir)
                print(f"      ✅ ADDED to unexplored exits! (Priority: {relative_dir})")
            else:
                print(f"      ❌ NOT added to unexplored exits")
        
        print(f"🎯 Final unexplored exits (ordered by priority): {node.unexploredExits}")
        
        # อัปเดต frontier queue
        has_unexplored = len(node.unexploredExits) > 0
        
        if has_unexplored and node.id not in self.frontierQueue:
            self.frontierQueue.append(node.id)
            print(f"🚀 Added {node.id} to frontier queue")
        elif not has_unexplored and node.id in self.frontierQueue:
            self.frontierQueue.remove(node.id)
            print(f"🧹 Removed {node.id} from frontier queue")
        
        # Dead end detection
        blocked_count = sum(1 for blocked in node.walls.values() if blocked)
        is_dead_end = blocked_count >= 3 and len(node.unexploredExits) == 0
        node.isDeadEnd = is_dead_end
        
        if is_dead_end:
            print(f"🚫 DEAD END CONFIRMED at {node.id} - {blocked_count} walls detected!")
            if node.id in self.frontierQueue:
                self.frontierQueue.remove(node.id)
                print(f"🧹 Removed dead end {node.id} from frontier queue")

    def find_next_exploration_direction(self):
        """Find the next ABSOLUTE direction to explore"""
        current_node = self.get_current_node()
        if not current_node:
            return None
        
        if self.is_dead_end(current_node):
            print(f"🚫 Current node is a dead end - no exploration directions available")
            return None
        
        # Return first unexplored exit (now using absolute directions)
        if current_node.unexploredExits:
            for unexplored_dir in current_node.unexploredExits:
                if self.can_move_to_direction_absolute(unexplored_dir):
                    return unexplored_dir
        
        return None
    
    def find_path_to_frontier(self, target_node_id):
        """Find shortest path to frontier node using BFS"""
        if target_node_id not in self.nodes:
            return None
        
        # BFS to find shortest path
        queue = deque([(self.currentPosition, [])])
        visited = set()
        visited.add(self.currentPosition)
        
        while queue:
            current_pos, path = queue.popleft()
            current_node_id = self.get_node_id(current_pos)
            
            # Check if we reached the target
            if current_node_id == target_node_id:
                return path
            
            # Explore neighbors
            if current_node_id in self.nodes:
                current_node = self.nodes[current_node_id]
                x, y = current_pos
                
                # Check all four directions
                directions = {
                    'north': (x, y + 1),
                    'south': (x, y - 1), 
                    'east': (x + 1, y),
                    'west': (x - 1, y)
                }
                
                for direction, neighbor_pos in directions.items():
                    neighbor_id = self.get_node_id(neighbor_pos)
                    
                    if (neighbor_pos not in visited and 
                        neighbor_id in self.nodes and
                        self.is_path_clear_absolute(current_pos, neighbor_pos, direction)):
                        
                        visited.add(neighbor_pos)
                        new_path = path + [direction]
                        queue.append((neighbor_pos, new_path))
        
        return None  # No path found
    
    def is_path_clear_absolute(self, from_pos, to_pos, direction):
        """Check if path between two adjacent nodes is clear using absolute directions"""
        from_node_id = self.get_node_id(from_pos)
        to_node_id = self.get_node_id(to_pos)
        
        if from_node_id not in self.nodes or to_node_id not in self.nodes:
            return False
        
        from_node = self.nodes[from_node_id]
        
        # Check if there's a wall blocking the path in absolute direction
        is_blocked = from_node.walls.get(direction, True)
        return not is_blocked
    
    def execute_path_to_frontier_with_reverse(self, path, movement_controller, attitude_handler):
        """NEW: Execute path using reverse movements for backtracking WITH NODE COUNTING"""
        print(f"🗺️ Executing REVERSE path to frontier: {path}")
        
        # เช็คสถานะการเคลื่อนไหวก่อนเริ่ม backtrack
        movement_status = movement_controller.get_movement_status()
        print(f"📊 Movement status before backtracking: {movement_status}")

        drift_corrections_during_backtrack = 0
        initial_correction_count = movement_controller.total_drift_corrections

        for i, step_direction in enumerate(path):
            print(f"🔍 Step {i+1}/{len(path)}: Current position: {self.currentPosition}, moving {step_direction}")
            
            # Use reverse movement for backtracking (more efficient)
            success = self.reverse_to_absolute_direction(step_direction, movement_controller, attitude_handler)
            
            if not success:
                print(f"❌ Failed to reverse {step_direction} during backtracking!")
                return False
            
            # *** เพิ่มการนับโหนดสำหรับการ backtrack และเช็ค drift correction ***
            needs_correction = movement_controller.increment_node_visit_for_backtrack_with_correction(attitude_handler)
            
            # ถ้าเกิด drift correction ระหว่าง backtrack
            if needs_correction:
                drift_corrections_during_backtrack = movement_controller.total_drift_corrections - initial_correction_count
                print(f"✅ Backtrack drift correction #{drift_corrections_during_backtrack} completed during step {i+1}!")
            
            time.sleep(0.2)  # Brief pause between moves
        
        if drift_corrections_during_backtrack > 0:
            print(f"🔧 Total drift corrections during this backtrack: {drift_corrections_during_backtrack}")
        
        print(f"✅ Successfully reached frontier at {self.currentPosition}")
        return True
    
    def find_nearest_frontier(self):
        """Find frontier with better validation and prioritization"""
        print("🔍 === FINDING NEAREST FRONTIER ===")
        
        if not self.frontierQueue:
            print("🔄 No frontiers in queue - rebuilding...")
            self.rebuild_frontier_queue()
            
            if not self.frontierQueue:
                print("🎉 No unexplored areas found - exploration complete!")
                return None, None, None
        
        # Validate and prioritize frontiers
        valid_frontiers = []
        
        print(f"🔍 Checking {len(self.frontierQueue)} frontier candidates...")
        
        for frontier_id in self.frontierQueue[:]:
            if frontier_id not in self.nodes:
                print(f"❌ Removing non-existent frontier {frontier_id}")
                continue
                
            frontier_node = self.nodes[frontier_id]
            
            # Re-validate unexplored exits
            print(f"\n🔍 Validating frontier {frontier_id} at {frontier_node.position}:")
            print(f"   📋 Claimed unexplored exits: {frontier_node.unexploredExits}")
            
            # Re-check each unexplored exit
            valid_exits = []
            for exit_direction in frontier_node.unexploredExits[:]:
                target_pos = self.get_next_position_from(frontier_node.position, exit_direction)
                target_node_id = self.get_node_id(target_pos)
                
                # Check if this exit is still valid
                target_x, target_y = target_pos
                is_outside_boundary = (
                    target_x < self.min_x or target_x > self.max_x or
                    target_y < self.min_y or target_y > self.max_y
                )
                
                if is_outside_boundary:
                    print(f"      🚫 {exit_direction} -> {target_pos}: OUTSIDE BOUNDARY - removing from unexplored")
                    continue
                
                target_exists = target_node_id in self.nodes
                if target_exists:
                    target_node = self.nodes[target_node_id]
                    target_fully_explored = target_node.fullyScanned
                    target_has_unexplored = len(target_node.unexploredExits) > 0
                    print(f"      🎯 {exit_direction} -> {target_pos}: exists={target_exists}, fully_explored={target_fully_explored}, has_unexplored={target_has_unexplored}")
                    
                    # ✅ FIXED: More flexible validation - consider valid if target doesn't exist, isn't fully explored, OR still has unexplored exits
                    if not target_fully_explored or target_has_unexplored:
                        valid_exits.append(exit_direction)
                        print(f"         ✅ Still valid for exploration")
                    else:
                        print(f"         ❌ Target fully explored with no unexplored exits")
                else:
                    valid_exits.append(exit_direction)
                    print(f"      🎯 {exit_direction} -> {target_pos}: NEW AREA - valid for exploration")
            
            # Update the node's unexplored exits with validated list
            frontier_node.unexploredExits = valid_exits
            
            if valid_exits:
                valid_frontiers.append(frontier_id)
                print(f"   ✅ Frontier {frontier_id} is VALID with exits: {valid_exits}")
            else:
                print(f"   ❌ Frontier {frontier_id} has NO valid unexplored exits")
        
        # Update frontier queue with only valid frontiers
        self.frontierQueue = valid_frontiers
        
        if not valid_frontiers:
            print("🎉 No valid frontiers remaining - exploration complete!")
            return None, None, None
        
        print(f"\n🎯 Found {len(valid_frontiers)} valid frontiers: {valid_frontiers}")
        
        # Find the nearest valid frontier
        best_frontier = None
        best_direction = None
        shortest_path = None
        min_distance = float('inf')
        
        for frontier_id in valid_frontiers:
            frontier_node = self.nodes[frontier_id]
            
            # Find path to this frontier
            path = self.find_path_to_frontier(frontier_id)
            
            if path is not None:
                distance = len(path)
                print(f"   📍 {frontier_id}: distance={distance}, exits={frontier_node.unexploredExits}")
                
                if distance < min_distance:
                    min_distance = distance
                    best_frontier = frontier_id
                    best_direction = frontier_node.unexploredExits[0]  # Take first unexplored direction
                    shortest_path = path
            else:
                print(f"   ❌ {frontier_id}: No path found!")
        
        if best_frontier:
            print(f"\n🏆 SELECTED: {best_frontier} with direction {best_direction} (distance: {min_distance})")
            print(f"🗺️ Path: {shortest_path}")
        else:
            print(f"\n❌ No reachable frontiers found!")
        
        return best_frontier, best_direction, shortest_path
    
    def rebuild_frontier_queue(self):
        """Rebuild frontier queue with comprehensive validation"""
        print("🔄 === REBUILDING FRONTIER QUEUE ===")
        self.frontierQueue = []
        
        for node_id, node in self.nodes.items():
            print(f"\n🔍 Checking node {node_id} at {node.position}:")
            
            # Re-validate unexplored exits for this node
            valid_exits = []
            
            if hasattr(node, 'unexploredExits'):
                print(f"   📋 Current unexplored exits: {node.unexploredExits}")
                
                for exit_direction in node.unexploredExits:
                    target_pos = self.get_next_position_from(node.position, exit_direction)
                    target_node_id = self.get_node_id(target_pos)
                    
                    # Check boundary first
                    target_x, target_y = target_pos
                    is_outside_boundary = (
                        target_x < self.min_x or target_x > self.max_x or
                        target_y < self.min_y or target_y > self.max_y
                    )
                    
                    if is_outside_boundary:
                        print(f"      🚫 {exit_direction} -> {target_pos}: OUTSIDE BOUNDARY")
                        continue
                    
                    # Validate this exit
                    target_exists = target_node_id in self.nodes
                    if target_exists:
                        target_node = self.nodes[target_node_id]
                        target_has_unexplored = len(target_node.unexploredExits) > 0
                        # ✅ FIXED: More flexible validation
                        if not target_node.fullyScanned or target_has_unexplored:
                            valid_exits.append(exit_direction)
                            print(f"      ✅ {exit_direction} -> {target_pos}: Target not fully explored or has unexplored exits")
                        else:
                            print(f"      ❌ {exit_direction} -> {target_pos}: Target fully explored with no unexplored exits")
                    else:
                        valid_exits.append(exit_direction)
                        print(f"      ✅ {exit_direction} -> {target_pos}: NEW AREA")
            
            # Update node's unexplored exits with validated list
            node.unexploredExits = valid_exits
            
            # Add to frontier queue if it has valid unexplored exits
            if valid_exits:
                self.frontierQueue.append(node_id)
                print(f"   🚀 ADDED to frontier queue with exits: {valid_exits}")
            else:
                print(f"   ❌ No valid unexplored exits - not added to frontier")
        
        print(f"\n✅ Frontier queue rebuilt: {len(self.frontierQueue)} frontiers found")
        print(f"🎯 Active frontiers: {self.frontierQueue}")
    
    def print_graph_summary(self):
        print("\n" + "="*60)
        print("📊 GRAPH MAPPING SUMMARY")
        print("="*60)
        print(f"🤖 Current Position: {self.currentPosition}")
        print(f"🧭 Current Direction: {self.currentDirection}")
        print(f"🗺️  Total Nodes: {len(self.nodes)}")
        print(f"🚀 Frontier Queue: {len(self.frontierQueue)} nodes")
        print("-"*60)
        
        for node_id, node in self.nodes.items():
            print(f"\n📍 Node: {node.id} at {node.position}")
            print(f"   🔍 Fully Scanned: {node.fullyScanned}")
            print(f"   🧱 Walls (absolute): {node.walls}")
            print(f"   🔍 Unexplored exits: {node.unexploredExits}")
            print(f"   ✅ Explored directions: {node.exploredDirections}")
            print(f"   🎯 Is dead end: {node.isDeadEnd}")
            
            # เช็คว่าสำรวจครบทุกทิศทางแล้วหรือยัง
            all_directions = ['north', 'south', 'east', 'west']
            explored_count = 0
            for direction in all_directions:
                if node.walls.get(direction, False):  # มีกำแพง
                    explored_count += 1
                elif direction in node.exploredDirections:  # สำรวจแล้ว
                    explored_count += 1
                elif direction in node.outOfBoundsExits:  # นอกขอบเขต
                    explored_count += 1
            
            completion_percentage = (explored_count / 4) * 100
            print(f"   📊 Exploration completion: {explored_count}/4 directions ({completion_percentage:.0f}%)")
            
            if node.sensorReadings:
                print(f"   📡 Sensor readings:")
                for direction, reading in node.sensorReadings.items():
                    print(f"      {direction}: {reading:.2f}cm")
            
            # NEW: Display marker information
            if hasattr(node, 'markersFound') and node.markersFound:
                print(f"   🎯 Markers found:")
                for direction, marker_ids in node.markersFound.items():
                    print(f"      {direction}: {marker_ids}")
            
            if hasattr(node, 'markerScanResults') and node.markerScanResults:
                print(f"   📊 Marker scan details:")
                for direction, scan_result in node.markerScanResults.items():
                    if scan_result['marker_ids']:
                        print(f"      {direction}: {scan_result['marker_ids']} at {scan_result['distance']:.1f}cm")
        
        print("-"*60)
        
        # แสดงสรุปการสำรวจโดยรวม
        total_directions_possible = len(self.nodes) * 4
        total_directions_resolved = 0
        
        for node in self.nodes.values():
            for direction in ['north', 'south', 'east', 'west']:
                if (node.walls.get(direction, False) or 
                    direction in node.exploredDirections or 
                    direction in node.outOfBoundsExits):
                    total_directions_resolved += 1
        
        overall_completion = (total_directions_resolved / total_directions_possible) * 100
        print(f"🎯 OVERALL EXPLORATION STATUS:")
        print(f"   📊 Total directions resolved: {total_directions_resolved}/{total_directions_possible}")
        print(f"   🎯 Overall completion: {overall_completion:.1f}%")
        
        if self.frontierQueue:
            print(f"🚀 Next exploration targets: {self.frontierQueue}")
        else:
            print("🎉 EXPLORATION COMPLETE - No more frontiers!")
        print("="*60)

# ===== ToF Sensor Handler =====
class ToFSensorHandler:
    def __init__(self):
        self.CALIBRATION_SLOPE = 0.0894 
        self.CALIBRATION_Y_INTERCEPT = 3.8409
        self.WINDOW_SIZE = 5
        self.tof_buffer = []
        self.WALL_THRESHOLD = 50.00
        
        self.readings = {
            'front': [],
            'left': [],
            'right': []
        }
        
        self.current_scan_direction = None
        self.collecting_data = False
        
    def calibrate_tof_value(self, raw_tof_mm):
        calibrated_cm = (self.CALIBRATION_SLOPE * raw_tof_mm) + self.CALIBRATION_Y_INTERCEPT
        return calibrated_cm
    
    def apply_median_filter(self, data, window_size):
        if len(data) == 0:
            return 0.0 
        if len(data) < window_size:
            return data[-1] 
        else:
            filtered = median_filter(data[-window_size:], size=window_size)
            return filtered[-1]
    
    def tof_data_handler(self, sub_info):
        if not self.collecting_data or not self.current_scan_direction:
            return
            
        raw_tof_mm = sub_info[0]
        
        if raw_tof_mm <= 0 or raw_tof_mm > 4000:
            return
            
        calibrated_tof_cm = self.calibrate_tof_value(raw_tof_mm)
        self.tof_buffer.append(calibrated_tof_cm)
        filtered_tof_cm = self.apply_median_filter(self.tof_buffer, self.WINDOW_SIZE)
        
        if len(self.tof_buffer) <= 20:
            self.readings[self.current_scan_direction].append({
                'filtered_cm': filtered_tof_cm,
                'timestamp': datetime.now().isoformat()
            })
        
        wall_status = "🧱 WALL" if filtered_tof_cm <= self.WALL_THRESHOLD else "🚪 OPEN"
        print(f"[{self.current_scan_direction.upper()}] {filtered_tof_cm:.2f}cm {wall_status}")
    
    def start_scanning(self, direction):
        self.current_scan_direction = direction
        self.tof_buffer.clear()
        if direction not in self.readings:
            self.readings[direction] = []
        else:
            self.readings[direction].clear()
        self.collecting_data = True
        
    def stop_scanning(self, unsub_distance_func):
        self.collecting_data = False
        try:
            unsub_distance_func()
        except:
            pass
    
    def get_average_distance(self, direction):
        if direction not in self.readings or len(self.readings[direction]) == 0:
            return 0.0
        
        filtered_values = [reading['filtered_cm'] for reading in self.readings[direction]]
        
        if len(filtered_values) > 4:
            q1 = np.percentile(filtered_values, 25)
            q3 = np.percentile(filtered_values, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            filtered_values = [x for x in filtered_values if lower_bound <= x <= upper_bound]
        
        return np.mean(filtered_values) if filtered_values else 0.0
    
    def is_wall_detected(self, direction):
        avg_distance = self.get_average_distance(direction)
        is_wall = avg_distance <= self.WALL_THRESHOLD and avg_distance > 0
        
        print(f"🔍 Wall check [{direction.upper()}]: {avg_distance:.2f}cm -> {'WALL' if is_wall else 'OPEN'}")
        
        return is_wall

# ===== Marker Scanning Helper Function =====
def scan_for_markers_at_direction(gimbal, sensor, angle, direction_name, marker_handler, tof_handler, speed=480):
    """
    สแกนหา marker ที่ทิศทางและมุมที่กำหนด
    
    Args:
        gimbal: Gimbal object
        sensor: Sensor object for ToF distance measurement
        angle: มุม gimbal yaw (0, -90, 90, 180)
        direction_name: ชื่อทิศทาง (front, left, right, back)
        marker_handler: MarkerVisionHandler object
        tof_handler: ToFSensorHandler object
        speed: ความเร็วการหมุน gimbal
        
    Returns:
        dict: {'marker_ids': [], 'distance': float, 'direction_name': str, 'angle': int, 'timestamp': str}
    """
    compass_dir = get_compass_direction(angle)
    print(f"🎯 Scanning {direction_name} | Gimbal Yaw: {angle}° | Compass: {compass_dir}")
    
    # หมุน gimbal ไปยังมุมที่กำหนด
    gimbal.moveto(pitch=-20, yaw=angle, pitch_speed=speed, yaw_speed=speed).wait_for_completed()
    time.sleep(0.3)  # รอให้กิมบอลเสถียร
    
    # วัดระยะทาง
    tof_handler.start_scanning('temp_scan')
    sensor.sub_distance(freq=50, callback=tof_handler.tof_data_handler)
    time.sleep(0.15)
    tof_handler.stop_scanning(sensor.unsub_distance)
    
    distance = tof_handler.get_average_distance('temp_scan')
    print(f"   📐 Distance: {distance:.2f}cm at {angle}°")
    
    marker_ids = []
    
    # ตรวจ marker เฉพาะถ้าระยะใกล้พอ และมี ToF reading ที่ถูกต้อง
    if distance > 0 and distance <= 40.0:
        print("✅ Distance OK - Scanning for markers...")
        
        # รีเซ็ตและเริ่มสแกน marker
        marker_handler.reset_detection()
        detected = marker_handler.wait_for_markers(timeout=1.0)
        
        if detected and marker_handler.markers:
            marker_ids = [m.id for m in marker_handler.markers]
            print(f"🎯 FOUND MARKERS: {marker_ids}")
            print(f"   📍 Direction: {direction_name} ({angle}°)")
            print(f"   📏 Distance: {distance:.2f}cm")
            print(f"   🧭 Compass: {compass_dir}")
        else:
            print(f"❌ No markers found at {direction_name} ({angle}°)")
    else:
        if distance <= 0:
            print(f"❌ Invalid ToF reading ({distance:.2f}cm) at {angle}° - Skipping marker detection")
        else:
            print(f"❌ Distance too far ({distance:.2f}cm > 40cm) at {angle}° - Skipping marker detection")
    
    return {
        'marker_ids': marker_ids,
        'distance': distance,
        'direction_name': direction_name,
        'angle': angle,
        'compass_direction': compass_dir,
        'timestamp': datetime.now().isoformat()
    }

# ===== Red Color Detection Function =====
def detect_red(ep_camera, threshold_area=100, attempts=3):
    """
    ตรวจจับสีแดงจากกล้องหน้า
    อ่าน frame หลายครั้ง (attempts) ป้องกัน queue empty
    คืนค่า True/False ถ้าเจอสีแดง
    """
    try:
        for _ in range(attempts):
            try:
                frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
                if frame is None:
                    continue
                    
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # ช่วงสีแดง (ปรับปรุงแล้วให้ครอบคลุมมากขึ้น)
                lower_red1 = np.array([0, 120, 70])
                upper_red1 = np.array([10, 255, 255])
                lower_red2 = np.array([170, 120, 70])
                upper_red2 = np.array([180, 255, 255])

                mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                mask = mask1 | mask2

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if cv2.contourArea(cnt) > threshold_area:
                        return True
                        
                # ถ้าไม่เจอสีแดงใน frame นี้ ให้ลองใหม่
                time.sleep(0.05)
                
            except Exception as e:
                print(f"❌ Error reading frame: {e}")
                time.sleep(0.1)
                
        return False
    except Exception as e:
        print(f"❌ detect_red error: {e}")
        return False

def scan_red_then_marker_enhanced(ep_robot, gimbal, chassis, sensor, marker_handler, tof_handler):
    """
    Enhanced version: สแกนหาสีแดงก่อน แล้วสแกน marker เฉพาะทิศทางที่เจอสีแดง
    ใช้เป็นทางเลือกแทนการสแกน marker ทุกทิศทาง
    """
    yaw_angles = [0, -90, 90]  # หน้า, ซ้าย, ขวา
    red_angles = []
    results = {}

    ep_camera = ep_robot.camera
    
    # เปิด Video Stream และรอให้เสถียร
    try:
        ep_camera.start_video_stream(display=False, resolution="720p")
        print("📹 Starting camera stream for red detection...")
        time.sleep(1.0)  # รอให้ frame มาเสถียร
    except Exception as e:
        print(f"❌ Error starting camera: {e}")
        return {}

    # ล็อคล้อหุ่นยนต์
    chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)

    print("\n🔴 === SCANNING FOR RED COLOR ===")
    # ตรวจสีแดงในแต่ละทิศทาง
    for yaw in yaw_angles:
        direction_name = get_direction_name(yaw)
        print(f"\n🔄 หมุน Gimbal ไปที่ {direction_name} ({yaw}°) เพื่อตรวจจับสีแดง")
        
        gimbal.moveto(pitch=0, yaw=yaw, pitch_speed=480, yaw_speed=480).wait_for_completed()
        time.sleep(0.3)  # รอให้กล้องเสถียร
        
        found_red = detect_red(ep_camera, threshold_area=100, attempts=5)
        if found_red:
            print(f"✅ เจอสีแดงที่ {direction_name} ({yaw}°)")
            red_angles.append(yaw)
        else:
            print(f"❌ ไม่เจอสีแดงที่ {direction_name} ({yaw}°)")

    # ปิด Video Stream หลังจากตรวจสีแดงเสร็จ
    try:
        ep_camera.stop_video_stream()
    except:
        pass

    if not red_angles:
        print("\n❌ ไม่เจอสีแดงในทิศทางใดเลย")
        # กลับไปตำแหน่งกลาง
        gimbal.moveto(pitch=0, yaw=0, pitch_speed=480, yaw_speed=480).wait_for_completed()
        chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
        return results

    print(f"\n🎯 === SCANNING MARKERS (เฉพาะทิศทางที่เจอสีแดง) ===")
    print(f"🔴 เจอสีแดงใน {len(red_angles)} ทิศทาง: {red_angles}")

    # scan marker เฉพาะ yaw ที่เจอสีแดง
    for yaw in red_angles:
        direction_name = get_direction_name(yaw)
        direction_key = direction_name.split(' ')[0].lower()  # Extract direction (front, left, right)
        
        print(f"\n🎯 สแกน Marker ที่ {direction_name} ({yaw}°)")
        
        marker_scan_result = scan_for_markers_at_direction(gimbal, sensor, yaw, direction_key, marker_handler, tof_handler, speed=480)
        
        results[direction_key] = {
            'direction_name': direction_name,
            'marker_ids': marker_scan_result['marker_ids'],
            'distance': marker_scan_result['distance'],
            'found_red': True,
            'angle': yaw,
            'compass_direction': marker_scan_result['compass_direction'],
            'timestamp': marker_scan_result['timestamp']
        }
        
        if marker_scan_result['marker_ids']:
            print(f"🎯 เจอ Marker: {marker_scan_result['marker_ids']} ที่ {direction_name} ({yaw}°)")
        else:
            print(f"❌ ไม่เจอ Marker ที่ {direction_name} ({yaw}°)")

    # กลับไปตำแหน่งกลาง
    print(f"\n🔄 กลับสู่ตำแหน่งกลาง...")
    gimbal.moveto(pitch=0, yaw=0, pitch_speed=480, yaw_speed=480).wait_for_completed()
    
    # ปลดล็อคล้อ
    chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0, timeout=0.1)

    return results

# ===== Main Exploration Functions =====
def scan_current_node_absolute(gimbal, chassis, sensor, tof_handler, graph_mapper, marker_handler=None, ep_robot=None):
    """NEW: Scan current node and update graph with ABSOLUTE directions + RED-FILTERED MARKER DETECTION"""
    print(f"\n🗺️ === Scanning Node at {graph_mapper.currentPosition} ===")
    
    current_node = graph_mapper.create_node(graph_mapper.currentPosition)
    
    # Check if node has been fully scanned before
    if current_node.fullyScanned:
        print(f"🔄 Node {current_node.id} already fully scanned - using cached data!")
        print(f"   🧱 Cached walls (absolute): {current_node.walls}")
        print(f"   🔍 Cached unexplored exits: {current_node.unexploredExits}")
        if current_node.sensorReadings:
            print(f"   📡 Cached sensor readings:")
            for direction, reading in current_node.sensorReadings.items():
                print(f"      {direction}: {reading:.2f}cm")
        # แสดงข้อมูล marker ที่เคยสแกนไว้
        if hasattr(current_node, 'markersFound') and current_node.markersFound:
            print(f"   🎯 Cached markers found:")
            for direction, marker_ids in current_node.markersFound.items():
                print(f"      {direction}: {marker_ids}")
        print("⚡ Skipping physical scan - using cached data")
        return current_node.sensorReadings
    
    # Only scan if node hasn't been fully scanned before
    print(f"🆕 First time visiting node {current_node.id} - performing full scan")
    print(f"🧭 Robot currently facing: {graph_mapper.currentDirection}")
    
    # Lock wheels
    chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
    time.sleep(0.2)
    
    speed = 480
    scan_results = {}
    red_directions = []  # เก็บทิศทางที่เจอสีแดง
    
    # ===== เปิด Video Stream สำหรับการตรวจสีแดง =====
    ep_camera = None
    if marker_handler and ep_robot:
        try:
            ep_camera = ep_robot.camera
            ep_camera.start_video_stream(display=False, resolution="720p")
            print("📹 Video stream started for red color detection")
            time.sleep(0.5)  # รอให้ camera เสถียร
        except Exception as e:
            print(f"⚠️ Cannot start video stream for red detection: {e}")
            ep_camera = None
    
    # ===== SCAN FRONT (0°) - ToF + Red Detection =====
    print("🔍 Scanning FRONT (0°) - ToF + Red Detection...")
    gimbal.moveto(pitch=0, yaw=0, pitch_speed=speed, yaw_speed=speed).wait_for_completed()
    time.sleep(0.2)
    
    # ToF scan
    tof_handler.start_scanning('front')
    sensor.sub_distance(freq=25, callback=tof_handler.tof_data_handler)
    time.sleep(0.2)
    tof_handler.stop_scanning(sensor.unsub_distance)
    
    front_distance = tof_handler.get_average_distance('front')
    front_wall = tof_handler.is_wall_detected('front')
    scan_results['front'] = front_distance
    
    print(f"📏 FRONT ToF result: {front_distance:.2f}cm - {'WALL' if front_wall else 'OPEN'}")
    
    # Red detection at FRONT
    if ep_camera and front_distance > 0 and front_distance <= 40.0:
        print("🔴 Checking for red color at FRONT...")
        found_red = detect_red(ep_camera, threshold_area=100, attempts=3)
        if found_red:
            print("✅ RED DETECTED at FRONT!")
            red_directions.append(('front', 0))
        else:
            print("❌ No red at FRONT")

    # Position adjustment if too close
    if front_distance <= 19.0:
        move_distance = -(23 - front_distance)
        print(f"⚠️ FRONT too close ({front_distance:.2f}cm)! Moving back {move_distance:.2f}m")
        ep_chassis.move(x=move_distance/100, y=0, xy_speed=0.2).wait_for_completed()
        time.sleep(0.2)

    # ===== SCAN LEFT (-90°) - ToF + Red Detection =====
    print("🔍 Scanning LEFT (-90°) - ToF + Red Detection...")
    gimbal.moveto(pitch=0, yaw=-90, pitch_speed=speed, yaw_speed=speed).wait_for_completed()
    time.sleep(0.2)
    
    # ToF scan
    tof_handler.start_scanning('left')
    sensor.sub_distance(freq=25, callback=tof_handler.tof_data_handler)
    time.sleep(0.2)
    tof_handler.stop_scanning(sensor.unsub_distance)
    
    left_distance = tof_handler.get_average_distance('left')
    left_wall = tof_handler.is_wall_detected('left')
    scan_results['left'] = left_distance
    
    print(f"📏 LEFT ToF result: {left_distance:.2f}cm - {'WALL' if left_wall else 'OPEN'}")
    
    # Red detection at LEFT
    if ep_camera and left_distance > 0 and left_distance <= 40.0:
        print("🔴 Checking for red color at LEFT...")
        found_red = detect_red(ep_camera, threshold_area=100, attempts=3)
        if found_red:
            print("✅ RED DETECTED at LEFT!")
            red_directions.append(('left', -90))
        else:
            print("❌ No red at LEFT")

    # Position adjustment if too close
    if left_distance < 15:
        move_distance = 20 - left_distance
        print(f"⚠️ LEFT too close ({left_distance:.2f}cm)! Moving right {move_distance:.2f}m")
        ep_chassis.move(x=0.01, y=move_distance/100, xy_speed=0.5).wait_for_completed()
        time.sleep(0.3)

    # ===== SCAN RIGHT (90°) - ToF + Red Detection =====
    print("🔍 Scanning RIGHT (90°) - ToF + Red Detection...")
    gimbal.moveto(pitch=0, yaw=90, pitch_speed=speed, yaw_speed=speed).wait_for_completed()
    time.sleep(0.2)
    
    # ToF scan
    tof_handler.start_scanning('right')
    sensor.sub_distance(freq=25, callback=tof_handler.tof_data_handler)
    time.sleep(0.2)
    tof_handler.stop_scanning(sensor.unsub_distance)
    
    right_distance = tof_handler.get_average_distance('right')
    right_wall = tof_handler.is_wall_detected('right')
    scan_results['right'] = right_distance

    print(f"📏 RIGHT ToF result: {right_distance:.2f}cm - {'WALL' if right_wall else 'OPEN'}")
    
    # Red detection at RIGHT
    if ep_camera and right_distance > 0 and right_distance <= 40.0:
        print("🔴 Checking for red color at RIGHT...")
        found_red = detect_red(ep_camera, threshold_area=100, attempts=3)
        if found_red:
            print("✅ RED DETECTED at RIGHT!")
            red_directions.append(('right', 90))
        else:
            print("❌ No red at RIGHT")

    # Position adjustment if too close
    if right_distance < 15:
        move_distance = -(21 - right_distance)
        print(f"⚠️ RIGHT too close ({right_distance:.2f}cm)! Moving left {move_distance:.2f}m")
        ep_chassis.move(x=0.01, y=move_distance/100, xy_speed=0.5).wait_for_completed()
        time.sleep(0.3)

    # ===== SPECIAL BACK SCAN FOR INITIAL NODE =====
    if graph_mapper.currentPosition == (0, 0) and current_node.initialScanDirection == graph_mapper.currentDirection:
        print("🔍 Special check: scanning BACK at start node...")
        gimbal.moveto(pitch=0, yaw=180, pitch_speed=speed, yaw_speed=speed).wait_for_completed()
        time.sleep(0.2)

        # ToF scan for back
        tof_handler.start_scanning('back')
        sensor.sub_distance(freq=25, callback=tof_handler.tof_data_handler)
        time.sleep(0.2)
        tof_handler.stop_scanning(sensor.unsub_distance)

        back_distance = tof_handler.get_average_distance('back')
        back_wall = tof_handler.is_wall_detected('back')
        scan_results['back'] = back_distance
        
        print(f"📏 BACK ToF result: {back_distance:.2f}cm - {'WALL' if back_wall else 'OPEN'}")

        # Red detection at BACK
        if ep_camera and back_distance > 0 and back_distance <= 40.0:
            print("🔴 Checking for red color at BACK...")
            found_red = detect_red(ep_camera, threshold_area=100, attempts=3)
            if found_red:
                print("✅ RED DETECTED at BACK!")
                red_directions.append(('back', 180))
            else:
                print("❌ No red at BACK")

        # Update walls for back direction
        direction_map = {
            'north': 'south',
            'south': 'north',
            'east': 'west',
            'west': 'east'
        }
        back_abs_dir = direction_map[graph_mapper.currentDirection]
        current_node.walls[back_abs_dir] = back_wall
        current_node.wallBack = back_wall

        if back_wall and back_abs_dir in current_node.unexploredExits:
            current_node.unexploredExits.remove(back_abs_dir)
            print(f"🧹 Removed BACK ({back_abs_dir}) from unexplored exits at start node")

    # ===== ปิด Video Stream =====
    if ep_camera:
        try:
            ep_camera.stop_video_stream()
            print("📹 Video stream stopped")
        except:
            pass

    # ===== MARKER SCANNING - เฉพาะทิศทางที่เจอสีแดง =====
    if marker_handler and red_directions:
        print(f"\n🎯 === MARKER SCANNING (Red-filtered) ===")
        print(f"🔴 Found red in {len(red_directions)} directions: {[d[0] for d in red_directions]}")
        
        for direction_name, angle in red_directions:
            print(f"\n🎯 Scanning markers at {direction_name.upper()} ({angle}°) where red was detected...")
            
            # หมุนไปที่มุมที่เจอสีแดงและก้มลง (-20°)
            gimbal.moveto(pitch=-20, yaw=angle, pitch_speed=speed, yaw_speed=speed).wait_for_completed()
            time.sleep(0.3)
            
            # วัดระยะทางใหม่ (เพราะก้มลงแล้ว)
            tof_handler.start_scanning('marker_scan')
            sensor.sub_distance(freq=50, callback=tof_handler.tof_data_handler)
            time.sleep(0.15)
            tof_handler.stop_scanning(sensor.unsub_distance)
            
            marker_distance = tof_handler.get_average_distance('marker_scan')
            print(f"   📐 Marker scan distance (tilted): {marker_distance:.2f}cm")
            
            # ตรวจ marker
            marker_ids = []
            if marker_distance > 0 and marker_distance <= 40.0:
                print("   ✅ Distance OK - Scanning for markers...")
                marker_handler.reset_detection()
                detected = marker_handler.wait_for_markers(timeout=1.0)
                
                if detected and marker_handler.markers:
                    marker_ids = [m.id for m in marker_handler.markers]
                    print(f"   🎯 FOUND MARKERS: {marker_ids}")
                else:
                    print(f"   ❌ No markers found")
            else:
                print(f"   ❌ Distance not suitable for marker detection: {marker_distance:.2f}cm")
            
            # เก็บผลลัพธ์
            current_node.markerScanResults[direction_name] = {
                'marker_ids': marker_ids,
                'distance': marker_distance,
                'direction_name': direction_name,
                'angle': angle,
                'compass_direction': get_compass_direction(angle),
                'timestamp': datetime.now().isoformat(),
                'found_red': True
            }
            
            if marker_ids:
                current_node.markersFound[direction_name] = marker_ids
                current_node.hasMarkers = True
                print(f"   ✅ Stored markers for {direction_name}: {marker_ids}")
    
    elif marker_handler:
        print(f"\n🔴 No red color detected in any direction - skipping marker scanning")
    
    # Return to center
    gimbal.moveto(pitch=0, yaw=0, pitch_speed=speed, yaw_speed=speed).wait_for_completed()
    time.sleep(0.2)
    
    # Unlock wheels
    chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0, timeout=0.1)
    time.sleep(0.2)
    
    # Update graph with wall information using ABSOLUTE directions
    try:
        print(f"🔄 Updating node walls and unexplored exits...")
        print(f"   📍 Position: {graph_mapper.currentPosition}")
        print(f"   🧭 Facing: {graph_mapper.currentDirection}")
        print(f"   🧱 Wall data: Left={left_wall}, Right={right_wall}, Front={front_wall}")
        
        # ✅ NEW: Use timeout protection for boundary processing
        with_timeout(graph_mapper.update_current_node_walls_absolute, 10, left_wall, right_wall, front_wall)
        current_node.sensorReadings = scan_results
        
        print(f"✅ Wall update completed successfully")
    except Exception as e:
        print(f"❌ Error updating node information: {e}")
        print("⚠️ Continuing with default wall configuration...")
        # Set safe defaults
        current_node.sensorReadings = scan_results
    
    print(f"✅ Node {current_node.id} scan complete:")
    print(f"   🧱 Walls detected (relative): Left={left_wall}, Right={right_wall}, Front={front_wall}")
    print(f"   🧱 Walls stored (absolute): {current_node.walls}")
    print(f"   📏 Distances: Left={left_distance:.1f}cm, Right={right_distance:.1f}cm, Front={front_distance:.1f}cm")
    
    # แสดงสรุป marker ที่พบ
    if hasattr(current_node, 'markersFound') and current_node.markersFound:
        print(f"   🎯 Markers summary:")
        for direction, marker_ids in current_node.markersFound.items():
            print(f"      {direction.upper()}: {marker_ids}")
    
    return scan_results

def explore_autonomously_with_absolute_directions(gimbal, chassis, sensor, tof_handler, graph_mapper, movement_controller, attitude_handler, marker_handler=None, ep_robot=None, max_nodes=20):
    """Main autonomous exploration with attitude drift correction INCLUDING BACKTRACKING + RED-FILTERED MARKER DETECTION"""
    print("\n🚀 === STARTING AUTONOMOUS EXPLORATION WITH RED-FILTERED MARKER DETECTION ===")
    print(f"🎯 Wall Detection Threshold: {tof_handler.WALL_THRESHOLD}cm")
    print(f"🔧 Attitude Drift Correction: Every {movement_controller.DRIFT_CORRECTION_INTERVAL} nodes (+{movement_controller.DRIFT_CORRECTION_ANGLE}° right)")
    if marker_handler:
        print(f"🔴 Red Color Filtering: Enabled - markers scanned only where red detected")
        print(f"📐 Marker Scan Mode: Gimbal tilt to -20° for optimal detection")
    else:
        print(f"⚠️ Marker Detection: Disabled")
    
    nodes_explored = 0
    scanning_iterations = 0
    dead_end_reversals = 0
    backtrack_attempts = 0
    reverse_backtracks = 0
    frontier_rebuild_attempts = 0  # นับจำนวนครั้งที่พยายาม rebuild frontier
    
    while nodes_explored < max_nodes:
        print(f"\n{'='*50}")
        print(f"--- EXPLORATION STEP {nodes_explored + 1} ---")
        print(f"🤖 Current position: {graph_mapper.currentPosition}")
        print(f"🧭 Current direction (absolute): {graph_mapper.currentDirection}")
        
        # แสดงสถานะ drift correction
        drift_status = movement_controller.get_drift_correction_status()
        print(f"🔧 Comprehensive Drift Correction Status:")
        print(f"   📊 Total nodes visited (including backtrack): {drift_status['nodes_visited']}")
        print(f"   ⏳ Next correction at node: {drift_status['next_correction_at']}")
        print(f"   ⏰ Nodes until correction: {drift_status['nodes_until_correction']}")
        print(f"   🔄 Total corrections done: {drift_status['total_corrections']}")
        print(f"   📍 Last correction at node: {drift_status['last_correction_at']}")
        print(f"{'='*50}")
        
        # *** เพิ่มจำนวนโหนดสำหรับ main exploration และเช็ค drift correction ***
        needs_drift_correction = movement_controller.increment_node_visit_main_exploration(attitude_handler)
        
        if needs_drift_correction:
            print(f"✅ Main exploration drift correction completed!")
        
        # Check if current node needs scanning
        current_node = graph_mapper.create_node(graph_mapper.currentPosition)
        
        if not current_node.fullyScanned:
            print("🔍 NEW NODE - Performing full scan...")
            try:
                # ✅ NEW: Add timeout protection for scanning
                scan_results = with_timeout(scan_current_node_absolute, 30, gimbal, chassis, sensor, tof_handler, graph_mapper, marker_handler, ep_robot)
                scanning_iterations += 1
            except Exception as e:
                print(f"❌ Error during node scanning: {e}")
                print("⚠️ Continuing with cached data or default values...")
                scan_results = {}
            
            # ✅ NEW: Mark the direction as explored from previous node after successful scan
            if hasattr(graph_mapper, 'previous_node') and graph_mapper.previous_node:
                # Find the direction we came from
                prev_pos = graph_mapper.previous_node.position
                curr_pos = current_node.position
                
                # Calculate the direction from previous to current
                dx = curr_pos[0] - prev_pos[0]
                dy = curr_pos[1] - prev_pos[1]
                
                if dx == 1 and dy == 0:
                    came_from_direction = 'east'
                elif dx == -1 and dy == 0:
                    came_from_direction = 'west'
                elif dx == 0 and dy == 1:
                    came_from_direction = 'north'
                elif dx == 0 and dy == -1:
                    came_from_direction = 'south'
                else:
                    came_from_direction = None
                
                if came_from_direction and came_from_direction not in graph_mapper.previous_node.exploredDirections:
                    graph_mapper.previous_node.exploredDirections.append(came_from_direction)
                    print(f"✅ Marked {came_from_direction} as explored from {graph_mapper.previous_node.id} after scanning")
            
            # Check if this scan revealed a dead end
            if graph_mapper.is_dead_end(current_node):
                print(f"🚫 DEAD END DETECTED after scanning!")
                print(f"🔙 Initiating reverse maneuver...")
                
                success = graph_mapper.handle_dead_end(movement_controller)
                if success:
                    dead_end_reversals += 1
                    print(f"✅ Successfully reversed from dead end (Total reversals: {dead_end_reversals})")
                    nodes_explored += 1
                    continue
                else:
                    print(f"❌ Failed to reverse from dead end!")
                    break
        else:
            print("⚡ REVISITED NODE - Using cached scan data (no physical scanning)")
            graph_mapper.update_unexplored_exits_absolute(current_node)
            graph_mapper.build_connections()
        
        nodes_explored += 1
        
        # Print current graph state only every 5 nodes to reduce spam
        if nodes_explored % 5 == 0 or nodes_explored == 1:
            graph_mapper.print_graph_summary()
        
        # Find next direction to explore
        graph_mapper.previous_node = current_node
        
        # STEP 1: Try to find unexplored direction from current node
        next_direction = graph_mapper.find_next_exploration_direction()
        
        if next_direction:
            print(f"\n🎯 Next exploration direction (absolute): {next_direction}")
            
            can_move = graph_mapper.can_move_to_direction_absolute(next_direction)
            print(f"🚦 Movement check: {'ALLOWED' if can_move else 'BLOCKED'}")
            
            if can_move:
                try:
                    success = graph_mapper.move_to_absolute_direction(next_direction, movement_controller, attitude_handler)
                    if success:
                        print(f"✅ Successfully moved to {graph_mapper.currentPosition}")
                        time.sleep(0.2)
                        continue
                    else:
                        print(f"❌ Movement failed - wall detected!")
                        if current_node and next_direction in current_node.unexploredExits:
                            current_node.unexploredExits.remove(next_direction)
                        continue
                    
                except Exception as e:
                    print(f"❌ Error during movement: {e}")
                    break
            else:
                print(f"🚫 Cannot move to {next_direction} - blocked by wall!")
                if current_node and next_direction in current_node.unexploredExits:
                    current_node.unexploredExits.remove(next_direction)
                continue
        
        # STEP 2: Backtracking logic
        print(f"🔍 No unexplored directions from current node - initiating backtracking...")
        backtrack_attempts += 1
        
        try:
            frontier_id, frontier_direction, path = graph_mapper.find_nearest_frontier()
            
            if frontier_id and path is not None and frontier_direction:
                print(f"🎯 Found frontier node {frontier_id} with unexplored direction: {frontier_direction}")
                print(f"🗺️ Path to frontier: {path} (distance: {len(path)} steps)")
                print("🔙 REVERSE BACKTRACK: Using reverse movements WITH drift correction!")
                
                try:
                    # *** เปลี่ยนการเรียกใช้ให้ส่ง attitude_handler ไปด้วย ***
                    success = graph_mapper.execute_path_to_frontier_with_reverse(path, movement_controller, attitude_handler)
                    
                    if success:
                        reverse_backtracks += 1
                        print(f"✅ Successfully REVERSE backtracked to frontier at {graph_mapper.currentPosition}")
                        print(f"   📊 Total reverse backtracks: {reverse_backtracks}")
                        
                        # แสดงสถานะ drift correction หลัง backtrack
                        updated_drift_status = movement_controller.get_drift_correction_status()
                        print(f"   🔧 Total nodes after backtrack: {updated_drift_status['nodes_visited']}")
                        print(f"   🔄 Total corrections: {updated_drift_status['total_corrections']}")
                        
                        time.sleep(0.2)
                        continue
                        
                    else:
                        print(f"❌ Failed to execute reverse backtracking path!")
                        break
                        
                except Exception as e:
                    print(f"❌ Error during reverse backtracking: {e}")
                    break
        except Exception as e:
            print(f"❌ Error finding frontier for backtracking: {e}")
            print("⚠️ Continuing to completion check...")
        else:
            # STEP 3: Final check
            frontier_rebuild_attempts += 1
            print(f"🎉 No more frontiers found! (Rebuild attempt {frontier_rebuild_attempts})")
            
            if frontier_rebuild_attempts > 2:
                print("🛑 Maximum frontier rebuild attempts reached - exploration complete!")
                break
                
            print("🔄 Performing final frontier scan...")
            graph_mapper.rebuild_frontier_queue()
            
            # เช็คว่าสำรวจครบทุกโหนดในขอบเขตแล้วหรือยัง
            boundary_info = graph_mapper.get_boundary_status()
            expected_total_nodes = boundary_info['total_cells']
            actual_nodes = len(graph_mapper.nodes)
            
            print(f"📊 Node coverage check:")
            print(f"   🎯 Expected nodes in boundary: {expected_total_nodes}")
            print(f"   ✅ Actual nodes explored: {actual_nodes}")
            print(f"   📈 Coverage: {actual_nodes/expected_total_nodes*100:.1f}%")
            
            if actual_nodes >= expected_total_nodes:
                print("🎉 All nodes in boundary explored!")
                # ทำความสะอาด frontier queue ขั้นสุดท้าย
                graph_mapper.frontierQueue.clear()
                break
            
            # ✅ NEW: Additional safety check from patch version - if we have any valid unexplored exits
            total_unexplored = sum(len(node.unexploredExits) for node in graph_mapper.nodes.values())
            print(f"📊 Total unexplored exits across all nodes: {total_unexplored}")
            
            if total_unexplored == 0:
                print("🎉 No more unexplored exits found anywhere - exploration complete!")
                break
            
            if graph_mapper.frontierQueue:
                print(f"🚀 Found {len(graph_mapper.frontierQueue)} missed frontiers - continuing...")
                continue
            else:
                print("🎉 EXPLORATION DEFINITELY COMPLETE!")
                break
        
        if nodes_explored >= max_nodes:
            print(f"⚠️ Reached maximum nodes limit ({max_nodes})")
            break
    
    # Final statistics
    final_drift_status = movement_controller.get_drift_correction_status()
    
    print(f"\n🎉 === EXPLORATION COMPLETED ===")
    print(f"📊 PERFORMANCE SUMMARY:")
    print(f"   🗺️ Total exploration steps: {nodes_explored}")
    print(f"   📊 Total nodes visited (including backtrack): {final_drift_status['nodes_visited']}")
    print(f"   🔍 Physical scans performed: {scanning_iterations}")
    print(f"   🔙 Dead end reversals: {dead_end_reversals}")
    print(f"   🔄 Backtrack attempts: {backtrack_attempts}")
    print(f"   🔙 Reverse backtracks: {reverse_backtracks}")
    print(f"   ⚡ Scans saved by caching: {nodes_explored - scanning_iterations}")
    
    print(f"\n🔧 COMPREHENSIVE ATTITUDE DRIFT CORRECTION SUMMARY:")
    print(f"   📊 Total nodes counted: {final_drift_status['nodes_visited']} (exploration + backtracking)")
    print(f"   🔄 Total corrections performed: {final_drift_status['total_corrections']}")
    print(f"   🎯 Total angle corrected: {final_drift_status['total_corrections'] * final_drift_status['correction_angle']}°")
    print(f"   📍 Correction interval: Every {final_drift_status['correction_interval']} nodes")
    print(f"   🔧 Correction angle: +{final_drift_status['correction_angle']}° per correction")
    print(f"   📈 Last correction at node: {final_drift_status['last_correction_at']}")
    
    backtrack_node_count = final_drift_status['nodes_visited'] - nodes_explored
    if final_drift_status['nodes_visited'] > 0:
        drift_frequency = final_drift_status['total_corrections'] / final_drift_status['nodes_visited']
        backtrack_percentage = (backtrack_node_count / final_drift_status['nodes_visited']) * 100
        print(f"   📊 Drift correction frequency: {drift_frequency:.2f} corrections per total node")
        print(f"   🔙 Backtrack nodes: {backtrack_node_count} ({backtrack_percentage:.1f}% of total)")
    
    # แสดง final summary เฉพาะครั้งเดียวตอนจบ
    print(f"\n{'🎉'*20} FINAL EXPLORATION SUMMARY {'🎉'*20}")
    graph_mapper.print_graph_summary()
    generate_exploration_report_absolute(graph_mapper, nodes_explored, dead_end_reversals, reverse_backtracks, final_drift_status)


def generate_exploration_report_absolute(graph_mapper, nodes_explored, dead_end_reversals=0, reverse_backtracks=0, final_drift_status=None):
    """Generate comprehensive exploration report with absolute direction info"""
    print(f"\n{'='*60}")
    print("📋 FINAL EXPLORATION REPORT (ABSOLUTE DIRECTIONS)")
    print(f"{'='*60}")
    
    # Basic statistics
    total_nodes = len(graph_mapper.nodes)
    dead_ends = sum(1 for node in graph_mapper.nodes.values() if node.isDeadEnd)
    frontier_nodes = len(graph_mapper.frontierQueue)
    fully_scanned_nodes = sum(1 for node in graph_mapper.nodes.values() if node.fullyScanned)
    
    # NEW: Marker statistics
    nodes_with_markers = sum(1 for node in graph_mapper.nodes.values() if hasattr(node, 'hasMarkers') and node.hasMarkers)
    total_markers_found = 0
    markers_by_direction = {'front': 0, 'left': 0, 'right': 0, 'back': 0}
    
    for node in graph_mapper.nodes.values():
        if hasattr(node, 'markersFound') and node.markersFound:
            for direction, marker_ids in node.markersFound.items():
                total_markers_found += len(marker_ids)
                if direction in markers_by_direction:
                    markers_by_direction[direction] += len(marker_ids)
    
    print(f"📊 STATISTICS:")
    print(f"   🏁 Total nodes explored: {total_nodes}")
    print(f"   🎯 Node visits: {nodes_explored}")
    print(f"   🔍 Fully scanned nodes: {fully_scanned_nodes}")
    print(f"   🚫 Dead ends found: {dead_ends}")
    print(f"   🔙 Dead end reversals performed: {dead_end_reversals}")
    print(f"   🔙 Reverse backtracks performed: {reverse_backtracks}")
    print(f"   🚀 Remaining frontiers: {frontier_nodes}")
    
    # NEW: Red-filtered marker detection summary
    print(f"\n🎯 RED-FILTERED MARKER DETECTION SUMMARY:")
    print(f"   📍 Nodes with markers: {nodes_with_markers}")
    print(f"   🎯 Total markers found: {total_markers_found}")
    
    # Count red detections from marker scan results
    red_detections = 0
    for node in graph_mapper.nodes.values():
        if hasattr(node, 'markerScanResults') and node.markerScanResults:
            for direction, scan_result in node.markerScanResults.items():
                if scan_result.get('found_red', False):
                    red_detections += 1
    
    print(f"   🔴 Red color detections: {red_detections}")
    if total_markers_found > 0:
        print(f"   📊 Markers by direction:")
        for direction, count in markers_by_direction.items():
            if count > 0:
                print(f"      {direction.upper()}: {count} markers")
        print(f"   📈 Marker detection rate: {nodes_with_markers/total_nodes*100:.1f}% of nodes")
        if red_detections > 0:
            print(f"   🎯 Marker success rate from red: {total_markers_found/red_detections*100:.1f}% when red detected")
    
    # Efficiency metrics
    revisited_nodes = nodes_explored - total_nodes
    if revisited_nodes > 0:
        print(f"   🔄 Node revisits (backtracking): {revisited_nodes}")
        print(f"   ⚡ Scans saved by caching: {revisited_nodes}")
        print(f"   📈 Scanning efficiency: {(revisited_nodes / nodes_explored * 100):.1f}% improvement")
    
    # Map boundaries
    if graph_mapper.nodes:
        positions = [node.position for node in graph_mapper.nodes.values()]
        min_x = min(pos[0] for pos in positions)
        max_x = max(pos[0] for pos in positions)
        min_y = min(pos[1] for pos in positions)
        max_y = max(pos[1] for pos in positions)
        
        print(f"\n🗺️ MAP BOUNDARIES:")
        print(f"   X range: {min_x} to {max_x} (width: {max_x - min_x + 1})")
        print(f"   Y range: {min_y} to {max_y} (height: {max_y - min_y + 1})")
    
    # Wall statistics using absolute directions
    wall_stats = {'north': 0, 'south': 0, 'east': 0, 'west': 0}
    opening_stats = {'north': 0, 'south': 0, 'east': 0, 'west': 0}
    
    for node in graph_mapper.nodes.values():
        if hasattr(node, 'walls') and node.walls:
            for direction, is_wall in node.walls.items():
                if is_wall:
                    wall_stats[direction] += 1
                else:
                    opening_stats[direction] += 1
    
    print(f"\n🧱 WALL ANALYSIS (ABSOLUTE DIRECTIONS):")
    total_walls = sum(wall_stats.values())
    total_openings = sum(opening_stats.values())
    
    print(f"   Total walls detected: {total_walls}")
    print(f"   Total openings detected: {total_openings}")
    if (total_walls + total_openings) > 0:
        print(f"   Wall density: {total_walls/(total_walls+total_openings)*100:.1f}%")
    
    print(f"   Walls by direction:")
    for direction in ['north', 'south', 'east', 'west']:
        total_dir = wall_stats[direction] + opening_stats[direction]
        if total_dir > 0:
            wall_pct = wall_stats[direction] / total_dir * 100
            print(f"      {direction.upper()}: {wall_stats[direction]} walls, {opening_stats[direction]} openings ({wall_pct:.1f}% walls)")
    
    # Movement efficiency summary
    print(f"\n🔙 MOVEMENT EFFICIENCY:")
    print(f"   Dead end reversals: {dead_end_reversals}")
    print(f"   Reverse backtracks: {reverse_backtracks}")
    print(f"   Total reverse movements: {dead_end_reversals + reverse_backtracks}")
    print(f"   Time saved vs 180° turns: ~{(dead_end_reversals + reverse_backtracks) * 2:.1f} seconds")
    
    # Unexplored areas
    if graph_mapper.frontierQueue:
        print(f"\n🔍 UNEXPLORED AREAS:")
        for frontier_id in graph_mapper.frontierQueue:
            node = graph_mapper.nodes[frontier_id]
            print(f"   📍 {node.position}: {len(node.unexploredExits)} unexplored exits {node.unexploredExits}")
    
    print(f"\n⭐ ABSOLUTE DIRECTION BENEFITS:")
    print(f"   🔴 Red color filtering integrated with ToF scanning")
    print(f"   🎯 Smart marker detection only where red color detected")
    print(f"   🔄 Efficient scanning: Red detection at 0°, -90°, 90° (and 180° for initial node)")
    print(f"   📐 Gimbal tilt (-20°) for marker scanning only in red-filtered directions")
    print(f"   📊 Comprehensive marker tracking per node and direction")
    print(f"   ⚡ Performance optimized: Skip marker scanning where no red detected")
    
    print(f"\n{'='*60}")
    print("✅ ABSOLUTE DIRECTION EXPLORATION WITH RED-FILTERED MARKER DETECTION COMPLETE")
    print(f"{'='*60}")


# 4. แก้ไขการตั้งค่า boundary ใน main
if __name__ == '__main__':
    print("🤖 Connecting to robot...")
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    
    ep_gimbal = ep_robot.gimbal
    ep_chassis = ep_robot.chassis
    ep_sensor = ep_robot.sensor
    ep_vision = ep_robot.vision
    
    # Initialize components with STRICTER boundaries
    tof_handler = ToFSensorHandler()
    graph_mapper = GraphMapper(min_x=-3, min_y=-3, max_x=3, max_y=3)  # 3x3 grid
    movement_controller = MovementController(ep_chassis)
    attitude_handler = AttitudeHandler()
    attitude_handler.start_monitoring(ep_chassis)
    
    # NEW: Initialize marker detection system
    marker_handler = MarkerVisionHandler()
    print("🎯 Initializing marker detection system...")
    marker_detection_active = marker_handler.start_continuous_detection(ep_vision)
    if marker_detection_active:
        print("✅ Marker detection system activated")
    else:
        print("⚠️ Marker detection system failed to activate - continuing without markers")
        marker_handler = None
    
    # ✅ เพิ่มการแสดงข้อมูล boundary
    boundary_info = graph_mapper.get_boundary_status()
    print(f"🗺️ MAP BOUNDARIES CONFIGURED:")
    print(f"   📏 X range: [{boundary_info['min_x']}, {boundary_info['max_x']}]")
    print(f"   📏 Y range: [{boundary_info['min_y']}, {boundary_info['max_y']}]")
    print(f"   📐 Map size: {boundary_info['width']}x{boundary_info['height']} = {boundary_info['total_cells']} cells")
    print(f"   🎯 Valid positions: Only within these boundaries!")
    
    try:
        print("✅ Recalibrating gimbal...")
        ep_gimbal.recenter(pitch_speed=100, yaw_speed=100).wait_for_completed()
        ep_gimbal.moveto(pitch=0, yaw=0, pitch_speed=50, yaw_speed=50).wait_for_completed()
        time.sleep(0.3)
        
        print(f"🎯 Wall Detection Threshold: {tof_handler.WALL_THRESHOLD}cm")
        
        # Start autonomous exploration with absolute directions
        explore_autonomously_with_absolute_directions(ep_gimbal, ep_chassis, ep_sensor, tof_handler, 
                           graph_mapper, movement_controller, attitude_handler, marker_handler, ep_robot, max_nodes=49)
            
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            ep_sensor.unsub_distance()
            if marker_handler:
                marker_handler.stop_continuous_detection(ep_vision)
            movement_controller.cleanup()
            attitude_handler.stop_monitoring(ep_chassis)
        except:
            pass
        ep_robot.close()
        print("🔌 Connection closed")

# ===== Timeout Handler for Preventing Hangs =====
class TimeoutHandler:
    def __init__(self, timeout_seconds=30):
        self.timeout_seconds = timeout_seconds
        self.timer = None
        
    def timeout_callback(self):
        print(f"⚠️ TIMEOUT WARNING: Operation exceeded {self.timeout_seconds} seconds!")
        # Don't actually exit, just print warning
        
    def start_timeout(self):
        if self.timer:
            self.timer.cancel()
        self.timer = threading.Timer(self.timeout_seconds, self.timeout_callback)
        self.timer.start()
        
    def cancel_timeout(self):
        if self.timer:
            self.timer.cancel()
            self.timer = None

def with_timeout(func, timeout_seconds=15, *args, **kwargs):
    """Execute function with timeout protection"""
    timeout_handler = TimeoutHandler(timeout_seconds)
    
    try:
        print(f"⏱️ Starting operation with {timeout_seconds}s timeout...")
        timeout_handler.start_timeout()
        result = func(*args, **kwargs)
        timeout_handler.cancel_timeout()
        print(f"✅ Operation completed successfully within timeout")
        return result
    except Exception as e:
        timeout_handler.cancel_timeout()
        print(f"❌ Operation failed: {e}")
        raise e