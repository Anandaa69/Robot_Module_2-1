# Integrated Target Tracking System
# Merges detect_GPT_BEST.py detection with mode.py PID control
# Features: Target detection, gimbal tracking with 3-degree elevation, fire control

import cv2
import numpy as np
import time
import math
import threading
import queue
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
processed_output = {"details": [], "target_center": None, "fire_command": False}
output_lock = threading.Lock()
stop_event = threading.Event()

# =====================================================
# PID Controller for Gimbal
# =====================================================
class GimbalPID:
    def __init__(self, kp_yaw=0.3, ki_yaw=0.01, kd_yaw=0.01, kp_pitch=0.25, ki_pitch=0.15, kd_pitch=0.03):
        self.kp_yaw = kp_yaw
        self.ki_yaw = ki_yaw
        self.kd_yaw = kd_yaw
        self.kp_pitch = kp_pitch
        self.ki_pitch = ki_pitch
        self.kd_pitch = kd_pitch
        
        self.prev_error_yaw = 0
        self.prev_error_pitch = 0
        self.integral_yaw = 0
        self.integral_pitch = 0
        self.last_time = time.time()
        
    def update(self, error_yaw, error_pitch):
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0:
            dt = 0.01
            
        # Yaw PID
        self.integral_yaw += error_yaw * dt
        derivative_yaw = (error_yaw - self.prev_error_yaw) / dt
        yaw_speed = self.kp_yaw * error_yaw + self.ki_yaw * self.integral_yaw + self.kd_yaw * derivative_yaw
        
        # Pitch PID (with 3-degree elevation offset)
        error_pitch_with_offset = error_pitch - 3.0  # 3 degrees up
        self.integral_pitch += error_pitch_with_offset * dt
        derivative_pitch = (error_pitch_with_offset - self.prev_error_pitch) / dt
        pitch_speed = self.kp_pitch * error_pitch_with_offset + self.ki_pitch * self.integral_pitch + self.kd_pitch * derivative_pitch
        
        # Anti-windup
        self.integral_yaw = np.clip(self.integral_yaw, -200, 200)
        self.integral_pitch = np.clip(self.integral_pitch, -200, 200)
        
        self.prev_error_yaw = error_yaw
        self.prev_error_pitch = error_pitch
        self.last_time = current_time
        
        return yaw_speed, pitch_speed

# =====================================================
# AWB / Night enhance
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
# Object Detection (from detect_GPT_BEST.py)
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
# Connection Manager
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
            self._safe_close()

    def get_camera(self):
        with self._lock:
            return None if self._robot is None else self._robot.camera

    def get_gimbal(self):
        with self._lock:
            return None if self._robot is None else self._robot.gimbal

    def close(self):
        with self._lock:
            self._safe_close()

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

def processing_thread_func(tracker: ObjectTracker, q: queue.Queue, target_shape, target_color, roi_coords):
    global processed_output
    print("🧠 Processing thread started.")
    ROI_X, ROI_Y, ROI_W, ROI_H = roi_coords
    frame_center_x = ROI_X + ROI_W // 2
    frame_center_y = ROI_Y + ROI_H // 2
    
    pid_controller = GimbalPID()
    target_detected = False
    fire_cooldown = 0

    while not stop_event.is_set():
        try:
            frame_to_process = q.get(timeout=1.0)
            roi_frame = frame_to_process[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
            detections = tracker.get_raw_detections(roi_frame)

            detailed_results = []
            target_center = None
            fire_command = False
            target_detected = False

            # Process detections
            for i, d in enumerate(detections):
                shape, color, (x,y,w,h) = d['shape'], d['color'], d['box']
                center_x = x + w // 2
                center_y = y + h // 2
                
                is_target = (shape == target_shape and color == target_color)
                
                detailed_results.append({
                    "id": i + 1,
                    "color": color,
                    "shape": shape,
                    "is_target": is_target,
                    "box": (x,y,w,h),
                    "center": (center_x, center_y)
                })
                
                if is_target:
                    target_center = (center_x + ROI_X, center_y + ROI_Y)
                    target_detected = True
                    
                    # Calculate error for PID
                    error_yaw = frame_center_x - target_center[0]
                    error_pitch = frame_center_y - target_center[1]
                    
                    # Update PID controller
                    yaw_speed, pitch_speed = pid_controller.update(error_yaw, error_pitch)
                    
                    # Fire control - only fire when target is centered and stable
                    if abs(error_yaw) < 20 and abs(error_pitch) < 20 and fire_cooldown <= 0:
                        fire_command = True
                        fire_cooldown = 30  # 1 second cooldown at 30fps

            # Update fire cooldown
            if fire_cooldown > 0:
                fire_cooldown -= 1

            with output_lock:
                processed_output = {
                    "details": detailed_results,
                    "target_center": target_center,
                    "fire_command": fire_command,
                    "target_detected": target_detected
                }

        except queue.Empty:
            continue
        except Exception as e:
            print(f"CRITICAL: Processing error: {e}")
            time.sleep(0.05)

    print("🛑 Processing thread stopped.")

# =====================================================
# Main Control Loop
# =====================================================
def main():
    target_shape, target_color = "Circle", "Red"
    print(f"🎯 Target set to: {target_color} {target_shape}")

    tracker = ObjectTracker(use_gpu=USE_GPU)
    
    # ROI settings
    ROI_Y, ROI_H, ROI_X, ROI_W = 264, 270, 10, 911

    manager = RMConnection()
    manager.connect()
    
    # Recenter gimbal at startup
    gimbal = manager.get_gimbal()
    if gimbal:
        print("🎯 Recentering gimbal...")
        gimbal.recenter(pitch_speed=200, yaw_speed=200).wait_for_completed()
        time.sleep(1.0)
        print("✅ Gimbal recentered")

    cap_t = threading.Thread(target=capture_thread_func, args=(manager, frame_queue), daemon=True)
    proc_t = threading.Thread(
        target=processing_thread_func,
        args=(tracker, frame_queue, target_shape, target_color, (ROI_X, ROI_Y, ROI_W, ROI_H)),
        daemon=True
    )
    cap_t.start(); proc_t.start()

    print("\n--- Integrated Target Tracking System ---")
    print("Controls: s: toggle detection, f: fire, r: recenter, q: quit")

    display_frame = None
    detection_enabled = True
    
    try:
        while not stop_event.is_set():
            try:
                display_frame = frame_queue.get(timeout=1.0)
            except queue.Empty:
                if display_frame is None:
                    print("Waiting for first frame...")
                time.sleep(0.2)
                continue

            # Draw ROI
            cv2.rectangle(display_frame, (ROI_X, ROI_Y), (ROI_X+ROI_W, ROI_Y+ROI_H), (255,0,0), 2)
            
            # Draw center crosshair
            center_x = ROI_X + ROI_W // 2
            center_y = ROI_Y + ROI_H // 2
            cv2.line(display_frame, (center_x-20, center_y), (center_x+20, center_y), (255,255,255), 2)
            cv2.line(display_frame, (center_x, center_y-20), (center_x, center_y+20), (255,255,255), 2)

            if detection_enabled:
                cv2.putText(display_frame, "MODE: TRACKING", (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                
                with output_lock:
                    details = processed_output["details"]
                    target_center = processed_output["target_center"]
                    fire_command = processed_output["fire_command"]
                    target_detected = processed_output["target_detected"]

                # Draw zone dividers
                d1_abs = ROI_X + int(ROI_W*0.33)
                d2_abs = ROI_X + int(ROI_W*0.66)
                cv2.line(display_frame, (d1_abs, ROI_Y), (d1_abs, ROI_Y+ROI_H), (255,255,0), 1)
                cv2.line(display_frame, (d2_abs, ROI_Y), (d2_abs, ROI_Y+ROI_H), (255,255,0), 1)

                # Draw detections
                for det in details:
                    x,y,w,h = det['box']
                    abs_x, abs_y = x + ROI_X, y + ROI_Y
                    
                    if det['is_target']:
                        box_color = (0,0,255)  # Red for target
                        thickness = 4
                        cv2.putText(display_frame, "TARGET!", (abs_x, abs_y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    elif det['shape'] == 'Uncertain':
                        box_color = (0,255,255)  # Yellow for uncertain
                        thickness = 2
                    else:
                        box_color = (0,255,0)  # Green for others
                        thickness = 2
                    
                    cv2.rectangle(display_frame, (abs_x,abs_y), (abs_x+w, abs_y+h), box_color, thickness)
                    cv2.putText(display_frame, f"ID{det['id']}", (abs_x+5, abs_y+25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

                # Draw target center if detected
                if target_center:
                    cv2.circle(display_frame, target_center, 8, (0,0,255), -1)
                    cv2.circle(display_frame, target_center, 12, (255,255,255), 2)

                # Fire indicator
                if fire_command:
                    cv2.putText(display_frame, "FIRE!", (20, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
                    # Actually fire
                    if gimbal:
                        gimbal.fire_laser(1)
                        print("🔥 FIRED!")

                # Status display
                y_pos = 70
                for obj in details:
                    target_str = " (TARGET!)" if obj['is_target'] else ""
                    line = f"ID {obj['id']}: {obj['color']} {obj['shape']}{target_str}"
                    cv2.putText(display_frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    y_pos += 25
            else:
                cv2.putText(display_frame, "MODE: VIEWING", (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

            # Connection status
            st = "CONNECTED" if manager.connected.is_set() else "DISCONNECTED"
            cv2.putText(display_frame, f"SDK: {st}", (20, 70 if not detection_enabled else 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            cv2.imshow("Integrated Target Tracking System", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                detection_enabled = not detection_enabled
                print(f"Detection {'ON' if detection_enabled else 'OFF'}")
            elif key == ord('f'):
                if gimbal:
                    gimbal.fire_laser(1)
                    print("🔥 Manual fire!")
            elif key == ord('r'):
                if gimbal:
                    print("🎯 Recentering gimbal...")
                    gimbal.recenter(pitch_speed=200, yaw_speed=200).wait_for_completed()
                    print("✅ Gimbal recentered")

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

if __name__ == "__main__":
    main()
