# zone_multithread_stable_fix.py
import cv2
import numpy as np
import sys
import robomaster
from robomaster import robot, camera as r_camera
import time
import math
import subprocess
import threading
import queue 

# =====================================================
# GPU Acceleration Check
# =====================================================
USE_GPU = False
try:
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print("✅ NVIDIA GPU detected and OpenCV has CUDA support. Enabling CUDA acceleration.")
        USE_GPU = True
    else:
        print("⚠️ NVIDIA GPU not found or OpenCV not compiled with CUDA. Falling back to CPU mode.")
except Exception:
    print("⚠️ Could not check for CUDA support. Falling back to CPU mode.")


# =====================================================
# Shared Resources & Threading Primitives
# =====================================================
frame_queue = queue.Queue(maxsize=1) 
processed_output = {"details": []}
output_lock = threading.Lock()
stop_event = threading.Event()


# =====================================================
# ### FIX: Re-added the missing helper functions ###
# =====================================================
def apply_awb(bgr):
    """Applies Auto White Balance. This was the missing function."""
    if hasattr(cv2, "xphoto") and hasattr(cv2.xphoto, "createLearningBasedWB"):
        wb = cv2.xphoto.createLearningBasedWB()
        try: wb.setSaturationThreshold(0.99)
        except Exception: pass
        return wb.balanceWhite(bgr)
    return bgr

def night_enhance_pipeline_cpu(bgr):
    """Full pipeline for CPU-based night enhancement."""
    den = cv2.fastNlMeansDenoisingColored(bgr, None, 7, 7, 7, 21)
    # This function now correctly calls the apply_awb function defined above
    awb = apply_awb(bgr) 
    return awb


# =====================================================
# Object Detection Logic
# =====================================================
class ObjectTracker:
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        if self.use_gpu: self.stream = cv2.cuda_Stream()
        print(f"🖼️  ObjectTracker initialized in {'GPU' if use_gpu else 'CPU'} mode.")
    def _get_angle(self, pt1, pt2, pt0):
        dx1=pt1[0]-pt0[0]; dy1=pt1[1]-pt0[1]; dx2=pt2[0]-pt0[0]; dy2=pt2[1]-pt0[1]
        dot=dx1*dx2+dy1*dy2; mag1=math.sqrt(dx1*dx1+dy1*dy1); mag2=math.sqrt(dx2*dx2+dy2*dy2)
        if mag1*mag2==0: return 0
        return math.degrees(math.acos(dot/(mag1*mag2)))
    def get_raw_detections(self, frame):
        if self.use_gpu:
            frame_gpu = cv2.cuda_GpuMat(); frame_gpu.upload(frame, self.stream)
            return self._get_raw_detections_gpu(frame_gpu)
        else: return self._get_raw_detections_cpu(frame)
    def _get_raw_detections_cpu(self, frame):
        enhanced = cv2.GaussianBlur(night_enhance_pipeline_cpu(frame), (5, 5), 0)
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        color_ranges = {'Red':([0,80,40],[10,255,255],[170,80,40],[180,255,255]),'Yellow':([20,60,40],[35,255,255]),'Green':([35,40,30],[85,255,255]),'Blue':([90,40,30],[130,255,255])}
        masks = {}
        masks['Red'] = cv2.inRange(hsv, np.array(color_ranges['Red'][0]), np.array(color_ranges['Red'][1])) | cv2.inRange(hsv, np.array(color_ranges['Red'][2]), np.array(color_ranges['Red'][3]))
        for name in ['Yellow','Green','Blue']: masks[name] = cv2.inRange(hsv, np.array(color_ranges[name][0]), np.array(color_ranges[name][1]))
        combined_mask = masks['Red'] | masks['Yellow'] | masks['Green'] | masks['Blue']
        kernel = np.ones((5,5), np.uint8)
        cleaned_mask = cv2.morphologyEx(cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
        return self._find_and_classify_contours(frame, cleaned_mask, masks)
    def _get_raw_detections_gpu(self, frame_gpu):
        bgr_gpu = cv2.cuda.fastNlMeansDenoisingColored(frame_gpu, h=7, hColor=7, templateWindowSize=7, searchWindowSize=21, stream=self.stream)
        blurred_gpu = cv2.cuda.GaussianBlur(bgr_gpu, (5, 5), 0, stream=self.stream)
        hsv_gpu = cv2.cuda.cvtColor(blurred_gpu, cv2.COLOR_BGR2HSV, stream=self.stream)
        color_ranges = {'Red':([0,80,40],[10,255,255],[170,80,40],[180,255,255]),'Yellow':([20,60,40],[35,255,255]),'Green':([35,40,30],[85,255,255]),'Blue':([90,40,30],[130,255,255])}
        masks_gpu = {}
        red1_gpu = cv2.cuda.inRange(hsv_gpu, tuple(color_ranges['Red'][0]), tuple(color_ranges['Red'][1]), stream=self.stream)
        red2_gpu = cv2.cuda.inRange(hsv_gpu, tuple(color_ranges['Red'][2]), tuple(color_ranges['Red'][3]), stream=self.stream)
        masks_gpu['Red'] = cv2.cuda.bitwise_or(red1_gpu, red2_gpu, stream=self.stream)
        for name in ['Yellow','Green','Blue']: masks_gpu[name] = cv2.cuda.inRange(hsv_gpu, tuple(color_ranges[name][0]), tuple(color_ranges[name][1]), stream=self.stream)
        combined_mask_gpu = cv2.cuda.bitwise_or(masks_gpu['Red'], masks_gpu['Yellow'], stream=self.stream); combined_mask_gpu = cv2.cuda.bitwise_or(combined_mask_gpu, masks_gpu['Green'], stream=self.stream); combined_mask_gpu = cv2.cuda.bitwise_or(combined_mask_gpu, masks_gpu['Blue'], stream=self.stream)
        kernel = np.ones((5,5), np.uint8)
        morph_open_gpu = cv2.cuda.morphologyEx(combined_mask_gpu, cv2.MORPH_OPEN, kernel); cleaned_mask_gpu = cv2.cuda.morphologyEx(morph_open_gpu, cv2.MORPH_CLOSE, kernel)
        cleaned_mask_cpu = cleaned_mask_gpu.download(); masks_cpu = {name: m.download() for name, m in masks_gpu.items()}; frame_cpu = frame_gpu.download()
        return self._find_and_classify_contours(frame_cpu, cleaned_mask_cpu, masks_cpu)
    def _find_and_classify_contours(self, frame, cleaned_mask, masks):
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_detections = []
        img_h, img_w = frame.shape[:2]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1500: continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w == 0 or h == 0: continue
            aspect_ratio = w / float(h)
            if aspect_ratio > 4.0 or aspect_ratio < 0.25: continue
            hull = cv2.convexHull(cnt); hull_area = cv2.contourArea(hull)
            if hull_area == 0: continue
            solidity = float(area) / hull_area
            if solidity < 0.85: continue
            if x <= 2 or y <= 2 or (x + w) >= (img_w - 2) or (y + h) >= (img_h - 2): continue
            contour_mask = np.zeros(frame.shape[:2], dtype="uint8"); cv2.drawContours(contour_mask, [cnt], -1, 255, -1)
            max_mean, found_color = 0, "Unknown"
            for color_name, m in masks.items():
                mean_val = cv2.mean(m, mask=contour_mask)[0]
                if mean_val > max_mean: max_mean, found_color = mean_val, color_name
            if max_mean <= 20: continue
            shape = "Uncertain"
            peri = cv2.arcLength(cnt, True); approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            circularity = (4 * np.pi * area) / (peri * peri) if peri > 0 else 0
            if circularity > 0.84: shape = "Circle"
            elif len(approx) == 4 and solidity > 0.9:
                points = [tuple(p[0]) for p in approx]
                angles = [self._get_angle(points[(i - 1) % 4], points[(i + 1) % 4], p) for i, p in enumerate(points)]
                if all(75 <= ang <= 105 for ang in angles):
                    _, (rect_w, rect_h), _ = cv2.minAreaRect(cnt)
                    if rect_w == 0 or rect_h == 0: continue
                    aspect_ratio_rect = max(rect_w, rect_h) / min(rect_w, rect_h)
                    if 0.90 <= aspect_ratio_rect <= 1.10: shape = "Square"
                    elif w > h: shape = "Rectangle_H"
                    else: shape = "Rectangle_V"
            raw_detections.append({'contour': cnt, 'shape': shape, 'color': found_color, 'box': (x,y,w,h)})
        return raw_detections

# =====================================================
# Stable Thread Functions using Queue
# =====================================================
def capture_thread_func(ep_camera, q):
    print("🚀 Capture thread started.")
    while not stop_event.is_set():
        try:
            frame = ep_camera.read_cv2_image(timeout=1.0)
            if frame is not None:
                if q.full():
                    q.get_nowait() 
                q.put(frame)
            else:
                time.sleep(0.01)
        except Exception as e:
            print(f"CRITICAL: Capture thread encountered an error: {e}. Will retry...")
            time.sleep(0.5) 
    print("🛑 Capture thread stopped.")

def processing_thread_func(tracker, q, target_shape, target_color, roi_coords, is_detecting_func):
    global processed_output
    print("🧠 Processing thread started.")
    ROI_X, ROI_Y, ROI_W, ROI_H = roi_coords

    while not stop_event.is_set():
        if not is_detecting_func():
            time.sleep(0.1)
            continue
        try:
            frame_to_process = q.get(timeout=1.0)
            
            roi_frame = frame_to_process[ROI_Y : ROI_Y + ROI_H, ROI_X : ROI_X + ROI_W]
            detections = tracker.get_raw_detections(roi_frame)
            
            detailed_results = []
            object_id_counter = 1
            DIVIDER_X1, DIVIDER_X2 = int(ROI_W * 0.33), int(ROI_W * 0.66)

            for det in detections:
                shape, color, (x, y, w, h) = det['shape'], det['color'], det['box']
                obj_end_x = x + w
                zone_label = "Center"
                if obj_end_x < DIVIDER_X1: zone_label = "Left"
                elif x >= DIVIDER_X2: zone_label = "Right"
                is_target = (shape == target_shape and color == target_color)
                detailed_results.append({
                    "id": object_id_counter, "color": color, "shape": shape,
                    "zone": zone_label, "is_target": is_target, "box": (x,y,w,h)
                })
                object_id_counter += 1
            
            with output_lock:
                processed_output = {"details": detailed_results}

        except queue.Empty:
            print("Processing thread: No new frame in queue.")
            continue
        except Exception as e:
            print(f"CRITICAL: Processing thread encountered an error: {e}")
            time.sleep(0.1)

    print("🛑 Processing thread stopped.")

# =====================================================
# Main Thread
# =====================================================
if __name__ == '__main__':
    target_shape, target_color = "Circle", "Red" 
    print(f"🎯 Target set to: {target_color} {target_shape}")

    tracker = ObjectTracker(use_gpu=USE_GPU)
    ep_robot = robot.Robot()
    is_detecting = False
    
    ROI_Y, ROI_H, ROI_X, ROI_W = 264, 270, 10, 911
    
    try:
        print("\n🤖 Connecting to Robomaster robot...")
        ep_robot.initialize(conn_type="ap")
        print("📷 Starting camera stream (540p)...")
        ep_robot.camera.start_video_stream(display=False, resolution=r_camera.STREAM_540P)
        time.sleep(2)

        capture_thread = threading.Thread(target=capture_thread_func, args=(ep_robot.camera, frame_queue))
        processing_thread = threading.Thread(target=processing_thread_func, args=(
            tracker, frame_queue, target_shape, target_color, (ROI_X, ROI_Y, ROI_W, ROI_H), lambda: is_detecting
        ))
        
        capture_thread.start()
        processing_thread.start()

        print("\n--- Real-time Scanner (Stable Version) ---")
        print("Press 's' to toggle detection ON/OFF.")
        print("Press 'q' to quit.")
        
        display_frame = None
        while not stop_event.is_set():
            try:
                display_frame = frame_queue.get(timeout=1.0)
            except queue.Empty:
                if display_frame is None: 
                    print("Waiting for first frame from camera...")
                    time.sleep(0.5)
                    continue
            
            cv2.rectangle(display_frame, (ROI_X, ROI_Y), (ROI_X + ROI_W, ROI_Y + ROI_H), (255, 0, 0), 2)
            if is_detecting:
                cv2.putText(display_frame, "MODE: DETECTING", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                with output_lock:
                    details = processed_output["details"]
                
                divider1_abs = ROI_X + int(ROI_W * 0.33); divider2_abs = ROI_X + int(ROI_W * 0.66)
                cv2.line(display_frame, (divider1_abs, ROI_Y), (divider1_abs, ROI_Y + ROI_H), (255, 255, 0), 1)
                cv2.line(display_frame, (divider2_abs, ROI_Y), (divider2_abs, ROI_Y + ROI_H), (255, 255, 0), 1)

                for det in details:
                    x, y, w, h = det['box']; abs_x, abs_y = x + ROI_X, y + ROI_Y
                    box_color = (0, 0, 255) if det['is_target'] else (0, 255, 255) if det['shape'] == 'Uncertain' else (0, 255, 0)
                    thickness = 4 if det['is_target'] else 2
                    cv2.rectangle(display_frame, (abs_x, abs_y), (abs_x + w, abs_y + h), box_color, thickness)
                    cv2.putText(display_frame, str(det['id']), (abs_x + 5, abs_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

                if details:
                    y_pos = 70
                    for obj in details:
                        target_str = " (TARGET!)" if obj['is_target'] else ""
                        line = f"ID {obj['id']}: {obj['color']} {obj['shape']}{target_str}"
                        cv2.putText(display_frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                        cv2.putText(display_frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        y_pos += 25
            else:
                cv2.putText(display_frame, "MODE: VIEWING", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow("Robomaster Real-time Scan (Stable)", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            if key == ord('s'): is_detecting = not is_detecting; print(f"Detection toggled {'ON' if is_detecting else 'OFF'}.")

    except Exception as e:
        print(f"❌ An error occurred in the main thread: {e}")
    finally:
        print("\n🔌 Stopping threads and closing connection...")
        stop_event.set()
        capture_thread.join(); processing_thread.join()
        try:
            cv2.destroyAllWindows(); ep_robot.camera.stop_video_stream(); ep_robot.close()
            print("✅ Cleanup complete.")
        except Exception as e: print(f"   (Note) Error during cleanup: {e}")