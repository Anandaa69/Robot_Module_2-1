# zone_multithread_gpu.py
import cv2
import numpy as np
import sys
import robomaster
from robomaster import robot, camera as r_camera
import time
import math
import subprocess
import threading

# =====================================================
# GPU Acceleration Check
# =====================================================
USE_GPU = False
try:
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print("✅ NVIDIA GPU detected. Enabling CUDA acceleration.")
        USE_GPU = True
    else:
        print("⚠️ NVIDIA GPU not found or OpenCV not compiled with CUDA. Falling back to CPU.")
except Exception:
    print("⚠️ Could not check for CUDA. Falling back to CPU.")

# =====================================================
# Shared Resources & Threading Locks
# =====================================================
latest_frame = None
processed_output = {"annotated_roi": None, "details": []}
frame_lock = threading.Lock()
output_lock = threading.Lock()
stop_event = threading.Event()

# =====================================================
# Night-robust color pre-processing (CPU Version)
# =====================================================
def apply_awb(bgr):
    # This function relies on cv2.xphoto which may not have a GPU equivalent
    if hasattr(cv2, "xphoto") and hasattr(cv2.xphoto, "createLearningBasedWB"):
        wb = cv2.xphoto.createLearningBasedWB()
        try: wb.setSaturationThreshold(0.99)
        except Exception: pass
        return wb.balanceWhite(bgr)
    return bgr

def night_enhance_pipeline_cpu(bgr):
    den = cv2.fastNlMeansDenoisingColored(bgr, None, 7, 7, 7, 21)
    awb = apply_awb(den)
    # Retinex is computationally expensive, a good candidate for parallelization if needed
    # but let's keep it simple for now.
    return awb # Simplified for real-time performance, can add retinex back if needed

# =====================================================
# Object Detection Logic (CPU and GPU versions)
# =====================================================
class ObjectTracker:
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.stream = cv2.cuda_Stream()
            self.clahe = cv2.cuda.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        print(f"🖼️  ObjectTracker initialized in {'GPU' if use_gpu else 'CPU'} mode.")

    def _get_angle(self, pt1, pt2, pt0): # This is always CPU
        dx1=pt1[0]-pt0[0]; dy1=pt1[1]-pt0[1]
        dx2=pt2[0]-pt0[0]; dy2=pt2[1]-pt0[1]
        dot=dx1*dx2+dy1*dy2
        mag1=math.sqrt(dx1*dx1+dy1*dy1)
        mag2=math.sqrt(dx2*dx2+dy2*dy2)
        if mag1*mag2==0: return 0
        return math.degrees(math.acos(dot/(mag1*mag2)))

    def get_raw_detections(self, frame):
        # This function acts as a dispatcher
        if self.use_gpu:
            frame_gpu = cv2.cuda_GpuMat()
            frame_gpu.upload(frame, self.stream)
            return self._get_raw_detections_gpu(frame_gpu)
        else:
            return self._get_raw_detections_cpu(frame)

    def _get_raw_detections_cpu(self, frame):
        enhanced = cv2.GaussianBlur(night_enhance_pipeline_cpu(frame), (5, 5), 0)
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        
        color_ranges = {
            'Red': ([0, 80, 40], [10, 255, 255], [170, 80, 40], [180, 255, 255]),
            'Yellow': ([20, 60, 40], [35, 255, 255]), 'Green': ([35, 40, 30], [85, 255, 255]),
            'Blue': ([90, 40, 30], [130, 255, 255])
        }
        
        masks = {}
        masks['Red'] = cv2.inRange(hsv, np.array(color_ranges['Red'][0]), np.array(color_ranges['Red'][1])) | \
                       cv2.inRange(hsv, np.array(color_ranges['Red'][2]), np.array(color_ranges['Red'][3]))
        for name in ['Yellow', 'Green', 'Blue']:
            masks[name] = cv2.inRange(hsv, np.array(color_ranges[name][0]), np.array(color_ranges[name][1]))

        combined_mask = masks['Red'] | masks['Yellow'] | masks['Green'] | masks['Blue']
        kernel = np.ones((5, 5), np.uint8)
        cleaned_mask = cv2.morphologyEx(cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
        
        return self._find_and_classify_contours(frame, cleaned_mask, masks)

    def _get_raw_detections_gpu(self, frame_gpu):
        # GPU accelerated pipeline
        bgr_gpu = cv2.cuda.fastNlMeansDenoisingColored(frame_gpu, h=7, hColor=7, templateWindowSize=7, searchWindowSize=21, stream=self.stream)
        blurred_gpu = cv2.cuda.GaussianBlur(bgr_gpu, (5, 5), 0, stream=self.stream)
        hsv_gpu = cv2.cuda.cvtColor(blurred_gpu, cv2.COLOR_BGR2HSV, stream=self.stream)

        color_ranges = {
            'Red': ([0, 80, 40], [10, 255, 255], [170, 80, 40], [180, 255, 255]),
            'Yellow': ([20, 60, 40], [35, 255, 255]), 'Green': ([35, 40, 30], [85, 255, 255]),
            'Blue': ([90, 40, 30], [130, 255, 255])
        }

        # Create GPU masks
        masks_gpu = {}
        red1_gpu = cv2.cuda.inRange(hsv_gpu, tuple(color_ranges['Red'][0]), tuple(color_ranges['Red'][1]), stream=self.stream)
        red2_gpu = cv2.cuda.inRange(hsv_gpu, tuple(color_ranges['Red'][2]), tuple(color_ranges['Red'][3]), stream=self.stream)
        masks_gpu['Red'] = cv2.cuda.bitwise_or(red1_gpu, red2_gpu, stream=self.stream)
        
        for name in ['Yellow', 'Green', 'Blue']:
             masks_gpu[name] = cv2.cuda.inRange(hsv_gpu, tuple(color_ranges[name][0]), tuple(color_ranges[name][1]), stream=self.stream)

        combined_mask_gpu = cv2.cuda.bitwise_or(masks_gpu['Red'], masks_gpu['Yellow'], stream=self.stream)
        combined_mask_gpu = cv2.cuda.bitwise_or(combined_mask_gpu, masks_gpu['Green'], stream=self.stream)
        combined_mask_gpu = cv2.cuda.bitwise_or(combined_mask_gpu, masks_gpu['Blue'], stream=self.stream)

        kernel = np.ones((5, 5), np.uint8)
        morph_open_gpu = cv2.cuda.morphologyEx(combined_mask_gpu, cv2.MORPH_OPEN, kernel)
        cleaned_mask_gpu = cv2.cuda.morphologyEx(morph_open_gpu, cv2.MORPH_CLOSE, kernel)
        
        # Download the final mask to CPU for contour finding
        cleaned_mask_cpu = cleaned_mask_gpu.download()
        
        # Download CPU masks for color classification later
        masks_cpu = {name: m.download() for name, m in masks_gpu.items()}
        frame_cpu = frame_gpu.download()

        return self._find_and_classify_contours(frame_cpu, cleaned_mask_cpu, masks_cpu)

    def _find_and_classify_contours(self, frame, cleaned_mask, masks):
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1500: continue
            
            contour_mask = np.zeros(frame.shape[:2], dtype="uint8")
            cv2.drawContours(contour_mask, [cnt], -1, 255, -1)
            max_mean, found_color = 0, "Unknown"
            for color_name, m in masks.items():
                mean_val = cv2.mean(m, mask=contour_mask)[0]
                if mean_val > max_mean:
                    max_mean, found_color = mean_val, color_name
            if max_mean <= 20: continue
            
            shape = "Uncertain"
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            
            if len(approx) == 4:
                shape = "Quadrilateral" # Simplified for performance
            elif len(approx) > 7 :
                circularity = (4 * np.pi * area) / (peri * peri) if peri > 0 else 0
                if circularity > 0.8: shape = "Circle"
                    
            raw_detections.append({'contour': cnt, 'shape': shape, 'color': found_color})
            
        return raw_detections

# =====================================================
# Thread Functions
# =====================================================
def capture_thread_func(ep_camera):
    global latest_frame
    print("🚀 Capture thread started.")
    while not stop_event.is_set():
        try:
            frame = ep_camera.read_cv2_image(timeout=0.5)
            if frame is not None:
                with frame_lock:
                    latest_frame = frame.copy()
            else:
                time.sleep(0.01) # Avoid busy-waiting
        except Exception as e:
            print(f"Capture thread error: {e}")
            time.sleep(0.1)
    print("🛑 Capture thread stopped.")

def processing_thread_func(tracker, target_shape, target_color, roi_coords, is_detecting_func):
    global processed_output
    print("🧠 Processing thread started.")
    ROI_X, ROI_Y, ROI_W, ROI_H = roi_coords

    while not stop_event.is_set():
        if not is_detecting_func():
            time.sleep(0.1) # Sleep when not detecting
            continue

        with frame_lock:
            if latest_frame is None:
                continue
            frame_to_process = latest_frame.copy()
        
        # Crop ROI for processing
        roi_frame = frame_to_process[ROI_Y : ROI_Y + ROI_H, ROI_X : ROI_X + ROI_W]
        
        # Get detections
        detections = tracker.get_raw_detections(roi_frame)
        
        # --- Drawing Logic (similar to before) ---
        annotated_roi = roi_frame.copy()
        detailed_results = []
        object_id_counter = 1
        
        DIVIDER_X1, DIVIDER_X2 = int(ROI_W * 0.33), int(ROI_W * 0.66)
        cv2.line(annotated_roi, (DIVIDER_X1, 0), (DIVIDER_X1, ROI_H), (255, 255, 0), 2)
        cv2.line(annotated_roi, (DIVIDER_X2, 0), (DIVIDER_X2, ROI_H), (255, 255, 0), 2)

        for det in detections:
            shape, color, contour = det['shape'], det['color'], det['contour']
            x, y, w, h = cv2.boundingRect(contour)
            obj_start_x, obj_end_x = x, x + w

            zone_label = "Center"
            if obj_end_x < DIVIDER_X1: zone_label = "Left"
            elif obj_start_x >= DIVIDER_X2: zone_label = "Right"

            is_target = (shape == target_shape and color == target_color)
            box_color = (0, 0, 255) if is_target else (0, 255, 0)
            thickness = 4 if is_target else 2
            
            detailed_results.append({
                "id": object_id_counter, "color": color, "shape": shape,
                "zone": zone_label, "is_target": is_target
            })
            cv2.rectangle(annotated_roi, (x, y), (x+w, y+h), box_color, thickness)
            cv2.putText(annotated_roi, str(object_id_counter), (x+5, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
            object_id_counter += 1

        with output_lock:
            processed_output = {"annotated_roi": annotated_roi, "details": detailed_results}
            
    print("🛑 Processing thread stopped.")

# =====================================================
# Main Execution Block
# =====================================================
if __name__ == '__main__':
    # --- Setup ---
    target_shape, target_color = "Circle", "Red" # Default values, can be changed
    tracker = ObjectTracker(use_gpu=USE_GPU)
    ep_robot = robot.Robot()
    is_detecting = False
    
    ROI_Y, ROI_H = 264, 270
    ROI_X, ROI_W = 10, 911
    
    try:
        print("\n🤖 Connecting to Robomaster robot...")
        ep_robot.initialize(conn_type="ap")
        print("📷 Starting camera stream (540p)...")
        ep_robot.camera.start_video_stream(display=False, resolution=r_camera.STREAM_540P)
        time.sleep(2) # Wait for camera to stabilize

        # --- Start Threads ---
        capture_thread = threading.Thread(target=capture_thread_func, args=(ep_robot.camera,))
        processing_thread = threading.Thread(target=processing_thread_func, args=(
            tracker, target_shape, target_color, (ROI_X, ROI_Y, ROI_W, ROI_H), lambda: is_detecting
        ))
        capture_thread.start()
        processing_thread.start()

        print("\n--- Real-time Scanner ---")
        print("Press 's' to toggle detection ON/OFF.")
        print("Press 'q' to quit.")
        
        while not stop_event.is_set():
            with frame_lock:
                if latest_frame is None:
                    time.sleep(0.01)
                    continue
                display_frame = latest_frame.copy()

            if is_detecting:
                cv2.putText(display_frame, "MODE: DETECTING", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                with output_lock:
                    annotated_roi = processed_output["annotated_roi"]
                    details = processed_output["details"]
                
                if annotated_roi is not None:
                    display_frame[ROI_Y : ROI_Y + ROI_H, ROI_X : ROI_X + ROI_W] = annotated_roi
                
                if details:
                    y_pos = 70
                    for obj in details:
                        target_str = " (TARGET!)" if obj['is_target'] else ""
                        line = f"ID {obj['id']}: {obj['color']} {obj['shape']}{target_str}"
                        cv2.putText(display_frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4) # Black outline
                        cv2.putText(display_frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        y_pos += 25
            else:
                cv2.putText(display_frame, "MODE: VIEWING", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.rectangle(display_frame, (ROI_X, ROI_Y), (ROI_X + ROI_W, ROI_Y + ROI_H), (255, 0, 0), 2)

            cv2.imshow("Robomaster Real-time Scan", display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("'q' pressed. Shutting down.")
                break
            
            if key == ord('s'):
                is_detecting = not is_detecting
                print(f"Detection toggled {'ON' if is_detecting else 'OFF'}.")

    except Exception as e:
        print(f"❌ An error occurred in the main thread: {e}")
    finally:
        print("\n🔌 Stopping threads and closing connection...")
        stop_event.set()
        capture_thread.join()
        processing_thread.join()
        try:
            cv2.destroyAllWindows()
            ep_robot.camera.stop_video_stream()
            ep_robot.close()
            print("✅ Cleanup complete.")
        except Exception as e:
            print(f"   (Note) Error during cleanup: {e}")